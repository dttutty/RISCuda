"""Qwenputer BF16 language stack -> native C calc -> Qwen, one host launch.

Use the existing environment with Torch, Transformers and Qwenputer's dependencies.
No per-token Python runs between submission and completion. CPU setup/capture and
post-run reference checks are deliberately outside the profiled request.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from benchmarks.compare_qwenputer import CCalculator, Capture, native_view
from gpu_rv32.resident import ResidentLoop, library


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


class Request:
    def __init__(self, chat, view, question, maximum):
        import torch
        from transformers import StaticCache
        from chipformer.rv.evalarith import NUDGE
        from chipformer.q35.evals import SUFFIX
        self.chat = chat
        self.messages = [{'role':'system','content':NUDGE}, {'role':'user','content':question+SUFFIX}]
        self.ids = chat.prompt_ids(self.messages)
        self.prompt = torch.tensor([self.ids], device='cuda')
        self.length = len(self.ids)
        self.cache_length = self.length+maximum
        self.cache = StaticCache(config=chat.model.config, max_cache_len=self.cache_length)
        b = {'input':torch.zeros((1,1), dtype=torch.int64, device='cuda'),
             'next':torch.zeros(1, dtype=torch.int64, device='cuda'),
             'position':torch.zeros((1,1), dtype=torch.int64, device='cuda'),
             'mask':torch.full((1,1,1,self.cache_length), torch.finfo(torch.bfloat16).min,
                               dtype=torch.bfloat16, device='cuda')}
        b['mask'][...,0] = 0
        flat, offsets = bytearray(), [0]
        for token in chat.T.bytes[:chat.v0]:
            flat.extend(token)
            offsets.append(len(flat))
        for name, data, dtype in (('bytes',list(flat),torch.uint8),('offsets',offsets,torch.int32),
                                   ('printable',chat.T.printable,torch.int32)):
            b[name] = torch.tensor(data, dtype=dtype, device='cuda')
        self.buffers = b
        self.decode = torch.cuda.CUDAGraph(keep_graph=True)
        self.prefill = torch.cuda.CUDAGraph(keep_graph=True)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        # Retain externally allocated inputs for the lifetime of the graph.
        self.prefill_position = torch.arange(self.length, device='cuda').unsqueeze(0)
        self.prefill_mask = torch.where(
            torch.arange(self.cache_length, device='cuda')[None,:] <= self.prefill_position[0,:,None],
            0., torch.finfo(torch.bfloat16).min)[None,None,:,:].to(torch.bfloat16)
        with torch.cuda.stream(self.stream):
            def decode():
                out = view(input_ids=b['input'], position_ids=b['position'],
                    attention_mask={'full_attention':b['mask'],'linear_attention':None},
                    past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                torch.argmax(out.logits[:,-1,:chat.v0], dim=-1, out=b['next'])
            def prefill():
                out = chat.model(input_ids=self.prompt, position_ids=self.prefill_position,
                    attention_mask={'full_attention':self.prefill_mask,'linear_attention':None},
                    past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                torch.argmax(out.logits[:,-1,:chat.v0], dim=-1, out=b['next'])
            for _ in range(2):
                decode()
            self.stream.synchronize()
            with torch.cuda.graph(self.decode, stream=self.stream):
                decode()
            self.cache.reset()
            prefill()  # Initialize all stock chip/language prefill cache layers before capture.
            self.stream.synchronize()
            self.cache.reset()
            with torch.cuda.graph(self.prefill, stream=self.stream):
                prefill()
            self.cache.reset()
        self.stream.synchronize()

    def run(self, image, maximum, *, host_loop=False, profile=False):
        import torch
        self.cache.reset()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.stream.synchronize()
        eos = sorted(self.chat.eos)
        with ResidentLoop(self.prefill, self.decode, self.buffers, image,
                vocab=self.chat.v0, prompt_length=self.length, cache_length=self.cache_length,
                eos=eos[0], eos2=eos[1], tco=self.chat.T.TCO, tcc=self.chat.T.TCC,
                max_generated=maximum, max_total=maximum) as loop:
            state = loop.run(self.stream, host_loop=host_loop, profile=profile)
            elapsed, launches = loop.request_seconds, loop.host_graph_launches
        return state, elapsed, launches


def record(chat, state, seconds, launches, question):
    from chipformer.rv import calc
    tokens = list(state.tokens[:state.total])
    generated = [q for q,f in zip(tokens,state.forced) if not f]
    tools = []
    for t in state.tools[:state.calls]:
        expression, output = t.expression.decode(), t.output.decode()
        reference = 'error (too long: split it)\n' if t.gather_error == 2 else 'error\n' if t.gather_error else calc.expected(expression)
        tools.append(dict(expression=expression, output=output, status=t.status, error=t.error,
            steps=t.steps, bigint_calls=t.bigint_calls, gather_error=t.gather_error,
            elapsed_ns=t.elapsed_ns, reference=reference, reference_equal=output==reference))
    # Replay the generated stream through the stock bridge parser after completion.
    calls = iter(tools)
    capture = Capture(chat.T, lambda expression: next(calls)['output'])
    forced = []
    for q, f in zip(tokens,state.forced):
        if f:
            if not forced or q != forced.pop(0):
                raise AssertionError('device feedback tokens differ from the stock protocol')
        else:
            if forced:
                raise AssertionError('model resumed before feedback finished')
            forced.extend(capture.feed(q))
    if forced and state.stop == 1:
        raise AssertionError('EOS before feedback completed')
    if [t['expression'] for t in capture.responses] != [t['expression'] for t in tools]:
        raise AssertionError('device parameter gathering differs from the stock protocol')
    return dict(question=question, seconds=seconds, normal_eos=state.stop==1, stop=state.stop,
        host_graph_launches=launches, text=chat.text_of(tokens), generated_text=chat.text_of(generated),
        generated=state.generated, injected=state.injected, total=state.total, forwards=state.forwards,
        tokens=tokens, forced=list(state.forced[:state.total]), tools=tools, feedback_protocol_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwenputer',type=Path,default=ROOT.parent/'qwenputer')
    parser.add_argument('--model',type=Path,default=ROOT.parent/'m5gp/examples/qwen_rv32/models/Qwen3.5-4B-Qwenputer')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['small','big','chain','all'],default='all')
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--host-control',action='store_true')
    parser.add_argument('--profile',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        parser.error('--max-tokens must be in 1..4096')
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.qwenputer/'experimental/riscv'))
    import torch
    from chipformer.rv.chat import RvChat
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
        model=str(args.model.resolve()), dtype='bfloat16', thinking=False, batch_size=1,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','triton','ziglang')},
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.qwenputer,text=True).strip(),
        checkpoint_manifest=json.loads((args.model/'download_manifest.json').read_text()),
        checkpoint_metadata_sha256={name:hashlib.sha256((args.model/name).read_bytes()).hexdigest()
            for name in ('config.json','chipformer.json','tokenizer.json','model.safetensors.index.json')},
        reference_firmware_sha256=hashlib.sha256((args.qwenputer/'experimental/riscv/tests/data/rv/calc-bridge.elf').read_bytes()).hexdigest(),
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).splitlines(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),ROOT/'src/gpu_rv32/resident.py',ROOT/'src/gpu_rv32/native/resident.cu',
             ROOT/'src/gpu_rv32/native/include/resident.h',ROOT/'src/gpu_rv32/native/include/rv32_core.cuh',
             ROOT/'src/gpu_rv32/native/include/bigint.cuh',ROOT/'benchmarks/compare_qwenputer.py']})
    save(args.output/'metadata.json',metadata)
    library()
    image = CCalculator().image
    save(args.output/'guest_metadata.json',image.metadata)
    save(args.output/'guest_hashes.json',dict(code_sha256=hashlib.sha256(image.code).hexdigest(),
        memory_sha256=hashlib.sha256(image.memory).hexdigest(),entry=image.entry,symbols=image.symbols))
    print('Loading shared BF16 Qwenputer language weights',flush=True)
    chat = RvChat(str(args.model),device='cuda',fast=False,executor=False)
    chat.d0 = chat.meta['chip_dims_offset']
    view = native_view(chat)
    cases = {
        'small':('What is 2*8?',16),
        'big':('What is 12345678901234567890*98765432109876543210?',12345678901234567890*98765432109876543210),
        'chain':('First use calc to multiply 123456789 by 987654321. After receiving that result, make a second, separate calc call adding 12345 to the returned integer. Do not combine these into one call. Return the final result.',123456789*987654321+12345)}
    results = []
    with torch.inference_mode():
        for label,(question,expected) in cases.items():
            if args.case not in ('all',label):
                continue
            print(f'Capturing {label}',flush=True)
            request = Request(chat,view,question,args.max_tokens)
            save(args.output/f'{label}_prompt.json',dict(messages=request.messages,tokens=request.ids))
            # Warm the executable, language operations and VM without profiling.
            warm, _, _ = request.run(image,args.max_tokens)
            print(f'Warmup {label}: {warm.calls} calls, stop {warm.stop}',flush=True)
            arms = ['host','gpu'] if args.host_control else ['gpu']
            pair = {}
            for arm in arms:
                state, seconds, launches = request.run(image,args.max_tokens,
                    host_loop=arm=='host',profile=args.profile and arm=='gpu')
                r = record(chat,state,seconds,launches,question)
                r.update(case=label,arm=arm,expected=str(expected))
                from chipformer.q35 import arith_bench as AB
                r.update(AB.exact_fields({'gold':str(expected)},r['text']))
                from benchmarks.audit_qwenputer import verify_stream
                r['reference_trace'] = verify_stream(chat.T,
                    args.qwenputer/'experimental/riscv/tests/data/rv/calc-bridge.elf',
                    request.ids,r['tokens'],chip=False,base=chat.v0)
                (args.output/f'{label}_{arm}.state.bin').write_bytes(bytes(state))
                save(args.output/f'{label}_{arm}.json',r)
                print(json.dumps({k:r[k] for k in ('case','arm','seconds','normal_eos','exact','generated','injected','host_graph_launches')}
                    | {'calls':len(r['tools']),'tools':[(t['expression'],t['output']) for t in r['tools']]},ensure_ascii=False),flush=True)
                if not r['normal_eos'] or not r['exact'] or not r['tools'] or not all(t['reference_equal'] and not t['status'] and not t['error'] for t in r['tools']):
                    raise AssertionError('request correctness gate failed; inspect saved transcript')
                if not r['reference_trace']['ok'] or r['reference_trace']['ended_inside_tool']:
                    raise AssertionError('original CPU reference firmware rejected the device trace')
                if label=='chain' and len(r['tools'])<2:
                    raise AssertionError('chain test did not exercise multiple tool calls')
                pair[arm] = r
                results.append(r)
            if len(pair)==2 and pair['gpu']['tokens'] != pair['host']['tokens']:
                raise AssertionError('GPU-controlled and host-controlled token streams differ')
            del request
            torch.cuda.empty_cache()
    save(args.output/'summary.json',results)
    save(args.output/'exit.json',dict(exit_status=0,finished_at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    main()
