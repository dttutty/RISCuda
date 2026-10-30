"""Public 5x40 arithmetic evaluation: one host graph/request vs stock Qwenputer.

Language parameters, precision, prompt IDs and calc protocol are shared. The
native arm uses static captured inference and a conditional GPU controller;
the baseline retains the author's eager language/Triton circuit driver.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]
from benchmarks.compare_qwenputer import CCalculator, native_view, save, sha, summary
from examples.autonomous_qwen_calc import Request, record

OUTPUT = None
DATASETS = ['arith','dmx_mul_big','dmx_div_big','dmx_mul_div_multiple_longer','dmx_add_or_sub_big']


class PublicRequest(Request):
    """Capture stock-length prefill, then full-capacity masked decode.

    Prefill must not attend over a padded static KV extent: its reduction shape
    and backend should match the original dynamic-cache prompt computation.
    The backing KV storage remains static; only the returned prefill view trims.
    """
    def __init__(self,chat,view,question,maximum):
        import torch
        from transformers import StaticCache
        from chipformer.rv.evalarith import NUDGE
        from chipformer.q35.evals import SUFFIX

        class PrefillCache(StaticCache):
            def update(self,key_states,value_states,layer_idx,*args,**kwargs):
                keys,values=super().update(key_states,value_states,layer_idx,*args,**kwargs)
                length=key_states.shape[-2]
                return (keys[:,:,:length],values[:,:,:length]) if length>1 else (keys,values)

        self.chat=chat
        self.messages=[{'role':'system','content':NUDGE},{'role':'user','content':question+SUFFIX}]
        self.ids=chat.prompt_ids(self.messages)
        self.prompt=torch.tensor([self.ids],device='cuda')
        self.length=len(self.ids)
        self.cache_length=self.length+maximum
        self.cache=PrefillCache(config=chat.model.config,max_cache_len=self.cache_length)
        b={'input':torch.zeros((1,1),dtype=torch.int64,device='cuda'),
           'next':torch.zeros(1,dtype=torch.int64,device='cuda'),
           'position':torch.zeros((1,1),dtype=torch.int64,device='cuda'),
           'mask':torch.full((1,1,1,self.cache_length),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16,device='cuda')}
        b['mask'][...,0]=0
        flat,offsets=bytearray(),[0]
        for token in chat.T.bytes[:chat.v0]:
            flat.extend(token);offsets.append(len(flat))
        for name,data,dtype in [('bytes',list(flat),torch.uint8),('offsets',offsets,torch.int32),('printable',chat.T.printable,torch.int32)]:
            b[name]=torch.tensor(data,dtype=dtype,device='cuda')
        self.buffers=b
        self.prefill_position=torch.arange(self.length,device='cuda').unsqueeze(0)
        self.decode=torch.cuda.CUDAGraph(keep_graph=True)
        self.prefill=torch.cuda.CUDAGraph(keep_graph=True)
        self.stream=torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            def decode():
                out=view(input_ids=b['input'],position_ids=b['position'],
                    attention_mask={'full_attention':b['mask'],'linear_attention':None},
                    past_key_values=self.cache,use_cache=True,logits_to_keep=1)
                torch.argmax(out.logits[:,-1,:chat.v0],dim=-1,out=b['next'])
            def prefill():
                out=chat.model(input_ids=self.prompt,position_ids=self.prefill_position,
                    attention_mask={'full_attention':None,'linear_attention':None},
                    past_key_values=self.cache,use_cache=True,logits_to_keep=1)
                torch.argmax(out.logits[:,-1,:chat.v0],dim=-1,out=b['next'])
                return out.logits
            for _ in range(2):decode()
            self.stream.synchronize()
            with torch.cuda.graph(self.decode,stream=self.stream):decode()
            self.cache.reset()
            actual=prefill()
            reference=chat.model(input_ids=self.prompt,use_cache=True,logits_to_keep=1).logits
            self.prefill_gate=dict(bit_equal=bool(torch.equal(actual,reference)),
                max_error=float((actual.float()-reference.float()).abs().max()))
            if not self.prefill_gate['bit_equal']:
                raise AssertionError(f'Captured prefill differs from original prompt computation: {self.prefill_gate}')
            del actual,reference
            self.cache.reset()
            with torch.cuda.graph(self.prefill,stream=self.stream):prefill()
            self.cache.reset()
        self.stream.synchronize()


def bind_prompt(request, messages, ids):
    """Change only fixed-shape input storage, outside the timed request."""
    import torch
    if len(ids) != request.length:
        raise ValueError('captured prefill requires identical token length')
    request.prompt.copy_(torch.tensor([ids],device='cuda'))
    request.ids, request.messages = ids, messages


def capture_groups(prepared):
    groups = defaultdict(list)
    for item in prepared:
        groups[len(item['prompt_ids'])].append(item)
    return [(length,groups[length]) for length in sorted(groups)]


def main():
    global OUTPUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qwenputer',type=Path,default=ROOT.parent/'qwenputer')
    ap.add_argument('--model',type=Path,default=ROOT.parent/'m5gp/examples/qwen_rv32/models/Qwen3.5-4B-Qwenputer')
    ap.add_argument('--data',type=Path,default=ROOT/'runs/qwenputer_math_data')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--shard',default='0/1')
    ap.add_argument('--budget',type=int,default=4096)
    ap.add_argument('--max-chip',type=int,default=200000)
    ap.add_argument('--n',type=int,default=40)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--ids',nargs='+',help='explicit smoke/debug subset; saved as such')
    ap.add_argument('--arms',nargs='+',choices=['riscuda','qwenputer'],default=['riscuda','qwenputer'])
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--profile',action='store_true',help='separate single-ID native-only Nsight capture')
    args = ap.parse_args()
    if not 1 <= args.budget <= 4096:
        ap.error('budget must be in 1..4096')
    if args.profile and (not args.ids or len(args.ids)!=1 or args.arms!=['riscuda']):
        ap.error('--profile requires one --ids value and --arms riscuda')
    shard_i,shard_k = map(int,args.shard.split('/'))
    if not 0 <= shard_i < shard_k:
        ap.error('invalid shard')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    OUTPUT = args.output
    sys.path.insert(0,str(args.qwenputer/'experimental/riscv'))
    import torch
    from chipformer.rv.chat import RvChat
    from chipformer.rv import bridge as BR,elf,ref
    from chipformer.rv.evalarith import NUDGE
    from chipformer.q35 import arith_bench as AB,evals as E
    AB.DATA = args.data
    canonical = []
    for dataset in DATASETS:
        selected = E.load_data('arith',args.n,args.seed) if dataset=='arith' else AB.load(dataset,args.n,args.seed)
        canonical.extend(dict(p,dataset=dataset) for p in selected)
    selected = canonical[shard_i::shard_k]
    if args.ids:
        selected = [p for p in selected if p['id'] in args.ids]
        if {p['id'] for p in selected} != set(args.ids):
            ap.error('some --ids are absent from this shard')
    save(args.output/'all_problems.json',canonical)
    save(args.output/'problems.json',selected)
    sources = [Path(__file__),ROOT/'examples/autonomous_qwen_calc.py',ROOT/'benchmarks/compare_qwenputer.py',
        *[ROOT/'src/gpu_rv32'/p for p in ('resident.py','native/resident.cu','native/include/resident.h',
            'native/include/rv32_core.cuh','native/include/bigint.cuh')],
        ROOT/'benchmarks/qwenputer_calc/calc.c',ROOT/'benchmarks/qwenputer_calc/port.h']
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
        qwenputer_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.qwenputer,text=True).strip(),
        model=json.loads((args.model/'download_manifest.json').read_text()),dtype='bfloat16',thinking=False,
        greedy=True,batch=1,seed=args.seed,n=args.n,budget=args.budget,max_chip=args.max_chip,
        arms=args.arms,profiled=args.profile,backend='c',native_control='single conditional CUDA graph, including stock prefill',
        baseline_control='upstream host text loop + Triton circuit executor',
        shard=args.shard,ids=args.ids,problem_count=len(selected),
        problem_sha256=sha(args.output/'problems.json'),all_problems_sha256=sha(args.output/'all_problems.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','triton','ziglang')},
        cpu_affinity=sorted(os.sched_getaffinity(0)),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv'],text=True),
        timing_scope='Actual prefill through EOS/cutoff, including tools and feedback; excludes load/build/capture/graph instantiation/cache reset/final trace download/reference checks.',
        limitations=['The comparison includes static captured inference and native tools, not just removal of CPU dispatch.',
            'Exact-length groups reuse captured prefill inputs without padding; one private static cache per active group.',
            'Original Qwenputer remains host-controlled; no claim that its in-weight CPU was converted to a single graph.',
            'One measurement per question/arm, batch one; optional causal-conv1d and FLA absent.'])
    for name in ('download_manifest.json','modelscope_identity.json'):
        metadata[name]=json.loads((args.data/name).read_text())
    if args.resume and (args.output/'metadata.json').exists():
        prior=json.loads((args.output/'metadata.json').read_text())
        for key in ('model','problem_sha256','source_sha256','budget','max_chip','arms','shard','cuda_visible_devices'):
            if prior[key]!=metadata[key]:
                raise ValueError('resume mismatch: '+key)
        metadata['started_at']=prior['started_at']
    save(args.output/'metadata.json',metadata)
    records_path=args.output/'records.jsonl'
    records=[json.loads(l) for l in records_path.read_text().splitlines()] if records_path.exists() else []
    done={(r['id'],r['arm']) for r in records}
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    calculator=CCalculator()
    image=calculator.image
    save(args.output/'guest_build.json',image.metadata)
    (args.output/'guest.elf').write_bytes(image.elf)
    assert sha(args.qwenputer/'experimental/riscv/chipformer/rv/env/calc/calc.c')==sha(ROOT/'benchmarks/qwenputer_calc/calc.c')
    print('Loading shared BF16 Qwenputer checkpoint',flush=True)
    chat=RvChat(str(args.model),device='cuda',fast=False,executor='qwenputer' in args.arms)
    chat.d0=chat.meta['chip_dims_offset']
    view=native_view(chat)
    firmware=args.qwenputer/'experimental/riscv/tests/data/rv/calc-bridge.elf'

    def messages_for(question):
        return [{'role':'system','content':NUDGE},{'role':'user','content':question+E.SUFFIX}]

    def baseline(ids):
        machine=ref.Machine.from_program(elf.program(str(firmware)))
        BR.park(machine)
        torch.cuda.synchronize()
        started=time.perf_counter()
        r=chat.generate(ids,machine,max_new=args.budget,max_chip=args.max_chip,lock=None)
        torch.cuda.synchronize()
        r['wall_seconds']=time.perf_counter()-started
        r['normal_eos']=bool(r['tokens']) and r['tokens'][-1] in chat.eos
        return r

    with torch.inference_mode():
        eq_ids=chat.prompt_ids([{'role':'user','content':'Reply with OK.'}],tools=())
        x=torch.tensor([eq_ids],device='cuda')
        a=chat.model(input_ids=x,use_cache=False).logits[...,:chat.v0]
        b=view(input_ids=x,use_cache=False).logits[...,:chat.v0]
        assert torch.equal(a,b),'Language parameter/view gate failed'
        save(args.output/'language_equivalence.json',dict(bit_equal=True,language_layers_shared=True,head_shared=True))
        del x,a,b
        # Exercise reused captured inputs with distinct same-length real questions.
        warm=PublicRequest(chat,view,'What is 2*8?',args.budget)
        warm_records=[]
        for question,gold in [('What is 2*8?','16'),('What is 3*9?','27')]:
            messages=messages_for(question)
            ids=chat.prompt_ids(messages)
            bind_prompt(warm,messages,ids)
            s,seconds,launches=warm.run(image,args.budget)
            r=record(chat,s,seconds,launches,question)
            assert r['normal_eos'] and r['host_graph_launches']==1
            assert AB.exact_fields({'gold':gold},r['text'])['exact']
            assert len(r['tools'])==1 and r['tools'][0]['reference_equal']
            warm_records.append(dict(question=question,prompt_ids=ids,result=r))
        save(args.output/'warmup_rebinding.json',warm_records)
        del warm
        torch.cuda.empty_cache()
        if 'qwenputer' in args.arms:
            assert baseline(chat.prompt_ids(messages_for('What is 2*8?')))['normal_eos']
        prepared=[]
        for p in selected:
            messages=messages_for(p['question'])
            prepared.append(dict(problem=p,messages=messages,prompt_ids=chat.prompt_ids(messages)))
        groups=capture_groups(prepared)
        save(args.output/'capture_plan.json',[dict(length=n,ids=[p['problem']['id'] for p in rows]) for n,rows in groups])
        index=0
        for length,items in groups:
            if all((item['problem']['id'],arm) in done for item in items for arm in args.arms):
                index+=len(items)
                continue
            print(f'CAPTURE prompt length {length}: {len(items)} problems',flush=True)
            request=PublicRequest(chat,view,items[0]['problem']['question'],args.budget)
            save(args.output/f'prefill_gate_{length}.json',request.prefill_gate)
            for item in items:
                p,messages,ids=item['problem'],item['messages'],item['prompt_ids']
                bind_prompt(request,messages,ids)
                order=args.arms if index%2==0 else list(reversed(args.arms))
                index+=1
                for arm in order:
                    if (p['id'],arm) in done:
                        continue
                    print(f'START {index}/{len(selected)} {p["id"]} {arm}',flush=True)
                    if arm=='riscuda':
                        state,seconds,launches=request.run(image,args.budget,profile=args.profile)
                        # Persist state before scoring so a diagnostic survives errors.
                        (args.output/f'{p["id"]}.riscuda.state.bin').write_bytes(bytes(state))
                        r=record(chat,state,seconds,launches,p['question'])
                        assert launches==1
                        r.update(wall_seconds=seconds,text_tokens=state.total-int(state.stop==1),chip_tokens=0,
                            gpu_tools=r['tools'],text_seconds=None,chip_seconds=0.)
                    else:
                        r=baseline(ids)
                        r['text']=chat.text_of(r['tokens'])
                    import re
                    responses=re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>',r['text'],re.S)
                    row=dict(id=p['id'],dataset=p['dataset'],arm=arm,question=p['question'],gold=p['gold'],
                        **AB.exact_fields(p,r['text']),normal_eos=r['normal_eos'],wall_seconds=r['wall_seconds'],
                        prefill_tokens=len(ids),text_tokens=r['text_tokens'],chip_tokens=r['chip_tokens'],
                        calls=len(responses),responses=responses,text=r['text'],
                        tool_seconds=sum(t['elapsed_ns'] for t in r['tools'])/1e9 if arm=='riscuda' else r['chip_seconds'],
                        text_seconds=r.get('text_seconds'),host_graph_launches=r.get('host_graph_launches'),
                        result_file=f'{p["id"]}.{arm}.json')
                    save(args.output/row['result_file'],dict(r,prompt_ids=ids,messages=messages))
                    with records_path.open('a') as f:
                        f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    records.append(row)
                    done.add((p['id'],arm))
                    save(args.output/'summary.json',summary(records))
                    print(json.dumps({k:row[k] for k in ('id','arm','exact','normal_eos','wall_seconds','calls','text_tokens','chip_tokens','host_graph_launches')}),flush=True)
            del request
            torch.cuda.empty_cache()
    save(args.output/'summary.json',summary(records))
    print('ALL COMPLETE',flush=True)


if __name__=='__main__':
    status=0
    try:
        main()
    except BaseException:
        status=1
        raise
    finally:
        if OUTPUT is not None:
            (OUTPUT/'exit_status').write_text(str(status)+'\n')
