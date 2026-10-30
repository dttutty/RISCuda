"""Paired public Qwenputer arithmetic evaluation, shared BF16 language weights.

The RISCuda arm runs the same calc protocol through the same C calculator
and GPU-native decimal primitives (or optionally real GPU MicroPython).
Both arms deliberately retain host token loops: this isolates runtime/tool
replacement from a claim about CPU-free end-to-end serving. Timed requests
include prefill and feedback; model/build/warmup and reference checks excluded.
"""
import argparse
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=None
sys.path.insert(0,str(ROOT/'src'))

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class GPUCalculator:
    def __init__(self):
        from gpu_rv32 import compile_micropython
        self.prefix=(ROOT/'benchmarks/exact_calc_guest.py').read_text()
        self.image=compile_micropython(self.prefix+'\nprint(calculate("2*8"))\n')
        self.calls=[]
    def __call__(self,expression):
        from gpu_rv32 import run_micropython
        source=self.prefix+'\ntry:\n    print(calculate('+repr(expression)+'))\nexcept Exception:\n    print("error")\n'
        data=source.encode()
        if len(data)>16383:
            raise ValueError('guest input exceeds source buffer')
        ram=bytearray(self.image.memory)
        at=self.image.symbols['mp_source']
        ram[at:at+16384]=data+b'\0'*(16384-len(data))
        started=time.perf_counter()
        result=run_micropython(replace(self.image,memory=bytes(ram)),instruction_budget=20000000)
        wall=time.perf_counter()-started
        result.update(expression=expression,tool_wall_seconds=wall)
        self.calls.append(result)
        if result['status'] or result['error']:
            return 'error\n'
        return result['stdout']

class CCalculator:
    def __init__(self):
        from gpu_rv32 import compile_c
        port=ROOT/'benchmarks/qwenputer_calc'
        original=(port/'calc.c').read_text()
        assert original.count('#include "bn.h"')==original.count('#include "rt.h"')==1
        source=(port/'port.h').read_text()+'\n'+original.replace('#include "bn.h"','').replace('#include "rt.h"','')
        self.image=compile_c(source)
        self.calls=[]
    def __call__(self,expression):
        from gpu_rv32 import run_c
        encoded=expression.encode()
        ram=bytearray(self.image.memory)
        at=self.image.symbols['calc_in']
        assert len(encoded)<256
        ram[at:at+256]=encoded+b'\0'*(256-len(encoded))
        started=time.perf_counter()
        result=run_c(replace(self.image,memory=bytes(ram)),instruction_budget=20000000)
        result.update(expression=expression,tool_wall_seconds=time.perf_counter()-started)
        self.calls.append(result)
        if result['status'] or result['error']: return 'error\n'
        return result['stdout']

class Capture:
    """The stock bridge's 80 token rows and first-parameter gather, no arithmetic."""
    def __init__(self,T,calculator):
        self.T,self.calculator=T,calculator
        self.armed=False
        self.rows=[b'\0'*16]*81
        self.cap=0
        self.responses=[]
    def feed(self,t):
        if t==self.T.TCO:
            self.armed=True
            self.cap=0
            self.rows=[b'\0'*16]*81
        elif t==self.T.TCC:
            self.armed=False
            n,gt,ended=bytearray(),0,False
            for row in self.rows[1:]:
                if not row[0]:
                    ended=True
                    break
                for c in row.split(b'\0',1)[0]:
                    if gt<2:
                        gt+=c==62
                    elif c==60:
                        ended=True
                        break
                    elif c!=10 and len(n)<255:
                        n.append(c)
                if ended: break
            expression=n.decode('utf-8','replace')
            response=('error\n' if gt<2 else 'error (too long: split it)\n' if not ended
                      else self.calculator(expression))
            self.responses.append({'expression':expression,'response':response})
            body=[10,256,10,*response.encode(),257,10,258,10,10,259,10,10]
            return [self.T.printable[b] for b in body]
        elif self.armed:
            self.cap+=1
            self.rows[min(self.cap,80)]=self.T.row_bytes(t)
        return []

def native_view(chat):
    import copy
    import torch
    from chipformer.rv.core import TAU_TEXT
    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original=chat.lm.embed_tokens
        def forward(self,ids):
            out=self.original(ids).clone()
            out[...,chat.d0:]=0
            out[...,chat.meta['tau_dim']]=TAU_TEXT
            return out
    class Identity(torch.nn.Module):
        def forward(self,hidden_states,**kwargs): return hidden_states
    model=copy.copy(chat.model)
    model._modules=chat.model._modules.copy()
    inner=copy.copy(chat.lm)
    inner._modules=chat.lm._modules.copy()
    inner.layers=torch.nn.ModuleList([*[Identity() for _ in range(chat.n_chip)],*chat.lm.layers[chat.n_chip:]])
    inner.embed_tokens=Embedding()
    model.model=inner
    assert model.lm_head is chat.model.lm_head and inner.norm is chat.lm.norm
    assert all(a is b for a,b in zip(inner.layers[chat.n_chip:],chat.lm.layers[chat.n_chip:]))
    return model.eval()

def run_native(chat,view,ids,calculator,budget):
    import torch
    calculator.calls=[]
    capture=Capture(chat.T,calculator)
    torch.cuda.synchronize()
    started=time.perf_counter()
    # Identical stock prefill in both arms, including any circuit control
    # markers present in the tool schema. Only subsequent execution differs.
    out=chat.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=True)
    cache=out.past_key_values
    nxt=int(out.logits[0,-1,:chat.v0].argmax())
    gen=[]
    forced=deque()
    text_seconds=0
    pos=len(ids)
    for _ in range(budget):
        gen.append(nxt)
        if len(gen)%128==0:
            print(f'RISCuda progress {len(gen)} text tokens, {len(capture.responses)} calls, {time.perf_counter()-started:.1f}s',flush=True)
            if OUTPUT is not None:
                save(OUTPUT/'current_progress.json',{'tokens':len(gen),'seconds':time.perf_counter()-started,'text':chat.text_of(gen),'responses':capture.responses})
        if nxt in chat.eos: break
        before=time.perf_counter()
        hc=view.model.embed_tokens(torch.tensor([nxt],device='cuda'))[0]
        logits=chat._qwen_step(hc,cache,pos)
        pos+=1
        q=int(logits[:chat.v0].argmax())
        text_seconds+=time.perf_counter()-before
        if not forced:
            forced.extend(capture.feed(nxt))
        nxt=forced.popleft() if forced else q
    torch.cuda.synchronize()
    wall=time.perf_counter()-started
    return {'tokens':gen,'seconds':wall,'text_tokens':len(gen)-int(bool(gen) and gen[-1] in chat.eos),'chip_tokens':0,
            'text_seconds':text_seconds,'chip_seconds':0,'responses':capture.responses,
            'gpu_tools':calculator.calls,'normal_eos':bool(gen) and gen[-1] in chat.eos}

def summary(records):
    out={}
    for dataset in sorted({r['dataset'] for r in records}):
        out[dataset]={}
        for arm in ('riscuda','qwenputer'):
            rows=[r for r in records if r['dataset']==dataset and r['arm']==arm]
            out[dataset][arm]={'n':len(rows),'correct':sum(r['exact'] for r in rows),
                'completed_correct':sum(r['exact'] and r['normal_eos'] for r in rows),
                'normal_eos':sum(r['normal_eos'] for r in rows),
                'total_seconds':sum(r['wall_seconds'] for r in rows),
                'median_seconds':statistics.median([r['wall_seconds'] for r in rows]) if rows else None,
                'calls':sum(r['calls'] for r in rows),'text_tokens':sum(r['text_tokens'] for r in rows),
                'chip_tokens':sum(r['chip_tokens'] for r in rows),
                'tool_seconds':sum(r['tool_seconds'] for r in rows)}
        a,b=out[dataset]['riscuda'],out[dataset]['qwenputer']
        out[dataset]['qwenputer_over_riscuda']=b['total_seconds']/a['total_seconds'] if a['total_seconds'] and a['n']==b['n'] else None
        paired={}
        for r in records:
            if r['dataset']==dataset: paired.setdefault(r['id'],{})[r['arm']]=r
        common=[v for v in paired.values() if len(v)==2 and all(r['exact'] and r['normal_eos'] for r in v.values())]
        out[dataset]['paired_completed_correct_n']=len(common)
        out[dataset]['paired_completed_correct_ratio']=(sum(v['qwenputer']['wall_seconds'] for v in common)/sum(v['riscuda']['wall_seconds'] for v in common)) if common else None
    return out

def main():
    global OUTPUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qwenputer',type=Path,default=ROOT.parent/'qwenputer')
    ap.add_argument('--model',type=Path,default=ROOT.parent/'m5gp/examples/qwen_rv32/models/Qwen3.5-4B-Qwenputer')
    ap.add_argument('--data',type=Path,default=ROOT/'runs/qwenputer_math_data')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--n',type=int,default=40)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--budget',type=int,default=4096)
    ap.add_argument('--max-chip',type=int,default=200000)
    ap.add_argument('--backend',choices=['c','micropython'],default='c')
    ap.add_argument('--shard',default='0/1',help='i/k: every k-th problem, both arms on this GPU')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--datasets',nargs='+',default=['arith','dmx_mul_big','dmx_div_big','dmx_mul_div_multiple_longer','dmx_add_or_sub_big'])
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    OUTPUT=args.output
    sys.path.insert(0,str(args.qwenputer/'experimental/riscv'))
    import torch
    from chipformer.rv.chat import RvChat
    from chipformer.rv import bridge as BR,elf,ref,calc
    from chipformer.rv.evalarith import NUDGE
    from chipformer.q35 import arith_bench as AB,evals as E
    AB.DATA=args.data
    problems=[]
    for name in args.datasets:
        rows=E.load_data('arith',args.n,args.seed) if name=='arith' else AB.load(name,args.n,args.seed)
        problems.extend(dict(p,dataset=name) for p in rows)
    save(args.output/'all_problems.json',problems)
    shard_i,shard_k=map(int,args.shard.split('/'))
    if not 0<=shard_i<shard_k: raise ValueError('invalid shard')
    problems=problems[shard_i::shard_k]
    save(args.output/'problems.json',problems)
    records_file=args.output/'records.jsonl'
    records=[json.loads(s) for s in records_file.read_text().splitlines()] if records_file.exists() else []
    done={(r['id'],r['arm']) for r in records}
    metadata={'started_at':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'qwenputer_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.qwenputer,text=True).strip(),
        'model':json.loads((args.model/'download_manifest.json').read_text()),
        'dtype':'bfloat16','thinking':False,'greedy':True,'batch':1,'seed':args.seed,'backend':args.backend,
        'budget':args.budget,'max_chip':args.max_chip,'problem_count':len(problems),
        'shard':args.shard,'all_problems_sha256':sha(args.output/'all_problems.json'),
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'problem_sha256':sha(args.output/'problems.json'),
        'gpu_mapping':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv'],text=True),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'benchmarks/exact_calc_guest.py',
            ROOT/'benchmarks/qwenputer_calc/calc.c',ROOT/'benchmarks/qwenputer_calc/port.h',
            ROOT/'src/gpu_rv32/native/include/bigint.cuh',ROOT/'src/gpu_rv32/native/runtime.cu')},
        'timing_scope':'Prompt prefill through final EOS, including tools, feedback and host dispatch; excludes model load, fixed firmware build, warmup, and post-run reference auditing.',
        'limitations':['Both arms use host token loops, not the autonomous CUDA graph.',
            'Same stock model prefill, widened BF16 language parameter objects and head; RISCuda decoding uses idle embeddings after replacing the circuit.',
            'C backend uses the same calc.c; bn.h/rt.h bindings target native CUDA row primitives and guest stdout. Optional MicroPython uses a separate rational adapter.',
            'Legacy Torch 2.9 CUDA12.8 environment; optional causal-conv1d/FLA absent.']}
    for name in ('download_manifest.json','modelscope_identity.json'):
        if (args.data/name).exists(): metadata[name]=json.loads((args.data/name).read_text())
    if args.resume and (args.output/'metadata.json').exists():
        prior=json.loads((args.output/'metadata.json').read_text())
        for key in ('problem_sha256','model','dtype','budget','max_chip','shard','backend','source_sha256','cuda_visible_devices'):
            if prior.get(key)!=metadata.get(key): raise ValueError(f'resume mismatch: {key}')
    save(args.output/'metadata.json',metadata)
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    calculator=CCalculator() if args.backend=='c' else GPUCalculator()
    if args.backend=='c':
        upstream=args.qwenputer/'experimental/riscv/chipformer/rv/env/calc/calc.c'
        assert sha(upstream)==sha(ROOT/'benchmarks/qwenputer_calc/calc.c'),'Calculator source differs from pinned upstream'
    save(args.output/'guest_build.json',calculator.image.metadata)
    (args.output/'guest.elf').write_bytes(calculator.image.elf)
    for expr,expected in [('2*8','16\n'),('84/18','14/3\n'),('-7//3','-3\n'),('1.25*-3.5','-4.375\n'),('gcd(84,18)','6\n')]:
        got=calculator(expr)
        assert got==expected,(expr,got,expected,calculator.calls[-1])
    print('Loading shared BF16 Qwenputer checkpoint',flush=True)
    chat=RvChat(str(args.model),device='cuda',executor=True)
    view=native_view(chat)
    firmware=args.qwenputer/'experimental/riscv/tests/data/rv/calc-bridge.elf'
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        eq_ids=chat.prompt_ids([{'role':'user','content':'Reply with OK.'}],tools=())
        x=torch.tensor([eq_ids],device='cuda')
        a=chat.model(input_ids=x,use_cache=False).logits[...,:chat.v0]
        b=view(input_ids=x,use_cache=False).logits[...,:chat.v0]
        assert torch.equal(a,b),'Language prefill equivalence gate failed'
        save(args.output/'language_equivalence.json',{'bit_equal':True,'max_error':float((a.float()-b.float()).abs().max()),'language_layers_shared':True,'head_shared':True})
        def qwenputer(ids):
            m=ref.Machine.from_program(elf.program(str(firmware)))
            BR.park(m)
            torch.cuda.synchronize()
            started=time.perf_counter()
            r=chat.generate(ids,m,max_new=args.budget,max_chip=args.max_chip,lock=None)
            torch.cuda.synchronize()
            r['wall_seconds']=time.perf_counter()-started
            r['normal_eos']=bool(r['tokens']) and r['tokens'][-1] in chat.eos
            return r
        warm=chat.prompt_ids([{'role':'system','content':NUDGE},{'role':'user','content':'What is 2*8?'+E.SUFFIX}])
        assert run_native(chat,view,warm,calculator,256)['normal_eos']
        assert qwenputer(warm)['normal_eos']
        for i,p in enumerate(problems):
            messages=[{'role':'system','content':NUDGE},{'role':'user','content':p['question']+E.SUFFIX}]
            ids=chat.prompt_ids(messages)
            for arm in (('riscuda','qwenputer') if i%2==0 else ('qwenputer','riscuda')):
                if (p['id'],arm) in done: continue
                print(f"START {i+1}/{len(problems)} {p['id']} {arm}",flush=True)
                r=run_native(chat,view,ids,calculator,args.budget) if arm=='riscuda' else qwenputer(ids)
                wall=r.get('wall_seconds',r['seconds'])
                text=chat.text_of(r['tokens'])
                responses=re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>',text,re.S)
                row={'id':p['id'],'dataset':p['dataset'],'arm':arm,'question':p['question'],'gold':p['gold'],
                    **AB.exact_fields(p,text),'normal_eos':r['normal_eos'],'wall_seconds':wall,
                    'prefill_tokens':len(ids),'text_tokens':r['text_tokens'],'chip_tokens':r['chip_tokens'],
                    'calls':len(responses),'responses':responses,'text':text,
                    'text_seconds':r['text_seconds'],'tool_seconds':sum(c['tool_wall_seconds'] for c in r['gpu_tools']) if arm=='riscuda' else r['chip_seconds'],
                    'result_file':f"{p['id']}.{arm}.json"}
                save(args.output/row['result_file'],dict(r,prompt_ids=ids,messages=messages))
                with records_file.open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
                records.append(row)
                done.add((p['id'],arm))
                save(args.output/'summary.json',summary(records))
                print(json.dumps({k:row[k] for k in ('id','arm','exact','normal_eos','wall_seconds','calls','text_tokens','chip_tokens')}),flush=True)
    save(args.output/'summary.json',summary(records))
    print('ALL COMPLETE',flush=True)

if __name__=='__main__':
    status=0
    try: main()
    except BaseException:
        status=1
        raise
    finally:
        if OUTPUT is not None: (OUTPUT/'exit_status').write_text(str(status)+'\n')
