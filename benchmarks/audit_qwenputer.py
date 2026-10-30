"""Independent CPU replay/scoring and paired latency summary after timed requests."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_stream(T,firmware,ids,tokens,*,chip,base):
    from chipformer.rv import bridge as BR,elf,ref,spec as S
    m=ref.Machine.from_program(elf.program(str(firmware)))
    BR.park(m)
    session=BR.Session(T,m)
    session.history(ids)
    stream=[t-base if t>=base else BR.text_token(t) for t in tokens]
    position,checked=0,0
    while position<len(stream):
        if not S.is_text(stream[position]):
            return {'ok':False,'position':position,'reason':'unexpected machine token'}
        before=len(session.toks)
        session.qwen([stream[position]-S.TEXT_TOKEN])
        position+=1
        expected=session.toks[before+1:]
        if not chip: expected=[t for t in expected if S.is_text(t)]
        available=min(len(expected),len(stream)-position)
        if stream[position:position+available]!=expected[:available]:
            return {'ok':False,'position':position,'reason':'firmware-owned token mismatch'}
        checked+=available
        position+=available
        if available<len(expected):
            return {'ok':True,'owned_checked':checked,'calls':len(session.runs),'ended_inside_tool':True}
    return {'ok':True,'owned_checked':checked,'calls':len(session.runs),'ended_inside_tool':False}

def ratio(rows):
    if not rows: return None
    return sum(p['qwenputer']['wall_seconds'] for p in rows)/sum(p['riscuda']['wall_seconds'] for p in rows)

def stats(rows):
    output={}
    for arm in ('riscuda','qwenputer'):
        times=[p[arm]['wall_seconds'] for p in rows]
        output[arm]={'n':len(rows),'correct':sum(p[arm]['exact'] for p in rows),
            'completed_correct':sum(p[arm]['exact'] and p[arm]['normal_eos'] for p in rows),
            'budget_cutoffs':sum(not p[arm]['normal_eos'] for p in rows),
            'total_seconds':sum(times),'mean_seconds':statistics.mean(times) if times else None,
            'median_seconds':statistics.median(times) if times else None,
            'calls':sum(p[arm]['calls'] for p in rows),
            'calls_started':sum(p[arm]['calls_started'] for p in rows),
            'text_tokens':sum(p[arm]['text_tokens'] for p in rows),
            'chip_tokens':sum(p[arm]['chip_tokens'] for p in rows),
            'tool_seconds':sum(p[arm]['tool_seconds'] for p in rows)}
    successful=[p for p in rows if all(p[a]['exact'] and p[a]['normal_eos'] for a in ('riscuda','qwenputer'))]
    output['all_requests_time_ratio']=ratio(rows)
    output['paired_completed_correct_n']=len(successful)
    output['paired_completed_correct_time_ratio']=ratio(successful)
    if successful:
        rng=random.Random(7181)
        bootstrap=sorted(ratio(rng.choices(successful,k=len(successful))) for _ in range(2000))
        output['paired_time_ratio_bootstrap95']=[bootstrap[49],bootstrap[1949]]
    return output

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--qwenputer',type=Path,default=ROOT.parent/'qwenputer')
    ap.add_argument('--model',type=Path,default=ROOT.parent/'m5gp/examples/qwen_rv32/models/Qwen3.5-4B-Qwenputer')
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    sys.path.insert(0,str(args.qwenputer/'experimental/riscv'))
    from chipformer.rv import bridge as BR
    from chipformer.q35 import arith_bench as AB,evals as E
    from chipformer.rv.evalarith import NUDGE
    meta=json.loads((args.model/'chipformer.json').read_text())
    base=meta['chip_token_offset']
    T=BR.Text(args.model/'tokenizer.json')
    firmware=args.qwenputer/'experimental/riscv/tests/data/rv/calc-bridge.elf'
    reference_context={'firmware':digest(firmware),'tokenizer':digest(args.model/'tokenizer.json'),
        **{name:digest(args.qwenputer/'experimental/riscv/chipformer/rv'/name)
           for name in ('ref.py','bridge.py','spec.py','elf.py')}}
    auditor_hash=digest(Path(__file__))
    audit_path=args.run/'independent_verification.json'
    prior=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    cache=prior.get('cases',{}) if prior.get('auditor_sha256')==auditor_hash and prior.get('reference_context')==reference_context else {}
    cases={}
    rows=[]
    raw={}
    hashes=set()
    complete=True
    for directory in sorted(args.run.glob('shard[0-9]*')):
        if not directory.is_dir(): continue
        metadata=json.loads((directory/'metadata.json').read_text())
        assert metadata['thinking'] is False and metadata['dtype']=='bfloat16'
        assert all(v['match'] for v in metadata['modelscope_identity.json']['checks'])
        assert json.loads((directory/'language_equivalence.json').read_text())['bit_equal']
        hashes.add(metadata['all_problems_sha256'])
        for name,expected_hash in metadata['source_sha256'].items():
            assert digest(ROOT/name)==expected_hash,('changed source',name)
        canonical=json.loads((directory/'all_problems.json').read_text())
        selected=json.loads((directory/'problems.json').read_text())
        assert digest(directory/'all_problems.json')==metadata['all_problems_sha256']
        assert digest(directory/'problems.json')==metadata['problem_sha256']
        shard_i,shard_k=map(int,metadata['shard'].split('/'))
        assert selected==canonical[shard_i::shard_k]
        build=json.loads((directory/'guest_build.json').read_text())
        assert digest(directory/'guest.elf')==build['elf_sha256']
        AB.DATA=ROOT/'runs/qwenputer_math_data'
        expected=[]
        for dataset in dict.fromkeys(p['dataset'] for p in canonical):
            data=E.load_data('arith',40,0) if dataset=='arith' else AB.load(dataset,40,0)
            expected.extend(dict(p,dataset=dataset) for p in data)
        assert expected==canonical,'dataset selection differs from author seed0/n40'
        by_id={p['id']:p for p in selected}
        file=directory/'records.jsonl'
        lines=file.read_text().splitlines() if file.exists() else []
        local=[]
        for i,line in enumerate(lines):
            try: local.append(json.loads(line))
            except json.JSONDecodeError:
                if not args.allow_partial or i!=len(lines)-1: raise
        complete &= len(local)==2*len(selected)
        complete &= (directory/'exit_status').exists() and (directory/'exit_status').read_text().strip()=='0'
        for row in local:
            key=row['arm']+':'+row['id']
            assert key not in raw,'duplicate record'
            path=directory/row['result_file']
            generation=json.loads(path.read_text())
            p=by_id[row['id']]
            assert row['question']==p['question'] and row['gold']==p['gold']
            assert generation['messages']==[{'role':'system','content':NUDGE},{'role':'user','content':p['question']+E.SUFFIX}]
            text=T.decode_bytes([t for t in generation['tokens'] if t<base]).decode('utf-8','replace')
            assert text==row['text']
            assert AB.exact_fields(p,text)['exact']==row['exact']
            ended=bool(generation['tokens']) and generation['tokens'][-1] in (248046,248044)
            assert ended==row['normal_eos']
            if row['arm']=='riscuda':
                assert all(c['status']==0 and c['error']==0 for c in generation['gpu_tools'])
                assert all(c['kernel_launches']==1 for c in generation['gpu_tools'])
            fingerprint=digest(path)
            old=cache.get(key,{})
            if old.get('raw_sha256')==fingerprint:
                trace=old['reference_trace']
            else:
                trace=verify_stream(T,firmware,generation['prompt_ids'],generation['tokens'],chip=row['arm']=='qwenputer',base=base)
                print(key,trace,flush=True)
            assert trace['ok'],(key,trace)
            cases[key]={'raw_sha256':fingerprint,'reference_trace':trace}
            row['calls_started']=sum(t==T.TCC for t in generation['tokens'])
            raw[key]=generation
            rows.append(row)
    assert len(hashes)==1,'shards did not use identical canonical problems'
    paired={}
    for row in rows: paired.setdefault(row['id'],{})[row['arm']]=row
    pairs=[p for p in paired.values() if len(p)==2]
    equality=[]
    for p in pairs:
        key=p['riscuda']['id']
        a,b=raw['riscuda:'+key],raw['qwenputer:'+key]
        assert a['prompt_ids']==b['prompt_ids']
        bt=[t for t in b['tokens'] if t<base]
        n=min(len(a['tokens']),len(bt))
        equality.append({'id':key,'text_tokens_identical':a['tokens']==bt,
                         'shared_prefix_identical':a['tokens'][:n]==bt[:n],
                         'tool_responses_identical':p['riscuda']['responses']==p['qwenputer']['responses']})
    output={'audited_at':datetime.now(timezone.utc).isoformat(),'auditor_sha256':auditor_hash,
        'reference_context':reference_context,
        'complete':bool(complete and len(rows)==400),'records':len(rows),'paired_problems':len(pairs),
        'cases':cases,'text_identical_pairs':sum(v['text_tokens_identical'] for v in equality),
        'shared_prefix_identical_pairs':sum(v['shared_prefix_identical'] for v in equality),
        'response_identical_pairs':sum(v['tool_responses_identical'] for v in equality),
        'pair_checks':equality,'overall':stats(pairs),'datasets':{name:stats([p for p in pairs if p['riscuda']['dataset']==name]) for name in sorted({r['dataset'] for r in rows})}}
    audit_path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('cases','pair_checks')},ensure_ascii=False,indent=2))
    if not args.allow_partial and not output['complete']: raise SystemExit('benchmark incomplete')

if __name__=='__main__': main()
