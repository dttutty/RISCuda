"""Independently audit the public single-graph benchmark, including failures."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from benchmarks.audit_qwenputer import stats, verify_stream
from benchmarks.compare_qwenputer import save, sha
from gpu_rv32.resident import State


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--qwenputer',type=Path,default=ROOT.parent/'qwenputer')
    ap.add_argument('--model',type=Path,default=ROOT.parent/'m5gp/examples/qwen_rv32/models/Qwen3.5-4B-Qwenputer')
    ap.add_argument('--data',type=Path,default=ROOT/'runs/qwenputer_math_data')
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    sys.path.insert(0,str(args.qwenputer/'experimental/riscv'))
    from chipformer.rv import bridge as BR,calc
    from chipformer.q35 import arith_bench as AB,evals as E
    from chipformer.rv.evalarith import NUDGE
    from benchmarks.autonomous_qwenputer import DATASETS
    AB.DATA=args.data
    firmware=args.qwenputer/'experimental/riscv/tests/data/rv/calc-bridge.elf'
    T=BR.Text(args.model/'tokenizer.json')
    base=json.loads((args.model/'chipformer.json').read_text())['chip_token_offset']
    context={'firmware':sha(firmware),'tokenizer':sha(args.model/'tokenizer.json'),
        **{name:sha(args.qwenputer/'experimental/riscv/chipformer/rv'/name)
           for name in ('ref.py','bridge.py','spec.py','elf.py')}}
    target=args.run/'independent_verification.json'
    prior=json.loads(target.read_text()) if target.exists() else {}
    auditor_sha=sha(Path(__file__))
    cache=prior.get('cases',{}) if prior.get('auditor_sha256')==auditor_sha and prior.get('reference_context')==context else {}
    cases,rows,raw,metadata,canonical_hashes={},{},{},[],set()
    complete=True
    oracle_disagreements=[]
    recoveries=json.loads((args.run/'recoveries.json').read_text()) if (args.run/'recoveries.json').exists() else []
    recovered={r['interrupted_segment']:r for r in recoveries}
    directories=sorted(p for p in args.run.glob('shard[0-9]*') if p.is_dir())
    if not directories:
        directories=[args.run]
    for d in directories:
        m=json.loads((d/'metadata.json').read_text())
        metadata.append(m)
        assert not m['profiled'] and m['thinking'] is False and m['dtype']=='bfloat16'
        assert not m['dirty'],'production benchmark must identify a clean code commit'
        assert all(c['match'] for c in m['modelscope_identity.json']['checks'])
        for name,digest in m['source_sha256'].items():
            committed=subprocess.check_output(['git','show',m['code_commit']+':'+name],cwd=ROOT)
            assert __import__('hashlib').sha256(committed).hexdigest()==digest,('unrecorded source',name)
        for name,info in m['download_manifest.json']['files'].items():
            assert sha(args.data/'dm'/name)==info['sha256']
        assert json.loads((d/'language_equivalence.json').read_text())['bit_equal']
        warm=json.loads((d/'warmup_rebinding.json').read_text())
        assert len(warm)==2 and warm[0]['prompt_ids']!=warm[1]['prompt_ids']
        assert len(warm[0]['prompt_ids'])==len(warm[1]['prompt_ids'])
        for item,gold in zip(warm,['16','27']):
            assert item['result']['normal_eos'] and item['result']['host_graph_launches']==1
            assert AB.exact_fields({'gold':gold},item['result']['text'])['exact']
        canonical=[]
        for dataset in DATASETS:
            ps=E.load_data('arith',m['n'],m['seed']) if dataset=='arith' else AB.load(dataset,m['n'],m['seed'])
            canonical.extend(dict(p,dataset=dataset) for p in ps)
        assert canonical==json.loads((d/'all_problems.json').read_text())
        assert sha(d/'all_problems.json')==m['all_problems_sha256']
        canonical_hashes.add(m['all_problems_sha256'])
        i,k=map(int,m['shard'].split('/'))
        selected=canonical[i::k]
        if m['ids']:
            selected=[p for p in selected if p['id'] in m['ids']]
        assert selected==json.loads((d/'problems.json').read_text())
        assert sha(d/'problems.json')==m['problem_sha256']
        assert sha(d/'guest.elf')==json.loads((d/'guest_build.json').read_text())['elf_sha256']
        plan=json.loads((d/'capture_plan.json').read_text())
        assert sorted(x for group in plan for x in group['ids'])==sorted(p['id'] for p in selected)
        for group in plan:
            gate=d/f'prefill_gate_{group["length"]}.json'
            if gate.exists():
                assert json.loads(gate.read_text())['bit_equal']
            elif not args.allow_partial and d.name not in recovered:
                raise AssertionError(f'missing prefill-equivalence check: {gate}')
        expected={p['id']:p for p in selected}
        lines=(d/'records.jsonl').read_text().splitlines() if (d/'records.jsonl').exists() else []
        local=[]
        for n,line in enumerate(lines):
            try:local.append(json.loads(line))
            except json.JSONDecodeError:
                if not args.allow_partial or n!=len(lines)-1:raise
        segment_complete=len(local)==len(selected)*len(m['arms']) and (d/'exit_status').exists() and (d/'exit_status').read_text().strip()=='0'
        if d.name in recovered:
            recovery=recovered[d.name]
            assert (d/'exit_status').read_text().strip()=='1'
            assert sha(args.run/(d.name+'.log'))==recovery['interrupted_log_sha256']
            assert 'NameError: name \'abs\' is not defined' in (args.run/(d.name+'.log')).read_text()
            successor=args.run/recovery['continuation_segment']
            next_m=json.loads((successor/'metadata.json').read_text())
            completed_ids={r['id'] for r in local}
            assert set(next_m['ids'])==set(expected)-completed_ids
            assert all(sum(r['id']==pid for r in local)==2 for pid in completed_ids)
            assert next_m['cuda_visible_devices']==m['cuda_visible_devices']
            assert next_m['model']==m['model'] and next_m['budget']==m['budget']
            assert all(next_m['source_sha256'][name]==digest for name,digest in m['source_sha256'].items()
                       if name!='examples/autonomous_qwen_calc.py')
            segment_complete=True  # Global pair coverage below must include every missing ID.
        complete &= segment_complete
        for row in local:
            key=row['arm']+':'+row['id']
            assert key not in rows,'duplicate record'
            p=expected[row['id']]
            assert row['question']==p['question'] and row['gold']==p['gold']
            path=d/row['result_file']
            r=json.loads(path.read_text())
            assert r['messages']==[{'role':'system','content':NUDGE},{'role':'user','content':p['question']+E.SUFFIX}]
            assert chat_ids(T,r['messages'],args.model)==r['prompt_ids']
            text=T.decode_bytes([t for t in r['tokens'] if t<base]).decode('utf-8','replace')
            assert text==row['text']==r['text']
            for name,value in AB.exact_fields(p,text).items():
                assert row[name]==value
            assert row['normal_eos']==(bool(r['tokens']) and r['tokens'][-1] in (248046,248044))
            if row['arm']=='riscuda':
                assert json.loads((d/f'prefill_gate_{row["prefill_tokens"]}.json').read_text())['bit_equal']
                state_path=d/(row['id']+'.riscuda.state.bin')
                s=State.from_buffer_copy(state_path.read_bytes())
                assert list(s.tokens[:s.total])==r['tokens']
                assert list(s.forced[:s.total])==r['forced']
                assert s.total==s.generated+s.injected
                assert s.position==len(r['prompt_ids'])+s.forwards
                assert s.forwards==s.total-(s.stop in (1,5))
                assert len(r['tools'])==s.calls
                assert row['host_graph_launches']==r['host_graph_launches']==1
                for t,actual in zip(r['tools'],s.tools):
                    assert t['expression']==actual.expression.decode()
                    assert t['output']==actual.output.decode()
                    assert t['status']==actual.status==0 and t['error']==actual.error==0
                    if not t['reference_equal']:
                        # Unbounded Python arithmetic omits the original C parser's
                        # register/capacity limits. The firmware is authoritative.
                        authoritative=calc.run(t['expression'])[0]
                        assert t['output']==authoritative,('firmware mismatch',row['id'],t)
                        oracle_disagreements.append(dict(id=row['id'],expression=t['expression'],
                            output=t['output'],numeric_reference=t['reference'],
                            firmware_reference=authoritative,firmware_equal=True))
            fingerprint=sha(path)
            old=cache.get(key,{})
            if old.get('raw_sha256')==fingerprint:
                trace=old['reference_trace']
            else:
                trace=verify_stream(T,firmware,r['prompt_ids'],r['tokens'],chip=row['arm']=='qwenputer',base=base)
                print(key,trace,flush=True)
            assert trace['ok'],(key,trace)
            cases[key]={'raw_sha256':fingerprint,'reference_trace':trace}
            row['calls_started']=sum(t==T.TCC for t in r['tokens'])
            row['state_stop']=r.get('stop')
            rows[key],raw[key]=row,r
            # Cache expensive reference replay even if interrupted before the final summary.
            save(target,dict(auditor_sha256=auditor_sha,reference_context=context,cases=cases,complete=False))
    assert len(canonical_hashes)==1
    if not args.allow_partial:
        assert complete,'benchmark is incomplete'
    pairs={}
    for row in rows.values():pairs.setdefault(row['id'],{})[row['arm']]=row
    paired=[pair for pair in pairs.values() if len(pair)==2]
    if recoveries and not args.allow_partial:
        assert len(paired)==len(canonical) and set(pairs)=={p['id'] for p in canonical}
    equal=prefixes=0
    differences=[]
    for pair in paired:
        pid=pair['riscuda']['id']
        a=raw['riscuda:'+pid]['tokens']
        b=[t for t in raw['qwenputer:'+pid]['tokens'] if t<base]
        equal+=a==b
        prefixes+=a[:min(len(a),len(b))]==b[:min(len(a),len(b))]
        if a!=b:
            first=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            differences.append(dict(id=pid,first_difference=first,native_tokens=len(a),baseline_tokens=len(b)))
    overall=stats(paired)
    by_dataset={name:stats([p for p in paired if p['riscuda']['dataset']==name]) for name in DATASETS}
    identical=[p for p in paired if raw['riscuda:'+p['riscuda']['id']]['tokens']==[t for t in raw['qwenputer:'+p['riscuda']['id']]['tokens'] if t<base]]
    output=dict(audited_at=datetime.now(timezone.utc).isoformat(),auditor_sha256=auditor_sha,
        reference_context=context,complete=complete,full_public_evaluation=complete and len(paired)==200,
        records=len(rows),paired_problems=len(paired),cases=cases,metadata=metadata,recoveries=recoveries,
        numeric_oracle_disagreements=oracle_disagreements,
        overall=overall,by_dataset=by_dataset,text_identical_pairs=equal,shared_prefix_equal_pairs=prefixes,
        text_identical_subset=stats(identical),text_differences=differences,
        native_graph_launches={str(n):sum(r['arm']=='riscuda' and r['host_graph_launches']==n for r in rows.values()) for n in (1,)},
        all_failures=[{k:r[k] for k in ('id','arm','exact','normal_eos','state_stop')} for r in rows.values() if not r['exact'] or not r['normal_eos']],
        interpretation='Same BF16 language weights/protocol; static/cache/backend differences can change greedy text. End-to-end speed includes these changes and cannot be attributed solely to eliminating CPU dispatch.')
    save(target,output)
    fields=['id','dataset','riscuda_exact','qwenputer_exact','riscuda_eos','qwenputer_eos','riscuda_seconds','qwenputer_seconds','native_host_graph_launches','native_total_tokens','baseline_text_tokens']
    with (args.run/'paired_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for pair in paired:
            a,b=pair['riscuda'],pair['qwenputer']
            writer.writerow(dict(id=a['id'],dataset=a['dataset'],riscuda_exact=a['exact'],qwenputer_exact=b['exact'],
                riscuda_eos=a['normal_eos'],qwenputer_eos=b['normal_eos'],riscuda_seconds=a['wall_seconds'],
                qwenputer_seconds=b['wall_seconds'],native_host_graph_launches=a['host_graph_launches'],
                native_total_tokens=len(raw['riscuda:'+a['id']]['tokens']),baseline_text_tokens=b['text_tokens']))
    print(json.dumps(dict(complete=complete,overall=overall,text_identical_pairs=equal),indent=2),flush=True)


_tokenizer=None
def chat_ids(T,messages,model):
    global _tokenizer
    from transformers import AutoTokenizer
    from chipformer.rv.chat import CALC_TOOL
    if _tokenizer is None:
        _tokenizer=AutoTokenizer.from_pretrained(model)
    out=_tokenizer.apply_chat_template(messages,tools=[CALC_TOOL],add_generation_prompt=True,enable_thinking=False,tokenize=True)
    return list(map(int,out if isinstance(out,list) else out['input_ids']))


if __name__=='__main__':
    main()
