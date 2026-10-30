"""Audit saved autonomous requests and the actual Nsight CUDA API/copy trace."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_trace(path, expected_calls, expected_forwards):
    with sqlite3.connect(path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        apis = dict(db.execute('SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
                              'JOIN StringIds s ON r.nameId=s.id GROUP BY s.value'))
        assert db.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE returnValue!=0').fetchone()[0] == 0
        launches = sum(n for name,n in apis.items() if name.startswith('cudaGraphLaunch'))
        assert launches == 1,apis
        assert not any(n for name,n in apis.items() if name.startswith(
            ('cudaLaunchKernel','cudaLaunchCooperativeKernel','cudaLaunchHostFunc','cudaMemcpy'))),apis
        assert sum(n for name,n in apis.items() if name.startswith('cudaStreamSynchronize')) == 1,apis
        copies = dict(db.execute('SELECT copyKind,count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind')) \
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else {}
        assert copies.get(1,0) == copies.get(2,0) == 0,copies
        graphs = list(db.execute('SELECT start,end,graphId,graphExecId FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE'))
        assert len(graphs) == 1,graphs
        return dict(sqlite_sha256=sha(path),runtime_api_calls=apis,copy_kinds=copies,
            host_graph_launches=launches,device_graph_activities=len(graphs),
            device_graph_duration_ns=graphs[0][1]-graphs[0][0],
            validated_state_decode_steps=expected_forwards,validated_state_tool_calls=expected_calls,
            no_host_token_or_tool_launches=True,no_host_memcpy_apis=True,
            granularity='graph; internal kernels/copy nodes are not individually traced')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--profile-run',type=Path,required=True)
    p.add_argument('--sqlite',type=Path,required=True)
    p.add_argument('--qwenputer',type=Path,default=ROOT.parent/'qwenputer')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    sys.path.insert(0,str(args.qwenputer/'experimental/riscv'))
    from chipformer.rv import bridge as BR
    from chipformer.q35 import arith_bench as AB
    from benchmarks.audit_qwenputer import verify_stream
    from gpu_rv32.resident import State
    import ctypes as ct
    summary, pairs, metadata = [], {}, []
    for directory in (args.runs,args.profile_run):
        m = json.loads((directory/'metadata.json').read_text())
        metadata.append(m)
        assert not m['dirty'],m['commit']
        for name,digest in m['source_sha256'].items():
            assert sha(ROOT/name) == digest,name
        assert json.loads((directory/'exit.json').read_text())['exit_status'] == 0
        T = BR.Text(Path(m['model'])/'tokenizer.json')
        base = json.loads((Path(m['model'])/'chipformer.json').read_text())['chip_token_offset']
        for path in sorted(directory.glob('*_gpu.json'))+sorted(directory.glob('*_host.json')):
            r = json.loads(path.read_text())
            prompt = json.loads((directory/(r['case']+'_prompt.json')).read_text())
            s = State.from_buffer_copy(path.with_suffix('.state.bin').read_bytes())
            assert r['tokens'] == list(s.tokens[:s.total])
            assert r['forced'] == list(s.forced[:s.total])
            assert s.stop == 1 and r['normal_eos'] and r['tokens'][-1] in (248046,248044)
            assert s.total == s.generated+s.injected and s.forwards == s.total-1
            assert s.position == len(prompt['tokens'])+s.forwards
            text = T.decode_bytes(r['tokens']).decode('utf-8','replace')
            assert text == r['text']
            assert AB.exact_fields({'gold':r['expected']},text)['exact']
            assert all(t['reference_equal'] and t['status'] == t['error'] == 0 for t in r['tools'])
            trace = verify_stream(T,args.qwenputer/'experimental/riscv/tests/data/rv/calc-bridge.elf',
                prompt['tokens'],r['tokens'],chip=False,base=base)
            assert trace['ok'] and not trace['ended_inside_tool'] and trace['calls'] == s.calls
            assert trace == r['reference_trace']
            if r['arm'] == 'gpu':
                assert r['host_graph_launches'] == 1
            else:
                assert r['host_graph_launches'] == s.total+1
            if r['case'] == 'chain':
                assert s.calls == 2
                assert r['tools'][0]['output'].strip() in r['tools'][1]['expression']
            item = {k:r[k] for k in ('case','arm','seconds','expected','generated','injected','total',
                'forwards','normal_eos','exact','host_graph_launches','tools','reference_trace')}
            item.update(raw_file=str(path),raw_sha256=sha(path),state_sha256=sha(path.with_suffix('.state.bin')))
            summary.append(item)
            if directory == args.runs:
                pairs.setdefault(r['case'],{})[r['arm']] = r
    assert len(pairs) == 3
    for pair in pairs.values():
        assert pair['gpu']['tokens'] == pair['host']['tokens']
        assert pair['gpu']['forced'] == pair['host']['forced']
    profiled = [r for r in summary if Path(r['raw_file']).parent == args.profile_run]
    assert len(profiled) == 1
    evidence = dict(audited_at=datetime.now(timezone.utc).isoformat(),complete=True,
        implementation_commit=metadata[0]['commit'],metadata=metadata,
        requests=summary,host_control_token_streams_equal=True,
        profiler=profile_trace(args.sqlite,len(profiled[0]['tools']),profiled[0]['forwards']),
        timing_scope='Prefill through EOS including tools/feedback; excludes load/build/capture/graph instantiation/cache reset/download/reference audit.',
        limitation='Three correctness/parity cases, one timing per arm; no statistical speedup or 27B claim. Host control reuses identical captured GPU steps.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(evidence['profiler'],indent=2),flush=True)
    print('All saved states, source hashes, exact answers, dependent calls and original reference traces passed.')


if __name__ == '__main__':
    main()
