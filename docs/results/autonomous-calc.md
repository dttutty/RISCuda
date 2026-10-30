# Autonomous Qwen and native C calculator

**English** | [简体中文](autonomous-calc.zh-CN.md) · [Home](../../README.md) · [Validation index](../../VALIDATION.md)

Implementation `15a00d4` connects the native C calculator to a GPU-controlled
conditional CUDA graph. With **Qwen3.5-4B-Qwenputer BF16, thinking disabled**,
all three real-model cases returned exact answers and normal EOS. The complete
text-token streams and feedback markers matched a host control that launches
the **same captured step** once per token.

| Case | Calls | CPU step-control seconds | GPU-controlled seconds | CPU step launches | GPU host launches |
| --- | --- | --- | --- | --- | --- |
| `2*8` | 1 | 1.925 | 1.924 | 66 | 1 |
| `12345678901234567890*98765432109876543210` | 1 | 5.939 | 5.943 | 210 | 1 |
| `123456789*987654321`, then add `12345` to the returned result | 2 | 6.833 | 6.843 | 241 | 1 |

The dependent case actually generated separate calls for `123456789*987654321`
and `121932631112635269+12345`, yielding `121932631112647614`. The second call
consumed the first tool's returned integer. All three runs and both control
arms were replayed independently on the original CPU reference firmware;
all tool-owned feedback tokens matched. The final state also passed exact
answer, ordering, source-hash, cache-position and token-accounting checks.

The GPU tool interval (gate, RAM restore, VM and feedback encoding) was
0.294 ms for `2*8`, 0.804 ms for the 40-digit product and 0.517/0.467 ms for
the two dependent calls. Parameter gathering and the model forward pass are
outside this interval. Language generation dominates these batch-one requests.
There is **no meaningful latency improvement over the identical captured-step
host control in this small sample**. These are single measurements, not a
statistical speedup study or a rerun of the preceding 200-question benchmark.

A graph-level Nsight capture of the dependent request recorded **one host
`cudaGraphLaunch` and one final `cudaStreamSynchronize`**, with **no host kernel
launch or memcpy API calls** during the request. The final trace contained two
successful calculator calls and 239 decode forwards. Graph-level tracing does
not expose individual internal kernel/copy nodes, so their absence is not
inferred from the report. The measured graph includes real prompt prefill;
initial model loading, tokenization, capture/setup/cache reset and final trace
download/reference checks remain CPU responsibilities outside the request.

The tested scope is a fixed C calculator, batch one, static cache/graph shapes
and bounded generation. It does not establish a GPU-controlled arbitrary
MicroPython router, 27B compatibility, serving concurrency or throughput.
Nsight 2025.1.3 node-level tracing with automatic event tracing failed with
an illegal memory access; the documented graph-level/event-tracing-disabled
configuration completed correctly. Profiled wall time is excluded from the
latency table. All **73 regression methods** passed; wheel/source build and
packaged controller-resource checks passed.

See the [controller and reproduction instructions](../../docs/autonomous-calc.md)
and [audited requests, metadata and profiler evidence](../../docs/evidence/autonomous_calc.json).
Unprofiled requests were captured at implementation commit `15a00d4`; the final
profile was captured from clean commit `65464b2` with identical runtime/model
example source hashes. All seven saved requests passed the independent audit.
Local raw states, transcripts and Nsight reports are under
`runs/autonomous_calc_15a00d4/`, `runs/autonomous_calc_profile_data_65464b2/`
and `runs/autonomous_calc_profile_65464b2.{nsys-rep,sqlite}`.
The earlier 200-question evaluation still uses its original CPU loops.
