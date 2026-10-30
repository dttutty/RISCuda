# Public arithmetic with one CUDA Graph per request

**English** | [简体中文](autonomous-public-200.zh-CN.md) · [Validation index](../../VALIDATION.md)

RISCuda's single-graph arm and the original Qwenputer each answered **195/200
(97.5%)** correctly. On the **192 paired requests with exact answers and normal
EOS in both arms**, request latency fell **41.48%**, a **1.7088×** time ratio
(Qwenputer/RISCuda). A paired-problem bootstrap with 2000 resamples gives a
95% reduction interval of **35.98%–45.94%**. All **200 native requests submitted
one host CUDA Graph**, including actual prompt prefill, generation, RISC-V
calculator execution and feedback through EOS or the declared cutoff.

| Dataset | Native exact | Qwenputer exact | Common exact + EOS pairs | Mean native seconds | Mean Qwenputer seconds | Latency reduction |
| --- | --- | --- | --- | --- | --- | --- |
| Synthetic `arith` | 38/40 | 39/40 | 38 | 6.68 | 10.06 | 33.61% |
| DeepMind `mul_big` | 39/40 | 39/40 | 39 | 4.29 | 7.27 | 40.95% |
| DeepMind `div_big` | 39/40 | 39/40 | 38 | 6.59 | 8.52 | 22.57% |
| DeepMind `mul_div_multiple_longer` | 40/40 | 40/40 | 39 | 37.85 | 69.24 | 45.33% |
| DeepMind `add_or_sub_big` | 39/40 | 38/40 | 38 | 6.43 | 10.49 | 38.68% |

The latency columns use the common correctly completed subset in each row.
Those 192 requests took **2392.70 s** cumulatively for RISCuda and **4088.59 s**
for Qwenputer. Including all failures and cutoffs, the totals were **3070.00 s**
and **5304.83 s**, a **42.13%** reduction. These are sums of individual request
durations, not the wall time of the two concurrent GPU jobs. Each arm had
five budget cutoffs, including one with an already-correct answer; each had
194 requests that were both exact and normally completed.

The aggregate accuracy ties, but the failures differ. Native `arith-18` failed
where Qwenputer succeeded; native `add_or_sub_big-6890` succeeded where
Qwenputer failed. Native `mul_div_multiple_longer-7815` was correct but reached
the text budget; Qwenputer's corresponding correctly answered cutoff was
`div_big-7961`. Both misinterpreted the subtraction in `add_or_sub_big-8541`.
All wrong answers, repeated calls and truncated continuations remain in the
evidence; none was removed from the all-request totals.

Complete text-token streams matched for **139/200 pairs**. Static-cache and
attention execution can change greedy continuations despite shared BF16
parameters; this is not a bit-identical decoding claim. On the **138
text-identical pairs that were also exact and normally completed**, native
latency fell **41.31%** (time ratio 1.7038). This additional subset checks that
the overall improvement is not explained only by different answer lengths.

## Configuration and timing

The model is the same **Qwen3.5-4B-Qwenputer BF16** checkpoint as the
[ModelScope model card](https://modelscope.ai/models/Spartan1925/Qwen3.5-4B-Qwenputer),
with all eight local shards and inference/tokenizer configurations freshly
hash-checked against that mirror. The cached revision is
`2d9068508f2b680badb7abcded96fdd107e9ba5d`. The author's five datasets, seed 0,
40 problems per set, prompt IDs, tool schema, `NUDGE`, boxed-answer suffix,
greedy decoding and disabled thinking were retained. No gold answer entered
model inputs or GPU execution. The native C source and its calculator limits
match the author; native decimal-row services replace the in-weight executor.

Two checked-idle RTX A6000 48 GB GPUs on markov ran disjoint question shards.
Each pair stayed on the same GPU with alternating arm order. The environment
used Torch 2.9/CUDA 12.8, Transformers 5.21.1, Triton 3.5, Zig 0.14.1 and
CUDA toolkit 12.9; optional causal-conv1d and FLA were absent. Both arms use
batch 1 and a 4096 text-token budget; original Qwenputer additionally retains
its 200000 chip-token cap. The native controller is bounded to 64 calls.
Full commands, GPU UUIDs, CPU affinity and source hashes are in the evidence.

Native inference uses captured static-cache decoding and the conditional GPU
controller. Original Qwenputer retains its upstream host decoding loop and
Triton circuit executor. Prompt-length groups reuse stable input storage
without padding, with a real-question rebinding check at startup. Prefill
returns prompt-length KV views over static backing storage, and its logits
match original prefill bit for bit for every captured length group. Measured
requests include actual prefill, all generation, calculator work and feedback.
Model loading, build, warmup, capture, graph instantiation, initial cache reset,
final trace download and CPU reference checks are excluded.

This is a comparison of the complete runtime configurations. Its speedup
includes captured language inference, static caching and native calculator
execution; it cannot be attributed solely to eliminating CPU dispatch.
The native stage timer total was 0.718 s, while the baseline chip-token timer
was 807.72 s over 2513123 chip tokens. Their boundaries and generated workloads
differ, so their quotient is not a standalone tool speedup. These are single
measurements per problem/arm, not a throughput or serving-concurrency study.
The closed loop uses a fixed C calculator, not arbitrary MicroPython routing.

## Independent validation and recovery

All **400 saved requests passed independent original-firmware trace/prefix
replay**, exact-answer rescoring, prompt checks and source/state verification.
The native launch count was one for every request; VM status and guest error
were zero for every native calculator invocation. Eighteen calls disagreed
with the unbounded Python arithmetic oracle because the C firmware rejected
complex expressions under its register limits. Their errors matched the
original firmware exactly. Such a tool-level error is distinct from a VM fault.

Initial benchmark code was commit `d0a9d75`. One shard stopped after 76 complete
pairs because its post-run Python oracle raised `NameError` on `abs(...)`.
The GPU request had already completed. Validation-only fix `274233a` resumed
the remaining 24 pairs on the same GPU, retaining all 152 recorded rows; the
unrecorded request was rerun because its timer had not been saved. Its token
and forcing streams matched the interrupted attempt exactly. AST and source
checks verified unchanged inference code. Audit fix `1588c95` also recognizes
original firmware capacity errors. The initial failed segment, source versions
and recovery manifest are retained; it is not reported as a clean initial exit.

## One-launch profiler proof

A separate, excluded-from-benchmark Nsight capture used the actual public
question `add_or_sub_big-6634`. The GPU returned `error` for
`abs(20412938679033416 - (-0.4))`; Qwen then generated
`20412938679033416 + 0.4`, received `20412938679033416.4`, and answered correctly.
Both tool calls, error feedback and model continuation occurred inside one
root graph. Its tokens matched the unprofiled measured request, and its
48 firmware-owned feedback tokens passed original CPU reference replay.

| Observation during the profiled request | Value |
| --- | --- |
| Host `cudaGraphLaunch` calls | 1 |
| Final host `cudaStreamSynchronize` calls | 1 |
| Host kernel-launch API calls | 0 |
| Host memcpy API calls | 0 |
| Device root graph activities | 1 |
| Decode forwards, from final state | 472 |
| Tool calls, from final state and reference replay | 2 |
| Device graph duration including prefill | 14.290827561 s |

The CUDA Graph contains many device kernels. Nsight used graph-level tracing
with event tracing disabled; it does not enumerate internal kernels or copy
nodes. The profiled wrapper took 18.358 s, versus 14.341 s for the unprofiled
measured request; profiling overhead is excluded from benchmark comparisons.
CPU initialization, submission and final collection remain required. All CPU
mathematical validation happens after GPU completion and never feeds the model.

See the [complete audit, metadata and profiler evidence](../evidence/qwenputer_autonomous_200.json),
[200 paired measurements](../evidence/qwenputer_autonomous_200.csv),
[benchmark reproduction](../../benchmarks/README.md) and
[controller details](../autonomous-calc.md). Full local raw artifacts remain
under `runs/autonomous_public_20261030/`, including initial and resumed logs,
binary states, transcripts, source snapshots and the Nsight report/SQLite.
