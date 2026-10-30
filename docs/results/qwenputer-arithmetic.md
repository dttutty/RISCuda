# Matched 200-problem Qwenputer arithmetic evaluation

**English** | [简体中文](qwenputer-arithmetic.zh-CN.md) · [Home](../../README.md) · [Validation index](../../VALIDATION.md)

Both implementations scored **195/200 (97.5%)** under the author's public
five-set, seed-0, 40-problems-per-set evaluation. On the **194 paired requests
with exact answers and normal EOS in both arms**, RISCuda reduced total request
latency by **15.92%** (Qwenputer/RISCuda time ratio 1.1893; paired problem
bootstrap 95% reduction interval **14.15%–17.68%**, 2000 resamples).

| Dataset | Exact answers, each arm | Common correct + EOS pairs | Mean RISCuda seconds | Mean Qwenputer seconds | Latency reduction |
| --- | --- | --- | --- | --- | --- |
| Synthetic `arith` | 39/40 | 39 | 9.15 | 10.14 | 9.81% |
| DeepMind `mul_big` | 39/40 | 39 | 6.49 | 7.37 | 11.91% |
| DeepMind `div_big` | 39/40 | 38 | 7.17 | 8.62 | 16.83% |
| DeepMind `mul_div_multiple_longer` | 40/40 | 40 | 59.69 | 72.73 | 17.93% |
| DeepMind `add_or_sub_big` | 38/40 | 38 | 9.66 | 10.68 | 9.58% |

The data include negative values, decimals and rational answers. The latency
columns use only the common correctly completed subset indicated
in the third column. All requests, including failures/truncations, are retained
in the data: RISCuda took **4589.96 s** cumulatively versus **5382.87 s** for
Qwenputer, a **14.73%** reduction. These sums are per-request durations, not the
wall time of two concurrently running GPU jobs. Each arm had five budget
cutoffs; one cutoff already contained a correct boxed answer. Thus exact-answer
accuracy and correctly completed-request counts are deliberately separate.

The language parameters, BF16 precision, stock prompt prefill, prompt IDs,
calculator protocol, greedy decoding and disabled thinking were aligned.
All eight local checkpoint shards and tokenizer/inference configuration hashes
matched `Spartan1925/Qwen3.5-4B-Qwenputer` on ModelScope. The cached author
checkpoint is revision `2d9068508f2b680badb7abcded96fdd107e9ba5d`.
RISCuda executed the same author `calc.c`, with CUDA decimal-row bindings;
the limits and calculator semantics were preserved. No gold answer entered
inference or tool execution. Both arms retained CPU token loops.

Two checked-idle RTX A6000s ran disjoint 100-question shards, each pair on the
same GPU with alternating arm order. The inference implementation commit is
`f82e40bc3411e2092b27814b6be38272f3dbb35e`; upstream Qwenputer is
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. Both jobs exited 0. Timings include
prompt prefill, generation, calculator execution and feedback; loading,
compilation, warmup and independent CPU validation are excluded. There is one
measured request per problem/arm, batch 1. Software versions and full commands
are recorded in the evidence. Optional causal-conv1d and FLA were absent in
the shared Torch 2.9/CUDA 12.8 environment.

Independent validation regenerated the selected problems, checked source,
ELF and shard hashes, re-scored exact rational answers and replayed every
firmware-owned token/response on the original CPU reference machine.
**All 400 runs passed execution/trace validation.** Complete text-token streams
matched for **199/200 pairs**, and their shared prefixes matched for **200/200**.
The one length mismatch was `div_big-4617`: Qwenputer reached its 200000 chip
token cap at 3513 text tokens, while RISCuda continued to its 4096 text cap.
Both answers were wrong/truncated; this pair is excluded from the headline
latency comparison. RISCuda VM execution status and guest error were zero for
every actual native calculator invocation.

Observed model failures include repeated calls (`arith-0`, `mul_big-1020`,
`div_big-4617`), generation continuing after a correct boxed answer
(`div_big-7961`), no calculator use while misreading “Put together”
(`add_or_sub_big-6890`), and reversed subtraction operands
(`add_or_sub_big-8541`). In the last case the calculator correctly evaluated
the model's reversed expression. The local add/sub score is 38/40 versus
39/40 in the public model card; both local arms share this result. This is a
controlled local rerun, not a claim of exact reproduction of every published
score under the author's software/hardware configuration.

Diagnostic stage totals were 3.61 s for RISCuda's calculator API and 804.33 s
for Qwenputer's chip-token stage (2513123 chip tokens). These timers have
different boundaries, and cutoff workloads differ; the report does not turn
that quotient into a standalone calculator speedup claim. Full paired request
latency is the primary measure. Language generation still dominates RISCuda's
request cost. This experiment does not establish CPU-free serving speedup,
batch throughput, or superiority to a local CPU tool.

A separate warmed Nsight capture of the native C calculator evaluated
`12345678901234567890*98765432109876543210` correctly. It recorded exactly
**one host kernel launch and one VM kernel**, **1870 guest instructions** and
**28 native decimal-row services**, with zero host/device copies overlapping
the kernel. Kernel duration was **0.758 ms**; the library's request timer was
8.67 ms including allocations/transfers, and the calculator wrapper timer was
12.28 ms. The driver/module warmup was
outside the `cudaProfilerApi` capture range. This standalone profile is not
an LLM latency measurement. All **69 regression methods** passed, and wheel
and source-distribution resource checks passed.

See the [complete evidence and audit](../../docs/evidence/qwenputer_arithmetic_200.json),
[200 paired measurements](../../docs/evidence/qwenputer_arithmetic_200.csv), and
[method/reproduction instructions](../../benchmarks/README.md). Full local raw
generation records and Nsight reports remain in `runs/public_math_20261030/`;
their hashes are retained in the evidence.
