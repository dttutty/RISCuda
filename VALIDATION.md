# Standalone validation

Validated on Linux x86_64, Python 3.12.13, Zig 0.14.1, CUDA 12.9 and an NVIDIA
RTX A6000. The native target was `sm_80`. The fresh project environment had
exactly two installed packages: `gpu-rv32-python` and `ziglang`. No PyTorch,
Transformers, bitsandbytes, model checkpoint or original project import was
used for execution.

## Matched 200-problem Qwenputer arithmetic evaluation

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

See the [complete evidence and audit](docs/evidence/qwenputer_arithmetic_200.json),
[200 paired measurements](docs/evidence/qwenputer_arithmetic_200.csv), and
[method/reproduction instructions](benchmarks/README.md). Full local raw
generation records and Nsight reports remain in `runs/public_math_20261030/`;
their hashes are retained in the evidence.

## C, MicroPython and native bigint extension

Implementation commit: `fb9e34099332a8f5b19312a485e0a7b50173c9ce`.
The extension's full regression suite passed **67 methods in 14.849 seconds**
on checked-idle RTX A6000 GPU 0, UUID
`GPU-23d8809e-ca5a-3d0c-7a0f-26ab756ae05a`, on markov. Exit status was 0.

```bash
CUDA_VISIBLE_DEVICES=0 GPU_RV32_TESTS=1 GPU_RV32_CACHE="$PWD/.cache" \
  uv run python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 c examples/bigint.c
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 micropython examples/bigint.py
```

The original 53 methods still pass. Fourteen added methods cover C/ELF
building and bounds; real MicroPython's fixed firmware; C loops and integer
printf (including minimum signed64); functions/classes, containers, generators,
exceptions, garbage collection with live objects, factorial/powers, signed
large integer arithmetic/division, output overflow, instruction budgets,
native limb multiplication through 4096-bit inputs, rejection without RAM
changes, unknown services, and exact guest fallback above 4096 bits.

`examples/bigint.c` returned success and one bigint service invocation.
`examples/bigint.py` calculated `100!` and `2**256` correctly, in **567066 guest
instructions with 97 native multiplication calls**. The committed-code
Nsight run contained exactly **one host cudaLaunchKernel and one VM kernel**.
SQLite audit found **zero host/device copies overlapping that kernel**;
initialization/result collection used five copies outside it. The kernel
took about 0.22 seconds; the profiled host API request took 0.707 seconds,
including initialization/transfers but excluding prior compilation. These
are demonstration timings, not a comparative speedup or LLM request latency.

The rebuilt wheel was installed with `uv pip install --no-deps --target` into
a temporary directory and imported from `/tmp`, outside the checkout, with
a fresh `GPU_RV32_CACHE`. Both C printf and real MicroPython factorial/power
execution passed. No additional virtual environment, PyTorch or model weights
were required. The wheel and source distribution include the pinned
MIT-licensed MicroPython source, custom port, C helpers and CUDA service.

The machine-readable [evidence](docs/evidence/c_micropython.json) records the
commit, complete profile command, GPU mapping, outputs, test status and raw
artifact hashes. Raw local artifacts are under
`runs/micropython_bigint_committed_01*` and `runs/c_micropython_tests_final.log`.
Guest features are standalone; the companion Qwen controller has not yet been
changed to use the larger MicroPython firmware or large-result feedback.
Native acceleration currently covers bounded multiplication; other bigint
operations execute as MicroPython guest code.

## Initial regression tests (before the C/MicroPython extension)

```bash
uv sync --locked
CUDA_VISIBLE_DEVICES=1 GPU_RV32_TESTS=1 GPU_RV32_CACHE="$PWD/.cache" \
  uv run --locked python -m unittest discover -s tests -v
```

All **53 test methods passed** in 6.915 s:

| Coverage | Methods |
| --- | --- |
| Host assembler encodings, labels, constants and invalid input | 6 |
| Interpreter compilation, image validation and packaged resources | 5 |
| Device assembler/core parity, full M extension, memory and budgets | 16 |
| Guest Python arithmetic, control flow, errors and limits | 21 |
| Native service resumption and cumulative budgets | 2 |
| Direct machine execution, input/RAM, subroutines and faults | 3 |

Without `GPU_RV32_TESTS=1`, the 11 CPU methods run and 42 GPU methods skip.
The default CI workflow uses that CPU-only configuration and also builds the
source distribution and wheel.

The device embedding example services two ECALL 42 requests in the same CUDA
kernel. It returns value 86, status 0, total 7 instructions and 2 service calls.
With budget 4 it stops at cumulative instruction 4, returning value 43/status 1.
The CPU does not service those guest calls.

## Installed-wheel execution

`uv build` produced the source archive and a Python wheel containing the
runtime CUDA translation unit, both device headers, and all five guest
C/assembly/linker files. The wheel was installed into a separate isolated
environment and run from `/tmp`, using a fresh native-build cache. The
imported package was in that environment's `site-packages`, with no checkout
on `sys.path`.

Both CPU-assembled and GPU-assembled execution of `12345*6789` returned
**83810205**, with identical machine code. Each invocation launched one VM
kernel. `examples/sum.py` returned **269859**, stdout `269859\n`, VM status 0
and guest error 0. The fixed interpreter contained **1813 RV32IM words** and
executed **1338897 instructions** for that program. Its single kernel took
516.753418 ms in this warm-context demonstration; the host API request took
517.528837 ms, excluding first compilation. These timings are evidence of
execution, not a performance comparison against native Python.

The CLI's `--compile-only` path was independently exercised. It saved the
guest ELF, initial RAM and compiler/source hashes without loading CUDA.
Runtime examples and test logs are generated artifacts and are not committed.
Extraction origins and original file hashes are recorded in
[`PROVENANCE.json`](PROVENANCE.json); standalone package/cache/API changes are
part of this repository's own history.

## LLM integration: one host graph launch

The RISC-V/Python execution layer was also validated in a companion model
integration: real Qwen3.8-27B INT8 weights, greedy decoding and thinking
disabled. The CPU prepared model weights, caches, interpreter RAM and captured
graphs, then launched one root CUDA Graph. That graph performed actual prompt
prefill, token generation, Python-tool detection, RISC-V execution, integer
result-token construction and model continuation through EOS.

Qwen generated a while/if program over integers 1..137. The guest executed
1,338,921 instructions and returned 269859; Qwen used that tool result in its
final answer. The request had 82 generated tokens, 35 injected tokens and 117
decode forwards, with 450 prefill tokens and 567 final cache positions.

The profiler's request window recorded:

| Observation | Value |
| --- | --- |
| Host CUDA Graph launches | **1** |
| Host kernel-launch API calls | **0** |
| Host memcpy API calls | **0** |
| Host/device copy activities | **0** |
| GPU root graph executions | **1** |
| GPU graph duration, including prefill | **8.224940565 s** |

The CUDA Graph contains multiple GPU kernels. One host launch is the measured
property; the inference-and-tool loop is not a single fused kernel. The
standalone Python API separately executes its entire guest invocation in one
kernel. Graph-level tracing does not collect individual kernel executions.

The [trace summary](docs/evidence/qwen_python_single_launch.json) records the
model revision, generated source, numerical check, profiler counts and source
hashes. Its companion experiment was captured from a working tree with local
changes; both its base commit and that dirty-tree state are recorded. The
standalone interpreter image is a separate build, so its 1813-word image and
standalone timings above are not substituted for the companion 1812-word run.
Setup, warmup and final output collection are outside the profiled window.
