# Standalone validation

Validated on Linux x86_64, Python 3.12.13, Zig 0.14.1, CUDA 12.9 and an NVIDIA
RTX A6000. The native target was `sm_80`. The fresh project environment had
exactly two installed packages: `gpu-rv32-python` and `ziglang`. No PyTorch,
Transformers, bitsandbytes, model checkpoint or original project import was
used for execution.

## Regression tests

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
