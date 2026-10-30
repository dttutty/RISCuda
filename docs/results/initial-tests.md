# Initial regression tests (before the C/MicroPython extension)

**English** | [简体中文](initial-tests.zh-CN.md) · [Home](../../README.md) · [Validation index](../../VALIDATION.md)

Validated on Linux x86_64, Python 3.12.13, Zig 0.14.1, CUDA 12.9 and an NVIDIA
RTX A6000. The native target was `sm_80`. The fresh project environment had
exactly two installed packages: `gpu-rv32-python` and `ziglang`. No PyTorch,
Transformers, bitsandbytes, model checkpoint or original project import was
used for execution.

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
