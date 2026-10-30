# Run MicroPython on the GPU

**English** | [简体中文](micropython.zh-CN.md) · [Home](../README.md)

Run commands from the repository root after [installation](quickstart.md).

```python
from gpu_rv32 import compile_micropython, run_micropython

image = compile_micropython("print(sum(i * i for i in range(1, 11)))\n")
answer = run_micropython(image)
assert answer["status"] == answer["error"] == 0
print(answer["stdout"], end="")  # 385
```

`micropython` runs **real MicroPython v1.27.0** compiled to RV32IM, with its
parser, bytecode compiler, VM, allocator and garbage collector all executing
inside the GPU kernel. Functions, classes, `for/range`, lists, dictionaries,
strings, generators, exceptions and arbitrary-precision integers are available.
The integer-only port disables floating point, native emitters, external file
imports, networking and file I/O; it is not CPython or a full desktop Python.
The pinned MIT-licensed upstream sources are packaged; see
[MicroPython provenance](../src/gpu_rv32/vendor/micropython/PROVENANCE.md).
Building requires GNU make in addition to Zig, with no runtime source download.

Multiplication automatically uses CUDA bigint services for operands up to 4096 bits each (8192-bit output). Larger products fall back to exact guest arithmetic; addition, division and remainder execute in the guest. `bigint_calls` counts native services. Operator overloading is enabled for exact rational adapters. See [device services](device-api.md) and [execution limits](guest-limits.md).

Each invocation launches one CUDA kernel, including all services. The current [Qwen controller](autonomous-calc.md) uses a fixed C calculator; the earlier Qwen/Python prototype uses the [lightweight subset](python-subset.md). Neither automatically routes to this MicroPython firmware.

```bash
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 micropython examples/bigint.py
```
