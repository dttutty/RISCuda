# Installed-wheel execution

**English** | [简体中文](installed-wheel.zh-CN.md) · [Home](../../README.md) · [Validation index](../../VALIDATION.md)

Validated on Linux x86_64, Python 3.12.13, Zig 0.14.1, CUDA 12.9 and an NVIDIA
RTX A6000. The native target was `sm_80`. The fresh project environment had
exactly two installed packages: `gpu-rv32-python` and `ziglang`. No PyTorch,
Transformers, bitsandbytes, model checkpoint or original project import was
used for execution.

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
[`PROVENANCE.json`](../../PROVENANCE.json); standalone package/cache/API changes are
part of this repository's own history.
