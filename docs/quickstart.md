# Quick start

**English** | [简体中文](quickstart.zh-CN.md) · [Home](../README.md)

## Requirements and installation

The current runtime targets Linux and Python 3.12+. GPU execution needs `nvcc`
(CUDA 12.x) and an Ampere-or-newer NVIDIA GPU.
The default compilation target is `sm_80`; set `GPU_RV32_CUDA_ARCH` to change it.
Native build products go to the user cache, configurable with `GPU_RV32_CACHE`.
The package includes all CUDA headers and guest C/assembly/linker sources.
The standalone runtime uses the Python standard library, the pinned Zig
toolchain and CUDA.

MicroPython also requires GNU make.

```bash
git clone https://github.com/dttutty/RISCuda.git
cd RISCuda
uv sync --locked
```

## Run an example

Run from the repository root after selecting an available GPU; adjust its index as needed. The CLI returns JSON; guest output is in `stdout`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 micropython examples/bigint.py
```

This computes `100!` and `2**256`. The first call builds and caches the runtime and interpreter.

## Choose a guest

| Path | Use it for |
| --- | --- |
| [MicroPython](micropython.md) | Functions, containers, exceptions and large integers |
| [C](c.md) | Freestanding C and native services |
| [RISC-V](rv32.md) | Assembly, machine code and private RAM |
| [Lightweight Python](python-subset.md) | Bounded 32-bit integer tools |

## Compile only, artifacts and timing

```bash
uv run gpu-rv32 python examples/sum.py --compile-only --output runs/compile_only
```

This command does not load CUDA.

Save artifacts with `--output runs/new_directory`; the CLI writes input source,
binary or interpreter ELF/RAM, hashes, invocation metadata and execution JSON.
GPU timings exclude compilation, allocation and input/output transfers;
request timings include allocations/transfers but exclude first library build.
This VM demonstrates execution/control flow, not faster sequential Python.
