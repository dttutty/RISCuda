# C, MicroPython and native bigint extension

**English** | [简体中文](c-micropython.zh-CN.md) · [Home](../../README.md) · [Validation index](../../VALIDATION.md)

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

The machine-readable [evidence](../../docs/evidence/c_micropython.json) records the
commit, complete profile command, GPU mapping, outputs, test status and raw
artifact hashes. Raw local artifacts are under
`runs/micropython_bigint_committed_01*` and `runs/c_micropython_tests_final.log`.
Guest features are standalone; the companion Qwen controller has not yet been
changed to use the larger MicroPython firmware or large-result feedback.
Native acceleration currently covers bounded multiplication; other bigint
operations execute as MicroPython guest code.
