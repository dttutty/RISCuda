# CUDA embedding and device services

**English** | [简体中文](device-api.zh-CN.md) · [Home](../README.md)

Run commands from the repository root.

`src/gpu_rv32/native/include/rv32_core.cuh` exposes `rv32::Machine`, `initialize`,
`execute_memory` and `resume_memory`. With `native=true`, ECALL with nonzero
`a7` yields status 6 after advancing PC; `a7=0` halts. Registers, RAM and total
instruction count survive resumption. The embedding GPU kernel services the
request, updates registers/memory and resumes the same guest.

```bash
nvcc -std=c++17 -O2 -arch=sm_80 \
  -I src/gpu_rv32/native/include examples/native_resume.cu \
  -o /tmp/native_resume
CUDA_VISIBLE_DEVICES=0 /tmp/native_resume
# {"value":86,"status":0,"steps":7,"services":2}
```

The example handles two custom service calls within one kernel, with no CPU
callbacks. Budget 4 stops at cumulative step 4, including work before the first
yield. Application-specific service handlers can embed this interface in a
CUDA execution graph. Matrix, filesystem and model orchestration remain
application-level integrations.

## Native bigint services

MicroPython multiplication automatically uses a **CUDA big-integer service**
for operands up to 4096 bits each (8192-bit output). Larger products fall back
to MicroPython's exact guest implementation; addition, division, remainder
and other integer operations also execute in the guest. This is a software
coprocessor in CUDA, with no CPU callback. C can call it through
`gpu_big_mul` in `gpu_guest.h`: unsigned little-endian base-65536 limbs,
1..256 limbs per input, distinct output buffer of `an+bn` limbs. ECALL `a7=0x200`
takes output/A/A-length/B/B-length in `a0`..`a4`, and returns used output limbs
or -1 on invalid bounds/alignment/count/overlap. Result `bigint_calls` counts
service invocations. `run_code(..., services=True)` enables this ABI; unknown
services stop with VM status 7 (`unknown_service`). Guest step budgets are
cumulative across service calls. Each multiplication is separately bounded
to at most 256×256 limb products; these native operations are not RV steps.

## Decimal-row services

The additional experimental ECALL `0x201` provides decimal digit-row MAC,
subtract and ASCII load over an 8×128-byte guest digit file; its register
binding is documented in `src/gpu_rv32/native/include/bigint.cuh` and the calculator's
[port](../benchmarks/qwenputer_calc/port.h). `bigint_calls` counts both base-65536
multiplication and decimal-row services. MicroPython operator overloading
is enabled for exact rational adapters.
