# GPU RV32 + Python

A standalone CUDA software interpreter for real RV32IM machine instructions,
with a freestanding integer Python-subset interpreter compiled to RV32IM.
It uses the Python standard library, the pinned Zig toolchain and CUDA;
there is no model runtime dependency.

```bash
git clone https://github.com/dttutty/gpu-rv32-python.git
cd gpu-rv32-python
uv sync --locked
uv run gpu-rv32 python examples/sum.py --compile-only
# Select a checked idle NVIDIA GPU before execution:
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 asm examples/multiply.S
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 python examples/sum.py
```

```python
from gpu_rv32 import compile_program, run_program, run_assembly

print(run_assembly("li a0, 16\necall")["value"])
image = compile_program("print(2 * 8)\n")
print(run_program(image)["stdout"])
```

The current runtime targets Linux and Python 3.12+. GPU execution needs `nvcc`
(CUDA 12.x) and an Ampere-or-newer NVIDIA GPU.
The default compilation target is `sm_80`; set `GPU_RV32_CUDA_ARCH` to change it.
Native build products go to the user cache, configurable with `GPU_RV32_CACHE`.
The package includes all CUDA headers and guest C/assembly/linker sources.

## What runs where

The CPU assembles raw input or builds the fixed C interpreter with Zig,
initializes guest memory, launches one CUDA kernel and collects the result.
Inside that kernel the VM fetches/decodes real RISC-V instructions and executes
all guest branches, loops, calls and memory accesses. Python parsing and
evaluation execute as guest RISC-V instructions. Host Python does not evaluate
the supplied program.

Each standalone API invocation launches one kernel. Programs and results still
cross the host/device boundary at invocation boundaries. To integrate a longer
GPU-controlled workflow, embed the device headers and use `resume_memory` as
shown below; the standalone CLI does not orchestrate an LLM.

## Assembly API

```python
from gpu_rv32 import assemble, run_code, run_assembly

# CPU assembly, followed by one GPU VM invocation.
answer = run_assembly("li t0, 8\nmul a0, a0, t0\necall", input=17)
assert answer["status"] == 0 and answer["value"] == 136

# Assembly and execution both happen in the same GPU kernel.
answer = run_assembly("li a0, 16\necall", device_assembler=True)

# Direct machine code, with caller-supplied private RAM.
answer = run_code(assemble("lw a0, 0(zero)\necall"), memory=(42).to_bytes(4, "little"))
assert answer["value"] == 42
```

Raw execution accepts up to 16384 little-endian words and 1..65536 RAM bytes.
Assembly convenience calls use fresh 1024-byte RAM, with `sp=1024` and input
in `a0`. `x0` stays zero; arithmetic wraps modulo 2^32. ECALL halts and returns
`a0`; EBREAK reports a breakpoint. Misaligned/out-of-bounds accesses fail.
Code and writable RAM are separate, so modifying RAM does not change the
instruction array.

The integer sandbox implements ADD/SUB, comparisons, shifts, bitwise logic,
integer immediates, LUI/AUIPC, branches, JAL/JALR, byte/halfword/word loads and
stores, and all eight M-extension operations:
MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU. FENCE is a no-op for private RAM.
There is no privileged mode, CSR/MMU, Linux syscall ABI, floating point,
atomics, compressed instructions or full ISA compliance certification.

Both assemblers support full 32-bit `li`, ABI names or `x0`..`x31`, labels and
common pseudos (`mv`, `nop`, `j`, `call`, `ret`, zero-test/comparison branches).
Memory syntax is `offset(base)`. The CPU assembler additionally accepts AUIPC
and `jalr rd, rs1, imm`; the GPU text syntax is narrower than the VM's ISA.
They accept up to 1024 emitted instructions. The GPU assembler also limits
source to 8191 bytes and labels to 64. No assembler directives or relocations.

## Python subset

The guest supports signed 32-bit integers, variables, assignment, expressions,
`print(integer)`, `if/else`, `while`, `pass`, comments and spaces for indentation.
Expressions support `+ - * // %`, comparisons/chains, `& | ^ ~ << >>`,
`and/or/not`, and `True/False`. `//` and `%` use Python's floor rule; arithmetic
wraps at 32 bits. Negative shifts fail; shifts of 32 or more saturate to zero or
the sign bit, rather than allocating arbitrary-precision integers.

This is a bounded language subset: no imports, function definitions, strings,
containers, floats, `for/range`, `break/continue`, or augmented assignment.
The limits are 4095 UTF-8 source bytes, 256 nonblank statements, 32 variables,
31-byte names, 16 nested blocks/expression levels, and 4095 stdout bytes.
Guest RAM is 64 KiB, with at least 32 KiB reserved for stack. Every source uses
the same compiled interpreter; only initial source/budget RAM changes.

`compile_program(source, statement_budget=10000)` returns the fixed ELF,
machine code, initialized RAM, symbols and build hashes. It works without CUDA.
`run_program(image, instruction_budget=20000000)` returns `value`, `stdout`,
guest `error/error_name`, VM `status/status_name`, executed `steps`, GPU kernel
time and host request time. Always check both VM status and guest error.

| VM status | Meaning |
| --- | --- |
| 0 | ECALL halt |
| 1 | Instruction budget exhausted |
| 2 | Invalid instruction fetch |
| 3 | Illegal instruction |
| 4 | Invalid/alignment-failing data access |
| 5 | EBREAK |
| 6 | Native service pending, device embedding only |

Guest errors 0..6 mean success, syntax, undefined name, division by zero,
capacity/nesting, statement budget and negative shift. GPU assembly errors
use status 6 for syntax/range and 7 for assembler capacity; their `status_name`
distinguishes these from VM execution faults.

Save artifacts with `--output runs/new_directory`; the CLI writes input source,
binary or interpreter ELF/RAM, hashes, invocation metadata and execution JSON.
GPU timings exclude compilation, allocation and input/output transfers;
request timings include allocations/transfers but exclude first library build.
This VM demonstrates execution/control flow, not faster sequential Python.

## Embed GPU-side services

`native/include/rv32_core.cuh` exposes `rv32::Machine`, `initialize`,
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

## Tests and packaging

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v  # CPU; GPU cases skip
# After checking GPU occupancy and health:
CUDA_VISIBLE_DEVICES=0 GPU_RV32_TESTS=1 \
  uv run python -m unittest discover -s tests -v
uv build
```

CI runs CPU assembly/build/resource checks without a GPU. The opt-in GPU suite
covers arithmetic/division edge cases, branches/calls, memory traps, CPU/GPU
assembler parity, Python control flow/errors/limits and native yield/resume.
See [VALIDATION.md](VALIDATION.md) for the recorded standalone validation and
[PROVENANCE.json](PROVENANCE.json) for extraction origins and source hashes.
