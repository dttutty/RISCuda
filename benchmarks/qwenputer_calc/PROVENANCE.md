`calc.c` is unmodified from divagr18/qwenputer, commit
ca13310e2f9f788d1dd3feeea8213c1b0c23913c, at
experimental/riscv/chipformer/rv/env/calc/calc.c. MIT license copied here.

For the paired benchmark only, the build substitutes its bn.h and rt.h
includes with port.h. The parser, arithmetic algorithm, register limits,
62-digit limits and formatting stay the same. GPU-native ECALL 0x201 implements
the original decimal MAC/subtract/load row primitives over a guest-memory digit
file; scalar reads/scans/printing use RV32IM. The entire calculator invocation
executes in one CUDA kernel, without a host arithmetic evaluator.
