# Lightweight integer Python subset

**English** | [简体中文](python-subset.zh-CN.md) · [Home](../README.md)

Run commands from the repository root after [installation](quickstart.md).

Use this bounded interpreter for small integer tools. For functions, containers or large integers, use [MicroPython](micropython.md).

```python
from gpu_rv32 import compile_program, run_program

image = compile_program("print(2 * 8)\n")
answer = run_program(image)
assert answer["status"] == answer["error"] == 0
print(answer["stdout"], end="")  # 16
```

The guest supports signed 32-bit integers, variables, assignment, expressions,
`print(integer)`, `if/else`, `while`, `pass`, comments and spaces for indentation.
Expressions support `+ - * // %`, comparisons/chains, `& | ^ ~ << >>`,
`and/or/not`, and `True/False`. `//` and `%` use Python's floor rule; arithmetic
wraps at 32 bits. Negative shifts fail; shifts of 32 or more produce zero for
left shifts, and zero or -1 for right shifts according to the sign of the input.

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

Guest errors 0..6 mean success, syntax, undefined name, division by zero,
capacity/nesting, statement budget and negative shift.

Check [VM status](rv32.md#vm-status) separately from guest errors. See [quick start](quickstart.md) for artifacts and timing.
