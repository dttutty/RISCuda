# Run C programs

**English** | [简体中文](c.zh-CN.md) · [Home](../README.md)

Run commands from the repository root after [installation](quickstart.md).

```bash
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 c examples/bigint.c
```

```python
from gpu_rv32 import compile_c, run_c

c = compile_c('#include <stdio.h>\nint main(void) { printf("%d\\n", 2*8); return 0; }')
assert run_c(c)["stdout"] == "16\n"
```

`c` compiles freestanding C with Zig on the CPU before GPU execution. The entry
point is `int main(void)`; the return value is in `value`. Packaged helpers
provide memory copying, `puts`, `putchar`, `guest_write` and integer `printf`
(`%d/%i/%u/%x/%X/%s/%c/%%`, optional `l/ll/z`, width and zero padding).
There is no general libc, OS, malloc, Linux syscall or host filesystem API.

C can call the CUDA bigint service through `gpu_big_mul` in `gpu_guest.h`; see [device services](device-api.md). See [execution limits](guest-limits.md) for RAM, source/output bounds, budgets and errors.
