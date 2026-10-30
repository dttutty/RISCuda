# 运行 C 程序

[English](c.md) | **简体中文** · [返回首页](../README.zh-CN.md)

命令从仓库根目录运行。先完成[安装](quickstart.zh-CN.md)。

```bash
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 c examples/bigint.c
```

```python
from gpu_rv32 import compile_c, run_c

c = compile_c('#include <stdio.h>\nint main(void) { printf("%d\\n", 2*8); return 0; }')
assert run_c(c)["stdout"] == "16\n"
```

`c` 在 GPU 执行前，使用 CPU 上的 Zig 编译 freestanding C。
入口为 `int main(void)`，返回值放在 `value`。
随包提供的辅助函数支持内存复制、`puts`、`putchar`、`guest_write` 和整数 `printf`
（`%d/%i/%u/%x/%X/%s/%c/%%`、可选的 `l/ll/z` 长度修饰符、字段宽度和补零）。
没有通用 libc、操作系统、malloc、Linux 系统调用或主机文件系统 API。

C 可通过 `gpu_guest.h` 中的 `gpu_big_mul` 调用 CUDA 大整数服务，详见[设备服务](device-api.zh-CN.md)。RAM、源码与输出上限、预算和错误码见[执行限制](guest-limits.zh-CN.md)。
