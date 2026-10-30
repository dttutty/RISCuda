# 快速开始

[English](quickstart.md) | **简体中文** · [返回首页](../README.zh-CN.md)

## 环境与安装

当前运行时面向 Linux 和 Python 3.12+。GPU 执行需要 `nvcc`（CUDA 12.x）
以及 NVIDIA Ampere 或更新架构的 GPU。
默认编译目标为 `sm_80`，可通过 `GPU_RV32_CUDA_ARCH` 修改。
原生编译产物存放在用户缓存目录，可通过 `GPU_RV32_CACHE` 指定位置。
安装包包含全部 CUDA 头文件，以及客户机程序的 C、汇编和链接脚本源码。
独立运行时使用 Python 标准库、锁定版本的 Zig 工具链和 CUDA。

MicroPython 还需要 GNU make。

```bash
git clone https://github.com/dttutty/RISCuda.git
cd RISCuda
uv sync --locked
```

## 运行示例

从仓库根目录运行，先选择空闲 GPU 并按需要修改设备编号。CLI 返回 JSON，程序输出在 `stdout` 字段中。

```bash
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 micropython examples/bigint.py
```

该例计算 `100!` 和 `2**256`。首次调用会构建运行库和解释器，构建产物会缓存。

## 选择执行路径

| 路径 | 适用场景 |
| --- | --- |
| [MicroPython](micropython.zh-CN.md) | 函数、容器、异常和大整数 |
| [C](c.zh-CN.md) | freestanding C 与原生服务 |
| [RISC-V](rv32.zh-CN.md) | 汇编、机器码与私有 RAM |
| [轻量 Python](python-subset.zh-CN.md) | 固定容量的 32 位整数工具 |

## 仅编译、保存产物与计时

```bash
uv run gpu-rv32 python examples/sum.py --compile-only --output runs/compile_only
```

该命令无需加载 CUDA。

使用 `--output runs/new_directory` 可以保存运行产物。
命令行工具会写入输入源码、机器码或解释器 ELF/RAM、哈希、调用元数据和执行结果 JSON。
GPU 耗时不包含编译、内存分配和输入/输出传输；
请求耗时包含分配与传输，但不包含首次构建运行库的时间。
这个虚拟机用于验证执行和控制流，不代表顺序 Python 程序能够获得加速。
