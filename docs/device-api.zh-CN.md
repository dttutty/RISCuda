# 嵌入 CUDA 与设备服务接口

[English](device-api.md) | **简体中文** · [返回首页](../README.zh-CN.md)

命令从仓库根目录运行。

`src/gpu_rv32/native/include/rv32_core.cuh` 提供 `rv32::Machine`、`initialize`、
`execute_memory` 和 `resume_memory`。
当 `native=true` 时，`a7` 非零的 ECALL 会在推进 PC 后返回状态 6；
`a7=0` 则停止执行。寄存器、RAM 和累计指令数在恢复执行时保持不变。
外层 GPU kernel 负责处理请求、更新寄存器/内存，并恢复同一个客户机。

```bash
nvcc -std=c++17 -O2 -arch=sm_80 \
  -I src/gpu_rv32/native/include examples/native_resume.cu \
  -o /tmp/native_resume
CUDA_VISIBLE_DEVICES=0 /tmp/native_resume
# {"value":86,"status":0,"steps":7,"services":2}
```

该示例在一个 kernel 中处理两次自定义服务调用，没有 CPU 回调。
预算为 4 时，会在累计执行 4 条指令后停止，其中包含首次暂停前已执行的指令。
应用可以提供自己的服务处理逻辑，并将该接口嵌入 CUDA 执行图。
矩阵、文件系统和模型编排仍由应用层集成。

## 原生大整数服务

MicroPython 乘法会自动使用 **CUDA 大整数服务**，每个输入最多 4096 位，
输出最多 8192 位。更大的乘法回退到 MicroPython 在客户机中执行的精确算法；
加法、除法、取余及其他整数运算也在客户机中执行。
这是 CUDA 中的软件协处理器，没有 CPU 回调。C 可以通过 `gpu_guest.h` 中的
`gpu_big_mul` 调用它：输入为无符号、小端序、以 65536 为基数的数位（limb），
每个输入包含 1～256 个 limb，独立的输出缓冲区包含 `an+bn` 个 limb。
ECALL `a7=0x200` 通过 `a0`～`a4` 接收输出地址、A 地址、A 长度、B 地址、B 长度，
返回有效输出 limb 数；边界、对齐、数量或缓冲区重叠不合法时返回 -1。
结果中的 `bigint_calls` 统计服务调用次数。`run_code(..., services=True)` 开启此 ABI；
未知服务以 VM 状态 7（`unknown_service`）停止。客户机指令预算在服务调用之间累计。
每次乘法另有限制，最多执行 256×256 个 limb 乘积；这些原生操作不计作 RISC-V 指令步数。

## 十进制行服务

实验性 ECALL `0x201` 提供 8×128 字节十进制寄存器文件的行乘加、减法和
ASCII 数字加载，寄存器接口见 `src/gpu_rv32/native/include/bigint.cuh` 和
[计算器移植层](../benchmarks/qwenputer_calc/port.h)。`bigint_calls` 同时统计
原有 base-65536 乘法及新增十进制行操作。MicroPython 已开启运算符重载，
便于实现精确分数适配层。
