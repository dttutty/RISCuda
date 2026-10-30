# RISCV_ON_CUDA

### RISC-V 核心搬进 CUDA。完整 Python 程序，一个 kernel 执行。

[English](README.md) | **简体中文**

[![CPU 测试](https://github.com/dttutty/RISCV_ON_CUDA/actions/workflows/test.yml/badge.svg)](https://github.com/dttutty/RISCV_ON_CUDA/actions/workflows/test.yml)

**给 NVIDIA GPU 一个自己的 RISC-V 执行核心。** 在 GPU 上执行真实的
RV32IM 机器码和完整的整数 Python 程序：解析、分支、循环、内存访问和
求值都在一个 CUDA kernel 内完成，无需 CPU 逐条调度指令，
也无需宿主机 Python 对输入程序求值。

| 工作负载 | CPU 提交方式 | 验证范围 |
| --- | --- | --- |
| 一段完整的 RISC-V 或整数 Python 程序 | **一次 CUDA kernel 调用** | 当前独立库 |
| 包含暂停/恢复的自定义工具服务 | **一个 CUDA kernel**，包含两次服务调用 | [设备端示例](examples/native_resume.cu) |
| Qwen 推理 → Python 执行 → Qwen 继续推理 | **一次主机 CUDA Graph 启动** | [配套推理原型](VALIDATION.md#llm-integration-one-host-graph-launch) |

推理集成的执行图内部包含多个 GPU kernel。一次主机启动覆盖提示词预填充、
生成、Python 工具执行、结果反馈，直到最终答案。
当前独立库提供其中的 RISC-V/Python 执行层和设备端服务接口。

**53 项本地测试通过 · 已验证安装 wheel 后独立执行 GPU 程序 ·
基础执行库无需 PyTorch 或模型权重**

## 快速开始

```bash
git clone https://github.com/dttutty/RISCV_ON_CUDA.git
cd RISCV_ON_CUDA
uv sync --locked
uv run gpu-rv32 python examples/sum.py --compile-only
# 执行前，请选择已经确认空闲且健康的 NVIDIA GPU：
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 asm examples/multiply.S
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 python examples/sum.py
```

```python
from gpu_rv32 import compile_program, run_program, run_assembly

print(run_assembly("li a0, 16\necall")["value"])
image = compile_program("print(2 * 8)\n")
print(run_program(image)["stdout"])
```

当前运行时面向 Linux 和 Python 3.12+。GPU 执行需要 `nvcc`（CUDA 12.x）
以及 NVIDIA Ampere 或更新架构的 GPU。
默认编译目标为 `sm_80`，可通过 `GPU_RV32_CUDA_ARCH` 修改。
原生编译产物存放在用户缓存目录，可通过 `GPU_RV32_CACHE` 指定位置。
安装包包含全部 CUDA 头文件，以及客户机解释器的 C、汇编和链接脚本源码。
独立运行时使用 Python 标准库、锁定版本的 Zig 工具链和 CUDA。

## CPU 和 GPU 各自执行什么

CPU 负责汇编输入，或使用 Zig 构建固定的 C 解释器，
随后初始化客户机内存、启动一次 CUDA kernel，并收集结果。
在该 kernel 内，虚拟机读取并解码真实的 RISC-V 指令，执行客户机程序的
全部分支、循环、函数调用和内存访问。Python 源码的解析与求值也通过
客户机 RISC-V 指令执行，宿主机 Python 不会直接求值输入程序。

每次调用独立 API 都会启动一个 kernel，程序和结果仍会在调用边界处
进行主机与设备之间的传输。如果需要接入由 GPU 控制的更长工作流，
可以嵌入设备端头文件并使用下文的 `resume_memory` 接口。
独立命令行工具本身不负责 LLM 编排。

## 面向推理的 GPU 内工具执行

设备端核心可以作为执行引擎，嵌入由 GPU 控制的推理循环：

```mermaid
flowchart LR
    Q[Qwen 推理] -->|生成 Python| P[RISC-V 核心 + Python 解释器]
    P -->|GPU 构造结果 token| Q
```

配套原型已使用真实 Qwen3.8-27B 权重跑通该循环：
**117 次解码前向、1,338,921 条客户机指令，仅需一次主机执行图启动**，
得到正确答案 **269859**。Nsight 记录显示，**请求期间主机 kernel 启动为零，
主机与设备之间的数据拷贝也为零**。GPU 执行图耗时 8.225 秒，包含提示词预填充。

这是单独的模型集成，其 CUDA 执行图内部包含多个 kernel。
验证范围和来源见[验证记录](VALIDATION.md#llm-integration-one-host-graph-launch)
与[轨迹摘要](docs/evidence/qwen_python_single_launch.json)。
独立库的单 kernel Python 执行路径，可以通过下文的命令行工具和 Python API 直接使用。

## 汇编 API

```python
from gpu_rv32 import assemble, run_code, run_assembly

# CPU 完成汇编，然后调用一次 GPU 虚拟机。
answer = run_assembly("li t0, 8\nmul a0, a0, t0\necall", input=17)
assert answer["status"] == 0 and answer["value"] == 136

# 汇编和执行都在同一个 GPU kernel 中完成。
answer = run_assembly("li a0, 16\necall", device_assembler=True)

# 直接执行机器码，并由调用方提供私有 RAM 初始内容。
answer = run_code(assemble("lw a0, 0(zero)\necall"), memory=(42).to_bytes(4, "little"))
assert answer["value"] == 42
```

直接执行机器码时，最多接受 16384 个小端序指令字，以及 1～65536 字节 RAM。
汇编便捷接口使用全新初始化的 1024 字节 RAM，`sp=1024`，输入值放在 `a0`。
`x0` 始终为零，算术运算按模 2^32 回绕。ECALL 停止执行并返回 `a0`，
EBREAK 返回断点状态。未对齐或越界的内存访问会失败。
代码与可写 RAM 相互独立，因此修改 RAM 不会改变指令数组。

整数沙箱实现了 ADD/SUB、比较、移位、位运算、整数立即数、LUI/AUIPC、
分支、JAL/JALR、字节/半字/字加载与存储，以及全部八条 M 扩展指令：
MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU。
对于私有 RAM，FENCE 不执行额外操作。
目前不包含特权模式、CSR/MMU、Linux 系统调用 ABI、浮点、原子操作、
压缩指令，也未通过完整 ISA 一致性认证。

两种汇编器均支持完整 32 位 `li`、ABI 寄存器名或 `x0`～`x31`、标签，
以及常见伪指令（`mv`、`nop`、`j`、`call`、`ret`、零值判断和比较分支）。
内存操作语法为 `offset(base)`。CPU 汇编器还支持 AUIPC 和
`jalr rd, rs1, imm`；GPU 汇编器开放的文本语法少于虚拟机可执行的指令。
两种汇编器最多生成 1024 条机器指令。GPU 汇编器还限制源码不超过
8191 字节、标签不超过 64 个。目前不支持汇编器指示符或重定位。

## Python 子集

客户机支持有符号 32 位整数、变量、赋值、表达式、`print(integer)`、
`if/else`、`while`、`pass`、注释和空格缩进。
表达式支持 `+ - * // %`、比较和链式比较、`& | ^ ~ << >>`、
`and/or/not` 和 `True/False`。`//` 与 `%` 遵循 Python 的向下取整规则，
算术运算在 32 位处回绕。负数移位量会报错；移位量为 32 或更大时，
左移结果为零，右移结果根据输入的符号为零或 -1。

这是一个有容量限制的语言子集：不支持导入、函数定义、字符串、容器、
浮点数、`for/range`、`break/continue` 或增量赋值。
限制为 4095 字节 UTF-8 源码、256 条非空语句、32 个变量、
31 字节变量名、16 层嵌套块/表达式，以及 4095 字节标准输出。
客户机 RAM 为 64 KiB，其中至少保留 32 KiB 用于栈。
所有源码均使用同一份已编译解释器，只修改初始 RAM 中的源码和预算。

`compile_program(source, statement_budget=10000)` 返回固定解释器的 ELF、
机器码、初始化 RAM、符号表和构建哈希。该步骤无需 CUDA。
`run_program(image, instruction_budget=20000000)` 返回 `value`、`stdout`、
客户机 `error/error_name`、虚拟机 `status/status_name`、已执行的 `steps`、
GPU kernel 耗时和主机请求耗时。使用结果时，应同时检查虚拟机状态和客户机错误。

| 虚拟机状态 | 含义 |
| --- | --- |
| 0 | ECALL 停止执行 |
| 1 | 指令预算耗尽 |
| 2 | 无效的取指地址 |
| 3 | 非法指令 |
| 4 | 无效或未对齐的数据访问 |
| 5 | EBREAK 断点 |
| 6 | 原生服务请求待处理，仅用于设备端嵌入 |

客户机错误码 0～6 分别表示成功、语法错误、未定义名称、除零、
容量/嵌套限制、语句预算耗尽和负数移位量。
GPU 汇编错误使用状态 6 表示语法/范围错误，状态 7 表示汇编器容量不足；
可通过 `status_name` 将其与虚拟机执行错误区分。

使用 `--output runs/new_directory` 可以保存运行产物。
命令行工具会写入输入源码、机器码或解释器 ELF/RAM、哈希、调用元数据和执行结果 JSON。
GPU 耗时不包含编译、内存分配和输入/输出传输；
请求耗时包含分配与传输，但不包含首次构建运行库的时间。
这个虚拟机用于验证执行和控制流，不代表顺序 Python 程序能够获得加速。

## 嵌入 GPU 端服务

`native/include/rv32_core.cuh` 提供 `rv32::Machine`、`initialize`、
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

## 测试与打包

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v  # CPU 检查；跳过 GPU 用例
# 检查 GPU 的占用和健康状态后再执行：
CUDA_VISIBLE_DEVICES=0 GPU_RV32_TESTS=1 \
  uv run python -m unittest discover -s tests -v
uv build
```

CI 无需 GPU，会运行 CPU 汇编、构建和资源检查。
手动启用的 GPU 测试覆盖算术/除法边界情况、分支/调用、内存异常、
CPU/GPU 汇编器一致性、Python 控制流/错误/容量限制，以及原生服务暂停/恢复。
独立验证记录见 [VALIDATION.md](VALIDATION.md)，
提取来源和源文件哈希见 [PROVENANCE.json](PROVENANCE.json)。
