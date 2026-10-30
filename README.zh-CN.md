# RISCuda

Qwen → 原生 C 计算器 → Qwen 现在也支持 **GPU 自主循环**：CPU 启动一次 CUDA
Graph 后，调用检测、参数收集、RISC-V 执行、结果 token 回填和后续生成均由
GPU 推进。CPU 仍负责初始化、提交请求和最终取回结果。
详见[实现与复现说明](docs/autonomous-calc.md)。

### RISC-V 核心搬进 CUDA。完整 Python 程序，一个 kernel 执行。

[English](README.md) | **简体中文**

[![CPU 测试](https://github.com/dttutty/RISCuda/actions/workflows/test.yml/badge.svg)](https://github.com/dttutty/RISCuda/actions/workflows/test.yml)

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

**已验证安装 wheel 后独立执行 GPU 程序 · 支持 C 和真正的 MicroPython ·
基础执行库无需 PyTorch 或模型权重**

## 快速开始

```bash
git clone https://github.com/dttutty/RISCuda.git
cd RISCuda
uv sync --locked
uv run gpu-rv32 python examples/sum.py --compile-only
# 执行前，请选择已经确认空闲且健康的 NVIDIA GPU：
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 asm examples/multiply.S
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 python examples/sum.py
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 c examples/bigint.c
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 micropython examples/bigint.py
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

直接执行机器码时，最多接受 262144 个小端序指令字，以及 1～4194304 字节 RAM。
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

## C、MicroPython 和大整数服务

```python
from gpu_rv32 import compile_c, run_c, compile_micropython, run_micropython

c = compile_c('#include <stdio.h>\nint main(void) { printf("%d\\n", 2*8); return 0; }')
assert run_c(c)["stdout"] == "16\n"
answer = run_micropython(compile_micropython("print(2**256)\n"))
assert answer["status"] == answer["error"] == 0
print(answer["stdout"], answer["bigint_calls"])
```

`c` 接受以 `int main(void)` 为入口的 freestanding C，由 CPU 上的 Zig
预先编译成 RV32IM，随后整个程序在 GPU 中执行。返回值放在 `value`。
提供内存复制、`puts`、`putchar`、`guest_write` 和整数 `printf`，支持
`%d/%i/%u/%x/%X/%s/%c/%%`、`l/ll/z` 长度、字段宽度和补零。
没有完整 libc、操作系统、malloc、Linux 系统调用或主机文件系统访问。

`micropython` 执行的是真正的 **MicroPython v1.27.0**。它的源码解析器、
字节码编译器、虚拟机、内存分配器和垃圾回收器都作为 RISC-V 程序在 GPU
kernel 内运行，支持函数、类、`for/range`、列表、字典、字符串、生成器、
异常和任意精度整数。当前移植版关闭浮点、原生代码生成、外部文件导入、
网络和文件 I/O；它不是 CPython，也不是完整的桌面 Python 环境。
固定版本的 MIT 上游源码已随包附带，无需运行时下载。构建额外需要 GNU make。
来源与改动见 [MicroPython 记录](src/gpu_rv32/vendor/micropython/PROVENANCE.md)。

MicroPython 会自动把每个输入不超过 **4096 位**的大整数乘法交给 CUDA
实现的软件协处理器，乘积最多 8192 位。更大的乘法回退到 MicroPython
原有的任意精度算法，仍在 GPU guest 内执行；加法、除法、取模等运算也在
guest 内执行。C 可以通过 `gpu_guest.h` 中的 `gpu_big_mul` 调用同一服务。
输入是低位在前的 16 位 limbs，每个输入 1～256 个，输出缓冲区需要 `an+bn`
个，不能与输入重叠。ECALL `a7=0x200`，`a0`～`a4` 分别是输出地址、
A 地址、A 长度、B 地址、B 长度；返回有效输出长度，参数错误返回 -1。
返回结果中的 `bigint_calls` 是实际服务调用次数，全程没有 CPU 回调。
`run_code(..., services=True)` 开启此 ABI；未知服务返回 VM 状态 7。
指令预算在各次服务之间累计，单次原生乘法限制为最多 256×256 个 limb
乘积；原生工作量不计作 guest 指令数。

两个新后端均使用 4 MiB RAM，至少保留 256 KiB 栈。
MicroPython 使用 1 MiB 堆，源码与 stdout 分别最多 16383 字节；C 源码最多
65535 字节，stdout 上限相同。`error=1` 为未捕获异常（stdout 含 traceback），
2 为输出溢出，3 为 guest 致命错误，4 为不支持的 C 格式；VM 故障和预算耗尽
另由 `status` 报告。默认指令预算为 2 亿。这些限制并不能消除 C 程序自身的内存错误。

每次执行仍然只需 **一次 CPU kernel 启动**，包含全部大整数服务调用。
现有配套 Qwen 执行图仍使用下面的轻量 Python 子集，新后端不会自动替换它。

## 轻量 Python 子集

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

## 对齐 Qwenputer 的评测

公开 5 组、共 200 道算术题中，两边都答对 **195/200**。
对齐 BF16 语言权重、提示词和计算器接口后，在双方答对并正常结束的
194 道题上，RISCuda 端到端总耗时降低 **15.9%**。
199/200 对的完整文字 token 一致，另一对仅停止位置不同。
参见[完整结果与限制](VALIDATION.md#matched-200-problem-qwenputer-arithmetic-evaluation)
及[200 道逐题对照数据](docs/evidence/qwenputer_arithmetic_200.csv)。

[配对评测说明](benchmarks/README.md)使用作者公开的 5 组算术题，每组 40 题，
对齐 BF16 语言权重、提示词、greedy 解码和关闭 thinking 的设置。
默认 C 后端执行作者的同一份计算器源码，把十进制大整数行操作接到 CUDA
ECALL。这个对照保留两边的 CPU 解码循环，用于比较执行引擎，不代表 GPU
自主编排性能。答案评分和固件轨迹回放在计时后进行，失败样本同样保留。

实验性 ECALL `0x201` 提供 8×128 字节十进制寄存器文件的行乘加、减法和
ASCII 数字加载，寄存器接口见 `native/include/bigint.cuh` 和
[计算器移植层](benchmarks/qwenputer_calc/port.h)。`bigint_calls` 同时统计
原有 base-65536 乘法及新增十进制行操作。MicroPython 已开启运算符重载，
便于实现精确分数适配层。

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

## 引用

如果你在工作中使用了 RISCuda，请引用：

```bibtex
@software{zhao2026riscuda,
  author  = {Zhao, Lei and Lee, Chul-Ho},
  title   = {{RISCuda}: a {RISC-V} core and {Python} interpreter on {CUDA}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/dttutty/RISCuda}
}
```

也可使用 [BibTeX 文件](CITATION.bib)，或通过 [CITATION.cff](CITATION.cff)
使用 GitHub 的 **Cite this repository** 功能获取引用格式。
