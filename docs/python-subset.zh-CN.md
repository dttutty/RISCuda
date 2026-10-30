# 轻量整数 Python 子集

[English](python-subset.md) | **简体中文** · [返回首页](../README.zh-CN.md)

命令从仓库根目录运行。先完成[安装](quickstart.zh-CN.md)。

适合小型、固定容量的整数工具；需要函数、容器或大整数时，使用 [MicroPython](micropython.zh-CN.md)。

```python
from gpu_rv32 import compile_program, run_program

image = compile_program("print(2 * 8)\n")
answer = run_program(image)
assert answer["status"] == answer["error"] == 0
print(answer["stdout"], end="")  # 16
```

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

客户机错误码 0～6 分别表示成功、语法错误、未定义名称、除零、
容量/嵌套限制、语句预算耗尽和负数移位量。

需要分别检查[虚拟机状态](rv32.zh-CN.md#虚拟机状态)与客户机错误。产物保存与计时见[快速开始](quickstart.zh-CN.md)。
