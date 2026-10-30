# C 与 MicroPython 执行限制

[English](guest-limits.md) | **简体中文** · [返回首页](../README.zh-CN.md)

两条客户机执行路径均使用 4 MiB RAM，至少保留 256 KiB 栈。
MicroPython 使用 1 MiB 堆，源码与 stdout 分别最多 16383 字节；C 源码最多
65535 字节，stdout 上限相同。`error=1` 为未捕获异常（stdout 含 traceback），
2 为输出溢出，3 为客户机致命错误，4 为不支持的 C 格式。
`status` 独立报告虚拟机故障和预算耗尽。默认指令预算为 2 亿。
这些边界限制可以防止虚拟机无限执行，但不能消除 C 程序自身的内存错误。

每次调用在一个 kernel 内完成，包含原生大整数服务。需要同时检查虚拟机状态和客户机错误。参见[虚拟机状态](rv32.zh-CN.md#虚拟机状态)、[C](c.zh-CN.md)和 [MicroPython](micropython.zh-CN.md)。
