# C and MicroPython execution limits

**English** | [简体中文](guest-limits.zh-CN.md) · [Home](../README.md)

Both guest paths use 4 MiB RAM with at least 256 KiB reserved for stack.
MicroPython has a 1 MiB heap, 16383-byte source and 16383-byte captured stdout;
C accepts 65535-byte source and the same stdout bound. `error=1` reports an
uncaught MicroPython exception (traceback in stdout), 2 stdout overflow, 3 a
fatal guest error, and 4 an unsupported C format. `status` independently
reports machine faults/budget exhaustion. Default instruction budget is
200 million. These bounds prevent unbounded VM execution, not C memory bugs.

Each invocation executes in one kernel, including native bigint services. Check both VM status and guest error. See [VM status](rv32.md#vm-status), [C](c.md), and [MicroPython](micropython.md).
