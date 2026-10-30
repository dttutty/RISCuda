# 在 GPU 上运行 MicroPython

[English](micropython.md) | **简体中文** · [返回首页](../README.zh-CN.md)

命令从仓库根目录运行。先完成[安装](quickstart.zh-CN.md)。

```python
from gpu_rv32 import compile_micropython, run_micropython

image = compile_micropython("print(sum(i * i for i in range(1, 11)))\n")
answer = run_micropython(image)
assert answer["status"] == answer["error"] == 0
print(answer["stdout"], end="")  # 385
```

`micropython` 执行编译为 RV32IM 的 **真正的 MicroPython v1.27.0**，
其解析器、字节码编译器、虚拟机、内存分配器和垃圾回收器都在 GPU kernel
内运行。支持函数、类、`for/range`、列表、字典、字符串、生成器、异常和任意精度整数。
当前整数移植版关闭浮点、原生代码生成、外部文件导入、网络和文件 I/O；
它不是 CPython，也不是完整的桌面 Python 环境。
固定版本、采用 MIT 许可证的上游源码已随包附带，详见
[MicroPython 来源记录](../src/gpu_rv32/vendor/micropython/PROVENANCE.zh-CN.md)。
构建除 Zig 外还需要 GNU make，无需在运行时下载源码。

乘法对每个输入最多 4096 位的整数自动调用 CUDA 大整数服务，输出最多 8192 位。更大的乘法回退到客户机中的精确算法，加法、除法和取余等操作在客户机中执行。`bigint_calls` 统计原生服务调用次数。支持运算符重载，可实现精确分数适配层。参见[设备服务](device-api.zh-CN.md)与[执行限制](guest-limits.zh-CN.md)。

每次调用只启动一个 CUDA kernel，包含全部服务调用。当前 [Qwen 控制器](autonomous-calc.zh-CN.md)使用固定 C 计算器；较早的 Qwen/Python 原型使用[轻量 Python 子集](python-subset.zh-CN.md)。两者均未自动接入此 MicroPython 固件。

```bash
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 micropython examples/bigint.py
```
