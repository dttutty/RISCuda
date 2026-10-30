# RISCuda

### RISC-V 核心搬进 CUDA。完整 Python 程序，一个 kernel 执行。

[English](README.md) | **简体中文**

[![CPU tests](https://github.com/dttutty/RISCuda/actions/workflows/test.yml/badge.svg)](https://github.com/dttutty/RISCuda/actions/workflows/test.yml)

**给 NVIDIA GPU 一个自己的 RISC-V 执行核心。** RISCuda 在 CUDA 中运行真实的 RV32IM 机器码，让 C 和 Python 程序在 GPU 上执行。

- **完整程序，一个 kernel：** 解析、分支、循环和求值在 GPU 内完成。
- **真正的 MicroPython：** 支持函数、容器、异常和任意精度整数。
- **GPU 内工具调用：** 可嵌入 Qwen 的 GPU 自主推理循环。

## 问 Qwen，让 GPU 执行 Python

下面的例子来自 Qwen3.8-27B 与轻量 Python 解释器的一次真实运行记录。

**1. 用户向 Qwen 提问：**

> 请用Python的while循环计算：i从1到137（含），如果i能被3整除，就把i的平方加入总和；否则从总和中减去2*i。总和初始为0，最后输出总和。

**2. Qwen 生成 Python，作为工具调用的程序：**

```python
i = 1
total = 0
while i <= 137:
    if i % 3 == 0:
        total = total + i * i
    else:
        total = total - 2 * i
    i = i + 1
print(total)
```

**3. RISCuda 的 GPU RISC-V 核心运行 Python 解释器**，执行上述程序，将结果 `269859` 反馈给 Qwen。

**4. Qwen 根据工具结果回答用户：** 最终总和为 **269859**。

这次记录中的循环只需**一次主机 CUDA Graph 启动**，生成、Python 执行和结果反馈均由 GPU 控制。该 Python 集成来自配套原型；当前仓库提供的 Qwen 控制器使用固定 C 计算器。详见[原始问题、生成代码与结果](docs/evidence/qwen_python_single_launch.json)和[验证范围](docs/results/qwen-python.zh-CN.md)。运行独立 Python 程序可从[快速开始](docs/quickstart.zh-CN.md)入手。

## 怎样执行

```mermaid
flowchart LR
    U[用户问题] --> Q[Qwen]
    Q -->|生成 Python| R[GPU 内的 RISC-V 与 Python]
    R -->|工具结果| Q
    Q -->|最终回答| A[用户]
```

CPU 负责准备、提交请求和最终取回答案。CUDA Graph 内包含多个 kernel；独立 Python 调用则在**一个 CUDA kernel** 内执行。独立 MicroPython 移植版支持整数计算。详见[执行架构](docs/architecture.zh-CN.md)与 [Qwen/C 集成](docs/autonomous-calc.zh-CN.md)。

## 实验结果

在 200 道配对算术题上，两组均答对 **195/200**；双方正确且正常结束的 194 个请求中，RISCuda 端到端总耗时降低 **15.9%**。该对比的两组都使用 CPU 解码循环，衡量执行引擎替换的效果。见[结果、方法与限制](docs/results/qwenputer-arithmetic.zh-CN.md)。

## 文档

| 我想…… | 阅读 |
| --- | --- |
| 安装并运行例子 | [快速开始](docs/quickstart.zh-CN.md) |
| 运行 Python | [MicroPython](docs/micropython.zh-CN.md) |
| 运行 C | [C 指南](docs/c.zh-CN.md) |
| 执行汇编或机器码 | [RV32 参考](docs/rv32.zh-CN.md) |
| 使用轻量整数解释器 | [Python 子集](docs/python-subset.zh-CN.md) |
| 嵌入 CUDA 或扩展服务 | [设备 API](docs/device-api.zh-CN.md) |
| 接入 Qwen | [自主推理](docs/autonomous-calc.zh-CN.md) |
| 复现性能对比 | [评测指南](benchmarks/README.zh-CN.md) |
| 查看验证证据 | [验证索引](VALIDATION.zh-CN.md) |
| 开发与测试 | [开发指南](docs/development.zh-CN.md) |

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

引用信息也提供 [BibTeX](CITATION.bib) 和
[Citation File Format](CITATION.cff) 格式，可用于 GitHub 的 **Cite this repository** 功能。
