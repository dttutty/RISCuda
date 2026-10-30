# 初始回归测试（C/MicroPython 扩展之前）

[English](initial-tests.md) | **简体中文** · [返回首页](../../README.zh-CN.md) · [验证索引](../../VALIDATION.zh-CN.md)

验证环境为 Linux x86_64、Python 3.12.13、Zig 0.14.1、CUDA 12.9 和 NVIDIA RTX A6000，
原生编译目标为 `sm_80`。全新的项目环境仅安装了两个包：`gpu-rv32-python` 和 `ziglang`。
执行过程中未使用 PyTorch、Transformers、bitsandbytes、模型检查点，
也未导入原项目。

```bash
uv sync --locked
CUDA_VISIBLE_DEVICES=1 GPU_RV32_TESTS=1 GPU_RV32_CACHE="$PWD/.cache" \
  uv run --locked python -m unittest discover -s tests -v
```

全部 **53 项测试在 6.915 s 内通过**：

| 覆盖范围 | 测试数 |
| --- | --- |
| 主机汇编器编码、标签、常量和无效输入 | 6 |
| 解释器编译、镜像验证和随包资源 | 5 |
| 设备汇编器与核心一致性、完整 M 扩展、内存和预算 | 16 |
| 客户机 Python 算术、控制流、错误和限制 | 21 |
| 原生服务恢复和累计预算 | 2 |
| 直接执行机器码、输入和 RAM、子程序和故障 | 3 |

未设置 `GPU_RV32_TESTS=1` 时，运行 11 项 CPU 测试，跳过 42 项 GPU 测试。
默认 CI 工作流使用此纯 CPU 配置，并构建源码分发包和 wheel。

设备端嵌入示例在同一个 CUDA kernel 中处理两次 ECALL 42 请求，
返回值为 86、状态为 0，总计 7 条指令、2 次服务调用。
预算为 4 时，在累计第 4 条指令停止，返回值为 43、状态为 1。
CPU 不处理这些客户机调用。
