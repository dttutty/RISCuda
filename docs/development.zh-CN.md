# 测试与打包

[English](development.md) | **简体中文** · [返回首页](../README.zh-CN.md)

命令从仓库根目录运行。

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
独立验证记录见 [VALIDATION.zh-CN.md](../VALIDATION.zh-CN.md)，
提取来源和源文件哈希见 [PROVENANCE.json](../PROVENANCE.json)。
