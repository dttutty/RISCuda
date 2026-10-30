# 安装 wheel 后执行

[English](installed-wheel.md) | **简体中文** · [返回首页](../../README.zh-CN.md) · [验证索引](../../VALIDATION.zh-CN.md)

验证环境为 Linux x86_64、Python 3.12.13、Zig 0.14.1、CUDA 12.9 和 NVIDIA RTX A6000，
原生编译目标为 `sm_80`。全新的项目环境仅安装了两个包：`gpu-rv32-python` 和 `ziglang`。
执行过程中未使用 PyTorch、Transformers、bitsandbytes、模型检查点，
也未导入原项目。

`uv build` 生成源码压缩包和 Python wheel，包含运行时 CUDA 编译单元、
两个设备端头文件及全部五个客户机 C、汇编和链接文件。
wheel 安装到单独的隔离环境中，从 `/tmp` 运行，使用全新的原生构建缓存。
导入的包来自该环境的 `site-packages`，`sys.path` 中没有仓库源码路径。

CPU 汇编和 GPU 汇编后执行 `12345*6789` 均返回 **83810205**，机器码相同。
每次调用启动一个虚拟机 kernel。`examples/sum.py` 返回 **269859**，
标准输出为 `269859\n`，虚拟机状态为 0，客户机错误为 0。
固定解释器包含 **1813 个 RV32IM 指令字**，该程序执行 **1338897 条指令**。
在这次已预热上下文的演示中，单个 kernel 耗时 516.753418 ms，
主机 API 请求耗时 517.528837 ms，不含首次编译。
这些计时用于证明执行完成，不是与原生 Python 的性能对比。

CLI 的 `--compile-only` 路径也已独立验证：无需加载 CUDA 即可保存客户机 ELF、
初始 RAM 及编译器和源码哈希。运行示例和测试日志属于生成产物，不提交到 Git。
提取来源和原文件哈希记录在 [`PROVENANCE.json`](../../PROVENANCE.json)；
独立包、缓存和 API 的修改记录在本仓库自身历史中。
