# C、MicroPython 与原生大整数扩展

[English](c-micropython.md) | **简体中文** · [返回首页](../../README.zh-CN.md) · [验证索引](../../VALIDATION.zh-CN.md)

实现提交为 `fb9e34099332a8f5b19312a485e0a7b50173c9ce`。
扩展的完整回归测试在 markov 上已确认空闲的 RTX A6000 GPU 0 上运行，
UUID 为 `GPU-23d8809e-ca5a-3d0c-7a0f-26ab756ae05a`，
**67 项测试在 14.849 秒内全部通过**，退出状态为 0。

```bash
CUDA_VISIBLE_DEVICES=0 GPU_RV32_TESTS=1 GPU_RV32_CACHE="$PWD/.cache" \
  uv run python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 c examples/bigint.c
CUDA_VISIBLE_DEVICES=0 uv run gpu-rv32 micropython examples/bigint.py
```

原有 53 项测试仍然通过。新增 14 项覆盖 C/ELF 构建和边界、真实 MicroPython 固定固件、
C 循环和整数 printf（含最小 signed64 值）、函数和类、容器、生成器、异常、
存在活跃对象时的垃圾回收、阶乘和幂、有符号大整数算术和除法、输出溢出、指令预算、
输入最多 4096 位的原生 limb 乘法、拒绝无效请求时不修改 RAM、未知服务，
以及超过 4096 位时的精确客户机回退算法。

`examples/bigint.c` 成功返回，并调用一次大整数服务。
`examples/bigint.py` 正确计算 `100!` 和 `2**256`，
共执行 **567066 条客户机指令，调用 97 次原生乘法**。
已提交代码的 Nsight 运行包含恰好**一次主机 cudaLaunchKernel 和一个虚拟机 kernel**。
SQLite 审计发现**没有与该 kernel 重叠的主机与设备间拷贝**；
初始化和结果收集在 kernel 之外使用五次拷贝。kernel 耗时约 0.22 秒，
采样中的主机 API 请求耗时 0.707 秒，包含初始化和传输，不包含此前编译。
这些是演示计时，不是对照加速比或 LLM 请求耗时。

重新构建的 wheel 通过 `uv pip install --no-deps --target` 安装到临时目录，
从仓库之外的 `/tmp` 导入，并使用全新的 `GPU_RV32_CACHE`。
C printf 和真实 MicroPython 的阶乘及幂运算均通过。
无需额外虚拟环境、PyTorch 或模型权重。wheel 和源码分发包包含锁定版本、
采用 MIT 许可证的 MicroPython 源码、自定义移植层、C 辅助函数和 CUDA 服务。

机器可读的[证据](../../docs/evidence/c_micropython.json)记录了提交、完整采样命令、GPU 映射、
输出、测试状态和原始产物哈希。本地原始产物位于 `runs/micropython_bigint_committed_01*`
和 `runs/c_micropython_tests_final.log`。客户机功能可独立使用；配套 Qwen 控制器
尚未改为使用更大的 MicroPython 固件或大结果反馈。
原生加速目前仅覆盖有上限的乘法，其他大整数操作作为 MicroPython 客户机代码执行。
