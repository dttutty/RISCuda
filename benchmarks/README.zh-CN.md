# Qwenputer 算术配对评测

[English](README.md) | **简体中文**

已完成的本地 200 题实验中，**两组均答对 195/200**。在双方都正确完成的
194 个请求上，RISCuda 端到端耗时降低 **15.92%**（按题目进行 bootstrap
得到的 95% 区间为 14.15%–17.68%）。计入失败和截断请求后，累计耗时降低 14.73%。
199/200 对的完整文字 token 一致，全部 200 对的共同前缀均一致。
两个作业均以状态 0 退出，全部 400 次运行通过独立固件执行和反馈重放校验。
详见[结果表与限制](../VALIDATION.zh-CN.md#对齐的-200-题-qwenputer-算术评测)、
[证据](../docs/evidence/qwenputer_arithmetic_200.json)和
[逐对数据 CSV](../docs/evidence/qwenputer_arithmetic_200.csv)。

本评测复现公开的五组算术题，每组 40 题，随机种子为 0：
`arith`、`dmx_mul_big`、`dmx_div_big`、`dmx_mul_div_multiple_longer` 和
`dmx_add_or_sub_big`。DeepMind 数据来自官方 v1.0 压缩包：
https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz。
题目选择和精确有理数评分使用锁定版本的原作者加载器。
本地数据清单记录各文件的 SHA256；ModelScope 一致性清单检查全部八个权重分片，
以及推理和分词器配置。

两组加载同一份 BF16 Qwenputer 检查点，共享实际的语言层、归一化层、嵌入和输出头。
两组均使用原版预填充、相同的提示词 ID 和工具 schema、greedy 解码、关闭 thinking、
batch 1，以及 4096 个文字 token 和 200000 个芯片 token 的限制。
系统提示词使用原作者的 `NUDGE`，并附带其 boxed 答案后缀，不添加我们此前额外的结束提示。
标准答案不会进入模型消息或工具执行。

默认 RISCuda 组在 CUDA RV32IM 虚拟机中执行原作者**未修改的 calc.c**。
仅修改 bn.h/rt.h 绑定：行乘加、减法和加载通过 CUDA ECALL 0x201 操作
包含八个寄存器的十进制数字文件，标准输出使用客户机 RAM。
这保留了解析器、算术算法、寄存器限制、62 位十进制数限制和错误行为。
标量数字读取、扫描和打印作为客户机 RV32IM 指令执行。
详见 qwenputer_calc/PROVENANCE.zh-CN.md 和随附的上游 MIT 许可证。
可选的 `--backend micropython` 使用单独的精确有理数适配层；由于原作者的算术评测执行 C，
它不作为主要对照。

两组均保留主机 token 循环。这项对比测量的是替换权重内执行引擎的效果，
**并非无 CPU 参与的自主推理服务**。Qwenputer 组使用上游 RvChat 的 Triton 执行器；
RISCuda 在解码时替换为空闲文字嵌入，并使用相同的 80-token 捕获窗口注入字节 token 反馈。
文字一致性在生成结束后审计。

计时覆盖实际提示词预填充到 EOS，包括工具执行、反馈和主机调度。
加载、固定固件编译、预热和 CPU 参考检查不计入耗时。失败和截断请求仍保留在结果中。
同一实例上的工具执行和精确评分分别检查。芯片阶段耗时采用上游的轨迹 token 计时器，
不包含文字 token 上的电路计算。RISCuda 的工具耗时包含其 CUDA 分配、传输和同步 API 调用，
因此两组阶段计时的边界并不完全对称，以完整请求耗时为主要指标。

实验复用现有的对齐模型环境，不向独立运行时环境添加模型依赖：

```bash
uv run python benchmarks/prepare_data.py \
  --model ../m5gp/examples/qwen_rv32/models/Qwen3.5-4B-Qwenputer
CUDA_VISIBLE_DEVICES=0 GPU_RV32_CACHE="$PWD/.cache" \
  uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python benchmarks/compare_qwenputer.py --output runs/public_math
```

`--shard 0/2` 和 `--shard 1/2` 将相同的 200 道已选题目分配到分别检查过的 GPU；
每个配对的两组始终放在一起。记录内容包括 CPU 亲和性、GPU UUID、检查点版本、
编译器和源码哈希、完整命令、选定题目、原始生成 token、逐项耗时和退出状态。
`--resume` 要求数据、限制、源码和设备配置兼容。每完成一个题目的一组运行，
都会更新 `summary.json`。

两个分片作业结束后，运行独立的 CPU 审计器：

```bash
uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python benchmarks/audit_qwenputer.py runs/public_math
```

审计器重新生成选定题目，重新计算精确答案得分，检查提示词一致性，
在原版 CPU 参考机上重放每个由固件生成的 token 和反馈，验证虚拟机状态，并统计文字完全一致的配对。
分别报告全部请求，以及双方答案正确且正常 EOS 的共同子集的配对耗时比，
并使用固定种子的 2000 次重采样 bootstrap 计算置信区间。
`--allow-partial` 可在长任务运行期间审计已完成项，并按原始生成数据和审计器源码哈希缓存检查结果；
此时会将该实验标记为未完成。

本地环境使用 Torch 2.9/CUDA 12.8，未安装可选的 causal-conv1d 和 flash-linear-attention。
两组使用相同环境，但这与原作者较新的依赖声明不同；
公开的绝对耗时和准确率不应当作已对齐的本地基线。
