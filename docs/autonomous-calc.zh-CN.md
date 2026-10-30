# GPU 控制的 Qwen → C 计算器 → Qwen

[English](autonomous-calc.md) | **简体中文**

`gpu_rv32.resident.ResidentLoop` 将原生 C 计算器接入 batch 1、greedy 解码的 Qwen 循环。
初始化后，主机提交**一个 CUDA 执行图**，其中包含提示词预填充和由 GPU 控制的条件 WHILE 循环：

1. 在 GPU 上选择模型生成的 token，或队列中的工具反馈 token。
2. 执行已捕获的语言模型前向计算，并推进静态缓存。
3. 遇到 `</tool_call>` 时收集表达式，恢复客户机 RAM，在 GPU 上执行固定的
   RV32IM C 镜像及其原生十进制行服务。
4. 按原版字节 token 反馈协议编码结果，并送入同一模型，直到恢复生成。
5. 重复执行，直到 EOS 或达到固定预算，然后收集完整轨迹。

条件 IF/WHILE 执行图节点在设备端选择工作，不使用 CPU 回调、Python token 循环、
主机工具调度器或逐次调用启动。模型运行时，也不需要一个持续运行的 kernel 占用 SM。
一个请求包含许多 GPU kernel；**一次主机启动**描述的是提交边界，而不是设备端 kernel 数量。
该实现已在 RTX A6000 和 CUDA 12.9 上验证。

仍需要 CPU 完成初始化：加载和构建、对问题分词、分配和上传内存、捕获静态模型执行图，
以及实例化执行图。CPU 提交请求，等待一次，再下载最终状态。
这些边界操作位于 GPU 控制的生成和工具循环之外。

## 复现

使用已有模型环境：Torch 2.9、Transformers 5.21.1、Triton 3.5 和 Zig 0.14.1。
独立库不强制依赖 Torch。未经修改的本地 Qwenputer 仓库提供模型加载器、分词器协议和独立参考检查。
示例的默认路径指向同级仓库；需要时可通过 `--qwenputer` 和 `--model` 覆盖。

```bash
# First check the selected GPU is idle and healthy.
GPU_RV32_CACHE="$PWD/.cache" CUDA_VISIBLE_DEVICES=0 \
  uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python examples/autonomous_qwen_calc.py --host-control --output runs/autonomous

# A separate profiled request with two dependent tool calls:
GPU_RV32_CACHE="$PWD/.cache" CUDA_VISIBLE_DEVICES=0 \
  nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --cuda-graph-trace=graph --cuda-event-trace=false --capture-range=cudaProfilerApi \
    --capture-range-end=stop --output=runs/autonomous_profile \
  uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python examples/autonomous_qwen_calc.py --case chain --profile \
    --output runs/autonomous_profile_data

nsys export --type sqlite --output runs/autonomous_profile.sqlite \
  runs/autonomous_profile.nsys-rep
uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python benchmarks/audit_autonomous.py --runs runs/autonomous \
    --profile-run runs/autonomous_profile_data --sqlite runs/autonomous_profile.sqlite \
    --output runs/autonomous_audit.json
```

模型使用 Qwen3.5-4B-Qwenputer 共享的 BF16 语言网络，关闭 thinking。
原版检查点的预填充完整捕获；解码视图保留相同的语言层、归一化层和输出头，
将权重内计算器替换为原生 C 后端。C 客户机使用原作者未修改的计算器源码及现有移植绑定。
表达式来自模型实际生成，预期答案不会进入模型或工具输入。

示例在计时前预热请求。测量区间包含实际提示词预填充、全部生成、计算器执行和反馈；
不包含模型加载、编译、捕获、执行图实例化、初始缓存重置以及最终轨迹下载和验证。
采样器在原生提交包装函数内部启动和停止，仅对 GPU 组采样。

请使用上面的整图跟踪设置。所测试的 Nsight Systems 2025.1.3 在启用节点级执行图跟踪
和自动事件跟踪时，出现了非法内存访问；未采样的请求和整图跟踪均正确完成。
整图跟踪验证主机 API 调用及完整设备执行图区间，但不统计单独的图内 kernel 或拷贝节点。
每次工具执行和反馈通过最终状态及独立 CPU 参考重放校验。
采样时的耗时与未采样的请求耗时分开记录。

`--host-control` 用于隔离对照：CPU 每个 token 提交并检查一次**相同的已捕获 GPU 步骤**。
它不是此前的即时执行 Python 实现，工具检测和执行仍在 GPU 上完成。
将其 token 流与单次启动组比较，可检查控制器语义。比较二者耗时，可估计该捕获配置下
主机逐步控制的影响，但不能衡量相对于即时推理或本地 CPU 工具的综合收益。

## 接口与限制

调用方提供地址稳定的设备缓冲区、已捕获的预填充和解码执行图、词表字节和偏移及反馈表，
以及包含 `calc_in`、`guest_output`、`guest_error` 符号的 C 镜像。
`input`、`next` 和 `position` 为 int64 张量；单 token 全注意力掩码为 BF16。
位置从零开始。预填充必须写入首个下一 token ID；解码必须消费选定 token 并写入下一 ID。
所有执行图、缓冲区和缓存存储必须保留到请求完成，位于同一设备，且不被其他请求使用。
ctypes 接口无法保证无效或裸设备指针的安全性。

每个控制器仅可使用一次。上限为 4096 个生成 token、包含反馈在内的 8192 个总 token、
64 次调用、512 个排队反馈 token、4 MiB 客户机 RAM、1 MiB 代码，
并为每次调用设置累计客户机指令预算。示例将总 token 和生成 token 上限均设为 1024。
保留原版协议的 80 个 token 行、每行 16 字节、表达式解析器、62 位十进制数限制和错误消息。
后端是固定的 C 计算器，不是无限制的 Python 解释器或任意工具路由器。
MicroPython 仍可作为独立客户机运行；该控制器尚未路由到其固件。

停止码：1 为 EOS；2 为总 token 上限；3 为生成 token 上限；4 为缓存上限；
5 为工具调用上限；6 为输出或队列容量；7 为无效的下一 token ID。
非零客户机状态或错误会产生 `error` 工具反馈。调用记录包含源码、输出、虚拟机状态、
累计 RV32 指令数和原生服务次数。`elapsed_ns` 覆盖 GPU 工具门控、RAM 恢复、解释器和反馈编码，
不包含语言模型前向计算及最初的参数收集。

回归测试使用按脚本输出的已捕获解码器，检查实际反馈、顺序、掩码、位置、错误反馈、预算，
以及两次调用时主机和 GPU 控制的一致性。真实模型测试在完成后独立验证最终答案是否精确，
以及每次工具结果和原版反馈协议是否正确。
