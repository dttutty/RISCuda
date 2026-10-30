# LLM integration: one host graph launch

**English** | [简体中文](qwen-python.zh-CN.md) · [Home](../../README.md) · [Validation index](../../VALIDATION.md)

The RISC-V/Python execution layer was also validated in a companion model
integration: real Qwen3.8-27B INT8 weights, greedy decoding and thinking
disabled. The CPU prepared model weights, caches, interpreter RAM and captured
graphs, then launched one root CUDA Graph. That graph performed actual prompt
prefill, token generation, Python-tool detection, RISC-V execution, integer
result-token construction and model continuation through EOS.

Qwen generated a while/if program over integers 1..137. The guest executed
1,338,921 instructions and returned 269859; Qwen used that tool result in its
final answer. The request had 82 generated tokens, 35 injected tokens and 117
decode forwards, with 450 prefill tokens and 567 final cache positions.

The profiler's request window recorded:

| Observation | Value |
| --- | --- |
| Host CUDA Graph launches | **1** |
| Host kernel-launch API calls | **0** |
| Host memcpy API calls | **0** |
| Host/device copy activities | **0** |
| GPU root graph executions | **1** |
| GPU graph duration, including prefill | **8.224940565 s** |

The CUDA Graph contains multiple GPU kernels. One host launch is the measured
property; the inference-and-tool loop is not a single fused kernel. The
standalone Python API separately executes its entire guest invocation in one
kernel. Graph-level tracing does not collect individual kernel executions.

The [trace summary](../../docs/evidence/qwen_python_single_launch.json) records the
model revision, generated source, numerical check, profiler counts and source
hashes. Its companion experiment was captured from a working tree with local
changes; both its base commit and that dirty-tree state are recorded. The
standalone interpreter image is a separate build, so its 1813-word image and
standalone timings in the installed-wheel record are not substituted for the companion 1812-word run.
Setup, warmup and final output collection are outside the profiled window.
