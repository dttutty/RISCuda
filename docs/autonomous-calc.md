# GPU-controlled Qwen → C calculator → Qwen

`gpu_rv32.resident.ResidentLoop` integrates the native C calculator with a
batch-one, greedy Qwen decode loop. After setup, the host submits **one CUDA
graph** containing prompt prefill and a GPU-controlled conditional WHILE loop:

1. Select a generated token or a queued tool-response token on the GPU.
2. Run the captured language forward pass and advance its static cache.
3. On `</tool_call>`, gather the expression, restore guest RAM, run the fixed
   RV32IM C image and its native decimal-row services on the GPU.
4. Encode the result as the original byte-token feedback protocol and feed
   it through the same model until generation resumes.
5. Repeat until EOS or a fixed budget, then collect the completed trace.

Conditional IF/WHILE graph nodes select work on the device. This does not use
a CPU callback, Python token loop, host tool dispatcher or per-call launch.
It also does not require a persistent kernel to occupy an SM while the model
runs. A request contains many GPU kernels; **one host launch** describes the
submission boundary, not the number of device kernels.
The implementation is validated with CUDA 12.9 on an RTX A6000.

CPU initialization is still required: load/build, tokenize the question,
allocate/upload memory, capture static model graphs and instantiate the graph.
The CPU submits the request, waits once, then downloads the final state.
These boundaries are outside the GPU-controlled generation/tool loop.

## Reproduction

Use the existing model environment with Torch 2.9, Transformers 5.21.1,
Triton 3.5 and Zig 0.14.1. The standalone library keeps no mandatory Torch
dependency. An unmodified local Qwenputer checkout supplies the model loader,
tokenizer protocol and independent reference checks. The example's defaults
point to sibling checkouts; override `--qwenputer` and `--model` as necessary.

```bash
# First check the selected GPU is idle and healthy.
GPU_RV32_CACHE="$PWD/.cache" CUDA_VISIBLE_DEVICES=0 \
  uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python examples/autonomous_qwen_calc.py --host-control --output runs/autonomous

# A separate profiled request with two dependent tool calls:
GPU_RV32_CACHE="$PWD/.cache" CUDA_VISIBLE_DEVICES=0 \
  nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop --output=runs/autonomous_profile \
  uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python examples/autonomous_qwen_calc.py --case chain --profile \
    --output runs/autonomous_profile_data
```

The model is the shared BF16 language stack from Qwen3.5-4B-Qwenputer,
thinking disabled. Stock checkpoint prefill is captured intact; the decode
view retains the same language layers, norm and output head and replaces the
in-weight calculator with the native C backend. The C guest is the author's
unchanged calculator source with the existing port bindings. Expressions come
from actual model generation, and no expected answer enters model/tool inputs.

The example warms the request before measuring it. The measured interval
includes actual prompt prefill, all generation, calculator work and feedback;
it excludes model loading, compilation, capture, graph instantiation, initial
cache reset and final trace download/validation. Profiler start/stop are inside
the native submission wrapper. Only the GPU arm is profiled.

`--host-control` is an isolation control: it submits/checks **the same captured
GPU step** from the CPU once per token. It is not the earlier eager Python
implementation, and it still detects/executes tools on the GPU. Equality of
its token stream and the one-launch arm checks controller semantics. Comparing
their durations estimates host step control under this capture configuration;
it does not measure the combined benefit over eager inference or local CPU tools.

## Interface and limits

The caller supplies stable device buffers, captured prefill/decode graphs,
vocabulary byte/offset/feedback tables and a C image with `calc_in`,
`guest_output`, `guest_error` symbols. `input`, `next` and `position` are
int64 tensors; the one-token full-attention mask is BF16. Positions are zero
based. Prefill must write the first next-token ID; decode must consume the
selected token and write the next ID. All graphs/buffers/cache storage must
remain alive, on the same device, and unused by other requests until completion.
Invalid/raw device pointers cannot be made safe by the ctypes interface.

Each controller is single-use. Bounds are 4096 generated tokens, 8192 total
tokens including feedback, 64 calls, 512 queued response tokens, 4 MiB guest
RAM, 1 MiB code and a cumulative guest instruction budget per call. The example
uses 1024 total/generated tokens. The original protocol's 80 token rows and
16 bytes per row, expression parser, 62-digit calculator limits and error
messages are retained. The backend is a fixed C calculator, not an unrestricted
Python interpreter or arbitrary tool router. MicroPython remains available as
a standalone guest; this controller does not yet route to its firmware.

Stop codes: 1 EOS; 2 total token limit; 3 generated token limit; 4 cache limit;
5 tool call limit; 6 output/queue capacity; 7 invalid next-token ID. A nonzero
guest status/error yields an `error` tool response. Call records include source,
output, VM status, cumulative RV32 instruction count and native service count.
`elapsed_ns` spans the GPU tool gate, RAM restore, interpreter and feedback
encoding; it excludes the language forward pass and initial parameter gathering.

Regression tests use a scripted captured decoder to check actual feedback,
ordering, masks, positions, error responses, budgets and two-call host/GPU
parity. Real-model tests independently verify exact final answers, every tool
result and the stock feedback protocol after completion.
