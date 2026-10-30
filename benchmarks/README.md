# Paired Qwenputer arithmetic benchmark

This reproduces the public five arithmetic sets, 40 problems per set, seed 0:
`arith`, `dmx_mul_big`, `dmx_div_big`, `dmx_mul_div_multiple_longer`, and
`dmx_add_or_sub_big`. DeepMind files come from the official v1.0 archive at
https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz.
Problem selection and exact rational scoring use the pinned author's loaders.
The local data manifest records each file's SHA256; the ModelScope identity
manifest checks all eight checkpoint shards and inference/tokenizer configs.

Both arms load one BF16 Qwenputer checkpoint and share the actual language
layers, normalization, embeddings and output head. Both use the stock prefill,
same prompt IDs, tool schema, greedy decoding, disabled thinking, batch 1,
4096 text-token and 200000 chip-token limits. System prompt is the author's
`NUDGE`, with its boxed-answer suffix, without our previous extra finish hint.
No expected answer enters model messages or tool execution.

The default RISCuda arm runs the author's **unchanged calc.c** in the CUDA RV32IM
VM. Only bn.h/rt.h bindings change: row MAC/subtract/load use CUDA ECALL 0x201
over an eight-register decimal digit file, and stdout uses guest RAM. This
preserves its parser, arithmetic algorithms, register limits, 62-digit limits
and errors. Scalar digit reads, scans and printing run as guest RV32IM. See
qwenputer_calc/PROVENANCE.md and its upstream MIT license. The alternative
`--backend micropython` is a separate exact rational adapter; it is not the
primary comparison because the author's arithmetic benchmark executes C.

Both arms keep a host token loop. This comparison measures replacing the
in-weight execution engine, **not CPU-free autonomous serving**. The Qwenputer
arm uses upstream RvChat's Triton executor; RISCuda substitutes idle text
embeddings during decoding and injects byte-token feedback with the same
80-token capture window. Text equality is audited after generation.

Timings cover actual prompt prefill through EOS, tools, feedback and host
dispatch. Loading, fixed-firmware compilation, warmup and CPU reference checks
are excluded. Failed/truncated requests remain in the results. Same-instance
tool execution and exact scoring are checked separately. Chip time is the
upstream trace-token timer; it excludes circuit work on text tokens. RISCuda
tool time includes its CUDA allocation/transfer/synchronization API call, so
these stage timers are not perfectly symmetric; full request time is primary.

The existing matched environment is reused, rather than adding model packages
to the standalone runtime environment:

```bash
uv run python benchmarks/prepare_data.py \
  --model ../m5gp/examples/qwen_rv32/models/Qwen3.5-4B-Qwenputer
CUDA_VISIBLE_DEVICES=0 GPU_RV32_CACHE="$PWD/.cache" \
  uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python benchmarks/compare_qwenputer.py --output runs/public_math
```

`--shard 0/2` and `--shard 1/2` split the same 200 selected problems across
independently checked GPUs; both arms of a pair remain together. CPU affinity,
GPU UUIDs, checkpoint revisions, compiler/source hashes, complete command,
selected questions, raw generation tokens, per-item latency and exit status
are recorded. `--resume` requires compatible data, limits, source and device
configuration. `summary.json` is updated after every problem/arm.

After the two shard jobs finish, run the independent CPU auditor:

```bash
uv run --project ../m5gp/examples/qwen_rv32 --group python-vm --no-sync \
  python benchmarks/audit_qwenputer.py runs/public_math
```

It regenerates the selected problems, recomputes exact answer scores, checks
prompt identity, replays every firmware-owned token/response on the original
CPU reference machine, verifies VM status and counts text-identical pairs.
Paired latency ratios are reported for all requests and for the common subset
with correct answers and normal EOS, with a seeded 2000-resample bootstrap CI.
`--allow-partial` audits completed items during a long run and caches checks
by raw-generation and auditor-source hashes; it marks the run incomplete.

The local environment uses Torch 2.9/CUDA 12.8 and lacks optional causal-conv1d
and flash-linear-attention. This is shared by both arms and differs from the
author's newer dependency declarations; published absolute timings and accuracy
numbers should not be treated as a matched local baseline.
