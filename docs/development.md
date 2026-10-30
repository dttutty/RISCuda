# Tests and packaging

**English** | [简体中文](development.zh-CN.md) · [Home](../README.md)

Run commands from the repository root.

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v  # CPU; GPU cases skip
# After checking GPU occupancy and health:
CUDA_VISIBLE_DEVICES=0 GPU_RV32_TESTS=1 \
  uv run python -m unittest discover -s tests -v
uv build
```

CI runs CPU assembly/build/resource checks without a GPU. The opt-in GPU suite
covers arithmetic/division edge cases, branches/calls, memory traps, CPU/GPU
assembler parity, Python control flow/errors/limits and native yield/resume.
See [VALIDATION.md](../VALIDATION.md) for the recorded standalone validation and
[PROVENANCE.json](../PROVENANCE.json) for extraction origins and source hashes.
