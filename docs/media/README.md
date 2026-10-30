# Homepage animation

**English** | [简体中文](README.zh-CN.md) · [Home](../../README.md)

- [Animated banner](riscuda-hero.gif): 1440 × 500, 120 frames, 10-second loop.
- [Static poster](riscuda-hero.png): click the homepage banner to open this image.
- [Renderer](../../tools/render_hero.py): original artwork drawn with Pillow; no model or GPU needed.

The graphic illustrates Qwen generating `print(2 * 8)`, the GPU RISC-V core
running the Python interpreter, and the result returning to Qwen. The moving
activations, instruction ticks and playback speed are illustrative, not profiler
measurements. The one-host-graph-launch Python loop is a
[recorded companion prototype](../results/qwen-python.md); the repository's
current Qwen controller uses a C calculator.

To regenerate from the repository root, install Roboto and DejaVu fonts in
the Linux system font paths listed in the renderer, then run:

```bash
uv run --no-sync --with pillow==12.0.0 python tools/render_hero.py
```

Pillow is an optional rendering dependency, not a runtime dependency. Both
READMEs share the same assets. The renderer uses a shared GIF palette to avoid
color flicker, with an optional `--preview /tmp/riscuda-preview.png` contact sheet.
