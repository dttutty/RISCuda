# 首页动图

[English](README.md) | **简体中文** · [返回首页](../../README.zh-CN.md)

- [循环动图](riscuda-hero.gif)：1440 × 500，120 帧，10 秒循环。
- [静态封面](riscuda-hero.png)：点击首页动图即可打开。
- [生成脚本](../../tools/render_hero.py)：使用 Pillow 绘制的原创图形，无需模型或 GPU。

画面演示 Qwen 生成 `print(2 * 8)`，GPU RISC-V 核心运行 Python 解释器，
再将结果返回 Qwen。移动的激活点、指令活动和播放速度均为示意，并非 profiler 测量。
一次主机执行图启动的 Python 闭环来自[已有运行记录的配套原型](../results/qwen-python.zh-CN.md)；
仓库当前的 Qwen 控制器使用 C 计算器。

重新生成时，先在脚本列出的 Linux 系统字体路径中安装 Roboto 和 DejaVu 字体，
再从仓库根目录执行：

```bash
uv run --no-sync --with pillow==12.0.0 python tools/render_hero.py
```

Pillow 仅用于可选的图形生成，不是运行库依赖。中英文首页共用同一组图片。
生成器使用统一 GIF 调色板避免颜色闪烁；可加上
`--preview /tmp/riscuda-preview.png` 输出各阶段的预览拼图。
