# MicroPython 源码来源记录

[English](PROVENANCE.md) | **简体中文**

源码取自 https://github.com/micropython/micropython 的 v1.27.0，
提交为 78ff170de9e32c79db6e64d3e33d2bd60002bdcd（MIT 许可证，见 LICENSE）。

包含 py/、shared/libc/{string0.c,printf.c}、
shared/runtime/{gchelper_generic.c,gchelper.h}，以及
extmod/{virtpin.h,modplatform.h}。不需要上游子模块。
原有版权声明和许可证声明均已保留。

本地修改：当每个操作数的正数幅值最多包含 256 个以 65536 为基数的数位时，
py/mpz.c 调用移植层的 gpu_mpn_mul。MicroPython 的符号、内存分配和别名处理保持上游实现。
更大的乘积使用原有 mpn_mul 实现。py/makeqstrdefs.py 每次调用 Zig cc 预处理一个源文件，
最多使用四个工作进程。自定义移植层位于 guest/micropython/。
