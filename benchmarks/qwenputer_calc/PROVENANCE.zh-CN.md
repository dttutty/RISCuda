# 计算器源码来源记录

[English](PROVENANCE.md) | **简体中文**

`calc.c` 原样取自 divagr18/qwenputer，提交为
ca13310e2f9f788d1dd3feeea8213c1b0c23913c，路径为
experimental/riscv/chipformer/rv/env/calc/calc.c。MIT 许可证已复制到此目录。

仅在配对评测的构建中，将 bn.h 和 rt.h 的包含语句替换为 port.h。
解析器、算术算法、寄存器限制、62 位十进制数限制和格式保持不变。
GPU 原生 ECALL 0x201 在客户机内存中的数字文件上实现原版十进制行乘加、减法和加载操作；
标量读取、扫描和打印由 RV32IM 执行。整个计算器调用在一个 CUDA kernel 中完成，
无需主机算术求值器。
