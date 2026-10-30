.section .text.start,"ax",@progbits
.global _start
_start:
    call run_python
    li a7, 0
    ecall
1:  j 1b
