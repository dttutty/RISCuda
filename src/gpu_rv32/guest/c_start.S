.section .text.start,"ax",@progbits
.global _start
_start:
    call main
    li a7, 0
    ecall
1:  j 1b
