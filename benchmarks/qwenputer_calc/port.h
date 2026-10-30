#include <stdint.h>
#include "gpu_guest.h"
#define rv_putchar putchar
static uint8_t decimal_digits[8][128];
#define BN_OPS(n,off,m,d,s,ovw) ((uint32_t)(d)|(uint32_t)(s)<<3|(uint32_t)(ovw)<<6|(uint32_t)(n)<<7|(uint32_t)(off)<<13|(uint32_t)(m)<<20)
#define BN_OFF(off) ((uint32_t)(off)<<13)
#define BN_M(m) ((uint32_t)(m)<<20)
static inline uint32_t row(unsigned kind,uint32_t ops,const char* ascii) {
    register uintptr_t a0 __asm__("a0")=kind;
    register uintptr_t a1 __asm__("a1")=(uintptr_t)decimal_digits;
    register uintptr_t a2 __asm__("a2")=ops;
    register uintptr_t a3 __asm__("a3")=(uintptr_t)ascii;
    register uintptr_t a7 __asm__("a7")=0x201;
    __asm__ volatile("ecall":"+r"(a0):"r"(a1),"r"(a2),"r"(a3),"r"(a7):"memory");
    return a0;
}
static inline uint32_t bn_mac(uint32_t ops) { return row(0,ops,0); }
static inline uint32_t bn_msb(uint32_t ops) { return row(1,ops,0); }
static inline void bn_ld(const char* end,uint32_t ops) { row(2,ops,end); }
static inline uint32_t bn_get(uint32_t ops) { return decimal_digits[ops&7][(ops>>13)&127]; }
static inline uint32_t bn_scan(const char* s) {
    unsigned n=0; while(n<63 && s[n]>='0' && s[n]<='9') ++n; return n;
}
static inline void bn_print(int r,int n) { while(n) putchar('0'+decimal_digits[r][--n]); }
