#pragma once
#include <cstdint>

namespace rv32 {
// Decimal row primitives used by the public Qwenputer calc C firmware.
// ECALL 0x201: a0=kind(0 MAC,1 subtract,2 ASCII load), a1=8x128 byte
// digit-file pointer, a2=packed row operands, a3=ASCII end pointer for load.
__device__ inline uint32_t decimal_row(uint8_t* mem, uint32_t size, uint32_t kind,
                                      uint32_t base, uint32_t ops, uint32_t ascii) {
    uint32_t d=ops&7, s=(ops>>3)&7, overwrite=(ops>>6)&1;
    uint32_t n=(ops>>7)&63, offset=(ops>>13)&127, multiplier=(ops>>20)&15;
    if (multiplier>9) multiplier=9;
    if (kind>2 || size<1024 || base>size-1024 || n+offset>128) return 0xffffffffu;
    if (kind==2 && (ascii>size || ascii<n)) return 0xffffffffu;
    int32_t carry=0;
    for (uint32_t i=0;i<n;++i) {
        uint32_t at=base+128*d+offset+i;
        if (kind==2) { uint32_t v=mem[ascii-1-i]&15; mem[at]=v>9?9:v; }
        else if (kind==0) {
            uint32_t v=(overwrite?0:mem[at])+mem[base+128*s+i]*multiplier+carry;
            mem[at]=v%10; carry=v/10;
        } else {
            int32_t v=int32_t(mem[at])-int32_t(mem[base+128*s+i]*multiplier)-carry;
            carry=v<0?(9-v)/10:0;
            mem[at]=v+10*carry;
        }
    }
    return carry;
}
// Bounded CUDA-side multiplication service for unsigned base-2^16 limbs.
// Access as bytes: guest memory need not be host-aligned. Validate everything
// before touching RAM, reject output/input overlap, and leave RAM intact on error.
__device__ inline uint32_t big_mul(uint8_t* mem, uint32_t size, uint32_t out,
                                  uint32_t a, uint32_t an, uint32_t b, uint32_t bn) {
    if (!an || !bn || an > 256 || bn > 256) return 0xffffffffu;
    uint32_t n = an + bn, bytes = 2 * n;
    if (bytes > size || out > size - bytes || 2*an > size || a > size-2*an ||
        2*bn > size || b > size-2*bn || (out|a|b)&1) return 0xffffffffu;
    if ((out < a+2*an && a < out+bytes) || (out < b+2*bn && b < out+bytes)) return 0xffffffffu;
    for (uint32_t i=0; i<bytes; ++i) mem[out+i]=0;
    for (uint32_t j=0; j<bn; ++j) {
        uint32_t carry=0, bv=mem[b+2*j] | uint32_t(mem[b+2*j+1])<<8;
        for (uint32_t i=0; i<an; ++i) {
            uint32_t p=out+2*(i+j), av=mem[a+2*i] | uint32_t(mem[a+2*i+1])<<8;
            uint32_t t=av*bv + carry + (mem[p] | uint32_t(mem[p+1])<<8);
            mem[p]=t; mem[p+1]=t>>8; carry=t>>16;
        }
        uint32_t p=out+2*(j+an); mem[p]=carry; mem[p+1]=carry>>8;
    }
    while (n && !(mem[out+2*n-2] | mem[out+2*n-1])) --n;
    return n;
}
} // namespace rv32
