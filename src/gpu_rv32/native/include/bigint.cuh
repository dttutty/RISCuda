#pragma once
#include <cstdint>

namespace rv32 {
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
