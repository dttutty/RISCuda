#ifndef GPU_GUEST_H
#define GPU_GUEST_H
#include <stddef.h>
#include <stdint.h>

/* Unsigned base-65536 limbs, least significant first. Distinct output buffer.
 * 1..256 limbs/input; destination has an+bn limbs. Returns used limb count,
 * or -1 for invalid pointers/count/overlap. No CPU callback is involved. */
static inline int gpu_big_mul(uint16_t *out, const uint16_t *a, unsigned an,
                              const uint16_t *b, unsigned bn) {
    register uintptr_t x0 __asm__("a0") = (uintptr_t)out;
    register uintptr_t x1 __asm__("a1") = (uintptr_t)a;
    register uintptr_t x2 __asm__("a2") = an;
    register uintptr_t x3 __asm__("a3") = (uintptr_t)b;
    register uintptr_t x4 __asm__("a4") = bn;
    register uintptr_t id __asm__("a7") = 0x200;
    __asm__ volatile("ecall" : "+r"(x0) : "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(id) : "memory");
    return (int)x0;
}
/* Captured stdout, 16 KiB, including terminator. Overflow sets guest_error=2. */
extern char guest_output[16384];
extern unsigned guest_output_length;
extern int guest_error;
int putchar(int c);
int puts(const char *s);
void guest_write(const char *s, size_t n);
#endif
