#include "gpu_guest.h"
char guest_output[16384];
unsigned guest_output_length;
int guest_error;
void guest_write(const char *s, size_t n) {
    while (n--) {
        if (guest_output_length >= sizeof(guest_output) - 1) { guest_error = 2; return; }
        guest_output[guest_output_length++] = *s++;
    }
    guest_output[guest_output_length] = 0;
}
__attribute__((weak)) int putchar(int c) { char b = c; guest_write(&b, 1); return (unsigned char)c; }
__attribute__((weak)) int puts(const char *s) { while (*s) putchar(*s++); putchar('\n'); return 0; }
