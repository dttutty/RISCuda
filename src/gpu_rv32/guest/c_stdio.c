/* Small freestanding integer formatter: no allocation or OS calls. */
#include <stdarg.h>
#include <stdint.h>
#include "gpu_guest.h"

int printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int count = 0;
    while (*format) {
        if (*format != '%') { putchar(*format++); ++count; continue; }
        ++format;
        if (*format == '%') { putchar(*format++); ++count; continue; }
        unsigned width = 0;
        char padding = *format == '0' ? '0' : ' ';
        while (*format >= '0' && *format <= '9') {
            width = width*10 + *format++ - '0';
            if (width > 16384) { guest_error = 4; goto done; }
        }
        int length = 0;
        while (*format == 'l' && length < 2) { ++length; ++format; }
        if (*format == 'z') { length = 1; ++format; }
        char kind = *format;
        if (!kind) { guest_error = 4; break; }
        ++format;
        if (kind == 's') {
            const char *s = va_arg(args, const char *);
            if (!s) s = "(null)";
            while (*s) { putchar(*s++); ++count; }
        } else if (kind == 'c') {
            putchar(va_arg(args, int)); ++count;
        } else if (kind == 'd' || kind == 'i' || kind == 'u' || kind == 'x' || kind == 'X') {
            uint64_t value;
            int negative = 0;
            if (kind == 'd' || kind == 'i') {
                int64_t v = length == 2 ? va_arg(args, long long) :
                            length == 1 ? va_arg(args, long) : va_arg(args, int);
                negative = v < 0;
                value = negative ? 0-(uint64_t)v : (uint64_t)v;
            } else {
                value = length == 2 ? va_arg(args, unsigned long long) :
                        length == 1 ? va_arg(args, unsigned long) : va_arg(args, unsigned);
            }
            unsigned base = kind == 'x' || kind == 'X' ? 16 : 10;
            const char *digits = kind == 'X' ? "0123456789ABCDEF" : "0123456789abcdef";
            char buffer[64]; unsigned n = 0;
            do { buffer[n++] = digits[value % base]; value /= base; } while (value);
            unsigned spaces = width > n+negative ? width-n-negative : 0;
            if (negative && padding == '0') { putchar('-'); ++count; }
            while (spaces--) { putchar(padding); ++count; }
            if (negative && padding != '0') { putchar('-'); ++count; }
            while (n) { putchar(buffer[--n]); ++count; }
        } else { guest_error = 4; break; }
    }
done:
    va_end(args);
    return guest_error ? -1 : count;
}
