/* Freestanding compiler support; no OS, allocator or host Python calls. */
void *memset(void *dst, int value, unsigned size) {
    volatile unsigned char *d = dst;
    for (unsigned i = 0; i < size; ++i) d[i] = (unsigned char)value;
    return dst;
}
void *memcpy(void *dst, const void *src, unsigned size) {
    volatile unsigned char *d = dst;
    const volatile unsigned char *s = src;
    for (unsigned i = 0; i < size; ++i) d[i] = s[i];
    return dst;
}
void *memmove(void *dst, const void *src, unsigned size) {
    volatile unsigned char *d = dst;
    const volatile unsigned char *s = src;
    if (d < s) for (unsigned i = 0; i < size; ++i) d[i] = s[i];
    else for (unsigned i = size; i; --i) d[i - 1] = s[i - 1];
    return dst;
}
