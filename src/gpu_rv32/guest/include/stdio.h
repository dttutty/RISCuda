#pragma once
#include <stddef.h>
#include <stdarg.h>
typedef struct gpu_FILE FILE;
int printf(const char *, ...);
int snprintf(char *, size_t, const char *, ...);
int vsnprintf(char *, size_t, const char *, va_list);
int putchar(int);
int puts(const char *);
#define EOF (-1)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
