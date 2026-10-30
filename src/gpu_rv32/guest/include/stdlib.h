#pragma once
#include <stddef.h>
void abort(void) __attribute__((noreturn));
int abs(int);
long strtol(const char *, char **, int);
