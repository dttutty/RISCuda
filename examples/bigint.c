#include "gpu_guest.h"

/* (2^64-1)^2, returned as eight base-65536 limbs by the native GPU service. */
int main(void) {
    uint16_t a[4] = {65535, 65535, 65535, 65535};
    uint16_t product[8];
    int used = gpu_big_mul(product, a, 4, a, 4);
    if (used != 8 || product[0] != 1 || product[4] != 65534 || product[7] != 65535) return 1;
    puts("GPU bigint multiplication passed");
    return 0;
}
