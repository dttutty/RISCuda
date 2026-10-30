#include <stdint.h>
#include <alloca.h>
#define MICROPY_CONFIG_ROM_LEVEL MICROPY_CONFIG_ROM_LEVEL_CORE_FEATURES
#define MICROPY_ENABLE_COMPILER (1)
#define MICROPY_ENABLE_GC (1)
#define MICROPY_STACK_CHECK (1)
#define MICROPY_LONGINT_IMPL MICROPY_LONGINT_IMPL_MPZ
#define MICROPY_FLOAT_IMPL MICROPY_FLOAT_IMPL_NONE
#define MICROPY_ENABLE_EXTERNAL_IMPORT (0)
#define MICROPY_MODULE_FROZEN_MPY (0)
#define MICROPY_ROM_TEXT_COMPRESSION (0)
#define MICROPY_PY_IO (0)
#define MICROPY_PY_SYS (0)
#define MICROPY_PY_MICROPYTHON (0)
#define MICROPY_PY_ALL_SPECIAL_METHODS (1)
#define MICROPY_HELPER_REPL (0)
#define MICROPY_USE_INTERNAL_PRINTF (1)
#define MICROPY_USE_INTERNAL_ERRNO (1)
#define MICROPY_HW_BOARD_NAME "CUDA RV32IM"
#define MICROPY_HW_MCU_NAME "GPU software RISC-V"
#define MICROPY_ALLOC_PATH_MAX (128)
#define MP_STATE_PORT MP_STATE_VM
typedef long mp_off_t;
/* This port's only upstream change: bounded native multiplication fast path. */
unsigned gpu_mpn_mul(uint16_t *, const uint16_t *, unsigned, const uint16_t *, unsigned);
#define MICROPY_GPU_MPN_MUL (1)
