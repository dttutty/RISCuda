#include <string.h>
#include "gpu_guest.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/stackctrl.h"
#include "shared/runtime/gchelper.h"

char mp_source[16384];
static char heap[1048576] __attribute__((aligned(16)));
unsigned gpu_guest_bigint_calls;

unsigned gpu_mpn_mul(uint16_t *out, const uint16_t *a, unsigned an,
                      const uint16_t *b, unsigned bn) {
    int n = gpu_big_mul(out, a, an, b, bn);
    if (n < 0) mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("invalid GPU bigint request"));
    ++gpu_guest_bigint_calls;
    return n;
}

int main(void) {
    mp_stack_ctrl_init();
    mp_stack_set_limit(196608);
    gc_init(heap, heap+sizeof(heap));
    mp_init();
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, mp_source, strlen(mp_source), 0);
        qstr name = lex->source_name;
        mp_parse_tree_t tree = mp_parse(lex, MP_PARSE_FILE_INPUT);
        mp_obj_t module = mp_compile(&tree, name, false);
        mp_call_function_0(module);
        nlr_pop();
    } else {
        guest_error = 1;
        mp_obj_print_exception(&mp_plat_print, (mp_obj_t)nlr.ret_val);
    }
    mp_deinit();
    return guest_error;
}
void gc_collect(void) {
    gc_collect_start();
    gc_helper_collect_regs_and_stack();
    gc_collect_end();
}
mp_uint_t mp_hal_stdout_tx_strn(const char *s, size_t n) { guest_write(s, n); return n; }
void mp_hal_stdout_tx_strn_cooked(const char *s, size_t n) { guest_write(s, n); }
void nlr_jump_fail(void *val) {
    (void)val;
    guest_error = 3;
    __asm__ volatile("li a0, 3; li a7, 0; ecall" ::: "memory");
    for (;;) {}
}
void abort(void) { nlr_jump_fail(0); }
