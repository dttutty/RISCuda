#pragma once

#include <cstdint>

// Bounded, two-pass device assembler. No host parser participates in a tool call.
namespace gpu_asm {
constexpr int MAX_CODE = 1024, MAX_LABELS = 64, NAME = 40;
struct Label { char name[NAME]; int pc; };
__device__ inline bool equal(const char* a, const char* b) {
    while (*a && *a == *b) { ++a; ++b; }
    return *a == *b;
}
__device__ inline bool space(char c) { return c == ' ' || c == '\t' || c == '\r'; }
__device__ inline bool number(const char* s, int64_t& value) {
    bool neg = *s == '-';
    if (*s == '-' || *s == '+') ++s;
    int base = 10;
    if (s[0] == '0' && s[1] == 'x') { base = 16; s += 2; }
    if (!*s) return false;
    uint64_t v = 0;
    while (*s) {
        int d = *s >= '0' && *s <= '9' ? *s - '0' : *s >= 'a' && *s <= 'f' ? *s - 'a' + 10 : -1;
        if (d < 0 || d >= base || v > 0xffffffffull / base) return false;
        v = v * base + d;
        if (v > 0xffffffffull) return false;
        ++s;
    }
    value = neg ? -static_cast<int64_t>(v) : static_cast<int64_t>(v);
    return true;
}
__device__ inline int reg(const char* s) {
    if (s[0] == 'x') { int64_t n; if (number(s + 1, n) && n >= 0 && n < 32) return n; }
    const char* names[] = {"zero","ra","sp","gp","tp","t0","t1","t2","s0","s1",
        "a0","a1","a2","a3","a4","a5","a6","a7","s2","s3","s4","s5","s6","s7",
        "s8","s9","s10","s11","t3","t4","t5","t6"};
    for (int i = 0; i < 32; ++i) if (equal(s, names[i])) return i;
    return equal(s, "fp") ? 8 : -1;
}
__device__ inline bool fits(int64_t n, int bits) { return n >= -(1ll << (bits - 1)) && n < (1ll << (bits - 1)); }
__device__ inline bool memory(const char* operand, int& base, int64_t& offset) {
    char displacement[NAME] = {}, name[NAME] = {};
    int i = 0, n = 0;
    while (operand[i] && operand[i] != '(') {
        if (i == NAME - 1) return false;
        displacement[i] = operand[i]; ++i;
    }
    if (!i || operand[i] != '(') return false;
    ++i;
    while (operand[i] && operand[i] != ')') {
        if (n == NAME - 1) return false;
        name[n++] = operand[i++];
    }
    if (!n || operand[i] != ')' || operand[i + 1]) return false;
    base = reg(name);
    return base >= 0 && number(displacement, offset) && fits(offset, 12);
}
__device__ inline uint32_t iw(int rd, int rs, int64_t imm, int f3 = 0) {
    return ((uint32_t(imm) & 4095) << 20) | (rs << 15) | (f3 << 12) | (rd << 7) | 0x13;
}
__device__ inline bool emit(uint32_t* out, int& count, uint32_t w) {
    if (count >= MAX_CODE) return false;
    out[count++] = w;
    return true;
}

// Returns 0 on success, 6 on syntax/range errors, 7 on capacity errors.
__device__ inline int assemble(const char* text, uint32_t* out, int& length, int& error_line) {
    Label labels[MAX_LABELS];
    int label_count = 0;
    for (int pass = 0; pass < 2; ++pass) {
        int cursor = 0, count = 0, line = 0;
        while (text[cursor]) {
            ++line;
            error_line = line;
            char fields[4][NAME] = {};
            int n = 0, width = 0;
            bool comment = false;
            while (text[cursor] && text[cursor] != '\n') {
                char c = text[cursor++];
                if (comment) continue;
                if (c == '#') { comment = true; continue; }
                if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
                if (c == ':' && n == 0 && width) {
                    fields[0][width] = 0;
                    if (pass == 0) {
                        if (label_count == MAX_LABELS) return 7;
                        for (int k = 0; k < label_count; ++k) if (equal(labels[k].name, fields[0])) return 6;
                        for (int k = 0; k <= width; ++k) labels[label_count].name[k] = fields[0][k];
                        labels[label_count++].pc = count * 4;
                    }
                    fields[0][0] = 0; width = 0;
                } else if (space(c) || c == ',') {
                    if (width) { fields[n][width] = 0; ++n; width = 0; }
                } else {
                    if (n == 4 || width == NAME - 1) return 6;
                    fields[n][width++] = c;
                }
            }
            if (text[cursor] == '\n') ++cursor;
            if (width) { fields[n][width] = 0; ++n; }
            if (!n) continue;
            const char *op = fields[0], *a = fields[1], *b = fields[2], *c = fields[3];
            int rd = reg(a), rs = reg(b), rt = reg(c);
            int64_t imm = 0;
            uint32_t w = 0;
            if (equal(op, "li")) {
                if (n != 3 || rd < 0 || !number(b, imm) || imm < -2147483648ll || imm > 0xffffffffll) return 6;
                if (imm >= 2147483648ll) imm -= 4294967296ll;
                if (fits(imm, 12)) w = iw(rd, 0, imm);
                else {
                    int64_t upper = (imm + 2048) >> 12, lower = imm - upper * 4096;
                    if (!emit(out, count, ((uint32_t(upper) & 0xfffff) << 12) | (rd << 7) | 0x37)) return 7;
                    if (!lower) continue;
                    w = iw(rd, rd, lower);
                }
            } else if (equal(op, "mv")) {
                if (n != 3 || rd < 0 || rs < 0) return 6;
                w = iw(rd, rs, 0);
            } else if (equal(op, "addi") || equal(op, "slti") || equal(op, "sltiu") ||
                       equal(op, "xori") || equal(op, "ori") || equal(op, "andi")) {
                if (n != 4 || rd < 0 || rs < 0 || !number(c, imm) || !fits(imm, 12)) return 6;
                int f3 = equal(op, "addi") ? 0 : equal(op, "slti") ? 2 : equal(op, "sltiu") ? 3 :
                         equal(op, "xori") ? 4 : equal(op, "ori") ? 6 : 7;
                w = iw(rd, rs, imm, f3);
            } else if (equal(op, "slli") || equal(op, "srli") || equal(op, "srai")) {
                if (n != 4 || rd < 0 || rs < 0 || !number(c, imm) || imm < 0 || imm > 31) return 6;
                w = iw(rd, rs, imm | (equal(op, "srai") ? 0x400 : 0), equal(op, "slli") ? 1 : 5);
            } else if (equal(op, "add") || equal(op, "sub") || equal(op, "mul") ||
                       equal(op, "mulh") || equal(op, "mulhsu") || equal(op, "mulhu") ||
                       equal(op, "div") || equal(op, "divu") || equal(op, "rem") || equal(op, "remu") ||
                       equal(op, "sll") || equal(op, "slt") || equal(op, "sltu") ||
                       equal(op, "xor") || equal(op, "srl") || equal(op, "sra") ||
                       equal(op, "or") || equal(op, "and")) {
                if (n != 4 || rd < 0 || rs < 0 || rt < 0) return 6;
                int m3 = equal(op, "mul") ? 0 : equal(op, "mulh") ? 1 : equal(op, "mulhsu") ? 2 :
                         equal(op, "mulhu") ? 3 : equal(op, "div") ? 4 : equal(op, "divu") ? 5 :
                         equal(op, "rem") ? 6 : equal(op, "remu") ? 7 : -1;
                int f3 = m3 >= 0 ? m3 : equal(op, "sll") ? 1 : equal(op, "slt") ? 2 : equal(op, "sltu") ? 3 :
                         equal(op, "xor") ? 4 : (equal(op, "srl") || equal(op, "sra")) ? 5 :
                         equal(op, "or") ? 6 : equal(op, "and") ? 7 : 0;
                int f7 = m3 >= 0 ? 1 : (equal(op, "sub") || equal(op, "sra")) ? 32 : 0;
                w = (f7 << 25) | (rt << 20) | (rs << 15) | (f3 << 12) | (rd << 7) | 0x33;
            } else if (equal(op, "lb") || equal(op, "lbu") || equal(op, "lh") || equal(op, "lhu") ||
                       equal(op, "lw") || equal(op, "sb") || equal(op, "sh") || equal(op, "sw")) {
                int base = -1;
                if (n != 3 || rd < 0 || !memory(b, base, imm)) return 6;
                bool load = op[0] == 'l';
                int f3 = equal(op, "lb") || equal(op, "sb") ? 0 : equal(op, "lh") || equal(op, "sh") ? 1 :
                         equal(op, "lbu") ? 4 : equal(op, "lhu") ? 5 : 2;
                uint32_t u = uint32_t(imm) & 4095;
                if (load) w = (u << 20) | (base << 15) | (f3 << 12) | (rd << 7) | 3;
                else w = ((u >> 5) << 25) | (rd << 20) | (base << 15) | (f3 << 12) | ((u & 31) << 7) | 0x23;
            } else if (equal(op, "ecall") || equal(op, "nop")) {
                if (n != 1) return 6;
                w = equal(op, "ecall") ? 0x73 : 0x13;
            } else if (equal(op, "lui")) {
                if (n != 3 || rd < 0 || !number(b, imm) || imm < -(1 << 19) || imm >= (1 << 20)) return 6;
                w = ((uint32_t(imm) & 0xfffff) << 12) | (rd << 7) | 0x37;
            } else {
                const char* target = c;
                bool jump = equal(op, "j") || equal(op, "jal");
                int f3 = equal(op, "beq") ? 0 : equal(op, "bne") ? 1 : equal(op, "blt") ? 4 :
                         equal(op, "bge") ? 5 : equal(op, "bltu") ? 6 : equal(op, "bgeu") ? 7 : -1;
                bool zero_test = equal(op, "beqz") || equal(op, "bnez") || equal(op, "bltz") ||
                                 equal(op, "bgez") || equal(op, "bgtz") || equal(op, "blez");
                bool swapped = equal(op, "bgt") || equal(op, "ble") || equal(op, "bgtu") ||
                               equal(op, "bleu") || equal(op, "bgtz") || equal(op, "blez");
                if (equal(op, "beqz")) f3 = 0;
                else if (equal(op, "bnez")) f3 = 1;
                else if (equal(op, "bgt") || equal(op, "bltz") || equal(op, "bgtz")) f3 = 4;
                else if (equal(op, "ble") || equal(op, "bgez") || equal(op, "blez")) f3 = 5;
                else if (equal(op, "bgtu")) f3 = 6;
                else if (equal(op, "bleu")) f3 = 7;
                if (jump) {
                    if (equal(op, "j")) { if (n != 2) return 6; target = a; rd = 0; }
                    else { if (n != 3 || rd < 0) return 6; target = b; }
                } else {
                    if (zero_test) {
                        if (n != 3 || rd < 0) return 6;
                        target = b; rs = 0;
                    } else if (f3 < 0 || n != 4 || rd < 0 || rs < 0) return 6;
                    if (swapped) { int temporary = rd; rd = rs; rs = temporary; }
                }
                if (!number(target, imm)) {
                    bool found = false;
                    for (int k = 0; k < label_count; ++k) if (equal(labels[k].name, target)) {
                        imm = labels[k].pc - count * 4; found = true; break;
                    }
                    if (pass == 1 && !found) return 6;
                }
                if (pass == 0) imm = 0;
                if ((imm & 3) || !fits(imm, jump ? 21 : 13)) return 6;
                uint32_t u = uint32_t(imm);
                if (jump) w = (((u >> 20) & 1) << 31) | (((u >> 1) & 1023) << 21) |
                              (((u >> 11) & 1) << 20) | (((u >> 12) & 255) << 12) | (rd << 7) | 0x6f;
                else w = (((u >> 12) & 1) << 31) | (((u >> 5) & 63) << 25) | (rs << 20) |
                         (rd << 15) | (f3 << 12) | (((u >> 1) & 15) << 8) | (((u >> 11) & 1) << 7) | 0x63;
            }
            if (!emit(out, count, w)) return 7;
        }
        length = count;
    }
    error_line = 0;
    return length ? 0 : 6;
}
}  // namespace gpu_asm
