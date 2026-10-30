#pragma once

#include <cstdint>

// Shared device interpreter: host-driven and GPU-controlled execution use the same ISA.
namespace rv32 {
constexpr int RAM = 1024;
struct Result { int32_t value; uint32_t status, steps; };
// A yielded guest retains its registers, PC, stack and cumulative budget.
struct Machine { uint32_t r[32], pc, steps, status; };
__device__ inline void initialize(Machine& m, uint32_t ram_size, uint32_t entry = 0, int32_t input = 0) {
    for (int i = 0; i < 32; ++i) m.r[i] = 0;
    m.r[2] = ram_size; m.r[10] = static_cast<uint32_t>(input);
    m.pc = entry; m.steps = 0; m.status = 1;
}

__device__ inline int32_t sx(uint32_t value, int bits) {
    return static_cast<int32_t>(value << (32 - bits)) >> (32 - bits);
}

// Status: 0=ECALL exit, 1=budget, 2=instruction fetch, 3=illegal,
//         4=data access/alignment, 5=EBREAK, 6=native operation pending.
// With native=true, nonzero a7 yields after ECALL; a7=0 exits. The embedding
// kernel services the request, updates registers/RAM and calls resume again.
// The caller initializes RAM, including any data image or input buffers.
// Instruction addresses remain byte offsets into code, starting at VMA zero.
__device__ inline Result resume_memory(const volatile uint32_t* code, int length, int budget,
                                uint8_t* mem, uint32_t ram_size, Machine& machine,
                                bool native = false) {
    uint32_t r[32];
    for (int i = 0; i < 32; ++i) r[i] = machine.r[i];
    uint32_t pc = machine.pc, steps = machine.steps, status = 1;
    while (steps < static_cast<uint32_t>(budget)) {
        if ((pc & 3) || pc / 4 >= static_cast<uint32_t>(length)) { status = 2; break; }
        uint32_t w = code[pc / 4], op = w & 127, rd = (w >> 7) & 31;
        uint32_t f3 = (w >> 12) & 7, s1 = (w >> 15) & 31, s2 = (w >> 20) & 31;
        uint32_t f7 = w >> 25, a = r[s1], b = r[s2], value = 0, next = pc + 4;
        int32_t imm = sx(w >> 20, 12);
        bool write = false, illegal = false;
        ++steps;
        if (op == 0x37) { value = w & 0xfffff000u; write = true; }
        else if (op == 0x17) { value = pc + (w & 0xfffff000u); write = true; }
        else if (op == 0x6f) {
            uint32_t j = ((w >> 31) << 20) | (((w >> 12) & 255) << 12) |
                         (((w >> 20) & 1) << 11) | (((w >> 21) & 1023) << 1);
            value = pc + 4; next = pc + sx(j, 21); write = true;
        } else if (op == 0x67 && f3 == 0) {
            value = pc + 4; next = (a + imm) & ~1u; write = true;
        } else if (op == 0x63) {
            uint32_t j = ((w >> 31) << 12) | (((w >> 7) & 1) << 11) |
                         (((w >> 25) & 63) << 5) | (((w >> 8) & 15) << 1);
            bool take = false;
            switch (f3) {
                case 0: take = a == b; break;
                case 1: take = a != b; break;
                case 4: take = static_cast<int32_t>(a) < static_cast<int32_t>(b); break;
                case 5: take = static_cast<int32_t>(a) >= static_cast<int32_t>(b); break;
                case 6: take = a < b; break;
                case 7: take = a >= b; break;
                default: illegal = true;
            }
            if (take) next = pc + sx(j, 13);
        } else if (op == 0x13) {
            write = true;
            switch (f3) {
                case 0: value = a + imm; break;
                case 2: value = static_cast<int32_t>(a) < imm; break;
                case 3: value = a < static_cast<uint32_t>(imm); break;
                case 4: value = a ^ imm; break;
                case 6: value = a | imm; break;
                case 7: value = a & imm; break;
                case 1: value = a << s2; illegal = f7 != 0; break;
                case 5:
                    value = f7 == 32 ? static_cast<int32_t>(a) >> s2 : a >> s2;
                    illegal = f7 != 0 && f7 != 32; break;
            }
        } else if (op == 0x33) {
            write = true;
            if (f7 == 1) {
                int32_t sa = static_cast<int32_t>(a), sb = static_cast<int32_t>(b);
                bool overflow = a == 0x80000000u && b == 0xffffffffu;
                switch (f3) {
                    case 0: value = a * b; break;
                    case 1: value = static_cast<uint64_t>(int64_t(sa) * int64_t(sb)) >> 32; break;
                    case 2: value = static_cast<uint64_t>(int64_t(sa) * int64_t(b)) >> 32; break;
                    case 3: value = (uint64_t(a) * uint64_t(b)) >> 32; break;
                    case 4: value = !b ? 0xffffffffu : overflow ? a : static_cast<uint32_t>(sa / sb); break;
                    case 5: value = !b ? 0xffffffffu : a / b; break;
                    case 6: value = !b ? a : overflow ? 0 : static_cast<uint32_t>(sa % sb); break;
                    case 7: value = !b ? a : a % b; break;
                }
            } else {
                if (f7 != 0 && !(f7 == 32 && (f3 == 0 || f3 == 5))) illegal = true;
                switch (f3) {
                    case 0: value = f7 == 32 ? a - b : a + b; break;
                    case 1: value = a << (b & 31); break;
                    case 2: value = static_cast<int32_t>(a) < static_cast<int32_t>(b); break;
                    case 3: value = a < b; break;
                    case 4: value = a ^ b; break;
                    case 5: value = f7 == 32 ? static_cast<int32_t>(a) >> (b & 31) : a >> (b & 31); break;
                    case 6: value = a | b; break;
                    case 7: value = a & b; break;
                }
            }
        } else if (op == 3 || op == 0x23) {
            bool load = op == 3;
            uint32_t width = 0;
            if (load) {
                if (f3 == 0 || f3 == 4) width = 1;
                else if (f3 == 1 || f3 == 5) width = 2;
                else if (f3 == 2) width = 4;
            } else {
                imm = sx(((w >> 25) << 5) | ((w >> 7) & 31), 12);
                if (f3 <= 2) width = 1u << f3;
            }
            uint32_t address = a + imm;
            if (!width) illegal = true;
            else if (width > ram_size || address > ram_size - width || address % width) { status = 4; break; }
            else if (load) {
                for (uint32_t k = 0; k < width; ++k) value |= uint32_t(mem[address + k]) << (8 * k);
                if (f3 == 0) value = sx(value, 8);
                if (f3 == 1) value = sx(value, 16);
                write = true;
            } else {
                for (uint32_t k = 0; k < width; ++k) mem[address + k] = b >> (8 * k);
            }
        } else if (op == 0x0f && f3 == 0) {
            // FENCE is a no-op in this isolated single-hart RAM model.
        } else if (w == 0x73) {
            if (native && r[17] != 0) {
                status = 6; pc = next;
            } else status = 0;
            break;
        }
        else if (w == 0x100073) { status = 5; break; }
        else illegal = true;
        if (illegal) { status = 3; break; }
        if (write && rd) r[rd] = value;
        pc = next;
    }
    for (int i = 0; i < 32; ++i) machine.r[i] = r[i];
    machine.pc = pc; machine.steps = steps; machine.status = status;
    return {static_cast<int32_t>(r[10]), status, steps};
}

__device__ inline Result execute_memory(const volatile uint32_t* code, int length,
                                 int32_t input, int budget, uint8_t* mem,
                                 uint32_t ram_size, uint32_t entry = 0) {
    Machine machine;
    initialize(machine, ram_size, entry, input);
    return resume_memory(code, length, budget, mem, ram_size, machine);
}

__device__ inline Result execute(const volatile uint32_t* code, int length,
                          int32_t input, int budget) {
    uint8_t mem[RAM] = {};
    return execute_memory(code, length, input, budget, mem, RAM);
}

}  // namespace rv32
