"""Small, strict assembler for the RV32I instructions accepted by this demo.

Produces real little-endian RV32I instruction words, not a custom opcode format.
Labels are byte offsets from text address zero. No ELF, directives, or linker.
"""

import re
import struct

ABI = dict(zip(
    "zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 a6 a7 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 t3 t4 t5 t6".split(),
    range(32), strict=True,
))
ABI["fp"] = 8
REG_OPS = {"add": (0, 0), "sub": (0, 32), "sll": (1, 0), "slt": (2, 0),
           "sltu": (3, 0), "xor": (4, 0), "srl": (5, 0), "sra": (5, 32),
           "or": (6, 0), "and": (7, 0), "mul": (0, 1), "mulh": (1, 1),
           "mulhsu": (2, 1), "mulhu": (3, 1), "div": (4, 1), "divu": (5, 1),
           "rem": (6, 1), "remu": (7, 1)}
IMM_OPS = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
BRANCHES = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}
LOADS = {"lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
STORES = {"sb": 0, "sh": 1, "sw": 2}


def reg(token):
    if token in ABI:
        return ABI[token]
    if re.fullmatch(r"x(?:[0-9]|[12][0-9]|3[01])", token):
        return int(token[1:])
    raise ValueError(f"invalid register: {token}")


def signed(value, bits):
    if not -(1 << (bits - 1)) <= value < 1 << (bits - 1):
        raise ValueError(f"{value} does not fit signed {bits}-bit immediate")
    return value & ((1 << bits) - 1)


def encode(op, args, labels, pc):
    def arity(n):
        if len(args) != n:
            raise ValueError(f"{op} expects {n} operands, got {len(args)}")

    def offset(token, bits):
        value = labels[token] - pc if token in labels else int(token, 0)
        if value % 4:
            raise ValueError("jump/branch target must be 4-byte aligned")
        return signed(value, bits)

    def iword(rd, rs, imm, f3=0, opcode=0x13):
        return (signed(imm, 12) << 20) | (reg(rs) << 15) | (f3 << 12) | (reg(rd) << 7) | opcode

    if op == "mv":
        arity(2)
        return iword(args[0], args[1], 0)
    if op in {"nop", "ret", "ecall", "ebreak", "fence"}:
        arity(0)
        return {"nop": 0x13, "ret": 0x8067, "ecall": 0x73,
                "ebreak": 0x100073, "fence": 0x0ff0000f}[op]
    if op in IMM_OPS:
        arity(3)
        return iword(args[0], args[1], int(args[2], 0), IMM_OPS[op])
    if op in {"slli", "srli", "srai"}:
        arity(3)
        shift = int(args[2], 0)
        if not 0 <= shift <= 31:
            raise ValueError("shift amount must be in [0,31]")
        return iword(args[0], args[1], shift | (0x400 if op == "srai" else 0),
                     1 if op == "slli" else 5)
    if op in REG_OPS:
        arity(3)
        f3, f7 = REG_OPS[op]
        return (f7 << 25) | (reg(args[2]) << 20) | (reg(args[1]) << 15) | (f3 << 12) | (reg(args[0]) << 7) | 0x33
    if op in BRANCHES:
        arity(3)
        imm = offset(args[2], 13)
        return ((imm >> 12) << 31) | (((imm >> 5) & 63) << 25) | (reg(args[1]) << 20) | (reg(args[0]) << 15) | (BRANCHES[op] << 12) | (((imm >> 1) & 15) << 8) | (((imm >> 11) & 1) << 7) | 0x63
    if op in {"beqz", "bnez"}:
        arity(2)
        return encode("beq" if op == "beqz" else "bne", [args[0], "zero", args[1]], labels, pc)
    if op in {"bgtz", "blez", "bltz", "bgez"}:
        arity(2)
        branch = "blt" if op in {"bgtz", "bltz"} else "bge"
        operands = ["zero", args[0]] if op in {"bgtz", "blez"} else [args[0], "zero"]
        return encode(branch, [*operands, args[1]], labels, pc)
    if op in {"bgt", "ble", "bgtu", "bleu"}:
        arity(3)
        branch = {"bgt": "blt", "ble": "bge", "bgtu": "bltu", "bleu": "bgeu"}[op]
        return encode(branch, [args[1], args[0], args[2]], labels, pc)
    if op in {"j", "call", "jal"}:
        arity(2 if op == "jal" else 1)
        rd, target = args if op == "jal" else ("zero" if op == "j" else "ra", args[0])
        imm = offset(target, 21)
        return ((imm >> 20) << 31) | (((imm >> 1) & 1023) << 21) | (((imm >> 11) & 1) << 20) | (((imm >> 12) & 255) << 12) | (reg(rd) << 7) | 0x6f
    if op == "jalr":
        arity(3)  # jalr rd, rs1, imm
        return iword(args[0], args[1], int(args[2], 0), opcode=0x67)
    if op in {"lui", "auipc"}:
        arity(2)
        imm = int(args[1], 0)
        if not -(1 << 19) <= imm < (1 << 20):
            raise ValueError("upper immediate exceeds 20 bits")
        return ((imm & 0xfffff) << 12) | (reg(args[0]) << 7) | (0x37 if op == "lui" else 0x17)
    if op in LOADS or op in STORES:
        arity(2)
        match = re.fullmatch(r"([+-]?(?:0x[0-9a-f]+|[0-9]+))\((\w+)\)", args[1])
        if not match:
            raise ValueError("memory operand must have form offset(base)")
        value, base = match.groups()
        imm = signed(int(value, 0), 12)
        if op in LOADS:
            return iword(args[0], base, int(value, 0), LOADS[op], 3)
        return ((imm >> 5) << 25) | (reg(args[0]) << 20) | (reg(base) << 15) | (STORES[op] << 12) | ((imm & 31) << 7) | 0x23
    raise ValueError(f"unsupported instruction: {op}")


def expand_li(args):
    """Expand a signed integer or unsigned RV32 bit pattern before layout."""
    if len(args) != 2:
        raise ValueError(f"li expects 2 operands, got {len(args)}")
    rd, literal = args
    value = int(literal, 0)
    if not -(1 << 31) <= value < (1 << 32):
        raise ValueError(f"li constant {literal} is outside the 32-bit range")
    if value >= (1 << 31):
        value -= 1 << 32
    if -2048 <= value <= 2047:
        return [("addi", [rd, "zero", str(value)])]
    # ADDI sign-extends its low 12 bits, so round the upper part accordingly.
    upper = (value + 0x800) >> 12
    lower = value - (upper << 12)
    instructions = [("lui", [rd, str(upper & 0xfffff)])]
    if lower:
        instructions.append(("addi", [rd, rd, str(lower)]))
    return instructions


def assemble(source):
    labels, instructions = {}, []
    for lineno, line in enumerate(source.splitlines(), 1):
        line = line.split("#", 1)[0].strip().lower()
        while ":" in line:
            label, line = map(str.strip, line.split(":", 1))
            if not re.fullmatch(r"[a-z_][a-z_0-9]*", label) or label in labels:
                raise ValueError(f"line {lineno}: invalid or duplicate label {label!r}")
            labels[label] = len(instructions) * 4
        if line:
            parts = line.split(None, 1)
            op = parts[0]
            args = [p.strip() for p in parts[1].split(",")] if len(parts) == 2 else []
            try:
                expanded = expand_li(args) if op == "li" else [(op, args)]
            except ValueError as exc:
                raise ValueError(f"line {lineno}: {exc}") from exc
            instructions.extend((lineno, name, operands) for name, operands in expanded)
    if not 1 <= len(instructions) <= 1024:
        raise ValueError("program must expand to 1..1024 machine instructions")
    words = []
    for pc, (lineno, op, args) in enumerate(instructions):
        try:
            words.append(encode(op, args, labels, pc * 4))
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    return struct.pack(f"<{len(words)}I", *words)
