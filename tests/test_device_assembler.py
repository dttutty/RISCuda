import os
import random
import struct
from types import SimpleNamespace
import unittest
from gpu_rv32 import assemble, run_assembly


@unittest.skipUnless(os.environ.get("GPU_RV32_TESTS") == "1", "set GPU_RV32_TESTS=1 on a checked idle GPU")
class GPUAssemblerTests(unittest.TestCase):
    def run_program(self, source):
        result = run_assembly(source, device_assembler=True)
        code = result['machine_code']
        return SimpleNamespace(value=result['value'], status=result['status'], steps=result['steps'],
                               length=len(code)//4, code=struct.unpack(f'<{len(code)//4}I', code))

    def compare(self, source, expected):
        tool = self.run_program(source)
        self.assertEqual(tool.status, 0)
        binary = struct.pack(f"<{tool.length}I", *tool.code[:tool.length])
        self.assertEqual(binary, assemble(source))
        self.assertEqual(tool.value, expected)
        return tool

    def test_full_width_constants(self):
        rng = random.Random(20261029)
        values = [-2147483648, -2049, -2048, -1, 0, 2047, 2048, 4095, 4096, 2147483647, 0x80000000, 0xffffffff]
        values += [rng.randrange(-(1 << 31), 1 << 32) for _ in range(64)]
        for value in values:
            with self.subTest(value=value):
                expected = ((value + (1 << 31)) % (1 << 32)) - (1 << 31)
                self.compare(f"li a0, {value}\necall\n", expected)

    def test_arithmetic_and_labels(self):
        self.compare("li a0, 12345\nli t0,6789\nmul a0,a0,t0\nli t1,23456\nli t2,321\nmul t1,t1,t2\nsub a0,a0,t1\necall", 76280829)
        tool = self.compare("li a0,0\nli t0,1\nli t1,11\nloop: add a0,a0,t0\naddi t0,t0,1\nblt t0,t1,loop\necall", 55)
        self.assertEqual(tool.steps, 34)
        self.compare("jal ra,func\nj done\nfunc: li a0,0x80000800\njal zero,done\nli a0,0\ndone: ecall", -2147481600)
        self.compare("li a0,1\nli t0,2048\nbeq a0,zero,skip\nbne t0,zero,ok\nskip: li a0,9\nok: ecall", 1)

    def test_branch_pseudos(self):
        for op, a, b in [("bgt", 3, 2), ("ble", -3, 2), ("bgtu", -1, 2), ("bleu", 2, -1)]:
            self.compare(f"li a0,1\nli t0,{a}\nli t1,{b}\n{op} t0,t1,yes\nli a0,0\nyes: ecall", 1)
        for op, value in [("beqz", 0), ("bnez", 2), ("bltz", -2), ("bgez", 0), ("bgtz", 2), ("blez", 0)]:
            self.compare(f"li a0,1\nli t0,{value}\n{op} t0,yes\nli a0,0\nyes: ecall", 1)

    def test_multiply_high_signed_and_unsigned(self):
        for op, left, right, expected in [
            ("mulh", -1, 2, -1),
            ("mulh", -2147483648, -2147483648, 0x40000000),
            ("mulh", 2147483647, 2147483647, 0x3fffffff),
            ("mulhsu", -1, 0xffffffff, -1),
            ("mulhsu", -2, 0xffffffff, -2),
            ("mulhsu", -2147483648, 0xffffffff, -2147483648),
            ("mulhu", 0xffffffff, 0xffffffff, -2),
            ("mulhu", 0x80000000, 2, 1),
        ]:
            with self.subTest(op=op, left=left, right=right):
                self.compare(f"li t0,{left}\nli t1,{right}\n{op} a0,t0,t1\necall", expected)

    def test_divide_remainder_truncate_toward_zero(self):
        for op, left, right, expected in [
            ("div", 7, 3, 2), ("div", -7, 3, -2),
            ("div", 7, -3, -2), ("div", -7, -3, 2),
            ("rem", 7, 3, 1), ("rem", -7, 3, -1),
            ("rem", 7, -3, 1), ("rem", -7, -3, -1),
            ("divu", 0xffffffff, 2, 2147483647),
            ("remu", 0xffffffff, 2, 1),
            ("divu", 0x80000000, 1, -2147483648),
            ("remu", 0x80000000, 0xffffffff, -2147483648),
        ]:
            with self.subTest(op=op, left=left, right=right):
                self.compare(f"li t0,{left}\nli t1,{right}\n{op} a0,t0,t1\necall", expected)

    def test_divide_remainder_zero_and_overflow(self):
        for op in ["div", "divu", "rem", "remu"]:
            for dividend in [0, 7, -7, -2147483648]:
                with self.subTest(op=op, dividend=dividend):
                    expected = -1 if op.startswith("div") else dividend
                    self.compare(f"li a0,{dividend}\n{op} a0,a0,zero\necall", expected)
        self.compare("li a0,-2147483648\nli t0,-1\ndiv a0,a0,t0\necall", -2147483648)
        self.compare("li a0,-2147483648\nli t0,-1\nrem a0,a0,t0\necall", 0)

    def test_register_bitwise_shifts_and_comparisons(self):
        cases = [
            ("and", 0xf0, 0x3c, 0x30), ("or", 0xf0, 0x3c, 0xfc),
            ("xor", 0xf0, 0x3c, 0xcc),
            ("slt", -1, 1, 1), ("slt", 1, -1, 0),
            ("sltu", -1, 1, 0), ("sltu", 1, -1, 1),
            ("sll", 1, 31, -2147483648), ("sll", 7, 32, 7),
            ("sll", 7, 33, 14), ("srl", 0x80000001, 1, 0x40000000),
            ("sra", 0x80000001, 1, -1073741824),
            ("srl", 0x80000000, 63, 1), ("sra", 0x80000000, -1, -1),
        ]
        for op, left, right, expected in cases:
            with self.subTest(op=op, left=left, right=right):
                self.compare(f"li t0,{left}\nli t1,{right}\n{op} a0,t0,t1\necall", expected)

    def test_immediate_bitwise_shifts_and_comparisons(self):
        cases = [
            ("andi", 0x12345678, 255, 0x78),
            ("ori", 0, -2048, -2048), ("ori", 0, 2047, 2047),
            ("xori", 0x12345678, -1, -305419897),
            ("slti", -1, 1, 1), ("slti", 0, -1, 0),
            ("sltiu", -1, 1, 0), ("sltiu", 0, -1, 1),
            ("slli", 0x80000001, 0, -2147483647),
            ("slli", 1, 31, -2147483648),
            ("srli", 0x80000000, 31, 1),
            ("srai", 0x80000000, 31, -1),
            ("srli", -1, 0, -1), ("srai", -1, 0, -1),
        ]
        for op, value, immediate, expected in cases:
            with self.subTest(op=op, value=value, immediate=immediate):
                self.compare(f"li t0,{value}\n{op} a0,t0,{immediate}\necall", expected)

    def test_memory_load_widths_and_little_endian(self):
        # One word exposes distinct bytes and positive/negative halfwords.
        for op, offset, expected in [
            ("lb", 0, 1), ("lb", 1, 127), ("lb", 2, -1), ("lb", 3, -128),
            ("lbu", 2, 255), ("lbu", 3, 128),
            ("lh", 0, 0x7f01), ("lh", 2, -32513),
            ("lhu", 2, 0x80ff), ("lw", 0, -2130739455),
        ]:
            with self.subTest(op=op, offset=offset):
                self.compare(f"li t0,0x80ff7f01\nsw t0,0(zero)\n{op} a0,{offset}(zero)\necall", expected)

    def test_memory_store_widths_and_ram_boundaries(self):
        self.compare("li t0,0x1234\nsb t0,1023(zero)\nlbu a0,1023(zero)\necall", 0x34)
        self.compare("li t0,0xffffabcd\nsh t0,-2(sp)\nlhu a0,-2(sp)\necall", 0xabcd)
        self.compare("li t0,0xffffabcd\nsh t0,1022(zero)\nlh a0,1022(zero)\necall", -21555)
        self.compare("li t0,0x12345678\nsw t0,-4(sp)\nlw a0,-4(sp)\necall", 0x12345678)
        self.compare("li t0,86\nsw t0,+0(zero)\nlw a0,+0(zero)\necall", 86)
        self.compare("li t0,-8\nli t1,2048\nsw t0,-2048(t1)\nlw a0,-2048(t1)\necall", -8)
        self.compare("li t0,127\nli t1,-1024\nsb t0,2047(t1)\nlbu a0,2047(t1)\necall", 127)
        # Byte/halfword stores preserve neighboring bytes and truncate their input.
        self.compare("li t0,0x11223344\nsw t0,0(zero)\nli t0,0x1234aa\nsb t0,1(zero)\n"
                     "li t0,0x1234bbcc\nsh t0,2(zero)\nlw a0,0(zero)\necall", 0xbbccaa44 - (1 << 32))
        # Each invocation starts with isolated, zero-initialized RAM.
        self.compare("lw a0,0(zero)\necall", 0)

    def test_memory_array_loop(self):
        self.compare("li t0,0\nli t1,4\nfill: slli t2,t0,2\naddi t3,t0,1\nsw t3,0(t2)\n"
                     "addi t0,t0,1\nblt t0,t1,fill\nli t0,0\nli a0,0\n"
                     "sum: slli t2,t0,2\nlw t3,0(t2)\nadd a0,a0,t3\n"
                     "addi t0,t0,1\nblt t0,t1,sum\necall", 10)

    def test_memory_access_traps(self):
        for instruction in ["lb a0,-1(zero)", "lh a0,1024(zero)", "lw a0,1024(zero)",
                            "sb t0,1024(zero)", "sh t0,1023(zero)", "sw t0,1024(zero)",
                            "lh a0,1(zero)", "lw a0,2(zero)", "sh t0,1(zero)", "sw t0,2(zero)"]:
            with self.subTest(instruction=instruction):
                tool = self.run_program(instruction + "\necall")
                self.assertEqual((tool.status, tool.steps), (4, 1))
                self.assertEqual(struct.pack(f"<{tool.length}I", *tool.code[:tool.length]),
                                 assemble(instruction + "\necall"))

    def test_invalid_memory_and_bitwise_operands(self):
        instructions = [
            "lw a0,0t0", "lw a0,(t0)", "lw a0,0(t7)", "lw t7,0(zero)",
            "lw a0,0(zero))", "sw t0,0(zero)junk", "lw a0,0(zero", "sw t0,0zero)",
            "lw a0,0(zero),t1", "lw a0,2048(zero)", "sw t0,-2049(sp)",
            "and a0,t0,t7", "xor a0,t0", "andi a0,t0,2048", "ori a0,t0,-2049",
            "sltiu a0,t0,2048", "slli a0,t0,-1", "srli a0,t0,32", "srai a0,t0,32",
            "slli a0,t7,1", "srai a0,t0,1,t1",
        ]
        for instruction in instructions:
            with self.subTest(instruction=instruction):
                tool = self.run_program(instruction + "\necall")
                self.assertEqual((tool.status, tool.steps), (6, 0))

    def test_errors_before_execution(self):
        for source in ["li t7,2\necall", "li a0,4294967296\necall", "addi a0,zero,2048\necall",
                       "j missing\necall", "same: nop\nsame: ecall", "", "mul a0,a1", "li a0,wat"]:
            with self.subTest(source=source):
                tool = self.run_program(source)
                self.assertEqual(tool.status, 6)
                self.assertEqual(tool.steps, 0)

    def test_capacity(self):
        self.assertEqual(self.run_program("nop\n" * 1025).status, 7)
        self.assertEqual(self.run_program("\n".join(f"L{i}: nop" for i in range(65))).status, 7)
        self.compare("li a0,17\n" + "nop\n" * 1022 + "ecall", 17)

    def test_instruction_budget(self):
        tool = self.run_program("forever: j forever")
        self.assertEqual((tool.status, tool.steps), (1, 4096))

