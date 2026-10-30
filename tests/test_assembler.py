import struct
import unittest
from gpu_rv32 import assemble


class AssemblerTests(unittest.TestCase):
    def test_known_riscv_encodings(self):
            source = "addi a0, zero, 42\nadd a0, a0, a1\nret\necall"
            self.assertEqual(struct.unpack("<4I", assemble(source)),
                             (0x02a00513, 0x00b50533, 0x00008067, 0x00000073))
            self.assertEqual(struct.unpack("<2I", assemble("lw a0,+0(zero)\nsw t0,+4(sp)")),
                             (0x00002503, 0x00512223))

    def test_negative_backward_branch(self):
            self.assertEqual(assemble("again: beq zero, zero, again"), bytes.fromhex("63000000"))
            self.assertEqual(struct.unpack("<2I", assemble("again: nop\nbne a0,zero,again"))[1], 0xfe051ee3)

    def test_reject_bad_code(self):
            for source in ["li a0,0x100000000", "li a0,-2147483649", "li x32,12345",
                           "li a0", "li a0,1,2", "addi a0,zero,2048",
                           "addi x32,zero,0", "slli a0,a0,32", "j 2",
                           "j absent", "ecall a0", "x: nop\nx: nop", "divw a0,a0,a0", ".text", ""]:
                with self.subTest(source=source), self.assertRaises(ValueError):
                    assemble(source)

    def test_li_32bit_encodings(self):
            cases = {
                "12345": (0x00003537, 0x03950513),
                "2048": (0x00001537, 0x80050513),
                "-2049": (0xfffff537, 0x7ff50513),
                "4096": (0x00001537,),
                "0x7fffffff": (0x80000537, 0xfff50513),
                "-2147483648": (0x80000537,),
                "0x80000000": (0x80000537,),
                "0xffffffff": (0xfff00513,),
            }
            for literal, expected in cases.items():
                with self.subTest(literal=literal):
                    self.assertEqual(assemble(f"li a0,{literal}"), struct.pack(f"<{len(expected)}I", *expected))
            with self.assertRaisesRegex(ValueError, "line 2:.*32-bit"):
                assemble("nop\nli a0,0x100000000")

    def test_labels_follow_expanded_li(self):
            code = assemble("start: li a0,12345\nbeq zero,zero,done\nli a0,6789\ndone: bne a0,zero,start")
            words = struct.unpack("<6I", code)
            self.assertEqual(words[2], 0x00000663)  # PC 8 -> 20, +12 bytes
            self.assertEqual(words[5], 0xfe0516e3)  # PC 20 -> 0, -20 bytes
            code = assemble("j done\nli a0,4096\nli a0,2048\ndone: ecall")
            self.assertEqual(struct.unpack("<5I", code)[0], 0x0100006f)

    def test_instruction_limit_counts_expansion(self):
            self.assertEqual(len(assemble("\n".join(["li a0,12345"] * 512))), 4096)
            with self.assertRaises(ValueError):
                assemble("\n".join(["li a0,12345"] * 513))
