import math
import os
import random
import struct
import unittest

from gpu_rv32 import (compile_c, run_c, compile_micropython, run_micropython, assemble, run_code)
from gpu_rv32.elf import read_elf


class GuestBuildTests(unittest.TestCase):
    def test_c_build_and_elf_bounds(self):
        image = compile_c("int main(void) { return 2*8; }")
        self.assertEqual(image.metadata["isa"], "rv32im")
        self.assertEqual(len(image.memory), 4194304)
        corrupt = bytearray(image.elf)
        struct.pack_into("<I", corrupt, 32, len(corrupt))  # invalid section table
        with self.assertRaises(ValueError):
            read_elf(corrupt)

    def test_micropython_fixed_firmware(self):
        a, b = compile_micropython("print(16)"), compile_micropython("print(42)")
        self.assertEqual(a.code, b.code)
        self.assertEqual(a.elf, b.elf)
        self.assertNotEqual(a.memory, b.memory)
        self.assertEqual(a.metadata["version"], "v1.27.0")

    def test_source_validation(self):
        for compile_fn, limit in [(compile_c, 65536), (compile_micropython, 16384)]:
            for text in ["", "x\0y", "x"*limit]:
                with self.assertRaises(ValueError):
                    compile_fn(text)


@unittest.skipUnless(os.environ.get("GPU_RV32_TESTS") == "1", "GPU tests require resource checks")
class GuestExecutionTests(unittest.TestCase):
    def mp(self, text, expected=None, *, calls=False):
        result = run_micropython(compile_micropython(text))
        self.assertEqual((result["status"], result["error"]), (0, 0), result)
        if expected is not None:
            self.assertEqual(result["stdout"], expected, result)
        if calls:
            self.assertGreater(result["bigint_calls"], 0, result)
        self.assertEqual(result["kernel_launches"], 1)
        return result

    def test_c_control_and_stdout(self):
        result = run_c(compile_c('#include "gpu_guest.h"\nint main(void) { '
                                'int s=0; for(int i=1;i<=100;i++) s+=i; puts("done"); return s; }'))
        self.assertEqual((result["status"], result["value"], result["stdout"]), (0, 5050, "done\n"))
        result = run_c(compile_c('#include <stdio.h>\nint main(void) { '
                                'printf("%s %d %08x %lld\\n", "hello", -16, 0x1234u, -9223372036854775807LL-1); '
                                'return 0; }'))
        self.assertEqual((result["status"], result["error"]), (0, 0), result)
        self.assertEqual(result["stdout"], "hello -16 00001234 -9223372036854775808\n")

    def test_functions_containers_loops_and_exceptions(self):
        self.mp("def f(n):\n    return sum(i*i for i in range(n))\n"
                "class Box:\n    def __init__(self, value):\n        self.value=value\n"
                "d = {'answer': f(20)}\nprint(Box(d['answer']).value)\n"
                "try:\n    1//0\nexcept ZeroDivisionError:\n    print('caught')\n", "2470\ncaught\n")

    def test_factorial_and_power(self):
        self.mp("n=1\nfor i in range(1,101):\n    n*=i\nprint(n)\nprint(2**256)\n",
                f"{math.factorial(100)}\n{2**256}\n", calls=True)

    def test_signed_large_arithmetic_and_division(self):
        a, b = -123456789012345678901234567890, 98765432109876543210987654321
        self.mp(f"a={a}\nb={b}\nprint(a+b)\nprint(a-b)\nprint(a*b)\nprint(a//b)\nprint(a%b)\n",
                "".join(f"{v}\n" for v in [a+b, a-b, a*b, a//b, a%b]), calls=True)

    def test_garbage_collection_preserves_live_objects(self):
        self.mp("import gc\nkeep=list(range(50))\n"
                "for i in range(300):\n    garbage=[i]*100\n    gc.collect()\n"
                "print(sum(keep))\n", "1225\n")

    def test_large_product_falls_back_without_truncation(self):
        a, b = (1<<4200)+123, (1<<4100)+321
        result = self.mp("a=(1<<4200)+123\nb=(1<<4100)+321\nprint(a*b)\n", f"{a*b}\n")
        self.assertEqual(result["bigint_calls"], 0)

    def test_uncaught_exception_and_output_limit(self):
        result = run_micropython(compile_micropython("print(1//0)"))
        self.assertEqual((result["status"], result["error"]), (0, 1), result)
        self.assertIn("ZeroDivisionError", result["stdout"])
        result = run_micropython(compile_micropython("print('x'*20000)"))
        self.assertEqual((result["status"], result["error"]), (0, 2), result)
        self.assertEqual(len(result["stdout"]), 16383)

    def test_instruction_budget(self):
        result = run_micropython(compile_micropython("while True: pass"), instruction_budget=100000)
        self.assertEqual((result["status"], result["steps"]), (1, 100000))

    def test_bigint_limb_service_random_and_boundaries(self):
        code = assemble("li a0,2048\nli a1,0\nli a2,256\nli a3,512\nli a4,256\nli a7,512\necall\nli a7,0\necall")
        rng = random.Random(3701)
        for bits in [0, 1, 31, 64, 255, 4095, 4096]:
            a = (1<<bits)-1 if bits in (0, 4096) else rng.getrandbits(bits)
            b = rng.getrandbits(max(1, bits))
            initial = bytearray(4096)
            initial[:512], initial[512:1024] = a.to_bytes(512, "little"), b.to_bytes(512, "little")
            result = run_code(code, memory=initial, services=True)
            self.assertEqual(result["status"], 0, result)
            self.assertEqual(result["bigint_calls"], 1)
            self.assertEqual(int.from_bytes(result["memory"][2048:3072], "little"), a*b)

    def test_bigint_invalid_requests_do_not_write(self):
        initial = bytes([0x5a])*4096
        for out, a, an, b, bn in [(0, 0, 2, 512, 2), (2049, 0, 2, 512, 2),
                                   (4092, 0, 2, 512, 2), (2048, 0, 257, 512, 1),
                                   (2048, 0, 0, 512, 1), (2048, 4096, 1, 512, 1)]:
            code = assemble(f"li a0,{out}\nli a1,{a}\nli a2,{an}\nli a3,{b}\nli a4,{bn}\n"
                            "li a7,512\necall\nli a7,0\necall")
            result = run_code(code, memory=initial, services=True)
            self.assertEqual((result["status"], result["value"]), (0, -1))
            self.assertEqual(result["memory"], initial)

    def test_unknown_service(self):
        result = run_code(assemble("li a7,999\necall"), services=True)
        self.assertEqual(result["status_name"], "unknown_service")
