import os
import struct
import unittest

from gpu_rv32 import assemble, run_assembly, run_code


@unittest.skipUnless(os.environ.get("GPU_RV32_TESTS") == "1", "set GPU_RV32_TESTS=1 on a checked idle GPU")
class RuntimeTests(unittest.TestCase):
    def test_input_and_initial_memory(self):
        result = run_assembly("li a1,8\nmul a0,a0,a1\necall", input=17)
        self.assertEqual((result["value"], result["status"], result["kernel_launches"]), (136, 0, 1))
        initial = bytearray(1024)
        struct.pack_into("<I", initial, 4, 0x12345678)
        result = run_code(assemble("lw a0,4(zero)\nsw a0,8(zero)\necall"), memory=initial)
        self.assertEqual((result["value"], result["status"]), (0x12345678, 0))
        self.assertEqual(struct.unpack_from("<I", result["memory"], 8)[0], 0x12345678)
        self.assertEqual(run_assembly("lw a0,8(zero)\necall")["value"], 0)

    def test_subroutine_and_budget(self):
        self.assertEqual(run_assembly("call double\necall\ndouble: add a0,a0,a0\nret", input=-4)["value"], -8)
        result = run_assembly("again: j again", instruction_budget=11)
        self.assertEqual((result["status"], result["steps"]), (1, 11))

    def test_vm_faults(self):
        for code, status in [(b"\xff" * 4, 3), (assemble("nop"), 2),
                             (assemble("ebreak"), 5), (assemble("lw a0,2(zero)\necall"), 4)]:
            with self.subTest(status=status):
                self.assertEqual(run_code(code)["status"], status)
