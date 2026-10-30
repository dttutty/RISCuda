import unittest

from gpu_rv32 import compile_program, run_code
from gpu_rv32.build import GUEST, INCLUDE, ROOT
from gpu_rv32.python_vm import load_elf


class CompilerTests(unittest.TestCase):
    def test_fixed_interpreter_and_input_only_change(self):
        left, right = compile_program("print(16)\n"), compile_program("print(42)\n")
        self.assertEqual(left.code, right.code)
        self.assertEqual(left.elf, right.elf)
        self.assertNotEqual(left.memory, right.memory)
        self.assertEqual(left.metadata["isa"], "rv32im")
        self.assertEqual(len(left.memory), 65536)

    def test_source_and_budget_validation(self):
        for source in ["", "x" * 4096, "x\0y"]:
            with self.assertRaises(ValueError):
                compile_program(source)
        for count in [0, 1000001]:
            with self.assertRaises(ValueError):
                compile_program("pass\n", statement_budget=count)

    def test_reject_invalid_elf(self):
        for data in [b"", b"not an ELF", b"\x7fELF\x02\x01\x01" + bytes(45)]:
            with self.assertRaises(ValueError):
                load_elf(data)

    def test_all_native_resources_are_packaged(self):
        for name in ["mini_python.c", "firmware.c", "support.c", "start.S", "link.ld"]:
            self.assertTrue((GUEST / name).is_file(), name)
        self.assertTrue((INCLUDE / "rv32_core.cuh").is_file())
        self.assertTrue((INCLUDE / "gpu_assembler.cuh").is_file())
        self.assertTrue((ROOT / "native/runtime.cu").is_file())

    def test_machine_image_validation_before_cuda(self):
        for code in [b"", b"abc", bytes(65540)]:
            with self.assertRaises(ValueError):
                run_code(code)
        for kwargs in [{"entry": 2}, {"entry": 4}, {"ram_size": 0},
                       {"memory": b"a", "ram_size": 4}, {"instruction_budget": 0}]:
            with self.assertRaises(ValueError):
                run_code(bytes.fromhex("73000000"), **kwargs)
