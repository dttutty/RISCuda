"""Exercise Python source inside the RV32 interpreter on a selected idle GPU.

Run with ``uv run --locked --group python-vm verify_python_vm.py`` after the
resource checks. Expected results below are independent of the guest parser;
no host Python eval/exec participates in execution.
"""

import os
import unittest

from gpu_rv32 import compile_program, run_program


@unittest.skipUnless(os.environ.get("GPU_RV32_TESTS") == "1", "set GPU_RV32_TESTS=1 on a checked idle GPU")
class PythonVMTests(unittest.TestCase):
    def run_source(self, source, *, statement_budget=10000, instruction_budget=20000000):
        image = compile_program(source, statement_budget=statement_budget)
        return run_program(image, instruction_budget=instruction_budget)

    def check_value(self, source, expected, stdout=""):
        result = self.run_source(source)
        self.assertEqual(result["status"], 0, result)
        self.assertEqual(result["error"], 0, result)
        self.assertEqual(result["value"], expected, result)
        self.assertEqual(result["stdout"], stdout, result)
        self.assertGreater(result["steps"], 0)
        return result

    def check_error(self, source, expected, *, statement_budget=10000):
        result = self.run_source(source, statement_budget=statement_budget)
        self.assertEqual(result["status"], 0, result)
        self.assertEqual(result["error"], expected, result)
        return result

    def test_integer_arithmetic_and_precedence(self):
        for source, expected in [("2 * 8", 16), ("2 + 3 * 4 - 5", 9),
                                 ("(2 + 3) * (4 - 5)", -5),
                                 ("100 - 20 - 3", 77)]:
            with self.subTest(source=source):
                self.check_value(source + "\n", expected)

    def test_assignments_and_variable_reuse(self):
        self.check_value("a = 12\nb = 5\na = a * b\na + b\n", 65)

    def test_unary_signs(self):
        self.check_value("a = -7\nb = +3\n-(-a + b)\n", -10)

    def test_floor_division_signs(self):
        for left, right, expected in [(7, 3, 2), (-7, 3, -3),
                                      (7, -3, -3), (-7, -3, 2),
                                      (-6, 3, -2)]:
            with self.subTest(left=left, right=right):
                self.check_value(f"{left} // {right}\n", expected)

    def test_modulo_signs(self):
        for left, right, expected in [(7, 3, 1), (-7, 3, 2),
                                      (7, -3, -2), (-7, -3, -1),
                                      (-6, 3, 0)]:
            with self.subTest(left=left, right=right):
                self.check_value(f"{left} % {right}\n", expected)

    def test_if_else_and_comparisons(self):
        for condition, expected in [("3 < 4", 11), ("3 > 4", 22),
                                    ("3 <= 3", 11), ("3 >= 4", 22),
                                    ("3 == 3", 11), ("3 != 3", 22)]:
            with self.subTest(condition=condition):
                self.check_value(f"if {condition}:\n    result = 11\nelse:\n    result = 22\nresult\n",
                                 expected)
        # An inactive branch must not evaluate an undefined variable or division.
        self.check_value("if 0:\n    result = missing // 0\nelse:\n    result = 7\nresult\n", 7)

    def test_while_sum(self):
        self.check_value("i = 1\ntotal = 0\nwhile i <= 10:\n"
                         "    total = total + i\n    i = i + 1\ntotal\n", 55)

    def test_nested_loops(self):
        self.check_value("i = 0\ntotal = 0\nwhile i < 3:\n"
                         "    j = 0\n    while j < 4:\n"
                         "        total = total + i * j\n        j = j + 1\n"
                         "    i = i + 1\ntotal\n", 18)

    def test_print_stdout(self):
        self.check_value("print(7)\nprint(-12)\n7\n", 7, "7\n-12\n")

    def test_bitwise_operators(self):
        self.check_value("((240 & 60) | (1 << 8)) ^ 15\n", 319)
        self.check_value("~5\n", -6)

    def test_shifts(self):
        self.check_value("-8 >> 2\n", -2)
        self.check_value("(1 << 20) >> 4\n", 65536)
        self.check_value("-8 >> 40\n", -1)
        self.check_value("1 << 32\n", 0)
        self.check_error("1 << -1\n", 6)

    def test_short_circuit_and_chained_comparisons(self):
        self.check_value("not 0 and (3 or missing)\n", 3)
        self.check_value("0 and (1 // 0)\n", 0)
        self.check_value("1 < 2 < 3\n", 1)
        self.check_value("3 < 2 < missing\n", 0)

    def test_fixed_width_integers(self):
        self.check_value("2147483647 + 1\n", -2147483648)
        self.check_value("-2147483648 // -1\n", -2147483648)
        self.check_value("-2147483648 % -1\n", 0)

    def test_same_interpreter_for_different_sources(self):
        a = compile_program("2 * 8\n")
        b = compile_program("3 + 5\n", statement_budget=20)
        self.assertEqual(a.code, b.code)
        self.assertEqual(a.elf, b.elf)
        self.assertNotEqual(a.memory, b.memory)

    def test_capacity_limits(self):
        self.check_error("\n".join(f"v{i} = {i}" for i in range(33)), 4)
        self.check_value("(" * 16 + "1" + ")" * 16, 1)
        self.check_error("(" * 17 + "1" + ")" * 17, 4)

    def test_comments_and_blank_lines(self):
        self.check_value("# arithmetic example\n\na = 9  # input\n"
                         "\nb = 4\n# final expression\na * b - 1\n", 35)

    def test_undefined_name(self):
        self.check_error("known = 3\nknown + missing\n", 2)

    def test_division_by_zero(self):
        for operation in ["//", "%"]:
            with self.subTest(operation=operation):
                self.check_error(f"7 {operation} 0\n", 3)

    def test_malformed_source(self):
        for source in ["x =\n", "2 +\n", "if 1\n    x = 2\n", "print(1,2)\n",
                       "if 0:\n    x =\n", "  x = 1\n", "x = 1\n  y = 2\n",
                       "for = 7\n", "None = 7\n", "import = 1\n"]:
            with self.subTest(source=source):
                self.check_error(source, 1)

    def test_statement_budget(self):
        result = self.check_error("i = 0\nwhile 1:\n    i = i + 1\n", 5,
                                  statement_budget=20)
        self.assertLess(result["steps"], 20000000)

    def test_vm_instruction_budget(self):
        result = self.run_source("i = 0\nwhile 1:\n    i = i + 1\n",
                                 statement_budget=10000, instruction_budget=1)
        self.assertEqual(result["status"], 1, result)
        self.assertEqual(result["steps"], 1, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
