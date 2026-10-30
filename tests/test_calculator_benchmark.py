import hashlib
import os
from pathlib import Path
import unittest

from benchmarks.compare_qwenputer import CCalculator

class CalculatorBuildTests(unittest.TestCase):
    def test_pinned_calculator_and_port_build(self):
        path=Path(__file__).resolve().parents[1]/'benchmarks/qwenputer_calc/calc.c'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'b6c518356020b1c20af6e27fed08fae88a28d16e5dfd616f50aa2085e55e151b')
        image=CCalculator().image
        self.assertIn('calc_in',image.symbols)

@unittest.skipUnless(os.environ.get('GPU_RV32_TESTS')=='1','requires checked idle GPU')
class CalculatorExecutionTests(unittest.TestCase):
    def test_same_c_program_exact_integer_fraction_decimal_and_errors(self):
        c=CCalculator()
        cases=[('2*8','16\n'),('934878271422+44533617460','979411888882\n'),
               ('84/18','14/3\n'),('-7//3','-3\n'),('-7%3','2\n'),
               ('1.25*-3.5','-4.375\n'),('gcd(84,18)','6\n'),('1/0','error\n'),
               ('(1234567890123456789*9876543210987654321)/1234567890123456789',
                '9876543210987654321\n')]
        for source,expected in cases:
            with self.subTest(source=source):
                self.assertEqual(c(source),expected)
                self.assertEqual((c.calls[-1]['status'],c.calls[-1]['error']),(0,0))
                self.assertEqual(c.calls[-1]['kernel_launches'],1)
                self.assertGreater(c.calls[-1]['bigint_calls'],0)
