import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

from gpu_rv32.build import cache_dir, cuda_config
from gpu_rv32.resident import ResidentLoop

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(os.environ.get('GPU_RV32_TESTS') == '1', 'requires a checked idle CUDA GPU')
class ResidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmarks.compare_qwenputer import CCalculator
        cls.image = CCalculator().image
        nvcc, arch = cuda_config()
        source = ROOT/'tests/resident_fixture.cu'
        target = cache_dir()/('fixture_'+hashlib.sha256(source.read_bytes()+arch.encode()).hexdigest()[:16]+'.so')
        if not target.exists():
            subprocess.run([nvcc, '-O2', '-arch='+arch, '--shared', '-Xcompiler', '-fPIC',
                            str(source), '-o', str(target)], check=True)
        cls.lib = ct.CDLL(str(target))
        cls.lib.fixture_create.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_void_p,
                                           ct.c_int, ct.c_int, ct.c_int]
        cls.lib.fixture_create.restype = ct.c_void_p
        cls.lib.fixture_get.argtypes = [ct.c_void_p, ct.c_int]
        cls.lib.fixture_get.restype = ct.c_uint64
        cls.lib.fixture_errors.argtypes = cls.lib.fixture_close.argtypes = [ct.c_void_p]

    def run_script(self, expressions, responses, *, host_loop=False, max_total=4096,
                   max_generated=4096, max_calls=64, instruction_budget=20000000):
        # Header tokens are BPE-like chunks; the expression uses ASCII byte tokens.
        vocabulary = [bytes([i]) for i in range(256)] + [b'']*8 + [b'<function=calc>', b'<parameter=expre', b'ssion>']
        offsets, flat = [0], b''
        for row in vocabulary:
            flat += row
            offsets.append(len(flat))
        script = []
        for expression, response in zip(expressions, responses):
            script += [260, 264, 265, 266, *expression.encode(), ord('<'), ord('/'), ord('>'), 261]
            script += [10, 256, 10, *response.encode(), 257, 10, 258, 10, 10, 259, 10, 10]
        script += [ord('O'), ord('K'), 262]
        data = ct.create_string_buffer(flat)
        off = (ct.c_int32*len(offsets))(*offsets)
        seq = (ct.c_int32*len(script))(*script)
        f = self.lib.fixture_create(data, len(flat), off, len(vocabulary), seq, len(script), 16+max_total, 16)
        try:
            buffers = {name:self.lib.fixture_get(f,i) for i,name in enumerate(
                ('input','next','position','mask','bytes','offsets','printable'))}
            with ResidentLoop(self.lib.fixture_get(f,7), self.lib.fixture_get(f,8), buffers, self.image,
                              vocab=len(vocabulary), prompt_length=16, cache_length=16+max_total,
                              eos=262, eos2=263, tco=260, tcc=261, max_total=max_total,
                              max_generated=max_generated, max_calls=max_calls,
                              instruction_budget=instruction_budget) as loop:
                state = loop.run(host_loop=host_loop)
                launches = loop.host_graph_launches
                with self.assertRaisesRegex(RuntimeError, 'already used'):
                    loop.run()
            self.assertEqual(self.lib.fixture_errors(f), 0, 'wrong feedback, token ordering, mask or position')
            return state, launches, script
        finally:
            self.lib.fixture_close(f)

    def test_multicall_feedback_and_host_control_parity(self):
        expressions = ['12345678901234567890*98765432109876543210', '(84/18)+5/7']
        responses = ['1219326311370217952237463801111263526900\n', '113/21\n']
        a, launches, script = self.run_script(expressions, responses)
        b, host_launches, _ = self.run_script(expressions, responses, host_loop=True)
        self.assertEqual((a.stop,a.calls,launches), (1,2,1))
        self.assertEqual(list(a.tokens[:a.total]), script)
        self.assertEqual(list(a.tokens[:a.total]), list(b.tokens[:b.total]))
        self.assertEqual(list(a.forced[:a.total]), list(b.forced[:b.total]))
        self.assertEqual((a.forwards,a.position,host_launches), (a.total-1,16+a.total-1,a.total+1))
        self.assertEqual(a.generated+a.injected,a.total)
        for i in range(2):
            self.assertEqual(a.tools[i].output.decode(),responses[i])
            self.assertEqual((a.tools[i].status,a.tools[i].error), (0,0))
            self.assertGreater(a.tools[i].steps,0)
        self.assertGreater(a.tools[0].bigint_calls,0)

    def test_error_feedback_and_instruction_budget(self):
        a, _, _ = self.run_script(['1/0','(2+3'], ['error\n','error\n'])
        self.assertEqual((a.stop,a.calls), (1,2))
        self.assertTrue(all(t.status == 0 for t in a.tools[:a.calls]))
        a, _, _ = self.run_script(['2*8'], ['error\n'], instruction_budget=1)
        self.assertEqual((a.stop,a.tools[0].status,a.tools[0].steps), (1,1,1))

    def test_bounds(self):
        a, _, _ = self.run_script(['2*8'], ['16\n'], max_total=8)
        self.assertEqual((a.stop,a.total,a.calls), (2,8,0))
        a, _, _ = self.run_script(['2*8'], ['16\n'], max_generated=3)
        self.assertEqual((a.stop,a.generated,a.calls), (3,3,0))
        a, _, _ = self.run_script(['2*8','3*9'], ['16\n','27\n'], max_calls=1)
        self.assertEqual((a.stop,a.calls), (5,1))

    def test_capture_overflow_returns_protocol_error(self):
        a, _, _ = self.run_script(['1'*90], ['error (too long: split it)\n'])
        self.assertEqual((a.stop,a.calls,a.tools[0].gather_error), (1,1,2))
        self.assertEqual(a.tools[0].steps,0)
