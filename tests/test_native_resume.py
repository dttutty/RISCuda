import json
import os
from pathlib import Path
import subprocess
import unittest

from gpu_rv32.build import INCLUDE, cache_dir, cuda_config


@unittest.skipUnless(os.environ.get("GPU_RV32_TESTS") == "1", "set GPU_RV32_TESTS=1 on a checked idle GPU")
class NativeResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "examples/native_resume.cu"
        cls.binary = cache_dir() / "native_resume"
        nvcc, arch = cuda_config()
        subprocess.run([nvcc, "-std=c++17", "-O2", "-arch=" + arch, "-I", str(INCLUDE),
                        str(source), "-o", str(cls.binary)], check=True)

    def test_two_services_inside_one_guest_kernel(self):
        result = json.loads(subprocess.check_output([str(self.binary)]))
        self.assertEqual(result, {"value": 86, "status": 0, "steps": 7, "services": 2})

    def test_budget_is_cumulative_across_resume(self):
        result = json.loads(subprocess.check_output([str(self.binary), "4"]))
        self.assertEqual(result, {"value": 43, "status": 1, "steps": 4, "services": 1})
