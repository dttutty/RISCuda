"""Packaged resources and writable, content-addressed native build cache."""

import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
GUEST = ROOT / "guest"
INCLUDE = ROOT / "native/include"


def cache_dir():
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gpu-rv32-python"
    target = Path(os.environ.get("GPU_RV32_CACHE", default)).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def cuda_config():
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required for GPU execution; install the CUDA toolkit and add nvcc to PATH")
    arch = os.environ.get("GPU_RV32_CUDA_ARCH", "sm_80")
    if not re.fullmatch(r"sm_[0-9]{2,3}", arch):
        raise ValueError("GPU_RV32_CUDA_ARCH must look like sm_80 or sm_90")
    return nvcc, arch
