"""ctypes CUDA runtime; a complete guest invocation uses one kernel launch."""

import ctypes as ct
from functools import cache
import hashlib
import os
import struct
import subprocess
import tempfile
import time

from .assembler import assemble
from .build import ROOT, cache_dir, cuda_config

STATUSES = {0: "halt", 1: "instruction_budget", 2: "instruction_fetch", 3: "illegal_instruction",
            4: "memory_access", 5: "breakpoint"}


class Result(ct.Structure):
    _fields_ = [("value", ct.c_int32), ("status", ct.c_uint32), ("steps", ct.c_uint32)]


class AssemblyResult(ct.Structure):
    _fields_ = [(n, ct.c_int32) for n in ["value", "status", "steps", "length", "error_line"]] + [
        ("code", ct.c_uint32 * 1024)]


@cache
def library():
    nvcc, arch = cuda_config()
    sources = [ROOT / "native" / p for p in ["runtime.cu", "include/rv32_core.cuh", "include/gpu_assembler.cuh"]]
    version = subprocess.check_output([nvcc, "--version"])
    key = hashlib.sha256(version + arch.encode() + b"".join(p.read_bytes() for p in sources)).hexdigest()[:20]
    build = cache_dir() / "cuda"
    build.mkdir(exist_ok=True)
    target = build / ("librv32_" + key + ".so")
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=build) as staging:
            temporary = os.path.join(staging, target.name)
            subprocess.run([nvcc, "-std=c++17", "-O2", "-lineinfo", "-arch=" + arch, "--shared",
                            "-Xcompiler", "-fPIC", str(sources[0]), "-o", temporary], check=True)
            os.replace(temporary, target)
    lib = ct.CDLL(str(target))
    lib.rv32_error.restype = ct.c_char_p
    lib.rv32_assembly_size.restype = ct.c_size_t
    assert lib.rv32_assembly_size() == ct.sizeof(AssemblyResult)
    lib.rv32_run.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int,
                            ct.c_int, ct.POINTER(Result), ct.POINTER(ct.c_float)]
    lib.rv32_run_assembly.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_void_p, ct.POINTER(ct.c_float)]
    return lib


def integer(value):
    if not isinstance(value, int) or not -(1 << 31) <= value < (1 << 32):
        raise ValueError("input must be a signed32 integer or unsigned32 bit pattern")
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def budget(value):
    if not isinstance(value, int) or not 1 <= value <= 2000000000:
        raise ValueError("instruction budget must be in 1..2000000000")
    return value


def run_code(code, *, memory=None, ram_size=None, entry=0, input=0, instruction_budget=20000000):
    code = bytes(code)
    size = ram_size if ram_size is not None else len(memory) if memory is not None else 65536
    if not 1 <= size <= 65536 or not 4 <= len(code) <= 65536 or len(code) % 4:
        raise ValueError("require 1..65536 RAM bytes and 1..16384 machine words")
    if entry < 0 or entry % 4 or entry >= len(code):
        raise ValueError("entry must be an aligned byte offset inside code")
    memory = bytes(size) if memory is None else bytes(memory)
    if len(memory) != size:
        raise ValueError("initial memory length must equal RAM size")
    input, instruction_budget = integer(input), budget(instruction_budget)
    lib = library()
    device_code, ram = ct.create_string_buffer(code), ct.create_string_buffer(memory, size)
    result, milliseconds = Result(), ct.c_float()
    started = time.perf_counter()
    if lib.rv32_run(device_code, len(code) // 4, ram, size, entry, input, instruction_budget,
                    ct.byref(result), ct.byref(milliseconds)):
        raise RuntimeError(lib.rv32_error().decode())
    return {"value": result.value, "status": result.status, "status_name": STATUSES.get(result.status, "unknown"),
            "steps": result.steps, "memory": ram.raw, "gpu_milliseconds": milliseconds.value,
            "request_seconds": time.perf_counter() - started, "kernel_launches": 1}


def run_assembly(source, *, input=0, instruction_budget=4096, device_assembler=False):
    if not device_assembler:
        code = assemble(source)
        return dict(run_code(code, ram_size=1024, input=input, instruction_budget=instruction_budget), machine_code=code)
    encoded = source.encode("utf-8")
    if len(encoded) >= 8192 or b"\0" in encoded:
        raise ValueError("GPU assembly source must have at most 8191 bytes, without NUL")
    input, instruction_budget = integer(input), budget(instruction_budget)
    lib, result, milliseconds = library(), AssemblyResult(), ct.c_float()
    started = time.perf_counter()
    if lib.rv32_run_assembly(encoded, input, instruction_budget, ct.byref(result), ct.byref(milliseconds)):
        raise RuntimeError(lib.rv32_error().decode())
    return {"value": result.value, "status": result.status,
            "status_name": {6: "assembly_error", 7: "assembly_capacity"}.get(result.status, STATUSES.get(result.status, "unknown")),
            "steps": result.steps, "error_line": result.error_line,
            "machine_code": struct.pack(f"<{result.length}I", *result.code[:result.length]),
            "gpu_milliseconds": milliseconds.value, "request_seconds": time.perf_counter() - started,
            "kernel_launches": 1}
