"""Bounded, batch-one CUDA graph controller for Qwenputer's calc protocol.

The caller owns captured prefill/decode graphs and their stable device buffers.
Creation uploads a fixed C image; run submits one graph, then waits for completion.
Torch is optional: integer CUDA handles and device pointers are also accepted.
"""
import ctypes as ct
from functools import cache
import hashlib
import os
import subprocess
import tempfile
import time

from .build import ROOT, cache_dir, cuda_config
from .runtime import budget


class Tool(ct.Structure):
    _fields_ = [(n, ct.c_int32) for n in (
        "status", "error", "steps", "bigint_calls", "gather_error", "input_length", "output_length"
    )] + [("expression", ct.c_char * 256), ("output", ct.c_char * 256),
         ("started", ct.c_uint64), ("elapsed_ns", ct.c_uint64)]


class State(ct.Structure):
    _fields_ = [(n, ct.c_int32) for n in (
        "position", "forwards", "generated", "injected", "total", "calls", "stop", "pending",
        "armed", "cap", "queue_index", "queue_length"
    )] + [("tokens", ct.c_int32 * 8192), ("queue", ct.c_int32 * 512),
         ("forced", ct.c_uint8 * 8192), ("rows", (ct.c_uint8 * 16) * 81), ("tools", Tool * 64)]


class Options(ct.Structure):
    _fields_ = [("prefill_graph", ct.c_uint64), ("decode_graph", ct.c_uint64)] + [
        (n, ct.c_void_p) for n in ("input", "next", "position", "mask", "bytes", "offsets", "printable",
                                  "code", "initial")
    ] + [(n, ct.c_int32) for n in (
        "vocab", "cache_length", "prompt_length", "max_generated", "max_total", "max_calls",
        "eos", "eos2", "tco", "tcc", "words", "ram_size", "entry", "source", "output", "error", "budget"
    )]


@cache
def library():
    nvcc, arch = cuda_config()
    sources = [ROOT / "native" / p for p in (
        "resident.cu", "include/resident.h", "include/rv32_core.cuh", "include/bigint.cuh")]
    key = hashlib.sha256(subprocess.check_output([nvcc, "--version"]) + arch.encode()
                         + b"".join(p.read_bytes() for p in sources)).hexdigest()[:20]
    build = cache_dir() / "cuda"
    build.mkdir(exist_ok=True)
    target = build / ("libresident_" + key + ".so")
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=build) as staging:
            temporary = os.path.join(staging, target.name)
            subprocess.run([nvcc, "-std=c++17", "-O2", "-lineinfo", "-arch=" + arch, "--shared",
                            "-Xcompiler", "-fPIC", str(sources[0]), "-o", temporary], check=True)
            os.replace(temporary, target)
    lib = ct.CDLL(str(target))
    lib.resident_error.restype = ct.c_char_p
    lib.resident_state_size.restype = lib.resident_options_size.restype = ct.c_size_t
    if lib.resident_state_size() != ct.sizeof(State) or lib.resident_options_size() != ct.sizeof(Options):
        raise RuntimeError("resident controller ABI mismatch")
    lib.resident_create.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(Options)]
    lib.resident_run.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_int, ct.c_int, ct.POINTER(ct.c_uint64)]
    lib.resident_read.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    lib.resident_close.argtypes = [ct.c_void_p]
    lib.resident_close.restype = None
    return lib


def pointer(value):
    return int(value.data_ptr()) if hasattr(value, "data_ptr") else int(value)


class ResidentLoop:
    def __init__(self, prefill, decode, buffers, image, *, vocab, prompt_length, cache_length,
                 eos, tco, tcc, eos2=None, max_generated=4096, max_total=4096, max_calls=64,
                 instruction_budget=20000000):
        self.lib, self.handle = library(), ct.c_void_p()
        # Graphs and externally allocated input storage must outlive the executable.
        self.resources = (prefill, decode, buffers)
        o = Options()
        for name, graph in (("prefill_graph", prefill), ("decode_graph", decode)):
            setattr(o, name, int(graph.raw_cuda_graph()) if hasattr(graph, "raw_cuda_graph") else int(graph))
        for name in ("input", "next", "position", "mask", "bytes", "offsets", "printable"):
            setattr(o, name, pointer(buffers[name]))
        if not len(image.code) or len(image.code) % 4:
            raise ValueError("image code must contain complete RV32 machine words")
        code, initial = ct.create_string_buffer(image.code), ct.create_string_buffer(image.memory)
        o.code, o.initial = ct.addressof(code), ct.addressof(initial)
        for name, value in dict(vocab=vocab, prompt_length=prompt_length, cache_length=cache_length,
                                eos=eos, eos2=eos if eos2 is None else eos2, tco=tco, tcc=tcc,
                                max_generated=max_generated, max_total=max_total, max_calls=max_calls,
                                words=len(image.code)//4, ram_size=len(image.memory), entry=image.entry,
                                source=image.symbols['calc_in'], output=image.symbols['guest_output'],
                                error=image.symbols['guest_error'], budget=budget(instruction_budget)).items():
            if not isinstance(value, int) or not -(1 << 31) <= value < (1 << 31):
                raise ValueError(f"{name} must be an int32")
            setattr(o, name, value)
        self._check(self.lib.resident_create(ct.byref(self.handle), ct.byref(o)))

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.resident_error().decode())

    def run(self, stream=0, *, host_loop=False, profile=False):
        """Blocking submission; host_loop is a parity/control experiment only.

        Profiling starts after setup and stops before the final state download.
        CUDA context/device/stream must match those used by the captured model.
        """
        stream = int(stream.cuda_stream) if hasattr(stream, "cuda_stream") else int(stream)
        launches = ct.c_uint64()
        started = time.perf_counter()
        self._check(self.lib.resident_run(self.handle, stream, host_loop, profile, ct.byref(launches)))
        self.request_seconds = time.perf_counter() - started
        state = State()
        self._check(self.lib.resident_read(self.handle, ct.byref(state), ct.sizeof(state)))
        self.host_graph_launches = launches.value
        return state

    def close(self):
        if self.handle:
            self.lib.resident_close(self.handle)
            self.handle = ct.c_void_p()
            self.resources = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()
