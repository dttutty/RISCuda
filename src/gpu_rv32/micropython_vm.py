"""Real MicroPython v1.27.0 compiler/VM, running as a GPU RV32IM guest."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile

from .build import GUEST, ROOT, cache_dir
from .c_vm import FLAGS, run_c, source_hashes
from .elf import read_elf
from .python_vm import Image, compiler

VERSION = "v1.27.0"
REVISION = "78ff170de9e32c79db6e64d3e33d2bd60002bdcd"
VENDOR = ROOT/"vendor"/"micropython"


def compile_micropython(source):
    encoded = source.encode("utf-8")
    if not encoded or len(encoded) > 16383 or b"\0" in encoded:
        raise ValueError("MicroPython source must contain 1..16383 UTF-8 bytes without NUL")
    zig, version = compiler()
    inputs = [p for base in (VENDOR, GUEST/"micropython", GUEST/"include")
              for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    inputs += [GUEST/n for n in ("c_start.S", "c_support.c", "c_link.ld")]
    hashes = source_hashes(inputs, ROOT)
    key = hashlib.sha256(json.dumps([version, FLAGS, hashes], sort_keys=True).encode()).hexdigest()[:20]
    target = cache_dir()/"micropython"/key
    target.mkdir(parents=True, exist_ok=True)
    elf = target/"micropython.elf"
    if not elf.exists():
        # Build in an isolated writable directory; installed package stays read-only.
        with tempfile.TemporaryDirectory(dir=target) as directory:
            build = Path(directory)/"build"
            environment = dict(os.environ, MICROPY_GIT_TAG=VERSION, MICROPY_GIT_HASH=REVISION,
                               PYTHONDONTWRITEBYTECODE="1")
            command = ["make", "-j4", "-C", str(GUEST/"micropython"), "MP_TOP="+str(VENDOR),
                       "BUILD="+str(build), "CC="+shlex.quote(zig)+" cc", "PYTHON="+shlex.quote(sys.executable),
                       "RV_FLAGS="+shlex.join(FLAGS)]
            completed = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            (target/"build.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(f"MicroPython build failed; see {target/'build.log'}\n"+completed.stdout[-6000:])
            os.replace(build/"micropython.elf", elf)
    data = elf.read_bytes()
    code, memory, entry, symbols = read_elf(data)
    if not {"mp_source", "guest_output", "guest_error"} <= symbols.keys():
        raise ValueError("missing MicroPython guest symbols")
    memory = bytearray(memory)
    position = symbols["mp_source"]
    if position > len(memory)-16384:
        raise ValueError("MicroPython source outside RAM")
    memory[position:position+len(encoded)+1] = encoded+b"\0"
    metadata = {"language": "micropython", "version": VERSION, "upstream_revision": REVISION,
                "isa": "rv32im", "compiler": "zig "+version, "compiler_flags": FLAGS,
                "build_source_sha256": hashes, "source_sha256": hashlib.sha256(encoded).hexdigest(),
                "elf_sha256": hashlib.sha256(data).hexdigest(), "ram_bytes": len(memory),
                "code_words": len(code)//4, "bigint_native_operand_bits": 4096}
    return Image(code, bytes(memory), entry, symbols, data, metadata)


def run_micropython(image, *, instruction_budget=200000000):
    """Compile and evaluate source in the guest; one kernel, no host eval/exec."""
    return run_c(image, instruction_budget=instruction_budget)
