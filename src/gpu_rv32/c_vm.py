"""Compile freestanding C to actual RV32IM, then execute in one GPU kernel."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from .build import GUEST, cache_dir
from .elf import read_elf
from .python_vm import Image, compiler
from .runtime import run_code

FLAGS = ["-target", "riscv32-freestanding", "-mcpu=baseline_rv32+m-a-c-d-f", "-mabi=ilp32",
         "-Os", "-g", "-ffreestanding", "-fno-builtin", "-fno-stack-protector",
         "-msmall-data-limit=0", "-mno-relax", "-ffunction-sections", "-fdata-sections"]


def source_hashes(paths, root=GUEST):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def compile_c(source):
    """Build int main(void), no OS/libc except packaged memory/stdout helpers."""
    encoded = source.encode("utf-8")
    if not encoded or len(encoded) > 65535 or b"\0" in encoded:
        raise ValueError("C source must contain 1..65535 UTF-8 bytes without NUL")
    zig, version = compiler()
    names = [GUEST/n for n in ("c_start.S", "c_support.c", "support.c", "c_stdio.c", "c_link.ld")]
    hashes = source_hashes([*names, *list((GUEST/"include").glob("*"))])
    key = hashlib.sha256(json.dumps([version, FLAGS, hashes, source], sort_keys=True).encode()).hexdigest()[:20]
    target = cache_dir()/"c"/key
    target.mkdir(parents=True, exist_ok=True)
    elf = target/"program.elf"
    if not elf.exists():
        with tempfile.TemporaryDirectory(dir=target) as directory:
            path = Path(directory)
            (path/"program.c").write_bytes(encoded)
            subprocess.run([zig, "cc", *FLAGS, "-I"+str(GUEST/"include"), "-nostdlib",
                            "-Wl,--gc-sections,--build-id=none", "-Wl,-T,"+str(GUEST/"c_link.ld"),
                            str(path/"program.c"), *[str(p) for p in names[:-1]],
                            "-o", str(path/"program.elf")], check=True)
            os.replace(path/"program.elf", elf)
    data = elf.read_bytes()
    code, memory, entry, symbols = read_elf(data)
    metadata = {"language": "c", "isa": "rv32im", "compiler": "zig "+version,
                "compiler_flags": FLAGS, "source_sha256": hashlib.sha256(encoded).hexdigest(),
                "support_sha256": hashes, "elf_sha256": hashlib.sha256(data).hexdigest(),
                "ram_bytes": len(memory), "code_words": len(code)//4}
    return Image(code, memory, entry, symbols, data, metadata)


def run_c(image, *, instruction_budget=200000000):
    result = run_code(image.code, memory=image.memory, entry=image.entry, services=True,
                      instruction_budget=instruction_budget)
    memory = result.pop("memory")
    offset = image.symbols.get("guest_output")
    stdout = "" if offset is None else memory[offset:offset+16384].split(b"\0", 1)[0].decode("utf-8", "replace")
    offset = image.symbols.get("guest_error")
    error = 0 if offset is None else struct.unpack_from("<i", memory, offset)[0]
    return dict(result, stdout=stdout, error=error,
                error_name={0: "ok", 1: "exception", 2: "stdout_capacity", 3: "fatal",
                            4: "unsupported_format"}.get(error, "unknown"))
