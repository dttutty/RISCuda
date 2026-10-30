"""Compile a Python-subset interpreter to RV32IM and execute it on a CUDA VM.

Host work compiles C and embeds source bytes. Parsing/evaluation happen only
in the guest RISC-V interpreter; no host Python eval/exec implements the guest.
"""

from dataclasses import dataclass
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from .build import GUEST, cache_dir
from .runtime import run_code

SOURCES = GUEST
RAM = 65536
OUTPUT = 4096
ERRORS = {0: "ok", 1: "syntax", 2: "undefined_name", 3: "division_by_zero",
          4: "capacity", 5: "statement_budget", 6: "negative_shift"}


@dataclass
class Image:
    code: bytes
    memory: bytes
    entry: int
    symbols: dict
    elf: bytes
    metadata: dict


def load_elf(data):
    """Read the bounded, linked ELF32 image; no external ELF package needed."""
    if data[:7] != b"\x7fELF\x01\x01\x01" or len(data) < 52:
        raise ValueError("expected little-endian ELF32")
    h = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    if h[1] != 2 or h[2] != 243 or h[7] & 1 or h[11] != 40:
        raise ValueError(f"expected static RV32 ELF without compressed instructions: "
                         f"type={h[1]}, machine={h[2]}, flags={h[7]}, section_entry={h[11]}")
    sections = [struct.unpack_from("<IIIIIIIIII", data, h[6] + i * 40) for i in range(h[12])]
    memory, code, symbols = bytearray(RAM), bytearray(), {}
    for s in sections:
        _, kind, flags, address, offset, size, link, _, _, stride = s
        if flags & 2 and size:
            if address + size > 32768:
                raise ValueError("image must leave at least 32 KiB stack")
            if kind != 8:
                payload = data[offset:offset + size]
                if len(payload) != size:
                    raise ValueError("truncated ELF section")
                memory[address:address + size] = payload
            if flags & 4:
                if address % 4 or size % 4:
                    raise ValueError("RV32 text must be word-aligned")
                code.extend(bytes(max(0, address + size - len(code))))
                code[address:address + size] = memory[address:address + size]
        if kind == 2:
            if stride != 16:
                raise ValueError("unexpected symbol entry size")
            strings_section = sections[link]
            strings = data[strings_section[4]:strings_section[4] + strings_section[5]]
            for pos in range(offset, offset + size, 16):
                name, value, _, _, _, section = struct.unpack_from("<IIIBBH", data, pos)
                if name and section:
                    key = strings[name:].split(b"\0", 1)[0].decode("ascii")
                    symbols[key] = value
    required = {"py_source", "py_output", "py_value", "py_error", "py_budget", "_start"}
    if not required <= symbols.keys() or not code or h[4] != symbols["_start"]:
        raise ValueError(f"missing guest symbols or entry: missing={required - symbols.keys()}, "
                         f"entry={h[4]}, start={symbols.get('_start')}, code_bytes={len(code)}")
    for name, size in [("py_source", 4096), ("py_output", OUTPUT), ("py_value", 4),
                       ("py_error", 4), ("py_budget", 4)]:
        if not 0 <= symbols[name] <= RAM - size:
            raise ValueError("guest result outside RAM")
    return bytes(code), bytes(memory), h[4], symbols


@cache
def compiler():
    try:
        import ziglang
    except ImportError as exc:
        raise RuntimeError("install the pinned toolchain with uv sync --locked") from exc
    executable = Path(ziglang.__file__).parent / "zig"
    version = subprocess.check_output([str(executable), "version"], text=True).strip()
    return str(executable), version


def compile_program(source, *, statement_budget=10000):
    encoded = source.encode("utf-8")
    if not 0 < len(encoded) <= 4095 or b"\0" in encoded:
        raise ValueError("Python source must contain 1..4095 UTF-8 bytes, without NUL")
    if not 1 <= statement_budget <= 1000000:
        raise ValueError("statement budget must be in 1..1000000")
    zig, version = compiler()
    flags = ["-target", "riscv32-freestanding", "-mcpu=baseline_rv32+m-a-c-d-f", "-mabi=ilp32", "-Os", "-g",
             "-ffreestanding", "-fno-builtin", "-fno-stack-protector", "-msmall-data-limit=0", "-mno-relax"]
    names = ["mini_python.c", "support.c", "start.S", "firmware.c", "link.ld"]
    hashes = {n: hashlib.sha256((SOURCES / n).read_bytes()).hexdigest() for n in names}
    key = hashlib.sha256(json.dumps([version, flags, hashes], sort_keys=True).encode()).hexdigest()[:20]
    objects = cache_dir() / "guest" / key
    objects.mkdir(parents=True, exist_ok=True)
    for name in names[:-1]:
        target = objects / (name + ".o")
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=objects) as staging:
                temporary = Path(staging) / target.name
                subprocess.run([zig, "cc", *flags, "-c", str(SOURCES / name), "-o", str(temporary)], check=True)
                os.replace(temporary, target)
    elf = objects / "interpreter.elf"
    if not elf.exists():
        with tempfile.TemporaryDirectory(dir=objects) as staging:
            temporary = Path(staging) / elf.name
            command = [zig, "cc", *flags, "-nostdlib", "-Wl,--build-id=none",
                       "-Wl,-T," + str(SOURCES / "link.ld"),
                       *[str(objects / (n + ".o")) for n in names[:-1]], "-o", str(temporary)]
            subprocess.run(command, check=True)
            os.replace(temporary, elf)
    data = elf.read_bytes()
    code, memory, entry, symbols = load_elf(data)
    # Every program uses the same compiled interpreter; only input RAM changes.
    memory = bytearray(memory)
    begin = symbols["py_source"]
    memory[begin:begin + len(encoded) + 1] = encoded + b"\0"
    struct.pack_into("<I", memory, symbols["py_budget"], statement_budget)
    metadata = {"isa": "rv32im", "ram_bytes": RAM, "compiler": "zig " + version,
                "compiler_flags": flags, "source_sha256": hashes, "statement_budget": statement_budget,
                "python_sha256": hashlib.sha256(encoded).hexdigest(),
                "initial_ram_sha256": hashlib.sha256(memory).hexdigest(),
                "elf_sha256": hashlib.sha256(data).hexdigest(), "code_words": len(code) // 4}
    return Image(code, bytes(memory), entry, symbols, data, metadata)


def run_program(image, *, instruction_budget=20000000):
    """Execute the fixed guest interpreter; host Python never evaluates source."""
    result = run_code(image.code, memory=image.memory, entry=image.entry,
                      instruction_budget=instruction_budget)
    memory = result.pop("memory")
    value = struct.unpack_from("<i", memory, image.symbols["py_value"])[0]
    error = struct.unpack_from("<i", memory, image.symbols["py_error"])[0]
    offset = image.symbols["py_output"]
    stdout = memory[offset:offset + OUTPUT].split(b"\0", 1)[0].decode("utf-8")
    return dict(result, value=value, error=error, error_name=ERRORS.get(error, "unknown"),
                stdout=stdout, register_a0=result["value"])
