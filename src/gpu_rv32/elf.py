"""Strict static RV32IM ELF loader for freestanding C and MicroPython guests."""
import struct


def read_elf(data, *, ram_size=4194304, stack_reserve=262144):
    if len(data) < 52 or data[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("expected little-endian ELF32")
    h = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    if (h[1] != 2 or h[2] != 243 or h[3] != 1 or h[7] != 0 or h[8] != 52 or
            h[11] != 40 or not h[12] or h[6]+h[12]*40 > len(data)):
        raise ValueError("expected linked RV32IM ELF with a valid section table")
    if not 262144 <= ram_size <= 4194304 or not 65536 <= stack_reserve < ram_size:
        raise ValueError("invalid guest RAM/stack reservation")
    sections = [struct.unpack_from("<IIIIIIIIII", data, h[6]+i*40) for i in range(h[12])]
    memory, code, symbols, ranges = bytearray(ram_size), bytearray(), {}, []
    for s in sections:
        _, kind, flags, address, offset, size, link, _, _, stride = s
        if kind in (4, 6, 9):
            raise ValueError("dynamic images and relocations are unsupported")
        if flags & 2 and size:
            if address+size > ram_size-stack_reserve:
                raise ValueError("guest image overlaps reserved stack")
            if any(address < end and start < address+size for start, end in ranges):
                raise ValueError("overlapping allocated ELF sections")
            ranges.append((address, address+size))
            if kind != 8:
                if offset+size > len(data):
                    raise ValueError("truncated ELF section")
                memory[address:address+size] = data[offset:offset+size]
            if flags & 4:
                if address % 4 or size % 4 or address+size > 1048576:
                    raise ValueError("invalid RV32 code section")
                code.extend(bytes(max(0, address+size-len(code))))
                code[address:address+size] = memory[address:address+size]
        if kind == 2:
            if stride != 16 or size % 16 or offset+size > len(data) or link >= len(sections):
                raise ValueError("invalid ELF symbol table")
            st = sections[link]
            if st[4]+st[5] > len(data):
                raise ValueError("truncated ELF string table")
            strings = data[st[4]:st[4]+st[5]]
            for pos in range(offset, offset+size, 16):
                name, value, _, _, _, index = struct.unpack_from("<IIIBBH", data, pos)
                if name and index:
                    if name >= len(strings) or b"\0" not in strings[name:]:
                        raise ValueError("invalid ELF symbol name")
                    symbols[strings[name:].split(b"\0", 1)[0].decode("ascii")] = value
    if not code or h[4] % 4 or h[4] >= len(code) or h[4] != symbols.get("_start"):
        raise ValueError("missing or invalid guest entry point")
    return bytes(code), bytes(memory), h[4], symbols
