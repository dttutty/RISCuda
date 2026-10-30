"""Run assembly or integer Python from a standalone checkout or installed wheel."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .assembler import assemble
from .python_vm import compile_program, run_program
from .runtime import run_assembly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="language", required=True)
    for language in ["asm", "python"]:
        command = sub.add_parser(language)
        command.add_argument("source", type=Path)
        command.add_argument("--compile-only", action="store_true")
        command.add_argument("--output", type=Path, help="new artifact directory")
        command.add_argument("--instruction-budget", type=int, default=20000000 if language == "python" else 4096)
        if language == "asm":
            command.add_argument("--input", type=lambda s: int(s, 0), default=0)
            command.add_argument("--gpu-assembler", action="store_true")
        else:
            command.add_argument("--statement-budget", type=int, default=10000)
    args = parser.parse_args()
    source = args.source.read_text()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    metadata = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "command": sys.argv, "code_commit": git.stdout.strip() if git.returncode == 0 else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}
    if args.language == "python":
        image = compile_program(source, statement_budget=args.statement_budget)
        metadata["interpreter"] = image.metadata
        result = None if args.compile_only else run_program(image, instruction_budget=args.instruction_budget)
        if args.output:
            (args.output / "interpreter.elf").write_bytes(image.elf)
            (args.output / "initial_ram.bin").write_bytes(image.memory)
    else:
        if args.gpu_assembler and args.compile_only:
            parser.error("--gpu-assembler requires execution; CPU assembly supports --compile-only")
        code = None if args.gpu_assembler else assemble(source)
        result = None if args.compile_only else run_assembly(source, input=args.input,
            instruction_budget=args.instruction_budget, device_assembler=args.gpu_assembler)
        if result is not None:
            code = result["machine_code"]
        if args.output:
            (args.output / "program.bin").write_bytes(code)
    if args.output:
        (args.output / ("source.py" if args.language == "python" else "source.S")).write_text(source)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    display = {k: v for k, v in result.items() if k not in {"memory", "machine_code"}} if result is not None else metadata
    if args.output and result is not None:
        (args.output / "results.json").write_text(json.dumps(display, indent=2) + "\n")
    print(json.dumps(display, ensure_ascii=False, indent=2))
    if result is not None and (result["status"] or result.get("error", 0)):
        raise SystemExit(1)
