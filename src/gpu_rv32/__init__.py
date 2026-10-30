"""CUDA RV32IM VM and integer Python subset, independent of any LLM runtime."""

from .assembler import assemble
from .runtime import run_code, run_assembly
from .python_vm import Image, compile_program, run_program

__all__ = ["assemble", "run_code", "run_assembly", "Image", "compile_program", "run_program"]
__version__ = "0.1.0"
