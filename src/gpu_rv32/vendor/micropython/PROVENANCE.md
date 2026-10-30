# MicroPython source provenance

**English** | [简体中文](PROVENANCE.zh-CN.md)

Vendored from https://github.com/micropython/micropython at v1.27.0,
commit 78ff170de9e32c79db6e64d3e33d2bd60002bdcd (MIT; see LICENSE).

Included: py/, shared/libc/{string0.c,printf.c}, and
shared/runtime/{gchelper_generic.c,gchelper.h}, and
extmod/{virtpin.h,modplatform.h}. No upstream submodules required.
Original copyright/license notices are retained.

Local modifications: py/mpz.c calls the port's gpu_mpn_mul for positive
magnitudes of up to 256 base-65536 digits per operand. The MicroPython sign,
allocation and alias handling remain upstream. Larger products use the
original mpn_mul implementation. py/makeqstrdefs.py preprocesses one source per
Zig cc invocation, with at most four workers. The custom port lives in
guest/micropython/.
