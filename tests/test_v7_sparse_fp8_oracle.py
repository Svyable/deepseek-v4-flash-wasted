#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Differential-test the sparse Python FP8 linear used by Gate-H/V7 fixtures.

The real layer-4 chain found C BF16 -0 where the Python producer had +0 in KV.
Before allowing any zero-sign equivalence, require the optimized sparse oracle
to reproduce the scalar runtime exactly on constructions that include +0/-0,
empty blocks, exact cancellation, and extreme E8M0 powers of two.
"""

import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import make_v2_real_fixture as qref  # noqa: E402
import make_v6_attention_branch_fixture as producer  # noqa: E402

K = 256
ROWS = 137
KB = K // 128


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


def make_input():
    bits = [0] * K
    # Explicit signed zeros are semantically zero for amax but still carry a
    # bit pattern through the input quantizer.
    bits[1] = 0x8000
    bits[7] = qref.bf16_bits(1.0)
    bits[9] = qref.bf16_bits(-1.0)
    bits[17] = qref.bf16_bits(0.03125)
    bits[65] = qref.bf16_bits(-0.03125)
    # Second block: tiny dyadic values plus signed zero.
    bits[128] = 0x8000
    bits[129] = qref.bf16_bits(2.0 ** -20)
    bits[191] = qref.bf16_bits(-(2.0 ** -20))
    bits[255] = qref.bf16_bits(2.0 ** -18)
    return bits


def make_weights():
    data = bytearray(ROWS * K)
    # Deterministic finite E4M3 codes, including both signed zeros. Avoid 0x7f/ff NaNs.
    finite = [c for c in range(256) if (c & 0x7F) != 0x7F]
    for r in range(ROWS):
        for k in range(K):
            data[r * K + k] = finite[(r * 193 + k * 37 + 11) % len(finite)]
    # Make a few rows cancellation/zero-heavy around the active lanes.
    for r in (0, 1, 64, 128, 136):
        for k in range(K):
            if k not in (1, 7, 9, 17, 65, 128, 129, 191, 255):
                data[r * K + k] = 0x00 if (k & 1) == 0 else 0x80
    return bytes(data)


def make_scales():
    scales = bytearray(((ROWS + 127) // 128) * KB)
    # E8M0 values are powers of two. Exercise ordinary, very small and very
    # large scale rows without using 0xff NaN.
    pattern = [127, 90, 160, 1]
    for i in range(len(scales)):
        scales[i] = pattern[i % len(pattern)]
    return bytes(scales)


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77
    input_bits = make_input()
    weights = make_weights()
    scales = make_scales()
    want = producer.sparse_quantized_linear(weights, scales, ROWS, K, input_bits)

    with tempfile.TemporaryDirectory(prefix="v7-sparse-fp8-") as td:
        lib = os.path.join(td, "liblinear.so")
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", "-shared", "-fPIC",
               "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
               os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
               os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
               "-lm", "-o", lib]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            sys.stderr.write(p.stdout + p.stderr)
            print("FAIL: could not compile scalar FP8 differential")
            return 1
        so = ctypes.CDLL(lib)
        fn = so.waste_ds_v4_fp8_linear_e8m0_ref
        fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                       ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
        fn.restype = ctypes.c_int
        xvals = [qref.bf16_to_f32(b) for b in input_bits]
        x = (ctypes.c_float * K)(*xvals)
        w = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
        s = (ctypes.c_uint8 * len(scales)).from_buffer_copy(scales)
        out = (ctypes.c_float * ROWS)()
        if fn(x, 1, K, w, s, ROWS, out):
            print("FAIL: scalar FP8 linear rejected differential fixture")
            return 1
        got = [qref.bf16_bits(v) for v in out]

    if got != want:
        diffs = [(i, got[i], want[i]) for i in range(ROWS) if got[i] != want[i]]
        print(f"FAIL: sparse Python FP8 oracle differs from scalar C in {len(diffs)} / {ROWS} rows")
        for i, c, py in diffs[:16]:
            print(f"  row {i}: C=0x{c:04x} Python=0x{py:04x}")
        return 1

    print(f"PASS V7 sparse FP8 Python oracle matches scalar C for {ROWS} BF16 outputs")
    print("signed-zero/cancellation/extreme-scale constructions included")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
