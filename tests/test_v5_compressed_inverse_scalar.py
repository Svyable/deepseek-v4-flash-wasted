#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free mutation pin for inverse compressed YaRN, using pair 20."""

import ctypes
import math
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 512
ROPE = 64
NOPE = 448
PAIR = 20
POS = 255
BASE = 160000.0
FACTOR = 16.0
ORIGINAL = 65536
BETA_FAST = 32.0
BETA_SLOW = 1.0


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def bf16_bits(x):
    u = struct.unpack("<I", struct.pack("<f", f32(x)))[0]
    if (u & 0x7F800000) == 0x7F800000:
        if u & 0x007FFFFF:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16(x):
    return struct.unpack("<f", struct.pack("<I", bf16_bits(x) << 16))[0]


def correction_dim(rotations, base=BASE):
    return ROPE * math.log(ORIGINAL / (rotations * 2 * math.pi)) / (2 * math.log(base))


def yarn_freq(pair):
    freq = 1.0 / (BASE ** ((2 * pair) / ROPE))
    low = max(0.0, math.floor(correction_dim(BETA_FAST)))
    high = min(ROPE - 1.0, math.ceil(correction_dim(BETA_SLOW)))
    if low == high:
        high += 0.001
    ramp = min(1.0, max(0.0, (pair - low) / (high - low)))
    smooth = 1.0 - ramp
    return f32(freq / FACTOR * (1.0 - smooth) + freq * smooth)


def rotate_pair(re, im, angle):
    c = f32(math.cos(f32(angle)))
    s = f32(math.sin(f32(angle)))
    return (
        bf16(f32(f32(re * c) - f32(im * s))),
        bf16(f32(f32(re * s) + f32(im * c))),
    )


def compiler():
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def compile_lib(td):
    cc = compiler()
    if not cc:
        return None
    out = os.path.join(td, "libinverse.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_compressor_ref.c"),
        os.path.join(REPO, "src", "deepseek_v4_attention_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise AssertionError("inverse YaRN compile failed:\n" + p.stdout + p.stderr)
    return out


def main():
    with tempfile.TemporaryDirectory(prefix="inverse-yarn-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)
        fn = lib.waste_ds_v4_compressed_inverse_rope_ref
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float,
                       ctypes.c_float, ctypes.c_float]

        x = [0.0] * DIM
        i = NOPE + 2 * PAIR
        x[i] = bf16(1.25)
        x[i + 1] = bf16(-0.75)
        arr = (ctypes.c_float * DIM)(*x)
        rc = fn(arr, DIM, ROPE, POS, ORIGINAL,
                ctypes.c_float(BASE), ctypes.c_float(FACTOR),
                ctypes.c_float(BETA_FAST), ctypes.c_float(BETA_SLOW))
        if rc != 0:
            raise AssertionError("inverse compressed YaRN returned failure")

        freq = yarn_freq(PAIR)
        expected = rotate_pair(x[i], x[i + 1], f32(-POS * freq))
        got = (arr[i], arr[i + 1])
        if [bf16_bits(v) for v in got] != [bf16_bits(v) for v in expected]:
            raise AssertionError(
                f"pair20 inverse mismatch: got {[hex(bf16_bits(v)) for v in got]}, "
                f"want {[hex(bf16_bits(v)) for v in expected]}")

        # Known-wrong mutations must be visible at BF16: forward sign and plain
        # base-10000 RoPE. Pair 20 is in the YaRN blend interior, unlike pair 0.
        wrong_sign = rotate_pair(x[i], x[i + 1], f32(POS * freq))
        plain_freq = f32(1.0 / (10000.0 ** ((2 * PAIR) / ROPE)))
        wrong_plain = rotate_pair(x[i], x[i + 1], f32(-POS * plain_freq))
        got_bits = [bf16_bits(v) for v in got]
        if got_bits == [bf16_bits(v) for v in wrong_sign]:
            raise AssertionError("inverse-sign mutation survived at pair20")
        if got_bits == [bf16_bits(v) for v in wrong_plain]:
            raise AssertionError("plain base-10000 inverse mutation survived at pair20")

    print("PASS inverse compressed YaRN pair20: independent equation + sign/base mutations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
