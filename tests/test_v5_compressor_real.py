#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen real two-chunk ratio-128 compressor fixture."""

import ctypes
import json
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_compressor_ratio128_real")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"


def bf16_to_f32(b):
    return struct.unpack("<f", struct.pack("<I", b << 16))[0]


def f32_to_bf16(x):
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if (u & 0x7F800000) == 0x7F800000:
        if u & 0x007FFFFF:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def raw(name):
    with open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def u16(name):
    data = raw(name)
    return list(struct.unpack(f"<{len(data)//2}H", data))


def f32s(name):
    data = raw(name)
    return list(struct.unpack(f"<{len(data)//4}f", data))


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    out = os.path.join(td, "libcompressreal.so")
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
        raise AssertionError("compressor real replay compile failed:\n" + p.stdout + p.stderr)
    return out


def c_f32(xs):
    return (ctypes.c_float * len(xs))(*xs)


def c_u16(xs):
    return (ctypes.c_uint16 * len(xs))(*xs)


def exact(label, arr, expected):
    bits = [f32_to_bf16(arr[i]) for i in range(len(expected))]
    if bits == expected:
        return
    first = next(i for i,(a,b) in enumerate(zip(bits,expected)) if a != b)
    raise AssertionError(
        f"{label} mismatch at {first}: got 0x{bits[first]:04x}, want 0x{expected[first]:04x}")


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.exists(prov_path):
        print("SKIP: real ratio-128 compressor fixture not frozen")
        return 77
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("revision") != REV or prov.get("compress_ratio") != 128:
        raise AssertionError("compressor fixture provenance drifted")

    input_bits = u16("input.bf16.bin")
    if len(input_bits) != 256*4096:
        raise AssertionError("compressor input shape drifted")
    x = [bf16_to_f32(b) for b in input_bits]
    wkv = u16("wkv.bf16.bin")
    wgate = u16("wgate.bf16.bin")
    ape = f32s("ape.f32.bin")
    norm = f32s("norm.f32.bin")
    if len(wkv) != 512*4096 or len(wgate) != 512*4096 or len(ape) != 128*512 or len(norm) != 512:
        raise AssertionError("compressor parameter shape drifted")

    expected_pool = u16("pooled-before-norm.bf16.bin")
    expected_norm = u16("post-norm.bf16.bin")
    expected_rope = u16("post-yarn-rope.bf16.bin")
    expected_final = u16("compressed-kv.bf16.bin")
    if any(len(v) != 1024 for v in (expected_pool, expected_norm, expected_rope, expected_final)):
        raise AssertionError("compressor diagnostic shape drifted")

    with tempfile.TemporaryDirectory(prefix="v5-compressor-real-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)
        fn = lib.waste_ds_v4_compressor_ratio128_prefill_ref
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        pool = (ctypes.c_float * 1024)()
        norm_out = (ctypes.c_float * 1024)()
        rope = (ctypes.c_float * 1024)()
        final = (ctypes.c_float * 1024)()
        rc = fn(c_f32(x), 256, c_u16(wkv), c_u16(wgate), c_f32(ape), c_f32(norm),
                ctypes.c_float(1e-6), pool, norm_out, rope, final)
        if rc != 0:
            raise AssertionError("ratio-128 compressor reference returned failure")
        exact("pooled-before-norm", pool, expected_pool)
        exact("post-norm", norm_out, expected_norm)
        exact("post-yarn-rope", rope, expected_rope)
        exact("compressed-kv", final, expected_final)

    print("PASS Gate E ratio-128 real compressor: 4,096 exact BF16 diagnostics")
    print("pool chunk0:", " ".join(f"0x{x:04x}" for x in expected_pool[:8]))
    print("rope chunk1 tail:", " ".join(f"0x{x:04x}" for x in expected_rope[512+504:512+512]))
    print("final chunk1:", " ".join(f"0x{x:04x}" for x in expected_final[512:520]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
