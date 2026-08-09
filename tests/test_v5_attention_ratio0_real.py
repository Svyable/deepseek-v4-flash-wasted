#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen real two-token/two-head ratio-0 attention fixture."""

import ctypes
import json
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_attn_ratio0_real")
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


def read_raw(name):
    with open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def read_u16(name):
    data = read_raw(name)
    return list(struct.unpack(f"<{len(data)//2}H", data))


def read_f32(name):
    data = read_raw(name)
    return list(struct.unpack(f"<{len(data)//4}f", data))


def c_u8(data):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def c_f32(values):
    return (ctypes.c_float * len(values))(*values)


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    out = os.path.join(td, "libv5attn.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_attention_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise AssertionError("ratio-0 real replay compile failed:\n" + p.stdout + p.stderr)
    return out


def assert_bf16_exact(label, got, expected_bits):
    got_bits = [f32_to_bf16(got[i]) for i in range(len(expected_bits))]
    if got_bits == expected_bits:
        return
    first = next(i for i, (a, b) in enumerate(zip(got_bits, expected_bits)) if a != b)
    max_abs = max(
        abs(float(got[i]) - bf16_to_f32(expected_bits[i]))
        for i in range(len(expected_bits)))
    raise AssertionError(
        f"{label} mismatch at {first}: got 0x{got_bits[first]:04x} "
        f"want 0x{expected_bits[first]:04x}; max_abs={max_abs:.9g}")


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.exists(prov_path):
        print("SKIP: real Gate E ratio-0 fixture not frozen")
        return 77
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("revision") != REV or prov.get("compress_ratio") != 0:
        raise AssertionError("ratio-0 fixture provenance drifted")

    input_bits = read_u16("input.bf16.bin")
    if len(input_bits) != 2 * 4096:
        raise AssertionError("fixture input shape drifted")
    x = [bf16_to_f32(v) for v in input_bits]
    qnorm = [bf16_to_f32(v) for v in read_u16("q-norm.bf16.bin")]
    kvnorm = [bf16_to_f32(v) for v in read_u16("kv-norm.bf16.bin")]
    sinks = read_f32("attn-sink-head0-2.f32.bin")

    expected_q = read_u16("q-after-rope.bf16.bin")
    expected_kv = read_u16("kv-after-qat.bf16.bin")
    expected_attn = read_u16("attn-after-inverse-rope.bf16.bin")
    if [len(expected_q), len(expected_kv), len(expected_attn)] != [2048, 1024, 2048]:
        raise AssertionError("fixture diagnostic shape drifted")

    with tempfile.TemporaryDirectory(prefix="v5-attn-real-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)
        fn = lib.waste_ds_v4_attention_ratio0_prefill_ref
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        qout = (ctypes.c_float * 2048)()
        kvout = (ctypes.c_float * 1024)()
        aout = (ctypes.c_float * 2048)()
        rc = fn(
            c_f32(x), 2, 2,
            c_u8(read_raw("wq-a.fp8.bin")),
            c_u8(read_raw("wq-a-scale.e8m0.bin")),
            c_f32(qnorm),
            c_u8(read_raw("wq-b-head0-2.fp8.bin")),
            c_u8(read_raw("wq-b-scale-head0-2.e8m0.bin")),
            c_u8(read_raw("wkv.fp8.bin")),
            c_u8(read_raw("wkv-scale.e8m0.bin")),
            c_f32(kvnorm), c_f32(sinks),
            ctypes.c_float(1e-6), ctypes.c_float(10000.0),
            qout, kvout, aout,
        )
        if rc != 0:
            raise AssertionError("ratio-0 prefill reference returned failure")

        assert_bf16_exact("post-RoPE Q", qout, expected_q)
        assert_bf16_exact("post-QAT KV", kvout, expected_kv)
        assert_bf16_exact("post-inverse-RoPE attention", aout, expected_attn)

    pos1_head0 = 2 * 512
    print("PASS Gate E ratio-0 real two-token/two-head core: exact BF16")
    print("q pos0 head0 first8:", " ".join(f"0x{x:04x}" for x in expected_q[:8]))
    print("attn pos1 head0 first8:",
          " ".join(f"0x{x:04x}" for x in expected_attn[pos1_head0:pos1_head0+8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
