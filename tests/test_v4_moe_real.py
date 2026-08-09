#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the real layer-3 shared expert and complete Gate F combination."""

import ctypes
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v4_moe_real")


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


def read_bits(path, fmt):
    with open(path, "rb") as f:
        data = f.read()
    size = struct.calcsize(fmt)
    return list(struct.unpack(f"<{len(data)//size}{fmt}", data))


def raw(path):
    with open(path, "rb") as f:
        return f.read()


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        print("SKIP: no C compiler")
        return None
    out = os.path.join(td, "libgatef.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_expert_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout + p.stderr)
        raise RuntimeError("Gate F reference compilation failed")
    return out


def c_u8(data):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def c_f32(values):
    return (ctypes.c_float * len(values))(*values)


def main():
    if not os.path.exists(os.path.join(FIXTURE, "provenance.json")):
        print("SKIP: real Gate F MoE fixture not frozen")
        return 77

    with open(os.path.join(FIXTURE, "provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    if prov["revision"] != "9e165c30e2704aec5d9d593cce3eebd58bbef1cb":
        raise AssertionError("fixture revision drifted")

    input_bits = read_bits(os.path.join(FIXTURE, "input.bf16.bin"), "H")
    x = [bf16_to_f32(v) for v in input_bits]
    expected_shared = read_bits(os.path.join(FIXTURE, "shared-out8.bf16.bin"), "H")
    expected_hidden = read_bits(os.path.join(FIXTURE, "shared-hidden.bf16.bin"), "H")
    expected_combined = read_bits(os.path.join(FIXTURE, "combined-out8.bf16.bin"), "H")
    ids = read_bits(os.path.join(FIXTURE, "routed-ids.u32.bin"), "I")
    routed_bits = read_bits(os.path.join(FIXTURE, "routed-out8.bf16.bin"), "H")

    with tempfile.TemporaryDirectory(prefix="gate-f-moe-") as td:
        libpath = compile_lib(td)
        if libpath is None:
            return 77
        lib = ctypes.CDLL(libpath)

        shared = lib.waste_ds_v4_shared_expert_ref
        shared.restype = ctypes.c_int
        shared.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        gate = (ctypes.c_float * 2048)()
        up = (ctypes.c_float * 2048)()
        hidden = (ctypes.c_float * 2048)()
        shared_out = (ctypes.c_float * 8)()
        args = [
            c_f32(x), 4096, 2048,
            c_u8(raw(os.path.join(FIXTURE, "shared-w1.fp8.bin"))),
            c_u8(raw(os.path.join(FIXTURE, "shared-w1-scale.e8m0.bin"))),
            c_u8(raw(os.path.join(FIXTURE, "shared-w3.fp8.bin"))),
            c_u8(raw(os.path.join(FIXTURE, "shared-w3-scale.e8m0.bin"))),
            ctypes.c_float(10.0),
            c_u8(raw(os.path.join(FIXTURE, "shared-w2-rows0-8.fp8.bin"))),
            c_u8(raw(os.path.join(FIXTURE, "shared-w2-scale-row0.e8m0.bin"))),
            8, gate, up, hidden, shared_out,
        ]
        if shared(*args) != 0:
            raise AssertionError("shared expert C reference returned failure")
        got_shared = [f32_to_bf16(shared_out[i]) for i in range(8)]
        if got_shared != expected_shared:
            raise AssertionError(f"shared output mismatch: {got_shared} != {expected_shared}")
        got_hidden = [f32_to_bf16(hidden[i]) for i in range(2048)]
        if got_hidden != expected_hidden:
            first = next(i for i, (a, b) in enumerate(zip(got_hidden, expected_hidden)) if a != b)
            raise AssertionError(f"shared hidden mismatch at {first}: 0x{got_hidden[first]:04x} != 0x{expected_hidden[first]:04x}")

        combine = lib.waste_ds_v4_moe_combine_ref
        combine.restype = ctypes.c_int
        combine.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
        ]
        routed = [bf16_to_f32(v) for v in routed_bits]
        shared_vals = [bf16_to_f32(v) for v in expected_shared]
        combined = (ctypes.c_float * 8)()
        if combine((ctypes.c_uint32 * 6)(*ids), c_f32(routed), 6,
                   c_f32(shared_vals), 8, combined) != 0:
            raise AssertionError("MoE combination returned failure")
        got_combined = [f32_to_bf16(combined[i]) for i in range(8)]
        if got_combined != expected_combined:
            raise AssertionError(f"combined mismatch: {got_combined} != {expected_combined}")

        # Model-free accumulation-order pin. Top-k slot order [5,1,3] would do
        # 1 + 2^30 - 2^30 -> 0 in f32. Official ascending expert-ID order does
        # 2^30 - 2^30 + 1 -> 1. The difference survives the final BF16 cast.
        order_ids = (ctypes.c_uint32 * 3)(5, 1, 3)
        order_routed = c_f32([1.0, float(1 << 30), -float(1 << 30)])
        zero_shared = c_f32([0.0])
        order_out = (ctypes.c_float * 1)()
        if combine(order_ids, order_routed, 3, zero_shared, 1, order_out) != 0:
            raise AssertionError("order pin combine failed")
        if order_out[0] != 1.0:
            raise AssertionError(f"official expert-ID accumulation order lost: {order_out[0]}")

    print("PASS Gate F shared expert + six-routed/shared combination: exact BF16")
    print("selected IDs:", ids)
    print("combined:", " ".join(f"0x{x:04x}" for x in expected_combined))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
