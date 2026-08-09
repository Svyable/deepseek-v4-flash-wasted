#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the bounded real-checkpoint grouped output-projection fixture."""

import ctypes
import json
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_attn_output_group0_real")
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


def c_u8(data):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def c_f32(values):
    return (ctypes.c_float * len(values))(*values)


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    out = os.path.join(td, "libv5out.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_attention_output_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise AssertionError("output projection compile failed:\n" + p.stdout + p.stderr)
    return out


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.exists(prov_path):
        print("SKIP: Gate E grouped-output fixture not frozen")
        return 77
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("revision") != REV or prov.get("group_index") != 0:
        raise AssertionError("output fixture provenance drifted")

    group = [bf16_to_f32(v) for v in u16("group-input.bf16.bin")]
    expected_lora = u16("expected-group-lora.bf16.bin")
    expected_out = u16("expected-out8.bf16.bin")
    if len(group) != 4096 or len(expected_lora) != 1024 or len(expected_out) != 8:
        raise AssertionError("output fixture shape drifted")

    with tempfile.TemporaryDirectory(prefix="v5-out-real-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)
        fn = lib.waste_ds_v4_attention_output_group_ref
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        lora = (ctypes.c_float * 1024)()
        out = (ctypes.c_float * 8)()
        args = [
            c_f32(group), 0,
            c_u8(raw("wo-a-group0.fp8.bin")),
            c_u8(raw("wo-a-group0-scale.e8m0.bin")),
            c_u8(raw("wo-b-rows0-8.fp8.bin")),
            c_u8(raw("wo-b-scale-row0.e8m0.bin")),
            8, lora, out,
        ]
        if fn(*args) != 0:
            raise AssertionError("grouped output projection returned failure")
        got_lora = [f32_to_bf16(lora[i]) for i in range(1024)]
        got_out = [f32_to_bf16(out[i]) for i in range(8)]
        if got_lora != expected_lora:
            first = next(i for i, (a, b) in enumerate(zip(got_lora, expected_lora)) if a != b)
            raise AssertionError(
                f"wo_a group latent mismatch at {first}: got 0x{got_lora[first]:04x} "
                f"want 0x{expected_lora[first]:04x}")
        if got_out != expected_out:
            raise AssertionError(
                "wo_b output mismatch: " + " ".join(f"{a:04x}/{b:04x}" for a, b in zip(got_out, expected_out)))

        # Group placement is semantically meaningful because wo_b has distinct
        # K blocks per group. The same latent placed at group 1 must not silently
        # reproduce the group-0 expected output.
        shifted = (ctypes.c_float * 8)()
        if fn(c_f32(group), 1,
              c_u8(raw("wo-a-group0.fp8.bin")),
              c_u8(raw("wo-a-group0-scale.e8m0.bin")),
              c_u8(raw("wo-b-rows0-8.fp8.bin")),
              c_u8(raw("wo-b-scale-row0.e8m0.bin")),
              8, None, shifted) != 0:
            raise AssertionError("group-placement mutation call failed")
        shifted_bits = [f32_to_bf16(shifted[i]) for i in range(8)]
        if shifted_bits == expected_out:
            raise AssertionError("group-index mutation was not observable")

    print("PASS Gate E grouped wo_a/wo_b output seam: exact BF16")
    print("group lora first8:", " ".join(f"0x{x:04x}" for x in expected_lora[:8]))
    print("out8:", " ".join(f"0x{x:04x}" for x in expected_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
