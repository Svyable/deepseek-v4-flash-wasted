#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay frozen real ratio-128 compressed-history composition evidence."""

import ctypes
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_attention_history_ratio128_real")
COMP = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_compressor_ratio128_real")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"


def raw(base, name):
    with open(os.path.join(base, name), "rb") as f:
        return f.read()


def u16(base, name):
    data = raw(base, name)
    return list(struct.unpack(f"<{len(data)//2}H", data))


def i32(base, name):
    data = raw(base, name)
    return list(struct.unpack(f"<{len(data)//4}i", data))


def f32s(base, name):
    data = raw(base, name)
    return list(struct.unpack(f"<{len(data)//4}f", data))


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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    out = os.path.join(td, "libhistoryreal.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_attention_history_ref.c"),
        os.path.join(REPO, "src", "deepseek_v4_attention_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise AssertionError("history real compile failed:\n" + p.stdout + p.stderr)
    return out


def c_f32(xs):
    return (ctypes.c_float * len(xs))(*xs)


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.exists(prov_path):
        print("SKIP: real ratio-128 history fixture not frozen")
        return 77
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("revision") != REV or prov.get("query_position") != 255:
        raise AssertionError("history fixture provenance drifted")

    comp_path = os.path.join(COMP, "compressed-kv.bf16.bin")
    dep = prov["compressor_dependency"]
    got_hash = sha256(comp_path)
    if got_hash != dep["sha256"] or got_hash != dep["expected_sha256"]:
        raise AssertionError("history fixture compressor dependency drifted")

    q_bits = u16(FIX, "q-pos255-head0.bf16.bin")
    local_bits = u16(FIX, "local-kv-0-255.bf16.bin")
    comp_bits = u16(COMP, "compressed-kv.bf16.bin")
    want_idxs = i32(FIX, "topk-row255.i32.bin")
    want = u16(FIX, "expected-attn-row255.bf16.bin")
    sink = f32s(FIX, "attn-sink-head0.f32.bin")[0]
    if len(q_bits) != 512 or len(local_bits) != 256*512 or len(comp_bits) != 2*512:
        raise AssertionError("history fixture tensor shape drifted")
    if want_idxs != list(range(128,256)) + [256,257]:
        raise AssertionError("history fixture topk row drifted")
    if len(want) != 512:
        raise AssertionError("history expected output shape drifted")

    q = [bf16_to_f32(b) for b in q_bits]
    local = [bf16_to_f32(b) for b in local_bits]
    comp = [bf16_to_f32(b) for b in comp_bits]

    with tempfile.TemporaryDirectory(prefix="v5-history-real-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)

        row = lib.waste_ds_v4_ratio_history_topk_row_ref
        row.restype = ctypes.c_size_t
        row.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        idxbuf = (ctypes.c_int32 * 130)()
        n = row(128, 128, 256, 0, 255, idxbuf, 130)
        got_idxs = list(idxbuf[:n])
        if got_idxs != want_idxs:
            raise AssertionError(f"history topk mismatch: got tail {got_idxs[-8:]}, want {want_idxs[-8:]}")

        fn = lib.waste_ds_v4_ratio_history_sparse_attn_head_ref
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        out = (ctypes.c_float * 512)()
        rc = fn(c_f32(q), c_f32(local), 256, c_f32(comp), 2,
                128, 128, 256, 0, 255, 512,
                ctypes.c_float(sink), ctypes.c_float(512 ** -0.5), out)
        if rc != 0:
            raise AssertionError("ratio-128 history C replay returned failure")
        got = [f32_to_bf16(out[i]) for i in range(512)]
        if got != want:
            first = next(i for i,(a,b) in enumerate(zip(got,want)) if a != b)
            raise AssertionError(
                f"history attention mismatch at {first}: got 0x{got[first]:04x}, want 0x{want[first]:04x}")

    print("PASS Gate E ratio-128 real compressed-history composition: 130 exact indices + 512 BF16 attention values")
    print("topk tail:", want_idxs[-8:])
    print("attn first8:", " ".join(f"0x{x:04x}" for x in want[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())