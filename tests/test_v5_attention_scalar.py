#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free source-contract tests for Gate E ratio-0 attention primitives."""

import ctypes
import math
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def compiler():
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def compile_lib(td):
    cc = compiler()
    if not cc:
        return None
    out = os.path.join(td, "libattnref.so")
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
        raise AssertionError("attention scalar compile failed:\n" + p.stdout + p.stderr)
    return out


def arr_f(values):
    return (ctypes.c_float * len(values))(*[f32(x) for x in values])


def head_norm_python(values, eps):
    # Direct BF16 tensor expression from Attention.forward rather than the
    # learned RMSNorm module: square->BF16, mean->BF16, +eps->BF16,
    # rsqrt->BF16, in-place multiply->BF16.
    squares = [bf16(f32(bf16(v) * bf16(v))) for v in values]
    total = f32(0.0)
    for v in squares:
        total = f32(total + v)
    mean = bf16(f32(total / len(values)))
    shifted = bf16(f32(mean + eps))
    inv = bf16(f32(1.0 / math.sqrt(shifted)))
    return [bf16(f32(bf16(v) * inv)) for v in values]


def sparse_python(q, kv, idxs, sink, scale):
    d = len(q)
    max_score = -math.inf
    denom = f32(0.0)
    acc = [f32(0.0) for _ in range(d)]
    for base in range(0, len(idxs), 64):
        block = idxs[base:base + 64]
        scores = []
        new_max = max_score
        for idx in block:
            if idx < 0:
                scores.append(-math.inf)
                continue
            dot = f32(0.0)
            for a, b in zip(q, kv[idx]):
                dot = f32(dot + f32(a * b))
            score = f32(dot * scale)
            scores.append(score)
            new_max = max(new_max, score)
        rescale = f32(math.exp(max_score - new_max)) if math.isfinite(max_score) else f32(0.0)
        denom = f32(denom * rescale)
        acc = [f32(v * rescale) for v in acc]
        for idx, score in zip(block, scores):
            if not math.isfinite(score):
                continue
            e = f32(math.exp(score - new_max))
            denom = f32(denom + e)
            eb = bf16(e)
            for k in range(d):
                acc[k] = f32(acc[k] + f32(eb * kv[idx][k]))
        max_score = new_max
    denom = f32(denom + f32(math.exp(sink - max_score)))
    return [bf16(f32(v / denom)) for v in acc]


def main():
    with tempfile.TemporaryDirectory(prefix="gate-e-attn-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)

        rms = lib.waste_ds_v4_rmsnorm_ref
        rms.restype = ctypes.c_int
        rms.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_size_t, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
        x = [bf16(1.0), bf16(-2.0), bf16(3.0), bf16(-4.0)]
        w = [bf16(1.0), bf16(0.5), bf16(-1.0), bf16(2.0)]
        out = (ctypes.c_float * 4)()
        assert rms(arr_f(x), arr_f(w), 4, ctypes.c_float(1e-6), out) == 0
        mean_sq = f32(sum(f32(v * v) for v in x) / 4.0)
        s = f32(1.0 / math.sqrt(f32(mean_sq + 1e-6)))
        expected = [bf16(f32(f32(v * s) * ww)) for v, ww in zip(x, w)]
        assert [out[i] for i in range(4)] == expected

        head_norm = lib.waste_ds_v4_head_rmsnorm_bf16_ref
        head_norm.restype = ctypes.c_int
        head_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                              ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
        qh = [bf16(0.1015625), bf16(-0.3359375), bf16(1.234375), bf16(-2.71875)]
        hout = (ctypes.c_float * 4)()
        assert head_norm(arr_f(qh), 4, ctypes.c_float(1e-6), hout) == 0
        head_expected = head_norm_python(qh, 1e-6)
        assert [hout[i] for i in range(4)] == head_expected
        learned_unit = (ctypes.c_float * 4)()
        assert rms(arr_f(qh), None, 4, ctypes.c_float(1e-6), learned_unit) == 0
        assert [hout[i] for i in range(4)] != [learned_unit[i] for i in range(4)], \
            "fixture must distinguish direct BF16 q normalization from learned RMSNorm"

        rope = lib.waste_ds_v4_rope_ref
        rope.restype = ctypes.c_int
        rope.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.c_float, ctypes.c_int]
        r = arr_f([1.0, 0.0, 0.5, -0.25])
        assert rope(r, 4, 0, ctypes.c_float(10000.0), 0) == 0
        assert [r[i] for i in range(4)] == [bf16(1.0), bf16(0.0), bf16(0.5), bf16(-0.25)]
        r = arr_f([1.0, 0.0])
        assert rope(r, 2, 1, ctypes.c_float(10000.0), 0) == 0
        assert bf16_bits(r[0]) == bf16_bits(math.cos(1.0))
        assert bf16_bits(r[1]) == bf16_bits(math.sin(1.0))
        assert rope(r, 2, 1, ctypes.c_float(10000.0), 1) == 0
        assert abs(r[0] - 1.0) <= 1.0 / 128.0 and abs(r[1]) <= 1.0 / 128.0

        win = lib.waste_ds_v4_window_indices_ref
        win.restype = ctypes.c_size_t
        win.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        buf = (ctypes.c_int32 * 12)()
        assert win(3, 4, 0, buf, 12) == 12
        assert list(buf) == [0, -1, -1, 0, 1, -1, 0, 1, 2, 1, 2, 3]
        ring = (ctypes.c_int32 * 4)()
        assert win(4, 1, 2, ring, 4) == 4
        assert list(ring) == [0, 1, 2, -1]
        assert win(4, 1, 6, ring, 4) == 4
        assert list(ring) == [3, 0, 1, 2]

        sim = lib.waste_ds_v4_fp8_sim_inplace_ref
        sim.restype = ctypes.c_int
        sim.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t]
        values = [bf16(448.0)] + [bf16(0.0)] * 63 + [bf16(0.001)] * 64
        local64 = arr_f(values)
        global128 = arr_f(values)
        assert sim(local64, 128, 64) == 0
        assert sim(global128, 128, 128) == 0
        assert local64[64] != global128[64], "K64-vs-K128 mutation was not observable"
        assert local64[64] != 0.0, "K64 local scaling should retain the small second block"

        sparse = lib.waste_ds_v4_sparse_attn_head_ref
        sparse.restype = ctypes.c_int
        sparse.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
        ]
        q = [bf16(1.0), bf16(0.0)]
        kv = [[bf16(1.0), bf16(0.0)], [bf16(0.0), bf16(1.0)]]
        idxs = [0, 1, -1]
        expected = sparse_python(q, kv, idxs, 0.0, 1.0)
        got = (ctypes.c_float * 2)()
        flat_kv = [v for row in kv for v in row]
        cidx = (ctypes.c_int32 * len(idxs))(*idxs)
        assert sparse(arr_f(q), arr_f(flat_kv), 2, cidx, len(idxs), 2,
                      ctypes.c_float(0.0), ctypes.c_float(1.0), got) == 0
        assert [got[i] for i in range(2)] == expected

        no_sink_expected = sparse_python(q, kv, idxs, -100.0, 1.0)
        assert expected != no_sink_expected, "sink-denominator mutation should be visible"

        kv_many = [[bf16((i % 7 - 3) / 4.0), bf16((i % 5 - 2) / 4.0)] for i in range(65)]
        ids_many = list(range(65))
        exp_many = sparse_python(q, kv_many, ids_many, -0.75, 0.7)
        got_many = (ctypes.c_float * 2)()
        assert sparse(arr_f(q), arr_f([v for row in kv_many for v in row]), 65,
                      (ctypes.c_int32 * 65)(*ids_many), 65, 2,
                      ctypes.c_float(-0.75), ctypes.c_float(0.7), got_many) == 0
        assert [got_many[i] for i in range(2)] == exp_many

    print("PASS Gate E ratio-0 scalar contract: learned/head RMS, RoPE, K64 QAT, window, sink sparse attention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
