#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free source-contract tests for ratio-4 CSA/indexer primitives."""

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


def bf16_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def arr_f(xs):
    return (ctypes.c_float * len(xs))(*[f32(x) for x in xs])


def arr_u16(xs):
    return (ctypes.c_uint16 * len(xs))(*xs)


def compiler():
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def compile_lib(td):
    cc = compiler()
    if not cc:
        return None
    out = os.path.join(td, "libcsaref.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_csa_ref.c"),
        os.path.join(REPO, "src", "deepseek_v4_compressor_ref.c"),
        os.path.join(REPO, "src", "deepseek_v4_attention_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise AssertionError("CSA scalar compile failed:\n" + p.stdout + p.stderr)
    return out


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_encode(x):
    neg = math.copysign(1.0, x) < 0.0
    a = min(6.0, abs(f32(x)))
    best = 0
    best_err = abs(a - E2M1[0])
    for code in range(1, 8):
        err = abs(a - E2M1[code])
        if err < best_err or (err == best_err and code % 2 == 0 and best % 2 == 1):
            best, best_err = code, err
    return best | (8 if neg else 0)


def e2m1_decode(code):
    mag = E2M1[code & 7]
    return -mag if code & 8 else mag


def ceil_pow2_ratio(amax, divisor):
    ratio = f32(f32(amax) * f32(1.0 / divisor))
    bits = struct.unpack("<I", struct.pack("<f", ratio))[0]
    exp = ((bits >> 23) & 0xFF) - 127
    if bits & 0x7FFFFF:
        exp += 1
    return math.ldexp(1.0, exp)


def fp4_sim(xs, block):
    out = list(xs)
    for base in range(0, len(out), block):
        xb = out[base:base + block]
        amax = max(max(abs(v) for v in xb), math.ldexp(6.0, -126))
        scale = ceil_pow2_ratio(amax, 6.0)
        for j, value in enumerate(xb):
            code = e2m1_encode(max(-6.0, min(6.0, f32(value / scale))))
            out[base + j] = bf16(f32(e2m1_decode(code) * scale))
    return out


def overlap_python(kv, score, ape, chunks, d, ignore_overlap=False):
    two_d = 2 * d
    out = []
    for c in range(chunks):
        for j in range(d):
            logits = []
            values = []
            for t in range(4):
                if c == 0 or ignore_overlap:
                    logits.append(-math.inf)
                    values.append(0.0)
                else:
                    p = ((c - 1) * 4 + t) * two_d + j
                    logits.append(f32(score[p] + ape[t * two_d + j]))
                    values.append(kv[p])
                p = (c * 4 + t) * two_d + d + j
                logits.append(f32(score[p] + ape[t * two_d + d + j]))
                values.append(kv[p])
            m = max(logits)
            den = f32(0.0)
            num = f32(0.0)
            for z, v in zip(logits, values):
                if not math.isfinite(z):
                    continue
                e = f32(math.exp(f32(z - m)))
                den = f32(den + e)
                num = f32(num + f32(v * e))
            out.append(bf16(f32(num / den)))
    return out


def score_python(q, kv, weights, ncomp):
    out = []
    for t in range(ncomp):
        acc = f32(0.0)
        for h in range(64):
            dot = f32(0.0)
            for d in range(128):
                dot = f32(dot + f32(q[h * 128 + d] * kv[t * 128 + d]))
            dot = bf16(dot)
            dot = max(0.0, dot)
            product = bf16(f32(dot * weights[h]))
            acc = f32(acc + product)
        out.append(bf16(acc))
    return out


def main():
    with tempfile.TemporaryDirectory(prefix="v5-csa-scalar-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)

        # 1) Hadamard: direct H[i,j]=(-1)^popcount(i&j), not a butterfly oracle.
        had = lib.waste_ds_v4_hadamard_bf16_ref
        had.restype = ctypes.c_int
        had.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        x = [0.0] * 128
        x[0] = bf16(1.0)
        x[7] = bf16(0.5)
        arr = arr_f(x)
        assert had(arr, 128) == 0
        scale = f32(1.0 / math.sqrt(128.0))
        expected = []
        for row in range(128):
            raw = f32(x[0] + ((-1.0) if ((row & 7).bit_count() & 1) else 1.0) * x[7])
            expected.append(bf16(f32(raw * scale)))
        assert [bf16_bits(arr[i]) for i in range(128)] == [bf16_bits(v) for v in expected]

        # 2) K32 FP4 simulation. The K64 mutation must destroy the small block.
        fp4 = lib.waste_ds_v4_fp4_sim_inplace_ref
        fp4.restype = ctypes.c_int
        fp4.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t]
        vals = [bf16((1 + (i % 7)) / 16.0) for i in range(32)] + [bf16((i % 7) - 3.0) for i in range(32)]
        arr = arr_f(vals)
        assert fp4(arr, 64, 32) == 0
        want32 = fp4_sim(vals, 32)
        wrong64 = fp4_sim(vals, 64)
        got = [arr[i] for i in range(64)]
        assert [bf16_bits(v) for v in got] == [bf16_bits(v) for v in want32]
        assert [bf16_bits(v) for v in want32] != [bf16_bits(v) for v in wrong64], "K64 FP4 mutation survived"

        # 3) Overlap pooling: chunk1 must visibly depend on chunk0 FIRST half.
        overlap = lib.waste_ds_v4_csa_overlap_pool_ref
        overlap.restype = ctypes.c_int
        overlap.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_float)]
        chunks, d = 2, 2
        two_d = 4
        kv = [0.0] * (chunks * 4 * two_d)
        score = [0.0] * len(kv)
        ape = [0.0] * (4 * two_d)
        # previous chunk / token3 / first-half dim0: large, high score
        p = (3 * two_d) + 0
        kv[p] = 6.0; score[p] = 2.0
        # current chunk1 / token3 / second-half dim0: competing value
        p = ((4 + 3) * two_d) + d + 0
        kv[p] = -2.0; score[p] = 1.0
        # dim1 has a simple current-only marker.
        p = ((4 + 1) * two_d) + d + 1
        kv[p] = 3.0; score[p] = 1.5
        out = (ctypes.c_float * (chunks * d))()
        assert overlap(arr_f(kv), arr_f(score), arr_f(ape), chunks, d, out) == 0
        expected = overlap_python(kv, score, ape, chunks, d)
        wrong = overlap_python(kv, score, ape, chunks, d, ignore_overlap=True)
        got = [out[i] for i in range(chunks * d)]
        assert [bf16_bits(v) for v in got] == [bf16_bits(v) for v in expected]
        assert [bf16_bits(v) for v in expected[d:]] != [bf16_bits(v) for v in wrong[d:]], "ignore-overlap mutation survived"

        # 4) Index score BF16 boundaries + ReLU/negative head weighting.
        score_fn = lib.waste_ds_v4_csa_index_scores_ref
        score_fn.restype = ctypes.c_int
        score_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                             ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                             ctypes.POINTER(ctypes.c_float)]
        q = [0.0] * (64 * 128)
        q[0] = bf16(1.0)
        q[128] = bf16(-2.0)
        kv2 = [0.0] * (2 * 128)
        kv2[0] = bf16(2.0)
        kv2[128] = bf16(-1.0)
        weights = [0.0] * 64
        weights[0] = bf16(0.5)
        weights[1] = bf16(-0.25)
        scores = (ctypes.c_float * 2)()
        assert score_fn(arr_f(q), arr_f(kv2), 2, arr_f(weights), scores) == 0
        want_scores = score_python(q, kv2, weights, 2)
        assert [bf16_bits(scores[i]) for i in range(2)] == [bf16_bits(v) for v in want_scores]
        # Omitting ReLU makes t0 include the negative head and must differ.
        assert want_scores[0] != bf16(2.0 * 0.5 + (-4.0) * -0.25)

        # 5) BF16 weights_proj scale: sparse one-product rows make reduction exact.
        weights_fn = lib.waste_ds_v4_csa_indexer_weights_ref
        weights_fn.restype = ctypes.c_int
        weights_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint16),
                               ctypes.POINTER(ctypes.c_float)]
        xin = [0.0] * 4096
        xin[17] = bf16(2.0)
        wb = [0] * (64 * 4096)
        for h in range(64):
            wb[h * 4096 + 17] = bf16_bits((h % 7 - 3) / 8.0)
        wout = (ctypes.c_float * 64)()
        assert weights_fn(arr_f(xin), arr_u16(wb), wout) == 0
        ws = f32(1.0 / math.sqrt(128.0 * 64.0))
        for h in range(64):
            w = bf16_from_bits(wb[h * 4096 + 17])
            want = bf16(f32(bf16(f32(xin[17] * w)) * ws))
            assert bf16_bits(wout[h]) == bf16_bits(want)

        # 6) Top-512 cutoff actually prunes 8 of 520 distinct BF16 scores.
        topk = lib.waste_ds_v4_csa_topk_row_ref
        topk.restype = ctypes.c_size_t
        topk.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        unique = [bf16_from_bits(0x3C00 + i) for i in range(520)]
        idx = (ctypes.c_int32 * 512)()
        assert topk(arr_f(unique), 520, 520, 1000, idx, 512) == 512
        assert list(idx) == [1000 + i for i in range(519, 7, -1)]

        # Causal masking happens before top-k: future high scores must lose to visible lows.
        short = [bf16_from_bits(0x3D00 + i) for i in range(20)]
        idx20 = (ctypes.c_int32 * 20)()
        assert topk(arr_f(short), 20, 10, 80, idx20, 20) == 20
        assert list(idx20[:10]) == [89,88,87,86,85,84,83,82,81,80]
        assert list(idx20[10:]) == [-1] * 10

    print("PASS Gate E ratio-4 CSA scalar contract: Hadamard, K32 FP4, overlap, score/ReLU, weights, top-512/mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
