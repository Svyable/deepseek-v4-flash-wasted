#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free source-contract tests for ratio-128 compressed-history indexing."""

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


def arr_f(values):
    return (ctypes.c_float * len(values))(*[f32(x) for x in values])


def compiler():
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def compile_lib(td):
    cc = compiler()
    if not cc:
        return None
    out = os.path.join(td, "libhistoryref.so")
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
        raise AssertionError("history scalar compile failed:\n" + p.stdout + p.stderr)
    return out


def sparse_python(q, kv, idxs, sink, scale):
    d = len(q)
    best = -math.inf
    denom = f32(0.0)
    acc = [f32(0.0)] * d
    for base in range(0, len(idxs), 64):
        block = idxs[base:base + 64]
        scores = []
        new_best = best
        for idx in block:
            if idx < 0:
                scores.append(-math.inf)
                continue
            dot = f32(0.0)
            for a, b in zip(q, kv[idx]):
                dot = f32(dot + f32(a * b))
            score = f32(dot * scale)
            scores.append(score)
            new_best = max(new_best, score)
        rescale = f32(math.exp(best - new_best)) if math.isfinite(best) else f32(0.0)
        denom = f32(denom * rescale)
        acc = [f32(v * rescale) for v in acc]
        for idx, score in zip(block, scores):
            if not math.isfinite(score):
                continue
            e = f32(math.exp(score - new_best))
            denom = f32(denom + e)
            eb = bf16(e)
            for k in range(d):
                acc[k] = f32(acc[k] + f32(eb * kv[idx][k]))
        best = new_best
    denom = f32(denom + f32(math.exp(sink - best)))
    return [bf16(f32(v / denom)) for v in acc]


def main():
    with tempfile.TemporaryDirectory(prefix="gate-e-history-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)

        comp = lib.waste_ds_v4_compress_indices_ref
        comp.restype = ctypes.c_size_t
        comp.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]

        pre = (ctypes.c_int32 * (256 * 2))()
        assert comp(128, 256, 0, 256, pre, 512) == 512
        assert list(pre[126 * 2:127 * 2]) == [-1, -1]
        assert list(pre[127 * 2:128 * 2]) == [256, -1]
        assert list(pre[128 * 2:129 * 2]) == [256, -1]
        assert list(pre[255 * 2:256 * 2]) == [256, 257]
        assert list(pre[127 * 2:128 * 2]) != [128, -1], \
            "prefill compressed offset must be full seqlen, not window size"

        dec = (ctypes.c_int32 * 4)()
        assert comp(128, 1, 126, 128, dec, 4) == 0
        assert comp(128, 1, 127, 128, dec, 4) == 1 and dec[0] == 128
        assert comp(128, 1, 255, 128, dec, 4) == 2 and list(dec[:2]) == [128, 129]

        row = lib.waste_ds_v4_ratio_history_topk_row_ref
        row.restype = ctypes.c_size_t
        row.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]

        r127 = (ctypes.c_int32 * 130)()
        assert row(128, 128, 256, 0, 127, r127, 130) == 130
        assert list(r127[:128]) == list(range(128))
        assert list(r127[128:130]) == [256, -1]

        r128 = (ctypes.c_int32 * 130)()
        assert row(128, 128, 256, 0, 128, r128, 130) == 130
        assert list(r128[:128]) == list(range(1, 129))
        assert list(r128[128:130]) == [256, -1]

        r255 = (ctypes.c_int32 * 130)()
        assert row(128, 128, 256, 0, 255, r255, 130) == 130
        assert list(r255[:128]) == list(range(128, 256))
        assert list(r255[128:130]) == [256, 257]

        rdec = (ctypes.c_int32 * 129)()
        assert row(128, 128, 1, 128, 0, rdec, 129) == 129
        assert list(rdec[:128]) == list(range(1, 128)) + [0]
        assert rdec[128] == 128

        hist = lib.waste_ds_v4_ratio_history_sparse_attn_head_ref
        hist.restype = ctypes.c_int
        hist.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
        ]

        q = [bf16(1.0), bf16(-0.25)]
        local = [
            [bf16(0.25), bf16(0.0)],
            [bf16(0.5), bf16(0.25)],
            [bf16(-0.25), bf16(0.5)],
            [bf16(0.75), bf16(-0.5)],
        ]
        compressed = [
            [bf16(2.0), bf16(1.0)],
            [bf16(-1.5), bf16(2.0)],
        ]
        kv = local + compressed
        expected_idxs = [2, 3, 4, 5]
        expected = sparse_python(q, kv, expected_idxs, -0.5, 0.75)
        got = (ctypes.c_float * 2)()
        assert hist(arr_f(q), arr_f([v for x in local for v in x]), 4,
                    arr_f([v for x in compressed for v in x]), 2,
                    2, 2, 4, 0, 3, 2,
                    ctypes.c_float(-0.5), ctypes.c_float(0.75), got) == 0
        assert [got[i] for i in range(2)] == expected

        local_only = sparse_python(q, kv, [2, 3, -1, -1], -0.5, 0.75)
        wrong_offset = sparse_python(q, kv, [2, 3, 2, 3], -0.5, 0.75)
        assert expected != local_only, "dropping compressed history must be observable"
        assert expected != wrong_offset, "using local-window offset for prefill must be observable"

    print("PASS Gate E ratio-128 history scalar contract: prefill/decode offsets, visibility boundaries, combined sparse attention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
