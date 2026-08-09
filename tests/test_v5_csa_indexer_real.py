#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen real layer-2 ratio-4 CSA indexer fixture through C."""

import ctypes
import json
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_csa_indexer_real")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SEQLEN = 8
HIDDEN = 4096
Q_RANK = 1024
INDEX_HEADS = 64
INDEX_DIM = 128


def raw(name):
    with open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def u16(name):
    data = raw(name)
    return list(struct.unpack(f"<{len(data)//2}H", data))


def f32s(name):
    data = raw(name)
    return list(struct.unpack(f"<{len(data)//4}f", data))


def i32s(name):
    data = raw(name)
    return list(struct.unpack(f"<{len(data)//4}i", data))


def bf16_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", int(bits) << 16))[0]


def f32_to_bf16(value):
    u = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if (u & 0x7F800000) == 0x7F800000:
        if u & 0x007FFFFF:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def c_f32(values):
    return (ctypes.c_float * len(values))(*values)


def c_u8(data):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def c_u16(values):
    return (ctypes.c_uint16 * len(values))(*values)


def bits_from(arr, n):
    return [f32_to_bf16(arr[i]) for i in range(n)]


def exact(label, arr, want):
    got = bits_from(arr, len(want))
    if got == want:
        return
    first = next(i for i, (a,b) in enumerate(zip(got, want)) if a != b)
    raise AssertionError(
        f"{label} mismatch at {first}: got 0x{got[first]:04x}, want 0x{want[first]:04x}")


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    out = os.path.join(td, "libcsareal.so")
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
        raise AssertionError("CSA real compile failed:\n" + p.stdout + p.stderr)
    return out


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.exists(prov_path):
        print("SKIP: real ratio-4 CSA indexer fixture not frozen")
        return 77
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("revision") != REV or prov.get("layer") != 2 or prov.get("compress_ratio") != 4:
        raise AssertionError("CSA indexer provenance drifted")

    recipe = prov["input_recipe"]["active_tokens"]
    if [x["position"] for x in recipe] != [3,7]:
        raise AssertionError("CSA active-token recipe drifted")
    x = [0.0] * (SEQLEN * HIDDEN)
    for item in recipe:
        pos = int(item["position"]); lane = int(item["lane"])
        bits = int(item["bf16_hex"], 16)
        x[pos * HIDDEN + lane] = bf16_to_f32(bits)

    wq_a = c_u8(raw("wq-a.fp8.bin"))
    wq_a_scale = c_u8(raw("wq-a-scale.e8m0.bin"))
    q_norm = c_f32([bf16_to_f32(b) for b in u16("q-norm.bf16.bin")])
    index_wq_b = c_u8(raw("index-wq-b.fp8.bin"))
    index_wq_b_scale = c_u8(raw("index-wq-b-scale.e8m0.bin"))
    weights_proj = c_u16(u16("weights-proj.bf16.bin"))
    comp_wkv = c_u16(u16("index-comp-wkv.bf16.bin"))
    comp_wgate = c_u16(u16("index-comp-wgate.bf16.bin"))
    comp_ape = c_f32(f32s("index-comp-ape.f32.bin"))
    comp_norm = c_f32([bf16_to_f32(b) for b in u16("index-comp-norm.bf16.bin")])

    want_qr = u16("qr-pos7.bf16.bin")
    want_q = u16("index-q-pos7.bf16.bin")
    want_pool = u16("index-comp-pooled.bf16.bin")
    want_norm = u16("index-comp-post-norm.bf16.bin")
    want_rope = u16("index-comp-post-rope.bf16.bin")
    want_kv = u16("index-comp-kv.bf16.bin")
    want_weights = u16("index-weights.bf16.bin")
    want_scores = u16("index-scores.bf16.bin")
    want_topk = i32s("topk-row7.i32.bin")
    if want_topk not in ([8,9], [9,8]):
        raise AssertionError(f"real CSA top-k drifted: {want_topk}")

    with tempfile.TemporaryDirectory(prefix="v5-csa-real-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)

        linear = lib.waste_ds_v4_fp8_linear_e8m0_ref
        linear.restype = ctypes.c_int
        linear.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
        ]
        rms = lib.waste_ds_v4_rmsnorm_ref
        rms.restype = ctypes.c_int
        rms.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_size_t, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]

        x7 = c_f32(x[7*HIDDEN:8*HIDDEN])
        qr0 = (ctypes.c_float * Q_RANK)()
        if linear(x7, 1, HIDDEN, wq_a, wq_a_scale, Q_RANK, qr0) != 0:
            raise AssertionError("CSA main wq_a failed")
        qr = (ctypes.c_float * Q_RANK)()
        if rms(qr0, q_norm, Q_RANK, ctypes.c_float(1e-6), qr) != 0:
            raise AssertionError("CSA q_norm failed")
        exact("CSA qr", qr, want_qr)

        query = lib.waste_ds_v4_csa_indexer_query_ref
        query.restype = ctypes.c_int
        query.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                          ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_float)]
        q = (ctypes.c_float * (INDEX_HEADS * INDEX_DIM))()
        if query(qr, index_wq_b, index_wq_b_scale, 7, q) != 0:
            raise AssertionError("CSA indexer query failed")
        exact("CSA indexer q", q, want_q)

        compressor = lib.waste_ds_v4_csa_compressor_prefill_ref
        compressor.restype = ctypes.c_int
        compressor.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        pool = (ctypes.c_float * (2 * INDEX_DIM))()
        norm = (ctypes.c_float * (2 * INDEX_DIM))()
        rope = (ctypes.c_float * (2 * INDEX_DIM))()
        kv = (ctypes.c_float * (2 * INDEX_DIM))()
        if compressor(c_f32(x), SEQLEN, comp_wkv, comp_wgate, comp_ape, comp_norm,
                      INDEX_DIM, 1, pool, norm, rope, kv) != 0:
            raise AssertionError("CSA indexer compressor failed")
        exact("CSA comp pool", pool, want_pool)
        exact("CSA comp norm", norm, want_norm)
        exact("CSA comp rope", rope, want_rope)
        exact("CSA comp kv", kv, want_kv)

        weights_fn = lib.waste_ds_v4_csa_indexer_weights_ref
        weights_fn.restype = ctypes.c_int
        weights_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint16),
                               ctypes.POINTER(ctypes.c_float)]
        weights = (ctypes.c_float * INDEX_HEADS)()
        if weights_fn(x7, weights_proj, weights) != 0:
            raise AssertionError("CSA weights_proj failed")
        exact("CSA weights", weights, want_weights)

        score_fn = lib.waste_ds_v4_csa_index_scores_ref
        score_fn.restype = ctypes.c_int
        score_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                             ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                             ctypes.POINTER(ctypes.c_float)]
        scores = (ctypes.c_float * 2)()
        if score_fn(q, kv, 2, weights, scores) != 0:
            raise AssertionError("CSA index score failed")
        exact("CSA scores", scores, want_scores)

        topk = lib.waste_ds_v4_csa_topk_row_ref
        topk.restype = ctypes.c_size_t
        topk.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        out = (ctypes.c_int32 * 2)()
        n = topk(scores, 2, 2, SEQLEN, out, 2)
        if n != 2 or list(out) != want_topk:
            raise AssertionError(f"CSA top-k mismatch: got {list(out)[:n]}, want {want_topk}")

    print("PASS Gate E ratio-4 real CSA indexer: QR + overlap compressor + 64-head FP4 score + top-k exact")
    print("scores:", " ".join(f"0x{x:04x}" for x in want_scores), "topk:", want_topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
