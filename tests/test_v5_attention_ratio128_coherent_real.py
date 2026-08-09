#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen same-input layer-3 ratio-128 attention fixture in C."""

import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_attention_ratio128_coherent_real")
COMP = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v5_compressor_ratio128_real")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SEQLEN = 256
HIDDEN = 4096
HEAD = 512
Q_RANK = 1024
NOPE = 448


def raw(base, name):
    with open(os.path.join(base, name), "rb") as f:
        return f.read()


def u16(base, name):
    data = raw(base, name)
    return list(struct.unpack(f"<{len(data)//2}H", data))


def f32s(base, name):
    data = raw(base, name)
    return list(struct.unpack(f"<{len(data)//4}f", data))


def i32s(base, name):
    data = raw(base, name)
    return list(struct.unpack(f"<{len(data)//4}i", data))


def bf16_to_f32(b):
    return struct.unpack("<f", struct.pack("<I", int(b) << 16))[0]


def f32_to_bf16(x):
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if (u & 0x7F800000) == 0x7F800000:
        if u & 0x007FFFFF:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def c_f32(xs):
    return (ctypes.c_float * len(xs))(*xs)


def c_u8(data):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def c_u16(xs):
    return (ctypes.c_uint16 * len(xs))(*xs)


def bits_from(arr, n):
    return [f32_to_bf16(arr[i]) for i in range(n)]


def assert_bits(label, arr, want):
    got = bits_from(arr, len(want))
    if got == want:
        return
    first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
    raise AssertionError(
        f"{label} mismatch at {first}: got 0x{got[first]:04x}, want 0x{want[first]:04x}")


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    out = os.path.join(td, "libcoherent.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_compressor_ref.c"),
        os.path.join(REPO, "src", "deepseek_v4_attention_history_ref.c"),
        os.path.join(REPO, "src", "deepseek_v4_attention_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise AssertionError("coherent ratio-128 compile failed:\n" + p.stdout + p.stderr)
    return out


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.exists(prov_path):
        print("SKIP: coherent real ratio-128 fixture not frozen")
        return 77
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("revision") != REV or prov.get("query_position") != 255:
        raise AssertionError("coherent fixture provenance drifted")

    recipe = prov["input_recipe"]["active_tokens"]
    if [x["position"] for x in recipe] != [0, 255]:
        raise AssertionError("coherent active-token recipe drifted")
    x = [0.0] * (SEQLEN * HIDDEN)
    active = {}
    for item in recipe:
        pos = int(item["position"])
        lane = int(item["lane"])
        bits = int(item["bf16_hex"], 16)
        x[pos * HIDDEN + lane] = bf16_to_f32(bits)
        active[pos] = (lane, bits)

    # Immutable raw attention slices frozen by the fixture.
    wq_a = c_u8(raw(FIX, "wq-a.fp8.bin"))
    wq_a_scale = c_u8(raw(FIX, "wq-a-scale.e8m0.bin"))
    q_norm = c_f32([bf16_to_f32(b) for b in u16(FIX, "q-norm.bf16.bin")])
    wq_b = c_u8(raw(FIX, "wq-b-head0.fp8.bin"))
    wq_b_scale = c_u8(raw(FIX, "wq-b-scale-head0.e8m0.bin"))
    wkv = c_u8(raw(FIX, "wkv.fp8.bin"))
    wkv_scale = c_u8(raw(FIX, "wkv-scale.e8m0.bin"))
    kv_norm = c_f32([bf16_to_f32(b) for b in u16(FIX, "kv-norm.bf16.bin")])
    sink = f32s(FIX, "attn-sink-head0.f32.bin")[0]

    want_q = u16(FIX, "q-pos255-head0.bf16.bin")
    want_active_local = u16(FIX, "local-kv-active-0-255.bf16.bin")
    want_comp = u16(FIX, "compressed-kv-coherent.bf16.bin")
    want_pre = u16(FIX, "attn-pre-inverse.bf16.bin")
    want_post = u16(FIX, "attn-post-inverse.bf16.bin")
    want_topk = i32s(FIX, "topk-row255.i32.bin")
    if len(want_q) != HEAD or len(want_active_local) != 2 * HEAD or len(want_comp) != 2 * HEAD:
        raise AssertionError("coherent diagnostic shape drifted")
    if want_topk != list(range(128, 256)) + [256, 257]:
        raise AssertionError("coherent top-k row drifted")

    # Frozen compressor checkpoint payload, already independently proven in PR #12.
    comp_wkv_bits = u16(COMP, "wkv.bf16.bin")
    comp_wgate_bits = u16(COMP, "wgate.bf16.bin")
    comp_ape = f32s(COMP, "ape.f32.bin")
    comp_norm = [bf16_to_f32(b) for b in u16(COMP, "norm.bf16.bin")]
    dep = prov["compressor_dependency"]
    for key, name in (("wkv_sha256", "wkv.bf16.bin"),
                      ("wgate_sha256", "wgate.bf16.bin"),
                      ("ape_sha256", "ape.f32.bin"),
                      ("norm_sha256", "norm.bf16.bin")):
        if sha256_file(os.path.join(COMP, name)) != dep[key]:
            raise AssertionError(f"coherent compressor dependency drifted: {name}")

    with tempfile.TemporaryDirectory(prefix="v5-coherent-real-") as td:
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
        head_norm = lib.waste_ds_v4_head_rmsnorm_bf16_ref
        head_norm.restype = ctypes.c_int
        head_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                              ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
        rope = lib.waste_ds_v4_compressed_rope_ref
        rope.restype = ctypes.c_int
        rope.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float,
                         ctypes.c_float, ctypes.c_float]
        inverse = lib.waste_ds_v4_compressed_inverse_rope_ref
        inverse.restype = ctypes.c_int
        inverse.argtypes = rope.argtypes
        qat = lib.waste_ds_v4_fp8_sim_inplace_ref
        qat.restype = ctypes.c_int
        qat.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t]

        # Query head 0 from the exact position-255 token in the same input.
        x255 = c_f32(x[255 * HIDDEN:256 * HIDDEN])
        qa = (ctypes.c_float * Q_RANK)()
        if linear(x255, 1, HIDDEN, wq_a, wq_a_scale, Q_RANK, qa) != 0:
            raise AssertionError("coherent wq_a failed")
        qa_norm = (ctypes.c_float * Q_RANK)()
        if rms(qa, q_norm, Q_RANK, ctypes.c_float(1e-6), qa_norm) != 0:
            raise AssertionError("coherent q_norm failed")
        q = (ctypes.c_float * HEAD)()
        if linear(qa_norm, 1, Q_RANK, wq_b, wq_b_scale, HEAD, q) != 0:
            raise AssertionError("coherent wq_b failed")
        qn = (ctypes.c_float * HEAD)()
        if head_norm(q, HEAD, ctypes.c_float(1e-6), qn) != 0:
            raise AssertionError("coherent head q norm failed")
        if rope(qn, HEAD, 64, 255, 65536,
                ctypes.c_float(160000.0), ctypes.c_float(16.0),
                ctypes.c_float(32.0), ctypes.c_float(1.0)) != 0:
            raise AssertionError("coherent q compressed RoPE failed")
        assert_bits("coherent q", qn, want_q)

        # Local KV rows from the same input. Zero input rows are exact zero;
        # only positions 0 and 255 require the real quantized projection.
        local = [0.0] * (SEQLEN * HEAD)
        active_local = []
        for pos in (0, 255):
            xv = c_f32(x[pos * HIDDEN:(pos + 1) * HIDDEN])
            kv0 = (ctypes.c_float * HEAD)()
            if linear(xv, 1, HIDDEN, wkv, wkv_scale, HEAD, kv0) != 0:
                raise AssertionError(f"coherent wkv failed at {pos}")
            kv1 = (ctypes.c_float * HEAD)()
            if rms(kv0, kv_norm, HEAD, ctypes.c_float(1e-6), kv1) != 0:
                raise AssertionError(f"coherent kv_norm failed at {pos}")
            if rope(kv1, HEAD, 64, pos, 65536,
                    ctypes.c_float(160000.0), ctypes.c_float(16.0),
                    ctypes.c_float(32.0), ctypes.c_float(1.0)) != 0:
                raise AssertionError(f"coherent kv RoPE failed at {pos}")
            if qat(kv1, NOPE, 64) != 0:
                raise AssertionError(f"coherent kv QAT failed at {pos}")
            row = [kv1[i] for i in range(HEAD)]
            local[pos * HEAD:(pos + 1) * HEAD] = row
            active_local.extend(row)
        assert_bits("coherent active local KV", c_f32(active_local), want_active_local)

        # The same 256-token input now drives the real ratio-128 compressor.
        compressor = lib.waste_ds_v4_compressor_ratio128_prefill_ref
        compressor.restype = ctypes.c_int
        compressor.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        comp_out = (ctypes.c_float * (2 * HEAD))()
        if compressor(c_f32(x), SEQLEN, c_u16(comp_wkv_bits), c_u16(comp_wgate_bits),
                      c_f32(comp_ape), c_f32(comp_norm), ctypes.c_float(1e-6),
                      None, None, None, comp_out) != 0:
            raise AssertionError("coherent compressor failed")
        assert_bits("coherent compressor KV", comp_out, want_comp)

        topk = lib.waste_ds_v4_ratio_history_topk_row_ref
        topk.restype = ctypes.c_size_t
        topk.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        idxbuf = (ctypes.c_int32 * 130)()
        n = topk(128, 128, 256, 0, 255, idxbuf, 130)
        if list(idxbuf[:n]) != want_topk:
            raise AssertionError("coherent history indices mismatch")

        history = lib.waste_ds_v4_ratio_history_sparse_attn_head_ref
        history.restype = ctypes.c_int
        history.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
        ]
        pre = (ctypes.c_float * HEAD)()
        if history(qn, c_f32(local), SEQLEN, comp_out, 2,
                   128, 128, 256, 0, 255, HEAD,
                   ctypes.c_float(sink), ctypes.c_float(HEAD ** -0.5), pre) != 0:
            raise AssertionError("coherent history attention failed")
        assert_bits("coherent pre-inverse attention", pre, want_pre)

        post = (ctypes.c_float * HEAD)(*[pre[i] for i in range(HEAD)])
        if inverse(post, HEAD, 64, 255, 65536,
                   ctypes.c_float(160000.0), ctypes.c_float(16.0),
                   ctypes.c_float(32.0), ctypes.c_float(1.0)) != 0:
            raise AssertionError("coherent inverse compressed RoPE failed")
        assert_bits("coherent post-inverse attention", post, want_post)

        pair20 = NOPE + 2 * 20
        if bits_from(pre, HEAD)[pair20:pair20 + 2] == bits_from(post, HEAD)[pair20:pair20 + 2]:
            raise AssertionError("coherent inverse pair20 mutation survived")

        # Replacing the coherent compressed history with the older unrelated
        # compressor fixture must be visible through the exact C composition.
        old_comp_bits = u16(COMP, "compressed-kv.bf16.bin")
        old_comp = c_f32([bf16_to_f32(b) for b in old_comp_bits])
        mutated = (ctypes.c_float * HEAD)()
        if history(qn, c_f32(local), SEQLEN, old_comp, 2,
                   128, 128, 256, 0, 255, HEAD,
                   ctypes.c_float(sink), ctypes.c_float(HEAD ** -0.5), mutated) != 0:
            raise AssertionError("coherent compressed-history mutation failed to execute")
        if bits_from(mutated, HEAD) == want_pre:
            raise AssertionError("unrelated compressed history mutation survived")

    print("PASS Gate E coherent ratio-128: same input -> real Q/local KV + real compressor + 130-index attention + inverse YaRN")
    print("post-inverse first8:", " ".join(f"0x{x:04x}" for x in want_post[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
