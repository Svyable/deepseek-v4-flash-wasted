#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Require the real Gate-H layer-3 endpoint to use exact F32 HC semantics."""

import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import deepseek_v4_hc_oracle as hc  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
HCFIX = os.path.join(BASE, "v6_hc_composition_real")
ATTFIX = os.path.join(BASE, "v6_attention_branch_real")
MOEFIX = os.path.join(BASE, "v6_moe_branch_real")
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24
CORRECTED_SHA = "c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7"
OLD_SHA = "0e65c4ecb328d5067f2274e724f9f46e4a13218e7fdeb706f7d1a465c0ee4761"
DIFF_INDICES = [1186, 10135, 10503, 10877, 11565, 11672, 11675, 13648, 13717]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 2:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 2}")
    return list(struct.unpack(f"<{count}H", data))


def read_f32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}f", data))


def compiler():
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and shutil.which(name):
            return name
    return None


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)


def bits(values):
    return [chain.f32_bits(v) for v in values]


def exact_bf16(label, got, want):
    if got == want:
        return
    i = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
    raise AssertionError(f"{label}[{i}]: {got[i]:04x}!={want[i]:04x}")


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    hp = json.load(open(os.path.join(HCFIX, "provenance.json"), encoding="utf-8"))
    ap = json.load(open(os.path.join(ATTFIX, "provenance.json"), encoding="utf-8"))
    mp = json.load(open(os.path.join(MOEFIX, "provenance.json"), encoding="utf-8"))
    correction = mp.get("hc_precision_correction") or {}
    if correction.get("old_layer3_final_sha256") != OLD_SHA or \
       correction.get("corrected_layer3_final_sha256") != CORRECTED_SHA or \
       correction.get("bf16_diff_indices") != DIFF_INDICES or \
       correction.get("bf16_diff_count") != len(DIFF_INDICES):
        print("FAIL: Gate-H HC precision-correction provenance drifted")
        return 1

    def param(key, count):
        meta = hp["parameters"][key]
        path = os.path.join(HCFIX, meta["file"])
        want = meta.get("payload_sha256", meta.get("sha256"))
        if want and sha(path) != want:
            raise AssertionError(f"{key}: parameter SHA mismatch")
        return read_f32(path, count)

    residual_bits = read_u16(os.path.join(HCFIX, hp["states"]["residual"]["file"]), FLAT)
    attn_pre_frozen = read_u16(os.path.join(HCFIX, hp["states"]["attn_pre"]["file"]), DIM)
    attn_meta = ap["outputs"]["branch"]
    attn_path = os.path.join(ATTFIX, attn_meta["file"])
    if sha(attn_path) != attn_meta["sha256"]:
        raise AssertionError("attention branch SHA mismatch")
    attn_bits = read_u16(attn_path, DIM)
    ffn_meta = mp["input_dependency"]
    ffn_path = os.path.join(MOEFIX, ffn_meta["file"])
    if sha(ffn_path) != ffn_meta["sha256"]:
        raise AssertionError("ffn_pre SHA mismatch")
    ffn_frozen = read_u16(ffn_path, DIM)
    moe_meta = mp["outputs"]["moe_branch"]
    moe_path = os.path.join(MOEFIX, moe_meta["file"])
    if sha(moe_path) != moe_meta["sha256"]:
        raise AssertionError("MoE branch SHA mismatch")
    moe_bits = read_u16(moe_path, DIM)
    out_meta = mp["outputs"]["layer3_final"]
    out_path = os.path.join(MOEFIX, out_meta["file"])
    if sha(out_path) != CORRECTED_SHA or out_meta.get("sha256") != CORRECTED_SHA:
        raise AssertionError("corrected layer3 final SHA mismatch")
    frozen_final = read_u16(out_path, FLAT)

    residual = [chain.bf16_to_f32(v) for v in residual_bits]
    attn = [chain.bf16_to_f32(v) for v in attn_bits]
    moe = [chain.bf16_to_f32(v) for v in moe_bits]
    attn_fn = param("attn_fn", MIX * FLAT)
    attn_scale = param("attn_scale", 3)
    attn_base = param("attn_base", MIX)
    ffn_fn = param("ffn_fn", MIX * FLAT)
    ffn_scale = param("ffn_scale", 3)
    ffn_base = param("ffn_base", MIX)

    # Independent exact-F32 path.
    apre, apost, acomb, _ = hc.params(residual, attn_fn, attn_scale, attn_base, dim=DIM)
    py_attn_pre, _ = chain.cast_bf16(hc.pre_y(residual, apre, dim=DIM))
    exact_bf16("Python attn_pre", py_attn_pre, attn_pre_frozen)
    py_after_f = hc.post_y(attn, residual, apost, acomb, dim=DIM)
    py_after_bits, py_after = chain.cast_bf16(py_after_f)
    fpre, fpost, fcomb, _ = hc.params(py_after, ffn_fn, ffn_scale, ffn_base, dim=DIM)
    py_ffn, _ = chain.cast_bf16(hc.pre_y(py_after, fpre, dim=DIM))
    exact_bf16("Python ffn_pre", py_ffn, ffn_frozen)
    py_final_f = hc.post_y(moe, py_after, fpost, fcomb, dim=DIM)
    py_final, _ = chain.cast_bf16(py_final_f)
    exact_bf16("Python final", py_final, frozen_final)

    # Current C scalar path, including C-produced Sinkhorn coefficients.
    with tempfile.TemporaryDirectory(prefix="gate-h-real-hc-precision-") as td:
        lib = os.path.join(td, "libhc.so")
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", "-shared", "-fPIC", "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"), "-lm", "-o", lib]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            sys.stderr.write(p.stdout + p.stderr)
            return 1
        so = ctypes.CDLL(lib)
        pre = so.waste_ds_v4_hc_pre_ref
        pre.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint,
                        ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float)]
        pre.restype = ctypes.c_int
        post = so.waste_ds_v4_hc_post_ref
        post.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                         ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        post.restype = ctypes.c_int

        def c_pre(x, fn, scale, base):
            y = (ctypes.c_float * DIM)()
            po = (ctypes.c_float * HC)()
            co = (ctypes.c_float * (HC * HC))()
            rc = pre(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base),
                     ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6),
                     y, None, None, po, co)
            if rc:
                raise AssertionError("C hc_pre rejected real state")
            return list(y), list(po), [list(co[j * HC:(j + 1) * HC]) for j in range(HC)]

        def c_post(branch, residual, po, co):
            out = (ctypes.c_float * FLAT)()
            flat = [v for row in co for v in row]
            if post(as_cf(branch), as_cf(residual), DIM, as_cf(po), as_cf(flat), out):
                raise AssertionError("C hc_post rejected real state")
            return list(out)

        c_attn_pre_f, c_apost, c_acomb = c_pre(residual, attn_fn, attn_scale, attn_base)
        c_attn_pre, _ = chain.cast_bf16(c_attn_pre_f)
        exact_bf16("C attn_pre", c_attn_pre, attn_pre_frozen)
        c_after_f = c_post(attn, residual, c_apost, c_acomb)
        c_after_bits, c_after = chain.cast_bf16(c_after_f)
        c_ffn_pre_f, c_fpost, c_fcomb = c_pre(c_after, ffn_fn, ffn_scale, ffn_base)
        c_ffn, _ = chain.cast_bf16(c_ffn_pre_f)
        exact_bf16("C ffn_pre", c_ffn, ffn_frozen)
        c_final_f = c_post(moe, c_after, c_fpost, c_fcomb)
        c_final, _ = chain.cast_bf16(c_final_f)
        exact_bf16("C final", c_final, frozen_final)

    if bits(py_after_f) != bits(c_after_f) or bits(py_final_f) != bits(c_final_f):
        print("FAIL: independent and C real HC F32 paths are not bit-identical")
        return 1

    print("PASS Gate H corrected real layer-3 HC endpoint is exact F32 in Python and C")
    print("corrected sha256:", CORRECTED_SHA)
    print("old sha256 refused:", OLD_SHA)
    print("audited BF16 diff indices:", DIFF_INDICES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
