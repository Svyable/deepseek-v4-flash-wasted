#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Compose one complete real layer-3 Gate H transition offline.

This is the first permanent replay whose endpoints are both layer states:

    real residual
      -> HC-attn-pre
      -> frozen real 64-head attention branch
      -> HC-attn-post
      -> HC-FFN-pre
      -> frozen real six-routed-plus-shared MoE branch
      -> HC-FFN-post
      -> exact BF16 [4,4096] layer output

The attention and MoE arithmetic remain separately localizable. Here their
input bindings and the two HyperConnection transitions become one contract.
Post/comb matrices are derived from the independent Python Sinkhorn equations,
not reused from the C hc_pre call, so a normalization fault cannot cancel
between pre and post and leave a self-confirming composition.
"""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))
import test_v6_ffn_route_real as chain  # noqa: E402

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
HCFIX = os.path.join(BASE, "v6_hc_composition_real")
ATTFIX = os.path.join(BASE, "v6_attention_branch_real")
MOEFIX = os.path.join(BASE, "v6_moe_branch_real")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


def main():
    for d in (HCFIX, ATTFIX, MOEFIX):
        if not os.path.isfile(os.path.join(d, "provenance.json")):
            print(f"SKIP: {os.path.basename(d)} fixture is not frozen yet")
            return 77
    hc = json.load(open(os.path.join(HCFIX, "provenance.json"), encoding="utf-8"))
    att = json.load(open(os.path.join(ATTFIX, "provenance.json"), encoding="utf-8"))
    moe = json.load(open(os.path.join(MOEFIX, "provenance.json"), encoding="utf-8"))
    for label, prov in (("hc", hc), ("attention", att), ("moe", moe)):
        if prov.get("revision") != REV:
            print(f"FAIL: {label} fixture revision drifted")
            return 1

    params, states = hc["parameters"], hc["states"]
    residual_path = os.path.join(HCFIX, states["residual"]["file"])
    attn_pre_path = os.path.join(HCFIX, states["attn_pre"]["file"])
    attn_branch_meta = att["outputs"]["branch"]
    attn_branch_path = os.path.join(ATTFIX, attn_branch_meta["file"])
    moe_dep = moe.get("input_dependency") or {}
    moe_input_path = os.path.join(MOEFIX, moe_dep.get("file", ""))
    moe_branch_meta = moe["outputs"]["moe_branch"]
    moe_branch_path = os.path.join(MOEFIX, moe_branch_meta["file"])
    final_meta = moe["outputs"]["layer3_final"]
    final_path = os.path.join(MOEFIX, final_meta["file"])

    checks = [
        (residual_path, states["residual"]["sha256"], "residual"),
        (attn_pre_path, states["attn_pre"]["sha256"], "attn_pre"),
        (attn_branch_path, attn_branch_meta["sha256"], "attention branch"),
        (moe_input_path, moe_dep.get("sha256"), "MoE ffn_pre dependency"),
        (moe_branch_path, moe_branch_meta["sha256"], "MoE branch"),
        (final_path, final_meta["sha256"], "final layer state"),
    ]
    for path, want, label in checks:
        if not os.path.isfile(path) or sha(path) != want:
            print(f"FAIL: {label} missing or SHA mismatch")
            return 1

    def f32_param(key, count):
        meta = params[key]
        path = os.path.join(HCFIX, meta["file"])
        want = meta.get("payload_sha256", meta.get("sha256"))
        if not os.path.isfile(path) or (want and sha(path) != want):
            raise ValueError(f"{key}: HC parameter missing or SHA mismatch")
        return chain.read_f32(path, count)

    residual = [chain.bf16_to_f32(v) for v in chain.read_u16(residual_path, FLAT)]
    real_attn = [chain.bf16_to_f32(v) for v in chain.read_u16(attn_branch_path, DIM)]
    real_moe = [chain.bf16_to_f32(v) for v in chain.read_u16(moe_branch_path, DIM)]
    want_attn_pre = chain.read_u16(attn_pre_path, DIM)
    want_ffn_pre = chain.read_u16(moe_input_path, DIM)
    want_final = chain.read_u16(final_path, FLAT)

    attn_fn = f32_param("attn_fn", MIX * FLAT)
    attn_base = f32_param("attn_base", MIX)
    attn_scale = f32_param("attn_scale", 3)
    ffn_fn = f32_param("ffn_fn", MIX * FLAT)
    ffn_base = f32_param("ffn_base", MIX)
    ffn_scale = f32_param("ffn_scale", 3)

    # Independent Sinkhorn parameters for both transitions. These are used for
    # hc_post even though C hc_pre also returns a post/comb, deliberately.
    o_apre, o_apost, o_acomb = chain.oracle_hc_params(
        residual, attn_fn, attn_scale, attn_base)
    if chain.cast_bf16(chain.oracle_hc_pre_y(residual, o_apre))[0] != want_attn_pre:
        print("FAIL: independent attention hc_pre no longer matches the frozen boundary")
        return 1

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77
    with tempfile.TemporaryDirectory(prefix="v6-layer3-full-") as td:
        lib = os.path.join(td, "libhc.so")
        cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"),
               "-lm", "-o", lib]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile full layer HC replay")
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

        def c_pre(x, fn, base, scale):
            y = (ctypes.c_float * DIM)()
            po = (ctypes.c_float * HC)()
            co = (ctypes.c_float * (HC * HC))()
            rc = pre(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base),
                     ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6),
                     y, None, None, po, co)
            if rc:
                raise RuntimeError("C hc_pre rejected the layer state")
            return list(y)

        def c_post(branch, residual_, post_, comb_):
            y = (ctypes.c_float * FLAT)()
            flat_comb = [v for row in comb_ for v in row]
            rc = post(as_cf(branch), as_cf(residual_), DIM,
                      as_cf(post_), as_cf(flat_comb), y)
            if rc:
                raise RuntimeError("C hc_post rejected the layer state")
            return list(y)

        c_attn_pre_bits, _ = chain.cast_bf16(c_pre(residual, attn_fn, attn_base, attn_scale))
        if c_attn_pre_bits != want_attn_pre:
            print("FAIL: C attention hc_pre does not match the attention fixture input")
            return 1

        after_f = c_post(real_attn, residual, o_apost, o_acomb)
        after_bits, after = chain.cast_bf16(after_f)
        fpre, fpost, fcomb = chain.oracle_hc_params(after, ffn_fn, ffn_scale, ffn_base)
        if chain.cast_bf16(chain.oracle_hc_pre_y(after, fpre))[0] != want_ffn_pre:
            print("FAIL: independent FFN hc_pre does not match the frozen MoE input")
            return 1
        c_ffn_pre_bits, _ = chain.cast_bf16(c_pre(after, ffn_fn, ffn_base, ffn_scale))
        if c_ffn_pre_bits != want_ffn_pre:
            print("FAIL: C FFN hc_pre does not match the frozen MoE input")
            return 1

        final_f = c_post(real_moe, after, fpost, fcomb)
        final_bits, _ = chain.cast_bf16(final_f)
        if final_bits != want_final:
            for i, (got, want) in enumerate(zip(final_bits, want_final)):
                if got != want:
                    print(f"FAIL: final layer BF16 mismatch at {i}: {got:04x} != {want:04x}")
                    break
            return 1

        # Refusal 1: a plain residual add is not HyperConnection post.
        plain = list(after)
        for k in range(HC):
            for d in range(DIM):
                plain[k * DIM + d] = chain.f32(
                    plain[k * DIM + d] + real_moe[d])
        if chain.cast_bf16(plain)[0] == want_final:
            print("FAIL: ordinary residual add is indistinguishable from final hc_post")
            return 1

        # Refusal 2: the old deterministic FFN stub cannot stand in for the
        # checkpoint MoE branch and still arrive at the real final state.
        stub_path = os.path.join(HCFIX, states["ffn_branch"]["file"])
        stub = [chain.bf16_to_f32(v) for v in chain.read_u16(stub_path, DIM)]
        stub_final = chain.cast_bf16(c_post(stub, after, fpost, fcomb))[0]
        if stub_final == want_final:
            print("FAIL: stub FFN branch reaches the real checkpoint final state")
            return 1

    print("PASS Gate H/V6 complete real layer-3 composition, exact BF16 [4,4096]")
    print("after-attn first8:", " ".join(f"0x{x:04x}" for x in after_bits[:8]))
    print("ffn-pre first8:", " ".join(f"0x{x:04x}" for x in want_ffn_pre[:8]))
    print("final first8:", " ".join(f"0x{x:04x}" for x in want_final[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
