#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free Gate H/V6 composition contract for one DeepSeek V4 block.

This intentionally does not re-prove Attention or MoE arithmetic.  It proves the
load-bearing block wiring around independently produced branch values:

    HC-attn pre -> attention branch -> HC-attn post
      -> HC-FFN pre -> MoE branch -> HC-FFN post

The tiny deterministic branch functions stand in for already-independent
Attention/MoE implementations.  Mutations make it impossible to pass by
feeding either branch from the wrong HyperConnection state.
"""

import ctypes
import math
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 8
HC = 4
MIX = 24
FLAT = HC * DIM


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


def sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def split(mixes, scale, base, iters=20, eps=1e-6):
    pre = [sigmoid(mixes[j] * scale[0] + base[j]) + eps for j in range(HC)]
    post = [2.0 * sigmoid(mixes[j + HC] * scale[1] + base[j + HC]) for j in range(HC)]
    comb = [[0.0] * HC for _ in range(HC)]
    for j in range(HC):
        logits = [mixes[2 * HC + j * HC + k] * scale[2] + base[2 * HC + j * HC + k]
                  for k in range(HC)]
        m = max(logits)
        ex = [math.exp(v - m) for v in logits]
        s = sum(ex)
        comb[j] = [v / s + eps for v in ex]
    for k in range(HC):
        s = sum(comb[j][k] for j in range(HC)) + eps
        for j in range(HC):
            comb[j][k] /= s
    for _ in range(1, iters):
        for j in range(HC):
            s = sum(comb[j]) + eps
            comb[j] = [v / s for v in comb[j]]
        for k in range(HC):
            s = sum(comb[j][k] for j in range(HC)) + eps
            for j in range(HC):
                comb[j][k] /= s
    return pre, post, comb


def hc_pre(x, fn, scale, base):
    ss = sum(v * v for v in x)
    r = 1.0 / math.sqrt(ss / len(x) + 1e-6)
    mixes = []
    for row in range(MIX):
        off = row * FLAT
        mixes.append(sum(fn[off + c] * x[c] for c in range(FLAT)) * r)
    pre, post, comb = split(mixes, scale, base)
    y = [sum(pre[j] * x[j * DIM + d] for j in range(HC)) for d in range(DIM)]
    return y, post, comb


def hc_post(branch, residual, post, comb):
    out = [0.0] * FLAT
    for k in range(HC):
        for d in range(DIM):
            out[k * DIM + d] = post[k] * branch[d] + sum(
                comb[j][k] * residual[j * DIM + d] for j in range(HC))
    return out


def attention_stub(y):
    # Nonlinear, coordinate-coupled branch so a wrong HC-pre input is visible.
    return [math.tanh(0.45 * y[d] - 0.17 * y[(d + 3) % DIM] + 0.03 * (d + 1))
            for d in range(DIM)]


def moe_stub(y):
    return [math.sin(0.31 * y[d] + 0.11 * y[(d + 5) % DIM]) + 0.07 * y[d] * y[d]
            for d in range(DIM)]


def params(seed):
    fn = [(((i * (17 + seed) + 13 * seed) % 101) - 50) / 700.0 for i in range(MIX * FLAT)]
    base = [(((i * (7 + seed) + seed) % 31) - 15) / 23.0 for i in range(MIX)]
    scale = [0.73 + seed * 0.03, 1.11 - seed * 0.02, 0.57 + seed * 0.04]
    return fn, base, scale


def max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def as_c_float(values):
    return (ctypes.c_float * len(values))(*values)


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    residual = [(((i * 29 + 7) % 73) - 36) / 19.0 for i in range(FLAT)]
    attn_fn, attn_base, attn_scale = params(1)
    ffn_fn, ffn_base, ffn_scale = params(4)

    # Independent mathematical oracle.
    attn_in, attn_post, attn_comb = hc_pre(residual, attn_fn, attn_scale, attn_base)
    attn_branch = attention_stub(attn_in)
    after_attn = hc_post(attn_branch, residual, attn_post, attn_comb)
    ffn_in, ffn_post, ffn_comb = hc_pre(after_attn, ffn_fn, ffn_scale, ffn_base)
    ffn_branch = moe_stub(ffn_in)
    expected = hc_post(ffn_branch, after_attn, ffn_post, ffn_comb)

    # Required wiring mutations must be numerically load-bearing before C is
    # even involved.  This guards against choosing a vacuous construction.
    wrong_ffn_in, wp, wc = hc_pre(residual, ffn_fn, ffn_scale, ffn_base)
    wrong_original = hc_post(moe_stub(wrong_ffn_in), residual, wp, wc)
    if max_abs(expected, wrong_original) < 1e-3:
        raise SystemExit("FAIL: original-residual FFN mutation is not visible")

    sw_attn_in, sw_post, sw_comb = hc_pre(residual, ffn_fn, ffn_scale, ffn_base)
    sw_after = hc_post(attention_stub(sw_attn_in), residual, sw_post, sw_comb)
    sw_ffn_in, swp, swc = hc_pre(sw_after, attn_fn, attn_scale, attn_base)
    swapped = hc_post(moe_stub(sw_ffn_in), sw_after, swp, swc)
    if max_abs(expected, swapped) < 1e-3:
        raise SystemExit("FAIL: swapped HC parameter mutation is not visible")

    wrong_branch = hc_post(moe_stub(attn_in), after_attn, ffn_post, ffn_comb)
    if max_abs(expected, wrong_branch) < 1e-3:
        raise SystemExit("FAIL: wrong branch-input mutation is not visible")

    # Dropping either hc_post transition. The natural wrong implementation is
    # not "omit the step" — the shapes would not line up and it would fail
    # loudly — it is to write the ordinary residual add that every other
    # transformer uses, `out[k] = residual[k] + branch`, discarding the learned
    # post/comb mix. That compiles, runs, and produces plausible numbers, which
    # is exactly the class of error this gate exists to catch.
    def plain_residual_add(branch, residual_):
        return [residual_[k * DIM + d] + branch[d]
                for k in range(HC) for d in range(DIM)]

    dropped_final = plain_residual_add(ffn_branch, after_attn)
    if max_abs(expected, dropped_final) < 1e-3:
        raise SystemExit("FAIL: dropped FFN hc_post transition is not visible")

    # The attention transition, dropped the same way, then the rest of the
    # layer carried out correctly from that wrong state.
    da_after = plain_residual_add(attn_branch, residual)
    da_in, da_post, da_comb = hc_pre(da_after, ffn_fn, ffn_scale, ffn_base)
    dropped_attn = hc_post(moe_stub(da_in), da_after, da_post, da_comb)
    if max_abs(expected, dropped_attn) < 1e-3:
        raise SystemExit("FAIL: dropped attention hc_post transition is not visible")

    with tempfile.TemporaryDirectory(prefix="v6-layer-compose-") as td:
        lib = os.path.join(td, "libhc.so")
        cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"), "-lm", "-o", lib]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile Gate H composition probe")
            return 1
        so = ctypes.CDLL(lib)
        pre_fn = so.waste_ds_v4_hc_pre_ref
        pre_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint,
                           ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float)]
        pre_fn.restype = ctypes.c_int
        post_fn = so.waste_ds_v4_hc_post_ref
        post_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        post_fn.restype = ctypes.c_int

        def c_pre(x, fn, base, scale):
            y = (ctypes.c_float * DIM)()
            post = (ctypes.c_float * HC)()
            comb = (ctypes.c_float * (HC * HC))()
            rc = pre_fn(as_c_float(x), DIM, as_c_float(fn), as_c_float(scale), as_c_float(base),
                        ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6), y, None, None, post, comb)
            if rc:
                raise RuntimeError("C hc_pre rejected composition input")
            return list(y), list(post), [list(comb[j * HC:(j + 1) * HC]) for j in range(HC)]

        def c_post(branch, residual_, post, comb):
            out = (ctypes.c_float * FLAT)()
            flat_comb = [v for row in comb for v in row]
            rc = post_fn(as_c_float(branch), as_c_float(residual_), DIM,
                         as_c_float(post), as_c_float(flat_comb), out)
            if rc:
                raise RuntimeError("C hc_post rejected composition input")
            return list(out)

        c_attn_in, c_ap, c_ac = c_pre(residual, attn_fn, attn_base, attn_scale)
        c_after = c_post(attention_stub(c_attn_in), residual, c_ap, c_ac)
        c_ffn_in, c_fp, c_fc = c_pre(c_after, ffn_fn, ffn_base, ffn_scale)
        got = c_post(moe_stub(c_ffn_in), c_after, c_fp, c_fc)

    err = max_abs(expected, got)
    if err > 2e-5:
        print(f"FAIL: C two-HC composition max_abs {err:.9g}")
        return 1
    print("PASS Gate H model-free block wiring: HC-attn -> branch -> HC-FFN -> branch; mutations observable")
    print(f"final max_abs vs independent oracle: {err:.9g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
