#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Compose the real layer-3 attention branch through the real attention HC transition.

Gate H/V6, step 3 of the layer method: ``hc_post`` applied to the **real**
64-head attention branch instead of to a deterministic stub.

Two fixtures already exist and each is real on its own side:

  v6_hc_composition_real   real layer-3 hc_attn_* / hc_ffn_* parameters, but
                           both branches are stubs
  v6_attention_branch_real real 64-head attention from checkpoint wq/wkv/wo
                           bytes, but only as an isolated branch

Composing them is only legitimate because the second was computed **from the
first's** ``attn_pre`` state. That is not an assumption here: the attention
fixture records the producing file and its SHA-256, and this test recomputes
that digest and refuses to proceed if it does not match. Feeding an
independently correct branch output produced from the wrong branch input is a
wiring error the Gate H contract explicitly names, and hash equality is what
rules it out.

What this adds over the two fixtures separately: the attention half of the
layer is now real end to end — residual -> hc_pre -> real attention -> hc_post
— rather than real in two disconnected pieces.

What it does not add: no new checkpoint evidence about attention itself, which
`v6_attention_branch_real` already owns, and nothing about the FFN half, whose
branch remains a stub until the real router/MoE is composed at the ffn_pre
state.
"""

import ctypes
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))
from test_v6_layer_composition_scalar import split  # noqa: E402

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
HCFIX = os.path.join(BASE, "v6_hc_composition_real")
ATTFIX = os.path.join(BASE, "v6_attention_branch_real")
DIM = 4096
HC = 4
MIX = 24
FLAT = HC * DIM
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


def f32(v):
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", f32(v)))[0]


def bits_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def bf16_bits(v):
    u = f32_bits(v)
    exp, frac = u & 0x7F800000, u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16_to_f32(u):
    return bits_f32((u & 0xFFFF) << 16)


def cast_bf16(values):
    bits = [bf16_bits(v) for v in values]
    return bits, [bf16_to_f32(b) for b in bits]


def read_u16(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 2:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*2}")
    return list(struct.unpack(f"<{count}H", data))


def read_f32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*4}")
    return list(struct.unpack(f"<{count}f", data))


def attn_stub(x):
    """The stub v6_hc_composition_real froze, kept here only as a refusal."""
    return [f32(math.tanh(f32(0.5 * x[d] - 0.1875 * x[(d + 257) % DIM]
                              + ((d % 7) - 3) / 64.0))) for d in range(DIM)]


def oracle_hc_params(x, fn, scale, base):
    """Independent pre/post/comb: RMS-scaled mixes, then the Sinkhorn split.

    `split` is imported rather than transcribed a second time. It is the
    dimension-independent Python Sinkhorn from the model-free composition
    test, already agreeing with the C at 1.54e-07 there, and it is not the
    implementation under test — the C is. Deriving post/comb here rather than
    reading them out of the C's hc_pre is what lets this test see a fault
    inside the Sinkhorn normalization; taking them from hc_pre would make any
    such fault cancel on both sides of the comparison.
    """
    ss = sum(v * v for v in x)
    r = 1.0 / math.sqrt(ss / len(x) + 1e-6)
    mixes = [sum(fn[row * FLAT + c] * x[c] for c in range(FLAT)) * r
             for row in range(MIX)]
    return split(mixes, scale, base)


def oracle_hc_post(branch, residual, post, comb):
    """Independent hc_post: out[k,d] = post[k]*branch[d] + sum_j comb[j,k]*residual[j,d].

    Written from the source contract, not from src/deepseek_v4_mhc_ref.c, so
    the C path under test is compared against a separate implementation.
    """
    out = [0.0] * FLAT
    for k in range(HC):
        for d in range(DIM):
            acc = f32(post[k] * branch[d])
            for j in range(HC):
                acc = f32(acc + f32(comb[j][k] * residual[j * DIM + d]))
            out[k * DIM + d] = acc
    return out


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)


def max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    for path in (HCFIX, ATTFIX):
        if not os.path.isfile(os.path.join(path, "provenance.json")):
            print(f"SKIP: {os.path.basename(path)} fixture is not frozen yet")
            return 77

    hc = json.load(open(os.path.join(HCFIX, "provenance.json"), encoding="utf-8"))
    att = json.load(open(os.path.join(ATTFIX, "provenance.json"), encoding="utf-8"))
    for name, prov in (("hc-composition", hc), ("attention-branch", att)):
        if prov.get("revision") != REV or prov.get("layer") != 3:
            print(f"FAIL: {name} provenance is not canonical layer-3 0731")
            return 1

    # ---- the binding that makes composition legitimate -------------------
    dep = att.get("input_dependency") or {}
    dep_file = dep.get("file", "")
    want_dep_sha = dep.get("sha256", "")
    attn_pre_path = os.path.join(HCFIX, hc["states"]["attn_pre"]["file"])
    if os.path.normpath(os.path.join(REPO, dep_file)) != os.path.normpath(attn_pre_path):
        print("FAIL: attention branch does not declare the HC attn_pre state as its input")
        print(f"  declares: {dep_file}")
        return 1
    got_dep_sha = sha(attn_pre_path)
    if got_dep_sha != want_dep_sha:
        print("FAIL: attention branch was produced from a different attn_pre state")
        print(f"  declared {want_dep_sha}\n  actual   {got_dep_sha}")
        return 1

    # ---- every consumed byte is hash-pinned ------------------------------
    params = hc["parameters"]
    states = hc["states"]
    # Checkpoint parameters carry the payload digest of the tensor slice they
    # came from; derived states carry a plain file digest.
    for meta in list(params.values()) + list(states.values()):
        p = os.path.join(HCFIX, meta["file"])
        want = meta.get("payload_sha256", meta.get("sha256"))
        if want is None:
            print(f"FAIL: {meta['file']} has no recorded digest")
            return 1
        if not os.path.isfile(p) or sha(p) != want:
            print(f"FAIL: {meta['file']} missing or SHA mismatch")
            return 1
    branch_meta = att["outputs"]["branch"]
    branch_path = os.path.join(ATTFIX, branch_meta["file"])
    if not os.path.isfile(branch_path) or sha(branch_path) != branch_meta["sha256"]:
        print(f"FAIL: {branch_meta['file']} missing or SHA mismatch")
        return 1

    attn_fn = read_f32(os.path.join(HCFIX, params["attn_fn"]["file"]), MIX * FLAT)
    attn_base = read_f32(os.path.join(HCFIX, params["attn_base"]["file"]), MIX)
    attn_scale = read_f32(os.path.join(HCFIX, params["attn_scale"]["file"]), 3)
    ffn_fn = read_f32(os.path.join(HCFIX, params["ffn_fn"]["file"]), MIX * FLAT)
    ffn_base = read_f32(os.path.join(HCFIX, params["ffn_base"]["file"]), MIX)
    ffn_scale = read_f32(os.path.join(HCFIX, params["ffn_scale"]["file"]), 3)

    residual = [bf16_to_f32(v) for v in
                read_u16(os.path.join(HCFIX, states["residual"]["file"]), FLAT)]
    want_attn_pre = read_u16(os.path.join(HCFIX, states["attn_pre"]["file"]), DIM)
    stub_after = read_u16(os.path.join(HCFIX, states["after_attn"]["file"]), FLAT)

    real_branch_bits = read_u16(branch_path, DIM)
    real_branch = [bf16_to_f32(v) for v in real_branch_bits]
    if [f"0x{b:04x}" for b in real_branch_bits[:8]] != branch_meta["first8_hex"]:
        print("FAIL: real attention branch first8 disagrees with its provenance")
        return 1

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    with tempfile.TemporaryDirectory(prefix="v6-attn-compose-") as td:
        lib = os.path.join(td, "libhc.so")
        cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"), "-lm", "-o", lib]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile Gate H attention-composition replay")
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
                     ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6), y, None, None, po, co)
            if rc:
                raise RuntimeError("C hc_pre rejected the real fixture")
            return list(y), list(po), [list(co[j * HC:(j + 1) * HC]) for j in range(HC)]

        def c_post(branch, residual_, po, co):
            y = (ctypes.c_float * FLAT)()
            rc = post(as_cf(branch), as_cf(residual_), DIM, as_cf(po),
                      as_cf([v for row in co for v in row]), y)
            if rc:
                raise RuntimeError("C hc_post rejected the real fixture")
            return list(y)

        # The chain is re-derived rather than trusted: hc_pre of the frozen
        # residual must reproduce the exact attn_pre state the attention
        # branch was built from, or the hash binding above is describing a
        # state this code cannot actually reach.
        attn_pre_f, ap, ac = c_pre(residual, attn_fn, attn_base, attn_scale)
        got_attn_pre, _ = cast_bf16(attn_pre_f)
        if got_attn_pre != want_attn_pre:
            print("FAIL: recomputed attention HC-pre does not match the frozen state")
            return 1

        # post/comb derived independently, so a fault inside the Sinkhorn
        # normalization cannot cancel on both sides of the comparison.
        _, o_post, o_comb = oracle_hc_params(residual, attn_fn, attn_scale, attn_base)
        param_diag = max(max_abs(ap, o_post),
                         max_abs([v for r_ in ac for v in r_],
                                 [v for r_ in o_comb for v in r_]))
        if param_diag > 1e-5:
            print(f"FAIL: C hc_pre post/comb disagree with the independent "
                  f"Sinkhorn by {param_diag:.9g}")
            return 1

        got_f = c_post(real_branch, residual, ap, ac)
        got_bits, _ = cast_bf16(got_f)

        want_f = oracle_hc_post(real_branch, residual, o_post, o_comb)
        want_bits, _ = cast_bf16(want_f)
        if got_bits != want_bits:
            bad = next(i for i, (g, w) in enumerate(zip(got_bits, want_bits)) if g != w)
            print(f"FAIL: real attention composition BF16 mismatch at {bad}: "
                  f"0x{got_bits[bad]:04x} != 0x{want_bits[bad]:04x}")
            return 1
        diag = max_abs(got_f, want_f)

        # ---- refusals ----------------------------------------------------
        # Each is a wiring that produces a finite, plausible state.
        _, fp, fc = c_pre(residual, ffn_fn, ffn_base, ffn_scale)
        _, attn_pre_b = cast_bf16(attn_pre_f)
        refusals = {
            "stub branch instead of the real attention branch":
                c_post(cast_bf16(attn_stub(attn_pre_b))[1], residual, ap, ac),
            "FFN HC post/comb parameters on the attention transition":
                c_post(real_branch, residual, fp, fc),
            "plain residual add instead of hc_post":
                [residual[k * DIM + d] + real_branch[d]
                 for k in range(HC) for d in range(DIM)],
        }
        for label, wrong in refusals.items():
            if cast_bf16(wrong)[0] == got_bits:
                print(f"FAIL: refusal not observable — {label}")
                return 1

        # The real branch must actually move the state away from the stub
        # composition, or this test would be re-asserting the old fixture.
        if got_bits == stub_after:
            print("FAIL: real-branch composition equals the frozen stub composition")
            return 1

    print("PASS Gate H/V6 real attention composition: "
          "residual -> hc_pre -> real 64-head attention -> hc_post, exact BF16")
    print(f"hc_post f32 max_abs vs independent oracle: {diag:.9g}")
    print(f"hc_pre post/comb max_abs vs independent Sinkhorn: {param_diag:.9g}")
    print("after-attn first8: " + " ".join(f"0x{b:04x}" for b in got_bits[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
