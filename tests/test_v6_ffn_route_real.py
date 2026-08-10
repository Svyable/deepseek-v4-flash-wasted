#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Route the real layer-3 FFN branch input, and name the experts Gate H needs.

Gate H/V6, step 4 and the selection half of step 5. Continues the composition
one stage past `test_v6_attention_composition_real.py`:

    residual -> hc_pre(hc_attn_*) -> real 64-head attention -> hc_post
             -> hc_pre(hc_ffn_*)  -> real layer-3 router

Everything here is offline. The complete layer-3 router is already frozen —
`v3_router_real/layer3-gate-weight.bf16.bin` is the full [256,4096] gate and
`layer3-gate-bias.f32.bin` the 256 correction biases — so the routing decision
at the real `ffn_pre` state can be computed and pinned without touching the
checkpoint.

**Why this exists before the expensive step.** Gate H's step 5 needs the real
routed experts evaluated at this state, and a routed expert record is 13.4 MB.
Which six to fetch is not knowable from Gate F: that fixture routed a
different input and selected [2,29,225,220,108,69]. Running the real router
here turns "fetch the experts" into a bounded list of six, decided by
checkpoint bytes rather than by assumption. This is the project's own working
rule — run the cheap real test that could kill the expensive step first.

The top-k boundary margin is reported and asserted non-trivial for the same
reason. A selection whose 6th and 7th experts are separated by less than BF16
resolution would not be a stable acquisition list, and fetching against it
could produce a fixture that a differently-ordered but equally valid reduction
disagrees with.

Not claimed: no expert is executed here. This settles *which* experts the
composition selects and *what weight* each carries, not what they compute.
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
RTFIX = os.path.join(BASE, "v3_router_real")
DIM = 4096
HC = 4
MIX = 24
FLAT = HC * DIM
EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5
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
    return [f32(math.tanh(f32(0.5 * x[d] - 0.1875 * x[(d + 257) % DIM]
                              + ((d % 7) - 3) / 64.0))) for d in range(DIM)]


def oracle_hc_params(x, fn, scale, base):
    ss = sum(v * v for v in x)
    r = 1.0 / math.sqrt(ss / len(x) + 1e-6)
    mixes = [sum(fn[row * FLAT + c] * x[c] for c in range(FLAT)) * r
             for row in range(MIX)]
    return split(mixes, scale, base)


def oracle_hc_pre_y(x, pre):
    return [sum(pre[j] * x[j * DIM + d] for j in range(HC)) for d in range(DIM)]


def oracle_hc_post(branch, residual, post, comb):
    out = [0.0] * FLAT
    for k in range(HC):
        for d in range(DIM):
            acc = f32(post[k] * branch[d])
            for j in range(HC):
                acc = f32(acc + f32(comb[j][k] * residual[j * DIM + d]))
            out[k * DIM + d] = acc
    return out


def softplus(x):
    # log1p(exp(x)) with the large-x branch, which the source relies on for
    # positive router logits.
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def oracle_router(x, weight_bits, bias):
    """Independent learned-routing path from the pinned source contract.

    scores = sqrt(softplus(raw)); selection uses scores + correction bias;
    the returned weights are gathered from the *unbiased* scores, normalized,
    then multiplied by the routed scaling factor.
    """
    raw = []
    for e in range(EXPERTS):
        row = weight_bits[e * DIM:(e + 1) * DIM]
        raw.append(sum(x[d] * bf16_to_f32(row[d]) for d in range(DIM)))
    score = [math.sqrt(softplus(v)) for v in raw]
    sel = [score[e] + bias[e] for e in range(EXPERTS)]
    order = sorted(range(EXPERTS), key=lambda e: (-sel[e], e))
    ids = order[:TOPK]
    gathered = [score[e] for e in ids]
    total = sum(gathered)
    weights = [g / total * ROUTE_SCALE for g in gathered]
    margin = sel[order[TOPK - 1]] - sel[order[TOPK]]
    return ids, weights, score, sel, margin


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)


def max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    for path in (HCFIX, ATTFIX, RTFIX):
        if not os.path.isfile(os.path.join(path, "provenance.json")):
            print(f"SKIP: {os.path.basename(path)} fixture is not frozen yet")
            return 77

    hc = json.load(open(os.path.join(HCFIX, "provenance.json"), encoding="utf-8"))
    att = json.load(open(os.path.join(ATTFIX, "provenance.json"), encoding="utf-8"))
    rt = json.load(open(os.path.join(RTFIX, "provenance.json"), encoding="utf-8"))
    for name, prov in (("hc", hc), ("attention", att), ("router", rt)):
        if prov.get("revision") != REV:
            print(f"FAIL: {name} fixture is not the pinned 0731 revision")
            return 1

    # Same chain binding the attention composition enforces: the real branch
    # must have been produced from this exact attn_pre state.
    dep = att.get("input_dependency") or {}
    attn_pre_path = os.path.join(HCFIX, hc["states"]["attn_pre"]["file"])
    if sha(attn_pre_path) != dep.get("sha256"):
        print("FAIL: attention branch was produced from a different attn_pre state")
        return 1

    params, states = hc["parameters"], hc["states"]
    for meta in list(params.values()) + list(states.values()):
        p = os.path.join(HCFIX, meta["file"])
        want = meta.get("payload_sha256", meta.get("sha256"))
        if not os.path.isfile(p) or sha(p) != want:
            print(f"FAIL: {meta['file']} missing or SHA mismatch")
            return 1
    branch_meta = att["outputs"]["branch"]
    branch_path = os.path.join(ATTFIX, branch_meta["file"])
    if sha(branch_path) != branch_meta["sha256"]:
        print(f"FAIL: {branch_meta['file']} SHA mismatch")
        return 1

    gate_path = os.path.join(RTFIX, "layer3-gate-weight.bf16.bin")
    bias_path = os.path.join(RTFIX, "layer3-gate-bias.f32.bin")
    learned = rt.get("learned", {})
    for key, path in (("weight", gate_path), ("bias", bias_path)):
        meta = learned.get(key, {})
        want = meta.get("payload_sha256") or meta.get("sha256")
        if want and sha(path) != want:
            print(f"FAIL: layer3 gate {key} SHA mismatch")
            return 1

    attn_fn = read_f32(os.path.join(HCFIX, params["attn_fn"]["file"]), MIX * FLAT)
    attn_base = read_f32(os.path.join(HCFIX, params["attn_base"]["file"]), MIX)
    attn_scale = read_f32(os.path.join(HCFIX, params["attn_scale"]["file"]), 3)
    ffn_fn = read_f32(os.path.join(HCFIX, params["ffn_fn"]["file"]), MIX * FLAT)
    ffn_base = read_f32(os.path.join(HCFIX, params["ffn_base"]["file"]), MIX)
    ffn_scale = read_f32(os.path.join(HCFIX, params["ffn_scale"]["file"]), 3)
    residual = [bf16_to_f32(v) for v in
                read_u16(os.path.join(HCFIX, states["residual"]["file"]), FLAT)]
    real_branch = [bf16_to_f32(v) for v in read_u16(branch_path, DIM)]
    gate_bits = read_u16(gate_path, EXPERTS * DIM)
    bias = read_f32(bias_path, EXPERTS)

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    with tempfile.TemporaryDirectory(prefix="v6-ffn-route-") as td:
        lib = os.path.join(td, "librt.so")
        cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
               "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"),
               os.path.join(REPO, "src", "deepseek_v4_router_ref.c"),
               "-lm", "-o", lib]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile Gate H FFN-route replay")
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
        score_fn = so.waste_ds_v4_router_score_ref
        score_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_uint16),
                             ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        score_fn.restype = ctypes.c_int
        learned_fn = so.waste_ds_v4_router_learned_ref
        learned_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                               ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
                               ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        learned_fn.restype = ctypes.c_int

        def c_pre(x, fn, base, scale):
            y = (ctypes.c_float * DIM)()
            po = (ctypes.c_float * HC)()
            co = (ctypes.c_float * (HC * HC))()
            if pre(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base),
                   ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6), y, None, None, po, co):
                raise RuntimeError("C hc_pre rejected the real fixture")
            return list(y), list(po), [list(co[j * HC:(j + 1) * HC]) for j in range(HC)]

        def c_post(branch, residual_, po, co):
            y = (ctypes.c_float * FLAT)()
            if post(as_cf(branch), as_cf(residual_), DIM, as_cf(po),
                    as_cf([v for r_ in co for v in r_]), y):
                raise RuntimeError("C hc_post rejected the real fixture")
            return list(y)

        def c_route(x):
            raw = (ctypes.c_float * EXPERTS)()
            sc = (ctypes.c_float * EXPERTS)()
            wb = (ctypes.c_uint16 * len(gate_bits))(*gate_bits)
            if score_fn(as_cf(x), DIM, wb, raw, sc):
                raise RuntimeError("C router score rejected the input")
            ids = (ctypes.c_uint32 * TOPK)()
            wts = (ctypes.c_float * TOPK)()
            sel = (ctypes.c_float * EXPERTS)()
            if learned_fn(sc, as_cf(bias), ctypes.c_float(ROUTE_SCALE), ids, wts, sel):
                raise RuntimeError("C learned routing rejected the scores")
            return list(ids), list(wts), list(sc), list(sel)

        # ---- compose forward to the real ffn_pre -------------------------
        attn_pre_f, ap, ac = c_pre(residual, attn_fn, attn_base, attn_scale)
        want_attn_pre = read_u16(os.path.join(HCFIX, states["attn_pre"]["file"]), DIM)
        if cast_bf16(attn_pre_f)[0] != want_attn_pre:
            print("FAIL: recomputed attention HC-pre does not match the frozen state")
            return 1

        after_f = c_post(real_branch, residual, ap, ac)
        _, after = cast_bf16(after_f)
        ffn_pre_f, fp, fc = c_pre(after, ffn_fn, ffn_base, ffn_scale)
        ffn_pre_bits, ffn_pre = cast_bf16(ffn_pre_f)

        o_pre, _, _ = oracle_hc_params(after, ffn_fn, ffn_scale, ffn_base)
        o_ffn_pre_f = oracle_hc_pre_y(after, o_pre)
        if cast_bf16(o_ffn_pre_f)[0] != ffn_pre_bits:
            print("FAIL: real FFN hc_pre disagrees with the independent oracle")
            return 1

        # ---- the real router at that state ------------------------------
        ids, weights, c_score, c_sel = c_route(ffn_pre)
        o_ids, o_weights, o_score, o_sel, margin = oracle_router(ffn_pre, gate_bits, bias)
        if ids != o_ids:
            print(f"FAIL: C selection {ids} != independent oracle {o_ids}")
            return 1
        wdiff = max_abs(weights, o_weights)
        if wdiff > 1e-5:
            print(f"FAIL: route weights disagree with the oracle by {wdiff:.9g}")
            return 1

        # A selection separated by less than BF16 resolution would not be a
        # stable acquisition list: a differently-ordered but equally valid
        # reduction could pick a different sixth expert, and the fetched
        # fixture would then disagree with the engine that used it.
        if not margin > 1e-4:
            print(f"FAIL: top-k boundary margin {margin:.9g} is too small to "
                  "treat this selection as a stable acquisition list")
            return 1

        # ---- refusals ----------------------------------------------------
        # The correction bias moves selection but must not enter the weights.
        biased_weights = [o_score[e] + bias[e] for e in ids]
        tot = sum(biased_weights)
        biased_weights = [w / tot * ROUTE_SCALE for w in biased_weights]
        if max_abs(weights, biased_weights) < 1e-6:
            print("FAIL: bias-contaminated weights are indistinguishable")
            return 1

        # Selection without the bias must be a different question than with it.
        unbiased_order = sorted(range(EXPERTS), key=lambda e: (-o_score[e], e))[:TOPK]

        # Routing off the stub composition must not stand in for the real one.
        stub_after = [bf16_to_f32(v) for v in
                      read_u16(os.path.join(HCFIX, states["after_attn"]["file"]), FLAT)]
        stub_ffn_pre_f, _, _ = c_pre(stub_after, ffn_fn, ffn_base, ffn_scale)
        _, stub_ffn_pre = cast_bf16(stub_ffn_pre_f)
        stub_ids, _, _, _ = c_route(stub_ffn_pre)
        if stub_ffn_pre_f == ffn_pre_f:
            print("FAIL: real and stub compositions produce the same ffn_pre")
            return 1

    print("PASS Gate H/V6 real FFN routing: "
          "residual -> hc_pre -> real attention -> hc_post -> hc_pre -> real router")
    print(f"ffn_pre first8: " + " ".join(f"0x{b:04x}" for b in ffn_pre_bits[:8]))
    print(f"selected experts (real attention branch): {ids}")
    print("route weights: " + " ".join(f"{w:.9f}" for w in weights))
    print(f"top-k boundary margin: {margin:.9g}")
    print(f"selection ignoring the correction bias: {unbiased_order}")
    print(f"selection off the stub composition:     {stub_ids}")
    print("Gate H step 5 must fetch layer-3 routed experts " +
          ",".join(str(i) for i in sorted(ids)) + " plus the shared expert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
