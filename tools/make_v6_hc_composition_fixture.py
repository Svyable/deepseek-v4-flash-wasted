#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build Gate H/V6 real layer-3 two-HyperConnection composition fixture.

This fixture isolates the new full-block wiring from Attention/MoE arithmetic.
It uses both official layer-3 HC parameter sets and exact BF16 block boundaries,
while deterministic nonlinear branch functions make the branch-input binding
load-bearing.  Later Gate-H fixtures replace those stubs with real Attention and
MoE outputs from the exact same HC-pre states.
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys

import fetch_hf_tensor_slice as slice_fetch

HC = 4
DIM = 4096
FLAT = HC * DIM
MIX = 24
ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6

PARAMS = {
    "attn_fn": "layers.3.hc_attn_fn",
    "attn_base": "layers.3.hc_attn_base",
    "attn_scale": "layers.3.hc_attn_scale",
    "ffn_fn": "layers.3.hc_ffn_fn",
    "ffn_base": "layers.3.hc_ffn_base",
    "ffn_scale": "layers.3.hc_ffn_scale",
}

# Sparse dyadic state: all four HC lanes participate, and values are exactly
# representable in BF16 so acquisition does not invent a reduction-order issue.
RESIDUAL_SUPPORT = (
    (0, 0, 1.0), (0, 511, -0.25), (0, 2048, 0.75),
    (1, 17, -0.5), (1, 1023, 0.125), (1, 3333, -1.25),
    (2, 7, 0.25), (2, 999, 1.5), (2, 3071, -0.375),
    (3, 31, 0.625), (3, 2047, -0.75), (3, 4095, -2.0),
)


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
    return bits, [bf16_to_f32(v) for v in bits]


def sigmoid(v):
    v = f32(v)
    if v >= 0:
        z = f32(math.exp(-v))
        return f32(1.0 / f32(1.0 + z))
    z = f32(math.exp(v))
    return f32(z / f32(1.0 + z))


def read_f32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*4}")
    return list(struct.unpack(f"<{count}f", data))


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[f32(v) for v in values]))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sinkhorn(mixes, scale, base):
    pre = [f32(sigmoid(f32(f32(mixes[j] * scale[0]) + base[j])) + HC_EPS)
           for j in range(HC)]
    post = [f32(2.0 * sigmoid(f32(f32(mixes[j + HC] * scale[1]) + base[j + HC])))
            for j in range(HC)]
    comb = [[0.0] * HC for _ in range(HC)]
    for j in range(HC):
        logits = [f32(f32(mixes[2*HC + j*HC + k] * scale[2])
                      + base[2*HC + j*HC + k]) for k in range(HC)]
        m = max(logits)
        ex = [f32(math.exp(f32(v - m))) for v in logits]
        s = f32(0.0)
        for v in ex:
            s = f32(s + v)
        for k in range(HC):
            comb[j][k] = f32(f32(ex[k] / s) + HC_EPS)
    for k in range(HC):
        s = f32(0.0)
        for j in range(HC):
            s = f32(s + comb[j][k])
        d = f32(s + HC_EPS)
        for j in range(HC):
            comb[j][k] = f32(comb[j][k] / d)
    for _ in range(1, ITERS):
        for j in range(HC):
            s = f32(0.0)
            for k in range(HC):
                s = f32(s + comb[j][k])
            d = f32(s + HC_EPS)
            for k in range(HC):
                comb[j][k] = f32(comb[j][k] / d)
        for k in range(HC):
            s = f32(0.0)
            for j in range(HC):
                s = f32(s + comb[j][k])
            d = f32(s + HC_EPS)
            for j in range(HC):
                comb[j][k] = f32(comb[j][k] / d)
    return pre, post, [comb[j][k] for j in range(HC) for k in range(HC)]


def hc_pre(residual, fn, base, scale):
    # Keep the source's full reduction order, but sparse input means only a few
    # products are nonzero and all are reproducible across loop directions.
    ss = f32(0.0)
    for x in residual:
        ss = f32(ss + f32(x * x))
    rsqrt = f32(1.0 / math.sqrt(f32(f32(ss / FLAT) + NORM_EPS)))
    mixes = []
    for r in range(MIX):
        acc = f32(0.0)
        off = r * FLAT
        for c in range(FLAT):
            acc = f32(acc + f32(fn[off + c] * residual[c]))
        mixes.append(f32(acc * rsqrt))
    pre, post, comb = sinkhorn(mixes, scale, base)
    y = []
    for d in range(DIM):
        acc = f32(0.0)
        for j in range(HC):
            acc = f32(acc + f32(pre[j] * residual[j * DIM + d]))
        y.append(acc)
    return y, mixes, pre, post, comb


def hc_post(branch, residual, post, comb):
    y = [0.0] * FLAT
    for k in range(HC):
        for d in range(DIM):
            acc = f32(post[k] * branch[d])
            for j in range(HC):
                acc = f32(acc + f32(comb[j * HC + k] * residual[j * DIM + d]))
            y[k * DIM + d] = acc
    return y


def attn_stub(x):
    # Cheap but input-sensitive; every output coordinate depends on two HC-pre
    # coordinates.  This is deliberately not an Attention approximation.
    return [f32(math.tanh(f32(0.5 * x[d] - 0.1875 * x[(d + 257) % DIM]
                                 + ((d % 7) - 3) / 64.0))) for d in range(DIM)]


def ffn_stub(x):
    return [f32(math.sin(f32(0.3125 * x[d] + 0.125 * x[(d + 911) % DIM]))
                + f32(0.0625 * x[d] * x[d])) for d in range(DIM)]


def make_residual():
    bits = [0] * FLAT
    for lane, d, v in RESIDUAL_SUPPORT:
        bits[lane * DIM + d] = bf16_bits(v)
    return bits, [bf16_to_f32(v) for v in bits]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    info = {}
    paths = {}
    for key, tensor in PARAMS.items():
        path = os.path.join(args.out_dir, key.replace("_", "-") + ".f32.bin")
        info[key] = slice_fetch.fetch_slice(args.snapshot, tensor, None, path, token=args.token)
        paths[key] = path

    for key in ("attn_fn", "ffn_fn"):
        if info[key]["dtype"] != "F32" or info[key]["shape"] != [MIX, FLAT]:
            raise ValueError(f"{PARAMS[key]} metadata {info[key]['dtype']} {info[key]['shape']}")
    for key in ("attn_base", "ffn_base"):
        if info[key]["dtype"] != "F32" or info[key]["shape"] != [MIX]:
            raise ValueError(f"{PARAMS[key]} metadata {info[key]['dtype']} {info[key]['shape']}")
    for key in ("attn_scale", "ffn_scale"):
        if info[key]["dtype"] != "F32" or info[key]["shape"] != [3]:
            raise ValueError(f"{PARAMS[key]} metadata {info[key]['dtype']} {info[key]['shape']}")

    attn_fn = read_f32(paths["attn_fn"], MIX * FLAT)
    attn_base = read_f32(paths["attn_base"], MIX)
    attn_scale = read_f32(paths["attn_scale"], 3)
    ffn_fn = read_f32(paths["ffn_fn"], MIX * FLAT)
    ffn_base = read_f32(paths["ffn_base"], MIX)
    ffn_scale = read_f32(paths["ffn_scale"], 3)

    residual_bits, residual = make_residual()
    attn_pre_f, attn_mix, attn_pre, attn_post, attn_comb = hc_pre(
        residual, attn_fn, attn_base, attn_scale)
    attn_pre_bits, attn_pre_b = cast_bf16(attn_pre_f)
    attn_branch_bits, attn_branch = cast_bf16(attn_stub(attn_pre_b))
    after_attn_f = hc_post(attn_branch, residual, attn_post, attn_comb)
    after_attn_bits, after_attn = cast_bf16(after_attn_f)

    ffn_pre_f, ffn_mix, ffn_pre, ffn_post, ffn_comb = hc_pre(
        after_attn, ffn_fn, ffn_base, ffn_scale)
    ffn_pre_bits, ffn_pre_b = cast_bf16(ffn_pre_f)
    ffn_branch_bits, ffn_branch = cast_bf16(ffn_stub(ffn_pre_b))
    final_f = hc_post(ffn_branch, after_attn, ffn_post, ffn_comb)
    final_bits, final = cast_bf16(final_f)

    # Reject vacuous fixture constructions now, before freezing bytes.
    wrong_pre_f, _, _, wrong_post, wrong_comb = hc_pre(
        residual, ffn_fn, ffn_base, ffn_scale)
    _, wrong_pre = cast_bf16(wrong_pre_f)
    _, wrong_branch = cast_bf16(ffn_stub(wrong_pre))
    wrong_final = hc_post(wrong_branch, residual, wrong_post, wrong_comb)
    wrong_bits, _ = cast_bf16(wrong_final)
    if wrong_bits == final_bits:
        raise ValueError("original-residual FFN mutation is invisible")

    _, bad_branch = cast_bf16(ffn_stub(attn_pre_b))
    bad_final = hc_post(bad_branch, after_attn, ffn_post, ffn_comb)
    bad_bits, _ = cast_bf16(bad_final)
    if bad_bits == final_bits:
        raise ValueError("wrong FFN branch-input mutation is invisible")

    files = {
        "residual": ("input-residual.bf16.bin", residual_bits),
        "attn_pre": ("attn-pre.bf16.bin", attn_pre_bits),
        "attn_branch": ("attn-stub-branch.bf16.bin", attn_branch_bits),
        "after_attn": ("after-attn-post.bf16.bin", after_attn_bits),
        "ffn_pre": ("ffn-pre.bf16.bin", ffn_pre_bits),
        "ffn_branch": ("ffn-stub-branch.bf16.bin", ffn_branch_bits),
        "final": ("expected-final.bf16.bin", final_bits),
    }
    for _, (name, bits) in files.items():
        write_u16(os.path.join(args.out_dir, name), bits)

    diag = attn_mix + attn_pre + attn_post + attn_comb + ffn_mix + ffn_pre + ffn_post + ffn_comb
    diag_path = os.path.join(args.out_dir, "expected-diagnostics.f32.bin")
    write_f32(diag_path, diag)

    manifest = {
        "schema_version": 1,
        "gate": "H/V6 real layer-3 HC composition sub-seam",
        "model": info["attn_fn"]["model"],
        "revision": info["attn_fn"]["revision"],
        "layer": 3,
        "boundary": "two real HC transitions; deterministic branch stubs, not Attention/MoE proof",
        "official_block_order": ["hc_attn_pre", "attention_branch", "hc_attn_post",
                                 "hc_ffn_pre", "moe_branch", "hc_ffn_post"],
        "parameters": {},
        "states": {},
        "diagnostics": {"file": os.path.basename(diag_path), "sha256": sha(diag_path),
                        "float_count": len(diag)},
        "mutations": ["ffn_pre_from_original_residual", "ffn_branch_from_attn_pre"],
    }
    for key, tensor in PARAMS.items():
        manifest["parameters"][key] = {
            "tensor": tensor, "dtype": info[key]["dtype"], "shape": info[key]["shape"],
            "payload_sha256": sha(paths[key]), "file": os.path.basename(paths[key]),
            "shard": info[key]["shard"], "byte_range": info[key].get("byte_range"),
        }
    for key, (name, bits) in files.items():
        path = os.path.join(args.out_dir, name)
        manifest["states"][key] = {"file": name, "sha256": sha(path), "bf16_count": len(bits)}

    prov = os.path.join(args.out_dir, "provenance.json")
    with open(prov, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print("PASS generated real layer-3 two-HC composition fixture")
    print("final first8:", " ".join(f"{x:04x}" for x in final_bits[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
