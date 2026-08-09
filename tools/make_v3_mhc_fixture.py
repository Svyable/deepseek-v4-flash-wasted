#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a real-checkpoint Gate D/V3 layer-0 attention mHC fixture.

Uses real ``layers.0.hc_attn_{fn,base,scale}`` parameters and an independent
Python implementation of the pinned hc_pre / hc_split_sinkhorn / hc_post
source equations. The deterministic BF16 residual is sparse on purpose: it
exercises all four HC lanes and real learned columns while avoiding a fixture
whose expected answer depends on an incidental 16k-term GEMM reduction order.
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
MIX = (2 + HC) * HC  # 24
ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6

FN = "layers.0.hc_attn_fn"
BASE = "layers.0.hc_attn_base"
SCALE = "layers.0.hc_attn_scale"

# (hc lane, hidden dim, exact dyadic value)
RESIDUAL_SUPPORT = (
    (0, 0, 1.0),
    (0, 2048, 0.75),
    (1, 17, -0.5),
    (1, 3333, -1.25),
    (2, 7, 0.125),
    (2, 1023, 0.25),
    (3, 999, 1.5),
    (3, 4095, -2.0),
)

# Branch result supplied to hc_post; only source HC is absent because this is
# the single collapsed attention result [dim].
BRANCH_SUPPORT = (
    (0, 0.125),
    (7, -0.375),
    (17, 1.5),
    (999, 0.5),
    (1023, -0.75),
    (2048, -0.25),
    (3333, 0.625),
    (4095, 1.0),
)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def bits_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bf16_bits(value):
    u = f32_bits(value)
    exp = u & 0x7F800000
    frac = u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16_to_f32(bits):
    return bits_f32((bits & 0xFFFF) << 16)


def sigmoid(value):
    value = f32(value)
    if value >= 0.0:
        z = f32(math.exp(-value))
        return f32(1.0 / f32(1.0 + z))
    z = f32(math.exp(value))
    return f32(z / f32(1.0 + z))


def read_f32(path, count):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*4}")
    return list(struct.unpack(f"<{count}f", data))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[f32(v) for v in values]))


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_bf16_state():
    bits = [0] * FLAT
    for lane, dim, value in RESIDUAL_SUPPORT:
        bits[lane * DIM + dim] = bf16_bits(value)
    return bits, [bf16_to_f32(v) for v in bits]


def make_branch():
    bits = [0] * DIM
    for dim, value in BRANCH_SUPPORT:
        bits[dim] = bf16_bits(value)
    return bits, [bf16_to_f32(v) for v in bits]


def sinkhorn(mixes, scale, base):
    pre = [0.0] * HC
    post = [0.0] * HC
    for j in range(HC):
        pre[j] = f32(sigmoid(f32(f32(mixes[j] * scale[0]) + base[j])) + HC_EPS)
        post[j] = f32(2.0 * sigmoid(
            f32(f32(mixes[j + HC] * scale[1]) + base[j + HC])))

    comb = [[0.0] * HC for _ in range(HC)]
    for j in range(HC):
        logits = [
            f32(f32(mixes[2 * HC + j * HC + k] * scale[2])
                + base[2 * HC + j * HC + k])
            for k in range(HC)
        ]
        row_max = max(logits)
        exps = [f32(math.exp(f32(v - row_max))) for v in logits]
        row_sum = f32(0.0)
        for v in exps:
            row_sum = f32(row_sum + v)
        for k in range(HC):
            comb[j][k] = f32(f32(exps[k] / row_sum) + HC_EPS)

    for k in range(HC):
        col_sum = f32(0.0)
        for j in range(HC):
            col_sum = f32(col_sum + comb[j][k])
        denom = f32(col_sum + HC_EPS)
        for j in range(HC):
            comb[j][k] = f32(comb[j][k] / denom)

    for _ in range(ITERS - 1):
        for j in range(HC):
            row_sum = f32(0.0)
            for k in range(HC):
                row_sum = f32(row_sum + comb[j][k])
            denom = f32(row_sum + HC_EPS)
            for k in range(HC):
                comb[j][k] = f32(comb[j][k] / denom)
        for k in range(HC):
            col_sum = f32(0.0)
            for j in range(HC):
                col_sum = f32(col_sum + comb[j][k])
            denom = f32(col_sum + HC_EPS)
            for j in range(HC):
                comb[j][k] = f32(comb[j][k] / denom)

    return pre, post, [comb[j][k] for j in range(HC) for k in range(HC)]


def source_oracle(fn, base, scale, residual, branch):
    square_sum = f32(0.0)
    nonzero = []
    for lane, dim, _ in RESIDUAL_SUPPORT:
        idx = lane * DIM + dim
        x = residual[idx]
        square_sum = f32(square_sum + f32(x * x))
        nonzero.append((idx, x))
    mean = f32(square_sum / float(FLAT))
    rsqrt = f32(1.0 / math.sqrt(f32(mean + NORM_EPS)))

    mixes = []
    for row in range(MIX):
        acc = f32(0.0)
        off = row * FLAT
        for idx, x in nonzero:
            acc = f32(acc + f32(fn[off + idx] * x))
        mixes.append(f32(acc * rsqrt))

    pre, post, comb = sinkhorn(mixes, scale, base)

    pre_y = [0.0] * DIM
    for d in range(DIM):
        acc = f32(0.0)
        for lane in range(HC):
            acc = f32(acc + f32(pre[lane] * residual[lane * DIM + d]))
        pre_y[d] = acc

    post_y = [0.0] * FLAT
    for out_lane in range(HC):
        for d in range(DIM):
            acc = f32(post[out_lane] * branch[d])
            for source_lane in range(HC):
                acc = f32(acc + f32(
                    comb[source_lane * HC + out_lane]
                    * residual[source_lane * DIM + d]))
            post_y[out_lane * DIM + d] = acc

    return {
        "square_sum": square_sum,
        "rsqrt": rsqrt,
        "mixes": mixes,
        "pre": pre,
        "post": post,
        "comb": comb,
        "pre_y": pre_y,
        "post_y": post_y,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    fn_path = os.path.join(args.out_dir, "hc_attn_fn.f32.bin")
    base_path = os.path.join(args.out_dir, "hc_attn_base.f32.bin")
    scale_path = os.path.join(args.out_dir, "hc_attn_scale.f32.bin")

    fn_info = slice_fetch.fetch_slice(args.snapshot, FN, None, fn_path, token=args.token)
    base_info = slice_fetch.fetch_slice(args.snapshot, BASE, None, base_path, token=args.token)
    scale_info = slice_fetch.fetch_slice(args.snapshot, SCALE, None, scale_path, token=args.token)

    if fn_info["dtype"] != "F32" or fn_info["shape"] != [MIX, FLAT]:
        raise ValueError(f"unexpected {FN} metadata: {fn_info['dtype']} {fn_info['shape']}")
    if base_info["dtype"] != "F32" or base_info["shape"] != [MIX]:
        raise ValueError(f"unexpected {BASE} metadata: {base_info['dtype']} {base_info['shape']}")
    if scale_info["dtype"] != "F32" or scale_info["shape"] != [3]:
        raise ValueError(f"unexpected {SCALE} metadata: {scale_info['dtype']} {scale_info['shape']}")

    fn = read_f32(fn_path, MIX * FLAT)
    base = read_f32(base_path, MIX)
    scale = read_f32(scale_path, 3)
    residual_bits, residual = make_bf16_state()
    branch_bits, branch = make_branch()
    oracle = source_oracle(fn, base, scale, residual, branch)

    residual_path = os.path.join(args.out_dir, "residual.bf16.bin")
    branch_path = os.path.join(args.out_dir, "branch.bf16.bin")
    pre_path = os.path.join(args.out_dir, "expected-pre.bf16.bin")
    post_path = os.path.join(args.out_dir, "expected-post.bf16.bin")
    diag_path = os.path.join(args.out_dir, "expected-diagnostics.f32.bin")

    write_u16(residual_path, residual_bits)
    write_u16(branch_path, branch_bits)
    write_u16(pre_path, [bf16_bits(x) for x in oracle["pre_y"]])
    write_u16(post_path, [bf16_bits(x) for x in oracle["post_y"]])
    write_f32(diag_path, oracle["mixes"] + oracle["pre"] + oracle["post"] + oracle["comb"])

    provenance = {
        "schema_version": 1,
        "gate": "Gate D / V3",
        "fixture_class": "F3 official-source + real-checkpoint mHC primitive",
        "evidence_state": "CHECKPOINT-VERIFIED_SOURCE-ORACLE",
        "model": fn_info["model"],
        "revision": fn_info["revision"],
        "operation": "layers.0 attention hc_pre + hc_split_sinkhorn + hc_post",
        "source_contract": {
            "model_path": "inference/model.py Block.hc_pre/Block.hc_post",
            "kernel_path": "inference/kernel.py hc_split_sinkhorn",
            "hc_mult": HC,
            "mix_hc": MIX,
            "sinkhorn_iters": ITERS,
            "norm_eps": NORM_EPS,
            "hc_eps": HC_EPS,
            "comb_orientation": "comb[source_hc, output_hc]; hc_post sums source_hc"
        },
        "parameters": {
            "fn": fn_info,
            "base": base_info,
            "scale": scale_info
        },
        "inputs": {
            "residual": {
                "file": os.path.basename(residual_path),
                "dtype": "BF16",
                "shape": [HC, DIM],
                "sha256": sha256_file(residual_path),
                "support": [list(x) for x in RESIDUAL_SUPPORT]
            },
            "branch": {
                "file": os.path.basename(branch_path),
                "dtype": "BF16",
                "shape": [DIM],
                "sha256": sha256_file(branch_path),
                "support": [list(x) for x in BRANCH_SUPPORT]
            }
        },
        "diagnostics": {
            "file": os.path.basename(diag_path),
            "layout": "mixes[24], pre[4], post[4], comb[16] row-major",
            "sha256": sha256_file(diag_path),
            "rsqrt": oracle["rsqrt"],
            "pre": oracle["pre"],
            "post": oracle["post"],
            "comb": oracle["comb"],
            "row_sums": [sum(oracle["comb"][j*HC:(j+1)*HC]) for j in range(HC)],
            "column_sums": [sum(oracle["comb"][j*HC+k] for j in range(HC)) for k in range(HC)]
        },
        "expected": {
            "pre_file": os.path.basename(pre_path),
            "pre_sha256": sha256_file(pre_path),
            "post_file": os.path.basename(post_path),
            "post_sha256": sha256_file(post_path),
            "comparison": "final hc_pre/hc_post exact BF16; diagnostic f32 values use a fixed small absolute tolerance"
        },
        "non_claim": "The fixture evaluates pinned source equations independently on CPU; it does not execute the official CUDA/TileLang kernels or a complete attention block."
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    for path in (fn_path + ".json", base_path + ".json", scale_path + ".json"):
        if os.path.exists(path):
            os.remove(path)

    print("Gate D mHC fixture built")
    print("pre:", " ".join(f"{x:.9g}" for x in oracle["pre"]))
    print("post:", " ".join(f"{x:.9g}" for x in oracle["post"]))
    print("comb row sums:", " ".join(f"{x:.9g}" for x in provenance["diagnostics"]["row_sums"]))
    print("comb col sums:", " ".join(f"{x:.9g}" for x in provenance["diagnostics"]["column_sums"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
