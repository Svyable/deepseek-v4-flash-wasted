#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build real learned + bootstrap/hash router fixtures for DeepSeek V4.

The fixture uses sparse BF16 hidden input so the 256-way router projection is
independent of a long GEMM reduction order while still consuming real checkpoint
weights. Learned layer 3 exercises correction-bias selection; hash layer 0 uses
one real tid2eid row for exact IDs and real gate weights for routing weights.
"""

import argparse
import hashlib
import json
import math
import os
import struct

import fetch_hf_tensor_slice as slice_fetch

DIM = 4096
EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5
TOKEN_ID = 4242

LEARNED_WEIGHT = "layers.3.ffn.gate.weight"
LEARNED_BIAS = "layers.3.ffn.gate.bias"
HASH_WEIGHT = "layers.0.ffn.gate.weight"
HASH_TABLE = "layers.0.ffn.gate.tid2eid"

SUPPORT = (
    (0, 1.0),
    (7, -0.375),
    (17, 0.625),
    (511, -1.25),
    (1023, 0.1875),
    (2048, 1.5),
    (3071, -0.75),
    (4095, 0.5),
)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


def bf16_bits(x):
    u = f32_bits(x)
    exp = u & 0x7F800000
    frac = u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16_to_f32(b):
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]


def read_bf16(path, count):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != count * 2:
        raise ValueError(f"{path}: {len(data)} B, expected {count*2}")
    return list(struct.unpack(f"<{count}H", data))


def read_f32(path, count):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} B, expected {count*4}")
    return list(struct.unpack(f"<{count}f", data))


def read_i64(path, count):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != count * 8:
        raise ValueError(f"{path}: {len(data)} B, expected {count*8}")
    return list(struct.unpack(f"<{count}q", data))


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_u32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[f32(x) for x in values]))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_input():
    bits = [0] * DIM
    for dim, value in SUPPORT:
        bits[dim] = bf16_bits(value)
    return bits, [bf16_to_f32(x) for x in bits]


def softplus_sqrt(x):
    x = f32(x)
    if x > 20.0:
        sp = x
    elif x < 0.0:
        sp = f32(math.log1p(math.exp(x)))
    else:
        sp = f32(x + math.log1p(math.exp(-x)))
    return f32(math.sqrt(sp))


def score(weight_bits, x):
    support = [(d, x[d]) for d, _ in SUPPORT]
    raw, scores = [], []
    for expert in range(EXPERTS):
        off = expert * DIM
        acc = f32(0.0)
        for dim, value in support:
            w = bf16_to_f32(weight_bits[off + dim])
            acc = f32(acc + f32(w * value))
        raw.append(acc)
        scores.append(softplus_sqrt(acc))
    return raw, scores


def normalize(scores, ids):
    selected = [scores[i] for i in ids]
    total = f32(0.0)
    for x in selected:
        total = f32(total + x)
    return [f32(f32(x / total) * ROUTE_SCALE) for x in selected]


def learned_oracle(weight_bits, bias, x):
    raw, scores = score(weight_bits, x)
    selection = [f32(scores[i] + bias[i]) for i in range(EXPERTS)]
    ranked = sorted(range(EXPERTS), key=lambda i: (-selection[i], i))
    ids = ranked[:TOPK]
    # Reject an ambiguous fixture rather than pretending to know PyTorch's
    # equal-value topk tie behavior.
    margin = selection[ranked[TOPK - 1]] - selection[ranked[TOPK]]
    if not margin > 1e-5:
        raise ValueError(f"learned top-k boundary too close: margin={margin}")
    if len({selection[i] for i in ids}) != TOPK:
        raise ValueError("learned top-k contains an exact score tie")
    return {
        "raw": raw,
        "score": scores,
        "selection": selection,
        "ids": ids,
        "weights": normalize(scores, ids),
        "boundary_margin": margin,
    }


def hash_oracle(weight_bits, tid2eid, x):
    raw, scores = score(weight_bits, x)
    ids = [int(v) for v in tid2eid]
    if len(ids) != TOPK or len(set(ids)) != TOPK or not all(0 <= i < EXPERTS for i in ids):
        raise ValueError(f"invalid checkpoint tid2eid row: {ids}")
    return {
        "raw": raw,
        "score": scores,
        "ids": ids,
        "weights": normalize(scores, ids),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    learned_weight_path = os.path.join(args.out_dir, "layer3-gate-weight.bf16.bin")
    learned_bias_path = os.path.join(args.out_dir, "layer3-gate-bias.f32.bin")
    hash_weight_path = os.path.join(args.out_dir, "layer0-gate-weight.bf16.bin")
    hash_ids_path = os.path.join(args.out_dir, "layer0-tid2eid-token4242.i64.bin")

    lw_info = slice_fetch.fetch_slice(
        args.snapshot, LEARNED_WEIGHT, None, learned_weight_path, token=args.token)
    lb_info = slice_fetch.fetch_slice(
        args.snapshot, LEARNED_BIAS, None, learned_bias_path, token=args.token)
    hw_info = slice_fetch.fetch_slice(
        args.snapshot, HASH_WEIGHT, None, hash_weight_path, token=args.token)
    hi_info = slice_fetch.fetch_slice(
        args.snapshot, HASH_TABLE, f"{TOKEN_ID}:{TOKEN_ID+1}", hash_ids_path,
        token=args.token)

    expected_meta = [
        (lw_info, "BF16", [EXPERTS, DIM], LEARNED_WEIGHT),
        (lb_info, "F32", [EXPERTS], LEARNED_BIAS),
        (hw_info, "BF16", [EXPERTS, DIM], HASH_WEIGHT),
        (hi_info, "I64", [129280, TOPK], HASH_TABLE),
    ]
    for info, dtype, shape, name in expected_meta:
        if info["dtype"] != dtype or info["shape"] != shape:
            raise ValueError(
                f"{name}: got {info['dtype']} {info['shape']}, expected {dtype} {shape}")

    lw = read_bf16(learned_weight_path, EXPERTS * DIM)
    bias = read_f32(learned_bias_path, EXPERTS)
    hw = read_bf16(hash_weight_path, EXPERTS * DIM)
    tid = read_i64(hash_ids_path, TOPK)
    input_bits, x = make_input()

    learned = learned_oracle(lw, bias, x)
    hashed = hash_oracle(hw, tid, x)

    input_path = os.path.join(args.out_dir, "input.bf16.bin")
    learned_diag = os.path.join(args.out_dir, "learned-diagnostics.f32.bin")
    learned_ids = os.path.join(args.out_dir, "learned-ids.u32.bin")
    learned_weights = os.path.join(args.out_dir, "learned-weights.f32.bin")
    hash_diag = os.path.join(args.out_dir, "hash-diagnostics.f32.bin")
    hash_ids = os.path.join(args.out_dir, "hash-ids.u32.bin")
    hash_weights = os.path.join(args.out_dir, "hash-weights.f32.bin")

    write_u16(input_path, input_bits)
    write_f32(learned_diag, learned["raw"] + learned["score"] + learned["selection"])
    write_u32(learned_ids, learned["ids"])
    write_f32(learned_weights, learned["weights"])
    write_f32(hash_diag, hashed["raw"] + hashed["score"])
    write_u32(hash_ids, hashed["ids"])
    write_f32(hash_weights, hashed["weights"])

    provenance = {
        "schema_version": 1,
        "gate": "V3 router primitive -> Gate F / V4",
        "fixture_class": "F3 official-source + real-checkpoint router primitive",
        "evidence_state": "CHECKPOINT-VERIFIED_SOURCE-ORACLE",
        "model": lw_info["model"],
        "revision": lw_info["revision"],
        "source_contract": {
            "path": "inference/model.py Gate.forward",
            "score_func": "sqrt(softplus(raw_linear))",
            "selection": "learned: topk(score+bias); hash: tid2eid[input_id]",
            "weight_source": "unbiased transformed score",
            "normalization": "divide selected weights by sum then multiply 1.5",
            "topk": TOPK,
            "experts": EXPERTS,
        },
        "input": {
            "file": os.path.basename(input_path),
            "dtype": "BF16",
            "shape": [DIM],
            "support": [list(v) for v in SUPPORT],
            "sha256": sha256_file(input_path),
        },
        "learned": {
            "layer": 3,
            "weight": lw_info,
            "bias": lb_info,
            "ids": learned["ids"],
            "weights": learned["weights"],
            "topk_boundary_margin": learned["boundary_margin"],
            "diagnostics_file": os.path.basename(learned_diag),
            "diagnostics_layout": "raw[256], score[256], selection[256]",
            "diagnostics_sha256": sha256_file(learned_diag),
            "ids_file": os.path.basename(learned_ids),
            "ids_sha256": sha256_file(learned_ids),
            "weights_file": os.path.basename(learned_weights),
            "weights_sha256": sha256_file(learned_weights),
        },
        "hash": {
            "layer": 0,
            "token_id": TOKEN_ID,
            "weight": hw_info,
            "tid2eid": hi_info,
            "ids": hashed["ids"],
            "weights": hashed["weights"],
            "diagnostics_file": os.path.basename(hash_diag),
            "diagnostics_layout": "raw[256], score[256]",
            "diagnostics_sha256": sha256_file(hash_diag),
            "ids_file": os.path.basename(hash_ids),
            "ids_sha256": sha256_file(hash_ids),
            "weights_file": os.path.basename(hash_weights),
            "weights_sha256": sha256_file(hash_weights),
        },
        "non_claim": "Router primitive only. No routed/shared expert execution or full MoE block is claimed by this fixture."
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    for path in (
        learned_weight_path + ".json", learned_bias_path + ".json",
        hash_weight_path + ".json", hash_ids_path + ".json"):
        if os.path.exists(path):
            os.remove(path)

    print("router fixture built")
    print("learned IDs:", learned["ids"])
    print("learned weights:", [f"{x:.9g}" for x in learned["weights"]])
    print("top-k boundary margin:", learned["boundary_margin"])
    print("hash token", TOKEN_ID, "IDs:", hashed["ids"])
    print("hash weights:", [f"{x:.9g}" for x in hashed["weights"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
