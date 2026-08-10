#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Correct Gate H's frozen layer-3 final HC boundary without checkpoint I/O.

The routed/shared expert outputs and combined MoE branch were already verified
against the immutable checkpoint. The only stale evidence is the final
HyperConnection composition, whose earlier Python producer retained double
precision inside Sinkhorn before rounding its coefficients to F32. The model
and scalar C path perform those operations in F32.

This tool reuses the committed real attention/MoE branch outputs, recomputes the
full layer-3 HC path with the independent bit-exact F32 oracle, requires every
upstream BF16 seam (`attn_pre`, `ffn_pre`) to remain unchanged, then rewrites
only `expected-layer3-final.bf16.bin` and its provenance.
"""

import hashlib
import json
import os
import struct
import sys

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
OLD_SHA = "0e65c4ecb328d5067f2274e724f9f46e4a13218e7fdeb706f7d1a465c0ee4761"
NEW_SHA = "c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7"
DIFF_INDICES = [1186, 10135, 10503, 10877, 11565, 11672, 11675, 13648, 13717]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def f32_param(prov, key, count):
    meta = prov["parameters"][key]
    path = os.path.join(HCFIX, meta["file"])
    want = meta.get("payload_sha256", meta.get("sha256"))
    if not os.path.isfile(path) or (want and sha(path) != want):
        raise ValueError(f"HC parameter {key} missing or SHA mismatch")
    return read_f32(path, count)


def exact(label, got, want):
    if got == want:
        return
    i = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
    raise ValueError(f"{label} mismatch at {i}: {got[i]:04x}!={want[i]:04x}")


def main():
    hp = load(os.path.join(HCFIX, "provenance.json"))
    ap = load(os.path.join(ATTFIX, "provenance.json"))
    mp_path = os.path.join(MOEFIX, "provenance.json")
    mp = load(mp_path)

    residual = read_u16(os.path.join(HCFIX, hp["states"]["residual"]["file"]), FLAT)
    attn_pre_frozen = read_u16(os.path.join(HCFIX, hp["states"]["attn_pre"]["file"]), DIM)
    attn_meta = ap["outputs"]["branch"]
    attn_path = os.path.join(ATTFIX, attn_meta["file"])
    if sha(attn_path) != attn_meta["sha256"]:
        raise ValueError("real attention branch SHA mismatch")
    attn = read_u16(attn_path, DIM)

    ffn_meta = mp["input_dependency"]
    ffn_path = os.path.join(MOEFIX, ffn_meta["file"])
    if sha(ffn_path) != ffn_meta["sha256"]:
        raise ValueError("real FFN-pre SHA mismatch")
    ffn_frozen = read_u16(ffn_path, DIM)
    moe_meta = mp["outputs"]["moe_branch"]
    moe_path = os.path.join(MOEFIX, moe_meta["file"])
    if sha(moe_path) != moe_meta["sha256"]:
        raise ValueError("real MoE branch SHA mismatch")
    moe = read_u16(moe_path, DIM)

    params = {
        "attn_fn": f32_param(hp, "attn_fn", MIX * FLAT),
        "attn_scale": f32_param(hp, "attn_scale", 3),
        "attn_base": f32_param(hp, "attn_base", MIX),
        "ffn_fn": f32_param(hp, "ffn_fn", MIX * FLAT),
        "ffn_scale": f32_param(hp, "ffn_scale", 3),
        "ffn_base": f32_param(hp, "ffn_base", MIX),
    }

    residual_f = [chain.bf16_to_f32(v) for v in residual]
    attn_f = [chain.bf16_to_f32(v) for v in attn]
    moe_f = [chain.bf16_to_f32(v) for v in moe]

    apre, apost, acomb, _ = hc.params(
        residual_f, params["attn_fn"], params["attn_scale"], params["attn_base"], dim=DIM)
    attn_pre, _ = chain.cast_bf16(hc.pre_y(residual_f, apre, dim=DIM))
    exact("corrected HC-attn-pre", attn_pre, attn_pre_frozen)
    after_f = hc.post_y(attn_f, residual_f, apost, acomb, dim=DIM)
    after_bits, after = chain.cast_bf16(after_f)

    fpre, fpost, fcomb, _ = hc.params(
        after, params["ffn_fn"], params["ffn_scale"], params["ffn_base"], dim=DIM)
    ffn_pre, _ = chain.cast_bf16(hc.pre_y(after, fpre, dim=DIM))
    exact("corrected HC-FFN-pre", ffn_pre, ffn_frozen)
    final_f = hc.post_y(moe_f, after, fpost, fcomb, dim=DIM)
    final_bits, _ = chain.cast_bf16(final_f)
    raw = struct.pack(f"<{len(final_bits)}H", *final_bits)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != NEW_SHA:
        raise ValueError(f"corrected final SHA {digest} != audited {NEW_SHA}")

    out_meta = mp["outputs"]["layer3_final"]
    out_path = os.path.join(MOEFIX, out_meta["file"])
    old_digest = sha(out_path)
    if old_digest not in (OLD_SHA, NEW_SHA):
        raise ValueError(f"unexpected pre-correction layer3 final SHA {old_digest}")
    if old_digest == OLD_SHA:
        old_bits = read_u16(out_path, FLAT)
        diffs = [i for i, (a, b) in enumerate(zip(final_bits, old_bits)) if a != b]
        if diffs != DIFF_INDICES:
            raise ValueError(f"audited BF16 diff set drifted: {diffs}")
    with open(out_path, "wb") as f:
        f.write(raw)

    out_meta["sha256"] = NEW_SHA
    out_meta["first8_hex"] = [f"0x{x:04x}" for x in final_bits[:8]]
    mp["hc_precision_correction"] = {
        "reason": "Gate-H Python Sinkhorn retained double precision before the model's F32 HC coefficient boundary; corrected oracle performs every HC operation in F32 and is bit-exact with src/deepseek_v4_mhc_ref.c",
        "old_layer3_final_sha256": OLD_SHA,
        "corrected_layer3_final_sha256": NEW_SHA,
        "bf16_diff_count": len(DIFF_INDICES),
        "bf16_diff_indices": DIFF_INDICES,
        "attn_pre_bf16_unchanged": True,
        "ffn_pre_bf16_unchanged": True,
        "router_and_expert_acquisition_unchanged": True,
        "independent_f32_oracle": "tools/deepseek_v4_hc_oracle.py",
        "exact_c_differential_test": "tests/test_v6_hc_oracle_f32.py",
    }
    with open(mp_path, "w", encoding="utf-8") as f:
        json.dump(mp, f, indent=2, sort_keys=True)
        f.write("\n")

    print("PASS corrected Gate-H layer-3 final HC boundary without checkpoint I/O")
    print("old sha256:", OLD_SHA)
    print("new sha256:", NEW_SHA)
    print("BF16 diff indices:", DIFF_INDICES)
    print("first8:", " ".join(f"0x{x:04x}" for x in final_bits[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
