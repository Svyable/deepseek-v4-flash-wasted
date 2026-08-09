#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a compact real layer-3 Gate F MoE combination fixture.

All six routed experts selected by the frozen learned-router fixture are
computed from real checkpoint bytes with the independent Python quantization
oracle. Their large raw weights are temporary; only the eight-output BF16
results and source hashes are frozen. The distinct shared-expert implementation
is kept in full (w1/w3 plus eight w2 rows) so its scalar C path remains an
offline permanent regression.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile

import fetch_hf_tensor_slice as slice_fetch
import make_v4_routed_expert_fixture as oracle

LAYER = 3
INPUT = 4096
INTER = 2048
OUT_ROWS = 8
LIMIT = 10.0
ROUTER_FIXTURE = os.path.join("tests", "fixtures", "deepseek_v4", "v3_router_real")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_u32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[oracle.f32(v) for v in values]))


def read_router(repo_root):
    root = os.path.join(repo_root, ROUTER_FIXTURE)
    with open(os.path.join(root, "provenance.json"), encoding="utf-8") as f:
        p = json.load(f)
    ids = list(p["learned"]["ids"])
    weights = list(p["learned"]["weights"])
    if len(ids) != 6 or len(set(ids)) != 6 or len(weights) != 6:
        raise ValueError("router fixture is not a unique top-6 fixture")
    with open(os.path.join(root, "input.bf16.bin"), "rb") as f:
        raw_input = f.read()
    bits = struct.unpack(f"<{INPUT}H", raw_input)
    x = [oracle.bf16_to_f32(v) for v in bits]
    return p, ids, [oracle.f32(v) for v in weights], raw_input, x


def fetch_bytes(snapshot, tensor, rows, path, token):
    info = slice_fetch.fetch_slice(snapshot, tensor, rows, path, token=token)
    with open(path, "rb") as f:
        data = f.read()
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return info, data


def routed_out(snapshot, expert, route_weight, x, work, token):
    base = f"layers.{LAYER}.ffn.experts.{expert}"
    names = {
        "w1": base + ".w1.weight", "s1": base + ".w1.scale",
        "w3": base + ".w3.weight", "s3": base + ".w3.scale",
        "w2": base + ".w2.weight", "s2": base + ".w2.scale",
    }
    infos, data = {}, {}
    for key in ("w1", "s1", "w3", "s3"):
        infos[key], data[key] = fetch_bytes(
            snapshot, names[key], None, os.path.join(work, key + ".bin"), token)
    infos["w2"], data["w2"] = fetch_bytes(
        snapshot, names["w2"], f"0:{OUT_ROWS}", os.path.join(work, "w2.bin"), token)
    infos["s2"], data["s2"] = fetch_bytes(
        snapshot, names["s2"], f"0:{OUT_ROWS}", os.path.join(work, "s2.bin"), token)

    expected = {
        "w1": ("I8", [INTER, INPUT // 2]),
        "s1": ("F8_E8M0", [INTER, INPUT // 32]),
        "w3": ("I8", [INTER, INPUT // 2]),
        "s3": ("F8_E8M0", [INTER, INPUT // 32]),
        "w2": ("I8", [4096, INTER // 2]),
        "s2": ("F8_E8M0", [4096, INTER // 32]),
    }
    for key, (dtype, shape) in expected.items():
        if infos[key]["dtype"] != dtype or infos[key]["shape"] != shape:
            raise ValueError(f"expert {expert} {key}: got {infos[key]['dtype']} {infos[key]['shape']}, expected {dtype} {shape}")

    gate = oracle.fp4_linear_rows(data["w1"], data["s1"], INTER, INPUT, x)
    up = oracle.fp4_linear_rows(data["w3"], data["s3"], INTER, INPUT, x)
    hidden = []
    for g, u in zip(gate, up):
        g = min(g, LIMIT)
        u = max(-LIMIT, min(LIMIT, u))
        h = oracle.f32(oracle.silu_f32(g) * u)
        h = oracle.f32(h * route_weight)
        hidden.append(oracle.bf16_to_f32(oracle.bf16_bits(h)))
    out = oracle.fp4_linear_rows(data["w2"], data["s2"], OUT_ROWS, INTER, hidden)
    out_bits = [oracle.bf16_bits(v) for v in out]
    return out_bits, {
        "expert": expert,
        "route_weight": route_weight,
        "output_bf16_hex": [f"0x{x:04x}" for x in out_bits],
        "tensors": {k: {
            "name": names[k], "shard": infos[k]["shard"],
            "dtype": infos[k]["dtype"], "shape": infos[k]["shape"],
            "payload_sha256": hashlib.sha256(data[k]).hexdigest(),
        } for k in names},
    }


def fp8_linear_rows(weight, scales, rows, k, x):
    qx, act_scales = oracle.quantize_activation(x)
    k_blocks = k // 128
    if len(weight) != rows * k:
        raise ValueError("FP8 weight slice dimensions do not match")
    scale_rows = (rows + 127) // 128
    if len(scales) != scale_rows * k_blocks:
        raise ValueError(f"FP8 scale grid has {len(scales)} bytes, expected {scale_rows * k_blocks}")

    out = []
    rev_out = []
    for r in range(rows):
        wr = weight[r * k:(r + 1) * k]
        sr = scales[(r // 128) * k_blocks:(r // 128 + 1) * k_blocks]
        parts = []
        for kb in range(k_blocks):
            part = oracle.f32(0.0)
            base = kb * 128
            for j in range(128):
                a = oracle.e4m3_float(qx[base + j])
                b = oracle.e4m3_float(wr[base + j])
                part = oracle.f32(part + oracle.f32(a * b))
            scale = act_scales[kb] * oracle.e8m0_float(sr[kb])
            parts.append(oracle.f32(part * scale))
        acc = oracle.f32(0.0)
        for p in parts:
            acc = oracle.f32(acc + p)
        rev = oracle.f32(0.0)
        for p in reversed(parts):
            rev = oracle.f32(rev + p)
        if oracle.bf16_bits(acc) != oracle.bf16_bits(rev):
            raise ValueError(f"FP8 row {r} is not BF16-stable across reduction direction")
        out.append(oracle.bf16_to_f32(oracle.bf16_bits(acc)))
        rev_out.append(rev)
    return out


def build_shared(snapshot, x, out_dir, token):
    base = f"layers.{LAYER}.ffn.shared_experts"
    names = {
        "w1": base + ".w1.weight", "s1": base + ".w1.scale",
        "w3": base + ".w3.weight", "s3": base + ".w3.scale",
        "w2": base + ".w2.weight", "s2": base + ".w2.scale",
    }
    paths = {
        "w1": os.path.join(out_dir, "shared-w1.fp8.bin"),
        "s1": os.path.join(out_dir, "shared-w1-scale.e8m0.bin"),
        "w3": os.path.join(out_dir, "shared-w3.fp8.bin"),
        "s3": os.path.join(out_dir, "shared-w3-scale.e8m0.bin"),
        "w2": os.path.join(out_dir, "shared-w2-rows0-8.fp8.bin"),
        "s2": os.path.join(out_dir, "shared-w2-scale-row0.e8m0.bin"),
    }
    infos, data = {}, {}
    for key in ("w1", "s1", "w3", "s3"):
        infos[key], data[key] = fetch_bytes(snapshot, names[key], None, paths[key], token)
    infos["w2"], data["w2"] = fetch_bytes(snapshot, names["w2"], f"0:{OUT_ROWS}", paths["w2"], token)
    infos["s2"], data["s2"] = fetch_bytes(snapshot, names["s2"], "0:1", paths["s2"], token)

    expected = {
        "w1": ("F8_E4M3", [INTER, INPUT]), "s1": ("F8_E8M0", [16, 32]),
        "w3": ("F8_E4M3", [INTER, INPUT]), "s3": ("F8_E8M0", [16, 32]),
        "w2": ("F8_E4M3", [4096, INTER]), "s2": ("F8_E8M0", [32, 16]),
    }
    for key, (dtype, shape) in expected.items():
        if infos[key]["dtype"] != dtype or infos[key]["shape"] != shape:
            raise ValueError(f"shared {key}: got {infos[key]['dtype']} {infos[key]['shape']}, expected {dtype} {shape}")

    gate = fp8_linear_rows(data["w1"], data["s1"], INTER, INPUT, x)
    up = fp8_linear_rows(data["w3"], data["s3"], INTER, INPUT, x)
    gate_c, up_c, hidden_bits = [], [], []
    for g, u in zip(gate, up):
        gc = min(g, LIMIT)
        uc = max(-LIMIT, min(LIMIT, u))
        gate_c.append(gc); up_c.append(uc)
        hidden_bits.append(oracle.bf16_bits(oracle.f32(oracle.silu_f32(gc) * uc)))
    hidden = [oracle.bf16_to_f32(b) for b in hidden_bits]
    out = fp8_linear_rows(data["w2"], data["s2"], OUT_ROWS, INTER, hidden)
    out_bits = [oracle.bf16_bits(v) for v in out]

    write_u16(os.path.join(out_dir, "shared-hidden.bf16.bin"), hidden_bits)
    write_f32(os.path.join(out_dir, "shared-gate-clamped.f32.bin"), gate_c)
    write_f32(os.path.join(out_dir, "shared-up-clamped.f32.bin"), up_c)
    write_u16(os.path.join(out_dir, "shared-out8.bf16.bin"), out_bits)

    return out_bits, {
        "output_bf16_hex": [f"0x{x:04x}" for x in out_bits],
        "hidden_sha256": sha256_file(os.path.join(out_dir, "shared-hidden.bf16.bin")),
        "tensors": {k: {
            "name": names[k], "shard": infos[k]["shard"],
            "dtype": infos[k]["dtype"], "shape": infos[k]["shape"],
            "frozen_file": os.path.basename(paths[k]),
            "payload_sha256": hashlib.sha256(data[k]).hexdigest(),
        } for k in names},
    }


def combine(ids, routed_bits, shared_bits, order):
    y = [oracle.f32(0.0) for _ in range(OUT_ROWS)]
    slots = list(range(len(ids)))
    if order == "expert-id":
        slots.sort(key=lambda i: ids[i])
    for slot in slots:
        vals = [oracle.bf16_to_f32(b) for b in routed_bits[slot]]
        for d in range(OUT_ROWS):
            y[d] = oracle.f32(y[d] + vals[d])
    shared = [oracle.bf16_to_f32(b) for b in shared_bits]
    for d in range(OUT_ROWS):
        y[d] = oracle.f32(y[d] + shared[d])
    return [oracle.bf16_bits(v) for v in y]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    router_p, ids, weights, raw_input, x = read_router(repo_root)
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "input.bf16.bin"), "wb") as f:
        f.write(raw_input)

    routed_bits, routed_meta = [], []
    with tempfile.TemporaryDirectory(prefix="gate-f-six-") as td:
        for slot, (expert, weight) in enumerate(zip(ids, weights)):
            work = os.path.join(td, f"e{expert}")
            os.makedirs(work)
            bits, meta = routed_out(args.snapshot, expert, weight, x, work, args.token)
            routed_bits.append(bits); routed_meta.append(meta)
            print(f"routed {slot+1}/6 expert {expert}: " + " ".join(meta["output_bf16_hex"]))

    shared_bits, shared_meta = build_shared(args.snapshot, x, args.out_dir, args.token)
    combined = combine(ids, routed_bits, shared_bits, "expert-id")
    topk_combined = combine(ids, routed_bits, shared_bits, "topk")

    write_u32(os.path.join(args.out_dir, "routed-ids.u32.bin"), ids)
    write_f32(os.path.join(args.out_dir, "routed-weights.f32.bin"), weights)
    write_u16(os.path.join(args.out_dir, "routed-out8.bf16.bin"), [b for row in routed_bits for b in row])
    write_u16(os.path.join(args.out_dir, "combined-out8.bf16.bin"), combined)

    provenance = {
        "schema_version": 1,
        "gate": "Gate F / V4 complete MoE primitive",
        "fixture_class": "F3 pinned official source + real 0731 checkpoint",
        "evidence_state": "CHECKPOINT-VERIFIED_SOURCE-ORACLE",
        "model": router_p["model"], "revision": router_p["revision"],
        "layer": LAYER, "input_sha256": hashlib.sha256(raw_input).hexdigest(),
        "router": {"ids_topk_order": ids, "weights": weights},
        "official_accumulation": {
            "routed_loop_order": sorted(ids),
            "shared_added_after_routed": True,
            "accumulator": "float32 zeros_like",
            "final_cast": "BF16",
            "source": "inference/model.py MoE.forward",
        },
        "routed_experts": routed_meta,
        "shared_expert": shared_meta,
        "expected": {
            "routed_file": "routed-out8.bf16.bin",
            "shared_file": "shared-out8.bf16.bin",
            "combined_file": "combined-out8.bf16.bin",
            "combined_bf16_hex": [f"0x{x:04x}" for x in combined],
            "topk_order_bf16_hex": [f"0x{x:04x}" for x in topk_combined],
            "order_mutation_visible_at_bf16": combined != topk_combined,
        },
        "storage_policy": "All six routed experts were fetched and independently evaluated during fixture generation. Only their compact BF16 outputs/hashes are frozen; full real routed expert 3/2 is already a separate permanent fixture. Full shared w1/w3 + eight w2 rows are frozen because shared FP8 is a distinct arithmetic path.",
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True); f.write("\n")

    print("shared output:", " ".join(shared_meta["output_bf16_hex"]))
    print("combined output:", " ".join(provenance["expected"]["combined_bf16_hex"]))
    print("expert-id order:", sorted(ids))
    print("order mutation visible at BF16:", provenance["expected"]["order_mutation_visible_at_bf16"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
