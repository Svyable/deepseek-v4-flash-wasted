#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a bounded real-checkpoint grouped attention-output fixture.

This is a structural/output-projection sub-seam, not a claim that heads 2..7
come from real attention.  A sparse 8-head group vector is deterministically
seeded from the already-frozen real ratio-0 token-1 heads 0 and 1.  Real
checkpoint wo_a group-0 and wo_b rows are then evaluated independently.

Official source/converter contract:
  checkpoint wo_a E4M3*E8M0 -> dequantize to BF16 -> group einsum
  flatten 8 x 1024 group latents -> quantized wo_b linear
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
import shutil
import struct
import sys

import fetch_hf_tensor_slice as slice_fetch
import make_v2_real_fixture as qref

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
GROUPS = 8
HEADS_PER_GROUP = 8
HEAD_DIM = 512
GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
GROUP_RANK = 1024
FLAT_RANK = GROUPS * GROUP_RANK
OUT_ROWS = 8
BLOCK = 128

WO_A = "layers.0.attn.wo_a.weight"
WO_A_SCALE = "layers.0.attn.wo_a.scale"
WO_B = "layers.0.attn.wo_b.weight"
WO_B_SCALE = "layers.0.attn.wo_b.scale"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 2:
        raise ValueError(f"{path}: odd BF16 byte count")
    return list(struct.unpack(f"<{len(data)//2}H", data))


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def build_group_input(repo_root, out_dir):
    src_path = os.path.join(
        repo_root, "tests", "fixtures", "deepseek_v4", "v5_attn_ratio0_real",
        "attn-after-inverse-rope.bf16.bin")
    src = read_u16(src_path)
    if len(src) != 2 * 2 * HEAD_DIM:
        raise ValueError("ratio-0 attention source fixture shape drifted")
    token1_h0 = src[(1 * 2 + 0) * HEAD_DIM:(1 * 2 + 1) * HEAD_DIM]
    token1_h1 = src[(1 * 2 + 1) * HEAD_DIM:(1 * 2 + 2) * HEAD_DIM]

    # Keep the dot-products reduction-stable while still touching all eight
    # head segments and both non-RoPE/RoPE dimensions. Only eight lanes/head
    # are non-zero; their values are real attention BF16 values.
    lanes = [0, 63, 64, 127, 255, 447, 448, 511]
    bits = [0] * GROUP_IN
    nonzero = []
    for head in range(HEADS_PER_GROUP):
        source = token1_h0 if head % 2 == 0 else token1_h1
        for j, lane in enumerate(lanes):
            src_lane = (lane * 37 + head * 29 + j * 11) % HEAD_DIM
            b = source[src_lane]
            if (head + j) % 3 == 0:
                b ^= 0x8000
            dst = head * HEAD_DIM + lane
            bits[dst] = b
            nonzero.append({
                "head": head, "lane": lane, "source_head": head % 2,
                "source_lane": src_lane, "bf16_hex": f"0x{b:04x}",
            })

    path = os.path.join(out_dir, "group-input.bf16.bin")
    write_u16(path, bits)
    return bits, {
        "file": "group-input.bf16.bin",
        "shape": [HEADS_PER_GROUP, HEAD_DIM],
        "dtype": "BF16",
        "source": "token-1 heads 0/1 from v5_attn_ratio0_real",
        "construction": "8 real-seeded sparse lanes per synthetic head; deterministic sign/permutation transform",
        "nonzero_count": len(nonzero),
        "nonzero": nonzero,
        "sha256": sha256_file(path),
    }


def fetch(snapshot, name, rows, path, token, dtype, shape):
    info = slice_fetch.fetch_slice(snapshot, name, rows, path, token=token)
    if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
        raise ValueError(
            f"{name}: got rev={info['revision']} {info['dtype']} {info['shape']}, "
            f"expected {REV} {dtype} {shape}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return info


def wo_a_projection(group_bits, weight, scale):
    if len(weight) != GROUP_RANK * GROUP_IN:
        raise ValueError("wo_a group slice byte count mismatch")
    if len(scale) != (GROUP_RANK // BLOCK) * (GROUP_IN // BLOCK):
        raise ValueError("wo_a scale-grid byte count mismatch")

    x = [qref.bf16_to_f32(b) for b in group_bits]
    nonzero = [i for i, b in enumerate(group_bits) if (b & 0x7FFF) != 0]
    out_bits = []
    out_values = []
    scale_cols = GROUP_IN // BLOCK

    for r in range(GROUP_RANK):
        seq = qref.f32(0.0)
        rev = qref.f32(0.0)
        exact_sum = Fraction(0, 1)
        row = r * GROUP_IN
        sr = (r // BLOCK) * scale_cols

        def operand(c):
            raw = weight[row + c]
            s = qref.e8m0_float(scale[sr + c // BLOCK])
            # Official converter dequantizes to BF16 before einsum.
            wb = qref.bf16_to_f32(qref.bf16_bits(qref.f32(qref.e4m3_float(raw) * s)))
            return x[c], wb

        for c in nonzero:
            a, b = operand(c)
            seq = qref.f32(seq + qref.f32(a * b))
            exact_sum += Fraction.from_float(a) * Fraction.from_float(b)
        for c in reversed(nonzero):
            a, b = operand(c)
            rev = qref.f32(rev + qref.f32(a * b))

        seq_b = qref.bf16_bits(seq)
        rev_b = qref.bf16_bits(rev)
        exact_b = qref.bf16_bits(float(exact_sum))
        if not (seq_b == rev_b == exact_b):
            raise ValueError(
                f"wo_a row {r} not BF16 reduction-stable: "
                f"seq=0x{seq_b:04x} rev=0x{rev_b:04x} exact=0x{exact_b:04x}")
        out_bits.append(seq_b)
        out_values.append(qref.bf16_to_f32(seq_b))
    return out_values, out_bits


def quantize_vector(values):
    if len(values) % BLOCK:
        raise ValueError("quantized vector width not K128 aligned")
    q = []
    scales = []
    for base in range(0, len(values), BLOCK):
        xb = values[base:base + BLOCK]
        s = qref.next_pow2_scale(max(abs(v) for v in xb))
        scales.append(s)
        for v in xb:
            z = qref.f32(v / s)
            z = max(-448.0, min(448.0, z))
            q.append(qref.e4m3_encode(z))
    return q, scales


def wo_b_projection(flat, weight, scale):
    qx, act_scales = quantize_vector(flat)
    k_blocks = FLAT_RANK // BLOCK
    if len(weight) != OUT_ROWS * FLAT_RANK or len(scale) != k_blocks:
        raise ValueError("wo_b bounded slice shape mismatch")
    bits = []
    values = []
    for r in range(OUT_ROWS):
        seq = qref.f32(0.0)
        rev = qref.f32(0.0)
        exact_sum = Fraction(0, 1)
        row = r * FLAT_RANK
        parts = []
        exact_parts = []
        for b in range(k_blocks):
            base = b * BLOCK
            part = qref.f32(0.0)
            exact_part = Fraction(0, 1)
            for j in range(BLOCK):
                prod = qref.e4m3_fraction(qx[base+j]) * qref.e4m3_fraction(weight[row+base+j])
                part = qref.f32(part + qref.f32(float(prod)))
                exact_part += prod
            s = qref.e8m0_fraction(scale[b]) * Fraction(act_scales[b])
            parts.append(qref.f32(part * float(s)))
            exact_parts.append(exact_part * s)
        for p in parts:
            seq = qref.f32(seq + p)
        for p in reversed(parts):
            rev = qref.f32(rev + p)
        for p in exact_parts:
            exact_sum += p
        sb, rb, eb = qref.bf16_bits(seq), qref.bf16_bits(rev), qref.bf16_bits(float(exact_sum))
        if not (sb == rb == eb):
            raise ValueError(
                f"wo_b row {r} not BF16 reduction-stable: "
                f"seq=0x{sb:04x} rev=0x{rb:04x} exact=0x{eb:04x}")
        bits.append(sb)
        values.append(qref.bf16_to_f32(sb))
    return values, bits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    group_bits, group_meta = build_group_input(repo_root, args.out_dir)
    paths = {
        "wo_a": os.path.join(args.out_dir, "wo-a-group0.fp8.bin"),
        "wo_a_scale": os.path.join(args.out_dir, "wo-a-group0-scale.e8m0.bin"),
        "wo_b": os.path.join(args.out_dir, "wo-b-rows0-8.fp8.bin"),
        "wo_b_scale": os.path.join(args.out_dir, "wo-b-scale-row0.e8m0.bin"),
    }
    infos = {
        "wo_a": fetch(args.snapshot, WO_A, "0:1024", paths["wo_a"], args.token,
                      "F8_E4M3", [8192,4096]),
        "wo_a_scale": fetch(args.snapshot, WO_A_SCALE, "0:8", paths["wo_a_scale"], args.token,
                            "F8_E8M0", [64,32]),
        "wo_b": fetch(args.snapshot, WO_B, "0:8", paths["wo_b"], args.token,
                      "F8_E4M3", [4096,8192]),
        "wo_b_scale": fetch(args.snapshot, WO_B_SCALE, "0:1", paths["wo_b_scale"], args.token,
                            "F8_E8M0", [32,64]),
    }

    with open(paths["wo_a"], "rb") as f: wo_a = f.read()
    with open(paths["wo_a_scale"], "rb") as f: wo_as = f.read()
    with open(paths["wo_b"], "rb") as f: wo_b = f.read()
    with open(paths["wo_b_scale"], "rb") as f: wo_bs = f.read()

    lora_values, lora_bits = wo_a_projection(group_bits, wo_a, wo_as)
    flat = lora_values + [0.0] * (FLAT_RANK - GROUP_RANK)
    out_values, out_bits = wo_b_projection(flat, wo_b, wo_bs)

    lora_path = os.path.join(args.out_dir, "expected-group-lora.bf16.bin")
    out_path = os.path.join(args.out_dir, "expected-out8.bf16.bin")
    write_u16(lora_path, lora_bits)
    write_u16(out_path, out_bits)

    tensor_meta = {}
    for key, info in infos.items():
        tensor_meta[key] = {
            "name": info["tensor"], "shard": info["shard"],
            "dtype": info["dtype"], "shape": info["shape"],
            "rows": [info["row_start"], info["row_end"]],
            "absolute_range": [info["absolute_start"], info["absolute_end"]],
            "payload_bytes": info["payload_bytes"],
            "payload_sha256": info["payload_sha256"],
            "file": os.path.basename(paths[key]),
        }

    provenance = {
        "schema_version": 1,
        "gate": "Gate E / V5 grouped output-projection sub-seam",
        "evidence_state": "CHECKPOINT-WEIGHTS + STRUCTURAL-REAL-SEEDED-INPUT",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REV,
        "layer": 0,
        "group_index": 0,
        "input": group_meta,
        "tensors": tensor_meta,
        "source_contract": {
            "reshape": "8 heads x 512 -> one 4096-value group",
            "wo_a": "checkpoint E4M3*E8M0 block dequant -> BF16, then BF16 einsum [4096] x [1024,4096]",
            "flatten": "place 1024 group latent at group 0 of 8 x 1024; other seven groups zero for this bounded seam",
            "wo_b": "standard K128 activation-E4M3 quantized linear from 8192 to first eight real output rows",
        },
        "oracle": {
            "producer": "tools/make_v5_attention_output_fixture.py independent Python equations",
            "shares_waste_runtime_helpers": False,
            "wo_a_reduction": "sequential/reverse/exact all required to agree at BF16",
            "wo_b_reduction": "sequential/reverse/exact all required to agree at BF16",
        },
        "expected": {
            "group_lora_file": "expected-group-lora.bf16.bin",
            "group_lora_sha256": sha256_file(lora_path),
            "group_lora_first8_hex": [f"0x{x:04x}" for x in lora_bits[:8]],
            "out_file": "expected-out8.bf16.bin",
            "out_sha256": sha256_file(out_path),
            "out8_hex": [f"0x{x:04x}" for x in out_bits],
        },
        "non_claims": [
            "heads 2..7 are structural transforms seeded from real heads 0/1, not independently computed real attention heads",
            "other output groups are zero in the bounded wo_b input",
            "this proves grouped output projection orientation/arithmetic, not full 64-head end-to-end output",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Gate E grouped output projection fixture built")
    print("group lora first8:", " ".join(provenance["expected"]["group_lora_first8_hex"]))
    print("wo_b out8:", " ".join(provenance["expected"]["out8_hex"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
