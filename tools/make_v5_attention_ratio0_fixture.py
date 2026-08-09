#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a real two-token/two-head layer-0 ratio-0 attention fixture.

The fixture deliberately stops at the inverse-RoPE sparse-attention output,
before grouped wo_a/wo_b. Expected values come from tools/v5_attention_oracle.c,
a standalone source-equation implementation with no WASTE runtime linkage.
Before any attention result is accepted, token 0's first eight wq_a outputs
must exactly match the already-frozen Gate-C projection fixture.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

import fetch_hf_tensor_slice as slice_fetch

MODEL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SEQLEN = 2
HEADS = 2
HIDDEN = 4096

TENSORS = {
    "wq_a": ("layers.0.attn.wq_a.weight", None, "F8_E4M3", [1024, 4096], "wq-a.fp8.bin"),
    "wq_a_scale": ("layers.0.attn.wq_a.scale", None, "F8_E8M0", [8, 32], "wq-a-scale.e8m0.bin"),
    "q_norm": ("layers.0.attn.q_norm.weight", None, "BF16", [1024], "q-norm.bf16.bin"),
    "wq_b": ("layers.0.attn.wq_b.weight", "0:1024", "F8_E4M3", [32768, 1024], "wq-b-head0-2.fp8.bin"),
    "wq_b_scale": ("layers.0.attn.wq_b.scale", "0:8", "F8_E8M0", [256, 8], "wq-b-scale-head0-2.e8m0.bin"),
    "wkv": ("layers.0.attn.wkv.weight", None, "F8_E4M3", [512, 4096], "wkv.fp8.bin"),
    "wkv_scale": ("layers.0.attn.wkv.scale", None, "F8_E8M0", [4, 32], "wkv-scale.e8m0.bin"),
    "kv_norm": ("layers.0.attn.kv_norm.weight", None, "BF16", [512], "kv-norm.bf16.bin"),
    "attn_sink": ("layers.0.attn.attn_sink", "0:2", "F32", [64], "attn-sink-head0-2.f32.bin"),
}


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


def build_input(repo_root, out_dir):
    gate_c = os.path.join(
        repo_root, "tests", "fixtures", "deepseek_v4", "v2_wq_a_real")
    token0_path = os.path.join(gate_c, "input.bf16.bin")
    token0 = read_u16(token0_path)
    if len(token0) != HIDDEN:
        raise ValueError("Gate-C anchor input is not one 4096-wide BF16 token")

    # Token 1 is a bit-exact permutation/sign transform of the proven Gate-C
    # input. No floating conversion is involved, so it remains a deterministic
    # finite BF16 vector while producing a genuinely different position-1 Q/KV.
    token1 = []
    for i in range(HIDDEN):
        src = (i * 2053 + 17) % HIDDEN
        bits = token0[src]
        if i % 3 == 0:
            bits ^= 0x8000
        token1.append(bits)

    path = os.path.join(out_dir, "input.bf16.bin")
    write_u16(path, token0 + token1)
    return {
        "file": "input.bf16.bin",
        "dtype": "BF16",
        "shape": [SEQLEN, HIDDEN],
        "sha256": sha256_file(path),
        "token0_source": "tests/fixtures/deepseek_v4/v2_wq_a_real/input.bf16.bin",
        "token1_formula": "token1[i] = token0[(i*2053+17)%4096], sign bit flipped when i%3==0",
    }


def fetch_inputs(snapshot, out_dir, token):
    evidence = {}
    for key, (name, rows, dtype, shape, filename) in TENSORS.items():
        path = os.path.join(out_dir, filename)
        info = slice_fetch.fetch_slice(snapshot, name, rows, path, token=token)
        if info["revision"] != MODEL_REVISION:
            raise ValueError(f"{key}: revision drift {info['revision']}")
        if info["dtype"] != dtype or info["shape"] != shape:
            raise ValueError(
                f"{key}: got {info['dtype']} {info['shape']}, expected {dtype} {shape}")
        evidence[key] = {
            "name": name,
            "shard": info["shard"],
            "dtype": dtype,
            "shape": shape,
            "rows": [info["row_start"], info["row_end"]],
            "absolute_range": [info["absolute_start"], info["absolute_end"]],
            "payload_bytes": info["payload_bytes"],
            "payload_sha256": info["payload_sha256"],
            "file": filename,
        }
        sidecar = path + ".json"
        if os.path.exists(sidecar):
            os.remove(sidecar)
    return evidence


def compile_oracle(repo_root, out_dir):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("no C compiler for independent attention fixture oracle")
    src = os.path.join(repo_root, "tools", "v5_attention_oracle.c")
    exe = os.path.join(out_dir, ".v5_attention_oracle")
    cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           "-ffp-contract=off", src, "-lm", "-o", exe]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("attention oracle compile failed:\n" + p.stdout + p.stderr)
    return exe


def run_oracle(exe, out_dir):
    p = subprocess.run([exe, out_dir], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("attention oracle failed:\n" + p.stdout + p.stderr)
    os.remove(exe)


def bf16_hex(path, start=0, count=8):
    values = read_u16(path)
    return [f"0x{x:04x}" for x in values[start:start+count]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="header-only pinned 0731 snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    input_meta = build_input(repo_root, args.out_dir)
    tensors = fetch_inputs(args.snapshot, args.out_dir, args.token)
    exe = compile_oracle(repo_root, args.out_dir)
    run_oracle(exe, args.out_dir)

    # Mandatory independence anchor: the standalone oracle's quantized wq_a
    # path must reproduce the already-proven Gate C expectation before any
    # larger attention expectation can be trusted.
    gate_c_expected = os.path.join(
        repo_root, "tests", "fixtures", "deepseek_v4", "v2_wq_a_real", "expected.bf16.bin")
    anchor = os.path.join(args.out_dir, "q-a-token0-first8.bf16.bin")
    with open(gate_c_expected, "rb") as f:
        want = f.read()
    with open(anchor, "rb") as f:
        got = f.read()
    if got != want:
        raise ValueError(
            "attention oracle failed Gate-C wq_a anchor: "
            f"got {bf16_hex(anchor)}, want {bf16_hex(gate_c_expected)}")

    outputs = {
        "q_after_rope": ("q-after-rope.bf16.bin", [SEQLEN, HEADS, 512]),
        "kv_after_qat": ("kv-after-qat.bf16.bin", [SEQLEN, 512]),
        "attn_after_inverse_rope": ("attn-after-inverse-rope.bf16.bin", [SEQLEN, HEADS, 512]),
    }
    output_meta = {}
    for key, (filename, shape) in outputs.items():
        path = os.path.join(args.out_dir, filename)
        elements = 1
        for d in shape:
            elements *= d
        if os.path.getsize(path) != elements * 2:
            raise ValueError(f"{filename}: unexpected byte count")
        output_meta[key] = {
            "file": filename,
            "dtype": "BF16",
            "shape": shape,
            "sha256": sha256_file(path),
            "first8_hex": bf16_hex(path),
        }

    # Position-1/head-0 tail sits inside the RoPE dimensions and is a compact
    # human-reviewable signature that the fixture is not only position zero.
    qvals = read_u16(os.path.join(args.out_dir, "q-after-rope.bf16.bin"))
    q_pos1_head0 = (1 * HEADS + 0) * 512
    output_meta["q_after_rope"]["position1_head0_rope_tail8_hex"] = [
        f"0x{x:04x}" for x in qvals[q_pos1_head0+504:q_pos1_head0+512]]
    avals = read_u16(os.path.join(args.out_dir, "attn-after-inverse-rope.bf16.bin"))
    a_pos1_head0 = (1 * HEADS + 0) * 512
    output_meta["attn_after_inverse_rope"]["position1_head0_first8_hex"] = [
        f"0x{x:04x}" for x in avals[a_pos1_head0:a_pos1_head0+8]]

    provenance = {
        "schema_version": 1,
        "gate": "Gate E / V5 ratio-0 sub-seam",
        "evidence_state": "CHECKPOINT-VERIFIED_SOURCE-ORACLE",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": MODEL_REVISION,
        "layer": 0,
        "compress_ratio": 0,
        "scope": "two-token/two-head Attention.forward through inverse-RoPE sparse-attention output; wo_a/wo_b intentionally excluded",
        "input": input_meta,
        "tensors": tensors,
        "source_contract": {
            "q": "wq_a -> learned f32 RMSNorm -> first two heads of wq_b -> direct BF16 q square/mean/rsqrt norm -> base RoPE on last64",
            "kv": "wkv -> learned f32 RMSNorm -> base RoPE last64 -> inplace E4M3 quant/dequant on first448 in K64 blocks",
            "window": "causal sliding window 128; two-token rows [0,-1] and [0,1]",
            "attention": "BF16 q/kv, f32 online softmax, BF16 exp weights for value GEMM, f32 denominator with attn_sink only in denominator, BF16 output",
            "post": "inverse base RoPE on last64",
        },
        "oracle_independence": {
            "producer": "tools/v5_attention_oracle.c",
            "shares_waste_runtime_helpers": False,
            "gate_c_anchor": "token0 first eight wq_a outputs exactly equal tests/fixtures/deepseek_v4/v2_wq_a_real/expected.bf16.bin",
        },
        "outputs": output_meta,
        "non_claims": [
            "does not execute the official TileLang GPU kernels",
            "does not include grouped wo_a/wo_b output projection",
            "does not prove ratio-128 compressed attention",
            "does not prove ratio-4 CSA/indexer attention",
            "therefore does not complete Gate E/V5 by itself",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Gate E ratio-0 real fixture built")
    print("q pos0 head0 first8:", " ".join(output_meta["q_after_rope"]["first8_hex"]))
    print("q pos1 head0 rope tail8:", " ".join(output_meta["q_after_rope"]["position1_head0_rope_tail8_hex"]))
    print("attn pos1 head0 first8:", " ".join(output_meta["attn_after_inverse_rope"]["position1_head0_first8_hex"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
