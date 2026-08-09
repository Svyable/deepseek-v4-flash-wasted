#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a real layer-3 ratio-128 compressed-history composition fixture.

This fixture deliberately proves the history-composition seam, not a single
same-input Attention.forward. It combines:

* one real layer-3 head-0 Q derived from official checkpoint wq_a/q_norm/wq_b;
* all 256 real layer-3 local KV entries derived from official wkv/kv_norm;
* the two already-frozen real compressor KV entries from
  v5_compressor_ratio128_real/compressed-kv.bf16.bin;
* real layer-3 head-0 attn_sink;
* the official row-255 prefill index namespace: local 128..255 followed by
  compressed entries 256,257.

Expected values are produced by independent Python equations only. No WASTE C
runtime helper is imported or executed by this generator. Temporary checkpoint
weight slices are deleted after their immutable revision/range/SHA provenance
has been recorded, so the committed fixture stays small.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile

import fetch_hf_tensor_slice as slice_fetch
import make_v2_real_fixture as qref

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SEQLEN = 256
WINDOW = 128
RATIO = 128
HIDDEN = 4096
Q_RANK = 1024
HEAD = 512
ROPE = 64
NOPE = HEAD - ROPE
K128 = 128
K64 = 64
BASE = 160000.0
FACTOR = 16.0
ORIGINAL = 65536
BETA_FAST = 32.0
BETA_SLOW = 1.0
QUERY_POS = 255

TENSORS = {
    "wq_a": ("layers.3.attn.wq_a.weight", None, "F8_E4M3", [1024,4096]),
    "wq_a_scale": ("layers.3.attn.wq_a.scale", None, "F8_E8M0", [8,32]),
    "q_norm": ("layers.3.attn.q_norm.weight", None, "BF16", [1024]),
    "wq_b": ("layers.3.attn.wq_b.weight", "0:512", "F8_E4M3", [32768,1024]),
    "wq_b_scale": ("layers.3.attn.wq_b.scale", "0:4", "F8_E8M0", [256,8]),
    "wkv": ("layers.3.attn.wkv.weight", None, "F8_E4M3", [512,4096]),
    "wkv_scale": ("layers.3.attn.wkv.scale", None, "F8_E8M0", [4,32]),
    "kv_norm": ("layers.3.attn.kv_norm.weight", None, "BF16", [512]),
    "attn_sink": ("layers.3.attn.attn_sink", "0:1", "F32", [64]),
}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16(path):
    data = open(path, "rb").read()
    if len(data) % 2:
        raise ValueError(f"{path}: odd BF16 bytes")
    return list(struct.unpack(f"<{len(data)//2}H", data))


def read_f32(path):
    data = open(path, "rb").read()
    if len(data) % 4:
        raise ValueError(f"{path}: non-f32 byte count")
    return list(struct.unpack(f"<{len(data)//4}f", data))


def write_u16(path, xs):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(xs)}H", *xs))


def write_i32(path, xs):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(xs)}i", *xs))


def fetch(snapshot, key, td, token):
    name, rows, dtype, shape = TENSORS[key]
    path = os.path.join(td, key + ".bin")
    info = slice_fetch.fetch_slice(snapshot, name, rows, path, token=token)
    if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
        raise ValueError(
            f"{name}: got {info['revision']} {info['dtype']} {info['shape']}, "
            f"expected {REV} {dtype} {shape}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    meta = {
        "name": name,
        "shard": info["shard"],
        "dtype": dtype,
        "shape": shape,
        "rows": [info["row_start"], info["row_end"]],
        "absolute_range": [info["absolute_start"], info["absolute_end"]],
        "payload_bytes": info["payload_bytes"],
        "payload_sha256": info["payload_sha256"],
    }
    return path, meta


def sparse_token_bits(anchor, token):
    lane = (token * 2053 + 17) % HIDDEN
    src = (token * 1543 + lane) % HIDDEN
    bits = anchor[src]
    if token % 3 == 1:
        bits ^= 0x8000
    # Do not let the chosen structural lane become exact zero: fall back to
    # the proven nonzero Gate-C lane while preserving the token-dependent sign.
    if (bits & 0x7FFF) == 0:
        bits = anchor[17] ^ (0x8000 if token % 3 == 1 else 0)
    return lane, bits


def quantized_single_product(weight, scales, out_rows, k, lane, x_bf16):
    """FP8 linear for a one-nonzero-lane BF16 input; returns BF16 bits."""
    x = qref.bf16_to_f32(x_bf16)
    act_scale = qref.next_pow2_scale(abs(x))
    qx = qref.e4m3_encode(f32(x / act_scale))
    kb = lane // K128
    out = []
    for r in range(out_rows):
        w = weight[r * k + lane]
        ws = scales[(r // K128) * (k // K128) + kb]
        y = qref.e4m3_float(qx) * qref.e4m3_float(w)
        y = f32(y * f32(act_scale * qref.e8m0_float(ws)))
        out.append(qref.bf16_bits(y))
    return out


def learned_rms(bits, weight_bits):
    x = [qref.bf16_to_f32(b) for b in bits]
    w = [qref.bf16_to_f32(b) for b in weight_bits]
    acc = f32(0.0)
    for v in x:
        acc = f32(acc + f32(v * v))
    mean = f32(acc / len(x))
    inv = f32(1.0 / math.sqrt(f32(mean + 1e-6)))
    return [qref.bf16_bits(f32(f32(v * inv) * w[i])) for i, v in enumerate(x)]


def quantize_dense(bits, block=K128):
    vals = [qref.bf16_to_f32(b) for b in bits]
    q = []
    scales = []
    for base in range(0, len(vals), block):
        xb = vals[base:base+block]
        scale = qref.next_pow2_scale(max(abs(v) for v in xb))
        scales.append(scale)
        for v in xb:
            q.append(qref.e4m3_encode(f32(v / scale)))
    return q, scales


def quantized_dense_rows(weight, weight_scales, rows, k, input_bits):
    qx, act_scales = quantize_dense(input_bits, K128)
    kb_count = k // K128
    out = []
    for r in range(rows):
        acc = f32(0.0)
        for kb in range(kb_count):
            part = f32(0.0)
            start = kb * K128
            for j in range(K128):
                a = qref.e4m3_float(qx[start+j])
                w = qref.e4m3_float(weight[r*k + start+j])
                part = f32(part + f32(a*w))
            ws = qref.e8m0_float(weight_scales[(r // K128) * kb_count + kb])
            acc = f32(acc + f32(part * f32(act_scales[kb] * ws)))
        out.append(qref.bf16_bits(acc))
    return out


def head_norm(bits):
    vals = [qref.bf16_to_f32(b) for b in bits]
    acc = f32(0.0)
    for v in vals:
        sq = qref.bf16_to_f32(qref.bf16_bits(f32(v*v)))
        acc = f32(acc + sq)
    mean = qref.bf16_to_f32(qref.bf16_bits(f32(acc / len(vals))))
    shifted = qref.bf16_to_f32(qref.bf16_bits(f32(mean + 1e-6)))
    inv = qref.bf16_to_f32(qref.bf16_bits(f32(1.0 / math.sqrt(shifted))))
    return [qref.bf16_bits(f32(v * inv)) for v in vals]


def correction_dim(rotations):
    return ROPE * math.log(ORIGINAL / (rotations * 2 * math.pi)) / (2 * math.log(BASE))


def yarn_freq(pair):
    freq = 1.0 / (BASE ** ((2*pair)/ROPE))
    low = max(0.0, math.floor(correction_dim(BETA_FAST)))
    high = min(ROPE-1.0, math.ceil(correction_dim(BETA_SLOW)))
    if low == high:
        high += 0.001
    ramp = min(1.0, max(0.0, (pair-low)/(high-low)))
    smooth = 1.0-ramp
    return f32(freq/FACTOR*(1.0-smooth) + freq*smooth)


def rope(bits, position, inverse=False):
    vals = [qref.bf16_to_f32(b) for b in bits]
    for pair in range(ROPE//2):
        i = NOPE + 2*pair
        angle = f32(position * yarn_freq(pair))
        if inverse:
            angle = -angle
        c = f32(math.cos(angle)); s = f32(math.sin(angle))
        re, im = vals[i], vals[i+1]
        vals[i] = qref.bf16_to_f32(qref.bf16_bits(f32(f32(re*c)-f32(im*s))))
        vals[i+1] = qref.bf16_to_f32(qref.bf16_bits(f32(f32(re*s)+f32(im*c))))
    return [qref.bf16_bits(v) for v in vals]


def qat_nope(bits):
    vals = [qref.bf16_to_f32(b) for b in bits]
    for base in range(0, NOPE, K64):
        block = vals[base:base+K64]
        scale = qref.next_pow2_scale(max(abs(v) for v in block))
        for j, v in enumerate(block):
            code = qref.e4m3_encode(max(-448.0, min(448.0, f32(v/scale))))
            vals[base+j] = qref.bf16_to_f32(qref.bf16_bits(
                f32(qref.e4m3_float(code) * scale)))
    return [qref.bf16_bits(v) for v in vals]


def sparse_attention(q_bits, local_bits, compressed_bits, sink):
    q = [qref.bf16_to_f32(b) for b in q_bits]
    local = [[qref.bf16_to_f32(b) for b in local_bits[t*HEAD:(t+1)*HEAD]] for t in range(SEQLEN)]
    comp = [[qref.bf16_to_f32(b) for b in compressed_bits[t*HEAD:(t+1)*HEAD]] for t in range(2)]
    kv = local + comp
    idxs = list(range(128, 256)) + [256, 257]
    scale = f32(HEAD ** -0.5)
    best = -math.inf
    denom = f32(0.0)
    acc = [f32(0.0)] * HEAD
    for base in range(0, len(idxs), 64):
        block = idxs[base:base+64]
        scores = []
        next_best = best
        for idx in block:
            dot = f32(0.0)
            for a, b in zip(q, kv[idx]):
                dot = f32(dot + f32(a*b))
            score = f32(dot * scale)
            scores.append(score)
            if score > next_best:
                next_best = score
        rescale = f32(math.exp(best-next_best)) if math.isfinite(best) else f32(0.0)
        denom = f32(denom * rescale)
        acc = [f32(v * rescale) for v in acc]
        for idx, score in zip(block, scores):
            e = f32(math.exp(score-next_best))
            denom = f32(denom + e)
            eb = qref.bf16_to_f32(qref.bf16_bits(e))
            row = kv[idx]
            for k in range(HEAD):
                acc[k] = f32(acc[k] + f32(eb * row[k]))
        best = next_best
    denom = f32(denom + f32(math.exp(f32(sink-best))))
    return [qref.bf16_bits(f32(v/denom)) for v in acc], idxs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compressor_dir = os.path.join(repo, "tests", "fixtures", "deepseek_v4", "v5_compressor_ratio128_real")
    compressor_path = os.path.join(compressor_dir, "compressed-kv.bf16.bin")
    compressor_prov = os.path.join(compressor_dir, "provenance.json")
    if not os.path.exists(compressor_path) or not os.path.exists(compressor_prov):
        raise ValueError("real ratio-128 compressor fixture must be frozen first")
    with open(compressor_prov, encoding="utf-8") as f:
        cp = json.load(f)
    if cp.get("revision") != REV:
        raise ValueError("compressor fixture revision drifted")
    compressed_bits = read_u16(compressor_path)
    if len(compressed_bits) != 2*HEAD:
        raise ValueError("compressor KV shape drifted")

    gate_c_anchor = os.path.join(repo, "tests", "fixtures", "deepseek_v4", "v2_wq_a_real", "input.bf16.bin")
    anchor = read_u16(gate_c_anchor)
    if len(anchor) != HIDDEN:
        raise ValueError("Gate-C anchor shape drifted")

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="v5-history-weights-") as td:
        paths = {}; tensor_meta = {}
        for key in TENSORS:
            paths[key], tensor_meta[key] = fetch(args.snapshot, key, td, args.token)

        wq_a = open(paths["wq_a"], "rb").read()
        wq_a_scale = open(paths["wq_a_scale"], "rb").read()
        q_norm = read_u16(paths["q_norm"])
        wq_b = open(paths["wq_b"], "rb").read()
        wq_b_scale = open(paths["wq_b_scale"], "rb").read()
        wkv = open(paths["wkv"], "rb").read()
        wkv_scale = open(paths["wkv_scale"], "rb").read()
        kv_norm = read_u16(paths["kv_norm"])
        sink = read_f32(paths["attn_sink"])[0]

        local_all = []
        input_recipe = []
        query_input = None
        for pos in range(SEQLEN):
            lane, bits = sparse_token_bits(anchor, pos)
            input_recipe.append({"position": pos, "lane": lane, "bf16_hex": f"0x{bits:04x}"})
            kv0 = quantized_single_product(wkv, wkv_scale, HEAD, HIDDEN, lane, bits)
            kv1 = learned_rms(kv0, kv_norm)
            kv2 = rope(kv1, pos)
            kv3 = qat_nope(kv2)
            local_all.extend(kv3)
            if pos == QUERY_POS:
                query_input = (lane, bits)

        lane, bits = query_input
        qa0 = quantized_single_product(wq_a, wq_a_scale, Q_RANK, HIDDEN, lane, bits)
        qa1 = learned_rms(qa0, q_norm)
        qb = quantized_dense_rows(wq_b, wq_b_scale, HEAD, Q_RANK, qa1)
        qn = head_norm(qb)
        q = rope(qn, QUERY_POS)

        expected, idxs = sparse_attention(q, local_all, compressed_bits, sink)

    q_path = os.path.join(args.out_dir, "q-pos255-head0.bf16.bin")
    local_path = os.path.join(args.out_dir, "local-kv-0-255.bf16.bin")
    sink_path = os.path.join(args.out_dir, "attn-sink-head0.f32.bin")
    idx_path = os.path.join(args.out_dir, "topk-row255.i32.bin")
    exp_path = os.path.join(args.out_dir, "expected-attn-row255.bf16.bin")
    write_u16(q_path, q); write_u16(local_path, local_all); write_i32(idx_path, idxs); write_u16(exp_path, expected)
    with open(sink_path, "wb") as f:
        f.write(struct.pack("<f", sink))

    prov = {
        "schema_version": 1,
        "gate": "Gate E / V5 ratio-128 compressed-history composition sub-seam",
        "evidence_state": "CHECKPOINT-WEIGHTS + STRUCTURAL-SPARSE-INPUT + FROZEN-REAL-COMPRESSOR + SOURCE-ORACLE",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REV,
        "layer": 3,
        "compress_ratio": RATIO,
        "query_position": QUERY_POS,
        "source_contract": {
            "q": "real layer3 wq_a -> learned RMSNorm -> head0 wq_b -> direct BF16 head norm -> compressed YaRN at position255",
            "local_kv": "real layer3 wkv -> learned RMSNorm -> compressed YaRN per position -> K64 E4M3 QAT first448",
            "history_namespace": "prefill sparse-attn KV is [all 256 local KV, 2 compressed KV]; row255 indices local128..255 then compressed256,257",
            "attention": "BF16 q/kv, f32 block-online softmax, BF16 exp weights for value GEMM, f32 denominator plus real attn_sink",
        },
        "compressor_dependency": {
            "path": "tests/fixtures/deepseek_v4/v5_compressor_ratio128_real/compressed-kv.bf16.bin",
            "sha256": sha256_file(compressor_path),
            "expected_sha256": cp["outputs"]["compressed_kv"]["sha256"],
        },
        "checkpoint_tensors": tensor_meta,
        "structural_input": {
            "source": "Gate-C frozen BF16 input bits",
            "formula": "one nonzero lane per position; lane=(position*2053+17)%4096, source=(position*1543+lane)%4096, sign flip when position%3==1",
            "query_position_recipe": input_recipe[QUERY_POS],
            "all_positions_recipe_sha256": hashlib.sha256(json.dumps(input_recipe, sort_keys=True).encode()).hexdigest(),
        },
        "outputs": {
            "q": {"file": os.path.basename(q_path), "shape": [512], "dtype": "BF16", "sha256": sha256_file(q_path), "first8_hex": [f"0x{x:04x}" for x in q[:8]]},
            "local_kv": {"file": os.path.basename(local_path), "shape": [256,512], "dtype": "BF16", "sha256": sha256_file(local_path), "row255_first8_hex": [f"0x{x:04x}" for x in local_all[255*HEAD:255*HEAD+8]]},
            "attn_sink": {"file": os.path.basename(sink_path), "shape": [1], "dtype": "F32", "sha256": sha256_file(sink_path), "value": sink},
            "topk": {"file": os.path.basename(idx_path), "shape": [130], "dtype": "I32", "sha256": sha256_file(idx_path), "first8": idxs[:8], "last8": idxs[-8:]},
            "attention": {"file": os.path.basename(exp_path), "shape": [512], "dtype": "BF16", "sha256": sha256_file(exp_path), "first8_hex": [f"0x{x:04x}" for x in expected[:8]]},
        },
        "oracle_independence": {
            "producer": "tools/make_v5_attention_history_real_fixture.py independent Python equations",
            "shares_waste_runtime_helpers": False,
        },
        "non_claims": [
            "compressed KV comes from the independently frozen compressor fixture and not from the same structural token inputs used for q/local KV",
            "therefore this proves cache/index/sparse-attention composition, not a single same-input Attention.forward",
            "does not include inverse-RoPE or grouped output projection",
            "does not prove ratio-4 CSA/indexer",
            "does not complete Gate E/V5",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True); f.write("\n")

    print("Gate E ratio-128 real history fixture built")
    print("q pos255 first8:", " ".join(prov["outputs"]["q"]["first8_hex"]))
    print("local kv pos255 first8:", " ".join(prov["outputs"]["local_kv"]["row255_first8_hex"]))
    print("topk tail:", prov["outputs"]["topk"]["last8"])
    print("attn row255 first8:", " ".join(prov["outputs"]["attention"]["first8_hex"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())