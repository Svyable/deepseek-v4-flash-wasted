#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the first real layer-2 ratio-4 CSA/indexer fixture.

A single 8-token structural input has two non-zero BF16 tokens, positions 3 and
7. Position 3 is deliberately the last token of chunk 0: its indexer-compressor
FIRST half must become chunk 1's overlap half. Position 7 contributes chunk 1's
current SECOND half. The fixture therefore makes an "ignore overlap" mutation
observable while also producing a non-zero query at row 7.

The same row-7 token drives real layer-2 wq_a/q_norm, real indexer wq_b, and
real BF16 weights_proj. The indexer compressor uses its real BF16 wkv/wgate,
F32 APE and BF16 norm. Expected values come from independent Python source
equations; no WASTE C runtime helper is imported or executed here.

This is intentionally an Indexer sub-seam. The main 512-wide ratio-4 attention
compressor and final sparse attention are not claimed here. A separate
model-free test already proves the top-512 cutoff with 520 candidates; this
short real fixture proves checkpoint-backed score/ranking for two visible
compressed entries and rejects ties.
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
import make_v5_attention_history_real_fixture as hist

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
LAYER = 2
SEQLEN = 8
RATIO = 4
HIDDEN = 4096
Q_RANK = 1024
INDEX_HEADS = 64
INDEX_DIM = 128
ROPE = 64
ACTIVE_POS = (3, 7)
QUERY_POS = 7

TENSORS = {
    "wq_a": ("layers.2.attn.wq_a.weight", None, "F8_E4M3", [1024,4096], "wq-a.fp8.bin"),
    "wq_a_scale": ("layers.2.attn.wq_a.scale", None, "F8_E8M0", [8,32], "wq-a-scale.e8m0.bin"),
    "q_norm": ("layers.2.attn.q_norm.weight", None, "BF16", [1024], "q-norm.bf16.bin"),
    "index_wq_b": ("layers.2.attn.indexer.wq_b.weight", None, "F8_E4M3", [8192,1024], "index-wq-b.fp8.bin"),
    "index_wq_b_scale": ("layers.2.attn.indexer.wq_b.scale", None, "F8_E8M0", [64,8], "index-wq-b-scale.e8m0.bin"),
    "weights_proj": ("layers.2.attn.indexer.weights_proj.weight", None, "BF16", [64,4096], "weights-proj.bf16.bin"),
    "comp_wkv": ("layers.2.attn.indexer.compressor.wkv.weight", None, "BF16", [256,4096], "index-comp-wkv.bf16.bin"),
    "comp_wgate": ("layers.2.attn.indexer.compressor.wgate.weight", None, "BF16", [256,4096], "index-comp-wgate.bf16.bin"),
    "comp_ape": ("layers.2.attn.indexer.compressor.ape", None, "F32", [4,256], "index-comp-ape.f32.bin"),
    "comp_norm": ("layers.2.attn.indexer.compressor.norm.weight", None, "BF16", [128], "index-comp-norm.bf16.bin"),
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
    return list(struct.unpack(f"<{len(data)//2}H", data))


def read_f32(path):
    data = open(path, "rb").read()
    return list(struct.unpack(f"<{len(data)//4}f", data))


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_i32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}i", *values))


def fetch(snapshot, key, out_dir, token):
    name, rows, dtype, shape, filename = TENSORS[key]
    path = os.path.join(out_dir, filename)
    info = slice_fetch.fetch_slice(snapshot, name, rows, path, token=token)
    if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
        raise ValueError(
            f"{name}: got {info['revision']} {info['dtype']} {info['shape']}, "
            f"expected {REV} {dtype} {shape}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return path, {
        "name": name,
        "shard": info["shard"],
        "dtype": dtype,
        "shape": shape,
        "absolute_range": [info["absolute_start"], info["absolute_end"]],
        "payload_bytes": info["payload_bytes"],
        "payload_sha256": info["payload_sha256"],
        "file": filename,
    }


def build_recipe(anchor):
    recipe = []
    for pos in ACTIVE_POS:
        lane, bits = hist.sparse_token_bits(anchor, pos)
        recipe.append({"position": pos, "lane": lane, "bits": bits,
                       "bf16_hex": f"0x{bits:04x}"})
    return recipe


def learned_rms_stable(bits, weight_bits):
    vals = [qref.bf16_to_f32(b) for b in bits]
    weights = [qref.bf16_to_f32(b) for b in weight_bits]

    def reduce(order):
        acc = f32(0.0)
        for i in order:
            acc = f32(acc + f32(vals[i] * vals[i]))
        mean = f32(acc / len(vals))
        inv = f32(1.0 / math.sqrt(f32(mean + 1e-6)))
        return [qref.bf16_bits(f32(f32(v * inv) * weights[i]))
                for i, v in enumerate(vals)]

    seq = reduce(range(len(vals)))
    rev = reduce(reversed(range(len(vals))))
    if seq != rev:
        first = next(i for i, (a,b) in enumerate(zip(seq, rev)) if a != b)
        raise ValueError(f"RMS reduction order reaches BF16 at lane {first}: 0x{seq[first]:04x} vs 0x{rev[first]:04x}")
    return seq


def quantized_dense_rows_stable(weight, weight_scales, rows, k, input_bits):
    qx, act_scales = hist.quantize_dense(input_bits, 128)
    kb_count = k // 128

    def row_value(r, reverse=False):
        acc = f32(0.0)
        blocks = range(kb_count - 1, -1, -1) if reverse else range(kb_count)
        for kb in blocks:
            part = f32(0.0)
            lanes = range(127, -1, -1) if reverse else range(128)
            base = kb * 128
            for lane in lanes:
                a = qref.e4m3_float(qx[base + lane])
                w = qref.e4m3_float(weight[r*k + base + lane])
                part = f32(part + f32(a*w))
            ws = qref.e8m0_float(weight_scales[(r // 128) * kb_count + kb])
            acc = f32(acc + f32(part * f32(act_scales[kb] * ws)))
        return qref.bf16_bits(acc)

    seq = []
    for r in range(rows):
        a = row_value(r, False)
        b = row_value(r, True)
        if a != b:
            raise ValueError(f"indexer wq_b reduction order reaches BF16 at row {r}: 0x{a:04x} vs 0x{b:04x}")
        seq.append(a)
    return seq


def yarn_freq(pair):
    base = 160000.0
    original = 65536
    factor = 16.0
    beta_fast = 32.0
    beta_slow = 1.0
    freq = 1.0 / (base ** ((2 * pair) / ROPE))
    def corr(rot):
        return ROPE * math.log(original / (rot * 2 * math.pi)) / (2 * math.log(base))
    low = max(0.0, math.floor(corr(beta_fast)))
    high = min(ROPE - 1.0, math.ceil(corr(beta_slow)))
    if low == high:
        high += 0.001
    ramp = min(1.0, max(0.0, (pair - low) / (high - low)))
    smooth = 1.0 - ramp
    return f32(freq / factor * (1.0 - smooth) + freq * smooth)


def rope128(bits, position):
    vals = [qref.bf16_to_f32(b) for b in bits]
    for pair in range(ROPE // 2):
        i = INDEX_DIM - ROPE + 2 * pair
        angle = f32(position * yarn_freq(pair))
        c = f32(math.cos(angle)); s = f32(math.sin(angle))
        re, im = vals[i], vals[i+1]
        vals[i] = qref.bf16_to_f32(qref.bf16_bits(f32(f32(re*c) - f32(im*s))))
        vals[i+1] = qref.bf16_to_f32(qref.bf16_bits(f32(f32(re*s) + f32(im*c))))
    return [qref.bf16_bits(v) for v in vals]


def hadamard_direct(bits):
    vals = [qref.bf16_to_f32(b) for b in bits]
    n = len(vals)
    scale = f32(1.0 / math.sqrt(n))
    out = []
    for row in range(n):
        acc = f32(0.0)
        for col, value in enumerate(vals):
            sign = -1.0 if ((row & col).bit_count() & 1) else 1.0
            acc = f32(acc + f32(sign * value))
        out.append(qref.bf16_bits(f32(acc * scale)))
    return out


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_encode(value):
    negative = math.copysign(1.0, value) < 0.0
    a = min(6.0, abs(f32(value)))
    best = 0; best_err = abs(a - E2M1[0])
    for code in range(1, 8):
        err = abs(a - E2M1[code])
        if err < best_err or (err == best_err and code % 2 == 0 and best % 2 == 1):
            best, best_err = code, err
    return best | (8 if negative else 0)


def e2m1_decode(code):
    v = E2M1[code & 7]
    return -v if code & 8 else v


def ceil_pow2_ratio(value, divisor):
    ratio = f32(f32(value) * f32(1.0 / divisor))
    bits = struct.unpack("<I", struct.pack("<f", ratio))[0]
    exp = ((bits >> 23) & 0xFF) - 127
    if bits & 0x7FFFFF:
        exp += 1
    return math.ldexp(1.0, exp)


def fp4_sim(bits, block=32):
    vals = [qref.bf16_to_f32(b) for b in bits]
    for base in range(0, len(vals), block):
        xb = vals[base:base+block]
        amax = max(max(abs(v) for v in xb), math.ldexp(6.0, -126))
        scale = ceil_pow2_ratio(amax, 6.0)
        for j, v in enumerate(xb):
            z = max(-6.0, min(6.0, f32(v / scale)))
            code = e2m1_encode(z)
            vals[base+j] = qref.bf16_to_f32(qref.bf16_bits(f32(e2m1_decode(code) * scale)))
    return [qref.bf16_bits(v) for v in vals]


def single_product_rows(weight_bits, lane, x_bits, rows, cols):
    x = qref.bf16_to_f32(x_bits)
    out = []
    for r in range(rows):
        w = qref.bf16_to_f32(weight_bits[r*cols + lane])
        out.append(f32(x * w))
    return out


def overlap_pool(kv, score, ape, reverse=False, ignore_overlap=False):
    chunks = 2
    d = INDEX_DIM
    two_d = 2*d
    order = range(7, -1, -1) if reverse else range(8)
    out = []
    for c in range(chunks):
        for j in range(d):
            logits = [0.0] * 8
            values = [0.0] * 8
            valid = [True] * 8
            for t in range(4):
                if c == 0 or ignore_overlap:
                    valid[t] = False
                    logits[t] = -math.inf
                else:
                    p = ((c-1)*4+t)*two_d+j
                    logits[t] = f32(score[p] + ape[t*two_d+j])
                    values[t] = kv[p]
                p = (c*4+t)*two_d+d+j
                logits[4+t] = f32(score[p] + ape[t*two_d+d+j])
                values[4+t] = kv[p]
            m = max(z for z in logits if math.isfinite(z))
            den = f32(0.0); num = f32(0.0)
            for t in order:
                if not valid[t] or not math.isfinite(logits[t]):
                    continue
                e = f32(math.exp(f32(logits[t] - m)))
                den = f32(den + e)
                num = f32(num + f32(values[t] * e))
            out.append(qref.bf16_bits(f32(num / den)))
    return out


def indexer_compressor(recipe, wkv_bits, wgate_bits, ape, norm_bits):
    two_d = 256
    kv = [0.0] * (SEQLEN * two_d)
    score = [0.0] * (SEQLEN * two_d)
    for item in recipe:
        rows_kv = single_product_rows(wkv_bits, item["lane"], item["bits"], two_d, HIDDEN)
        rows_sc = single_product_rows(wgate_bits, item["lane"], item["bits"], two_d, HIDDEN)
        base = item["position"] * two_d
        kv[base:base+two_d] = rows_kv
        score[base:base+two_d] = rows_sc
    pooled = overlap_pool(kv, score, ape, False, False)
    pooled_rev = overlap_pool(kv, score, ape, True, False)
    if pooled != pooled_rev:
        first = next(i for i,(a,b) in enumerate(zip(pooled, pooled_rev)) if a != b)
        raise ValueError(f"overlap softmax reduction order reaches BF16 at {first}")
    wrong = overlap_pool(kv, score, ape, False, True)
    if pooled[INDEX_DIM:2*INDEX_DIM] == wrong[INDEX_DIM:2*INDEX_DIM]:
        raise ValueError("real overlap fixture failed to witness previous-half contribution")

    final = []; norms = []; ropes = []
    for chunk in range(2):
        p = pooled[chunk*INDEX_DIM:(chunk+1)*INDEX_DIM]
        n = learned_rms_stable(p, norm_bits)
        r = rope128(n, chunk*RATIO)
        h = hadamard_direct(r)
        q = fp4_sim(h, 32)
        norms.extend(n); ropes.extend(r); final.extend(q)
    return pooled, norms, ropes, final


def indexer_query(qr_bits, wq_b, wq_b_scale):
    linear = quantized_dense_rows_stable(wq_b, wq_b_scale, INDEX_HEADS*INDEX_DIM, Q_RANK, qr_bits)
    out = []
    for h in range(INDEX_HEADS):
        row = linear[h*INDEX_DIM:(h+1)*INDEX_DIM]
        row = rope128(row, QUERY_POS)
        row = hadamard_direct(row)
        row = fp4_sim(row, 32)
        out.extend(row)
    return out


def weights_from_sparse(x_item, weights_bits):
    x = qref.bf16_to_f32(x_item["bits"])
    scale = f32(1.0 / math.sqrt(INDEX_DIM * INDEX_HEADS))
    out = []
    for h in range(INDEX_HEADS):
        w = qref.bf16_to_f32(weights_bits[h*HIDDEN + x_item["lane"]])
        linear = qref.bf16_to_f32(qref.bf16_bits(f32(x*w)))
        out.append(qref.bf16_bits(f32(linear * scale)))
    return out


def index_scores_stable(q_bits, kv_bits, weights_bits):
    q = [qref.bf16_to_f32(b) for b in q_bits]
    kv = [qref.bf16_to_f32(b) for b in kv_bits]
    weights = [qref.bf16_to_f32(b) for b in weights_bits]

    def one(t, reverse=False):
        products = []
        heads = range(INDEX_HEADS-1, -1, -1) if reverse else range(INDEX_HEADS)
        for h in heads:
            dot = f32(0.0)
            dims = range(INDEX_DIM-1, -1, -1) if reverse else range(INDEX_DIM)
            for d in dims:
                dot = f32(dot + f32(q[h*INDEX_DIM+d] * kv[t*INDEX_DIM+d]))
            dot = qref.bf16_to_f32(qref.bf16_bits(dot))
            if dot < 0.0:
                dot = 0.0
            product = qref.bf16_to_f32(qref.bf16_bits(f32(dot * weights[h])))
            products.append(product)
        acc = f32(0.0)
        for p in products:
            acc = f32(acc + p)
        return qref.bf16_bits(acc)

    seq = []; rev = []
    for t in range(2):
        seq.append(one(t, False)); rev.append(one(t, True))
    if seq != rev:
        raise ValueError(f"index score reduction order reaches BF16: {seq} vs {rev}")
    if seq[0] == seq[1]:
        raise ValueError(f"real index scores tie at BF16: 0x{seq[0]:04x}")
    return seq


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inv_path = os.path.join(repo, "reference", "deepseek-v4-flash-0731.layer2-csa.json")
    with open(inv_path, encoding="utf-8") as f:
        inventory = json.load(f)
    if inventory.get("revision") != REV or inventory.get("layer") != 2 or inventory.get("compress_ratio") != 4:
        raise ValueError("frozen layer-2 CSA inventory drifted")

    anchor_path = os.path.join(repo, "tests", "fixtures", "deepseek_v4", "v2_wq_a_real", "input.bf16.bin")
    anchor = read_u16(anchor_path)
    if len(anchor) != HIDDEN:
        raise ValueError("Gate-C anchor input shape drifted")
    recipe = build_recipe(anchor)
    by_pos = {x["position"]: x for x in recipe}

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {}; tensor_meta = {}
    for key in TENSORS:
        paths[key], tensor_meta[key] = fetch(args.snapshot, key, args.out_dir, args.token)

    wq_a = open(paths["wq_a"], "rb").read()
    wq_a_scale = open(paths["wq_a_scale"], "rb").read()
    q_norm = read_u16(paths["q_norm"])
    index_wq_b = open(paths["index_wq_b"], "rb").read()
    index_wq_b_scale = open(paths["index_wq_b_scale"], "rb").read()
    weights_proj = read_u16(paths["weights_proj"])
    comp_wkv = read_u16(paths["comp_wkv"])
    comp_wgate = read_u16(paths["comp_wgate"])
    comp_ape = read_f32(paths["comp_ape"])
    comp_norm = read_u16(paths["comp_norm"])

    qitem = by_pos[QUERY_POS]
    qr0 = hist.quantized_single_product(wq_a, wq_a_scale, Q_RANK, HIDDEN, qitem["lane"], qitem["bits"])
    qr = learned_rms_stable(qr0, q_norm)
    index_q = indexer_query(qr, index_wq_b, index_wq_b_scale)
    pooled, comp_norm_out, comp_rope, comp_kv = indexer_compressor(
        recipe, comp_wkv, comp_wgate, comp_ape, comp_norm)
    weights = weights_from_sparse(qitem, weights_proj)
    scores = index_scores_stable(index_q, comp_kv, weights)
    score_values = [qref.bf16_to_f32(b) for b in scores]
    order = sorted(range(2), key=lambda i: (-score_values[i], i))
    topk = [SEQLEN + i for i in order]

    files = {
        "qr": ("qr-pos7.bf16.bin", qr, [Q_RANK]),
        "index_q": ("index-q-pos7.bf16.bin", index_q, [INDEX_HEADS,INDEX_DIM]),
        "comp_pooled": ("index-comp-pooled.bf16.bin", pooled, [2,INDEX_DIM]),
        "comp_norm": ("index-comp-post-norm.bf16.bin", comp_norm_out, [2,INDEX_DIM]),
        "comp_rope": ("index-comp-post-rope.bf16.bin", comp_rope, [2,INDEX_DIM]),
        "comp_kv": ("index-comp-kv.bf16.bin", comp_kv, [2,INDEX_DIM]),
        "weights": ("index-weights.bf16.bin", weights, [INDEX_HEADS]),
        "scores": ("index-scores.bf16.bin", scores, [2]),
    }
    outmeta = {}
    for key, (name, bits, shape) in files.items():
        path = os.path.join(args.out_dir, name)
        write_u16(path, bits)
        outmeta[key] = {
            "file": name, "shape": shape, "dtype": "BF16",
            "sha256": sha256_file(path),
            "first8_hex": [f"0x{x:04x}" for x in bits[:8]],
        }
    topk_path = os.path.join(args.out_dir, "topk-row7.i32.bin")
    write_i32(topk_path, topk)
    outmeta["topk"] = {
        "file": os.path.basename(topk_path), "shape": [2], "dtype": "I32",
        "sha256": sha256_file(topk_path), "values": topk,
    }

    provenance = {
        "schema_version": 1,
        "gate": "Gate E / V5 ratio-4 CSA indexer sub-seam",
        "evidence_state": "CHECKPOINT-WEIGHTS + STRUCTURAL-SPARSE-INPUT + SOURCE-ORACLE",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REV,
        "layer": LAYER,
        "compress_ratio": RATIO,
        "query_position": QUERY_POS,
        "input_recipe": {
            "seqlen": SEQLEN,
            "active_tokens": [
                {"position": x["position"], "lane": x["lane"], "bf16_hex": x["bf16_hex"]}
                for x in recipe
            ],
            "all_other_tokens": "exact BF16 zero",
            "purpose": "position3 exercises chunk1 previous first-half overlap; position7 exercises chunk1 current second-half and query",
            "anchor": "tests/fixtures/deepseek_v4/v2_wq_a_real/input.bf16.bin",
            "anchor_sha256": sha256_file(anchor_path),
        },
        "inventory_dependency": {
            "file": "reference/deepseek-v4-flash-0731.layer2-csa.json",
            "sha256": sha256_file(inv_path),
        },
        "checkpoint_tensors": tensor_meta,
        "source_contract": {
            "qr": "real layer2 wq_a FP8/E8M0 -> learned q_norm",
            "index_q": "real indexer wq_b FP8/E8M0 -> 64x128 -> compressed YaRN -> normalized Hadamard -> K32 FP4 simulation",
            "index_compressor": "real BF16 wkv/wgate + F32 APE -> ratio4 previous-first/current-second overlap pool -> BF16 -> learned norm -> compressed YaRN -> Hadamard -> K32 FP4 simulation",
            "weights": "real BF16 weights_proj(x7) * 1/sqrt(128*64), BF16-visible",
            "scores": "BF16 q/kv einsum -> ReLU -> BF16 per-head weight product -> head sum -> BF16",
            "topk": "row7 has two visible compressed entries; real BF16 scores must be non-tied and rank to offset 8",
        },
        "stability_guards": {
            "q_norm": "forward/reverse f32 sum must give identical BF16 output",
            "index_wq_b": "forward/reverse K128 and lane reductions must agree for all 8192 BF16 outputs",
            "overlap_pool": "forward/reverse eight-slot softmax reductions must agree at BF16",
            "index_scores": "forward/reverse head/dim reductions must agree at BF16",
            "real_score_tie": "fixture rejected if the two real scores tie at BF16",
        },
        "mutations": {
            "ignore_previous_overlap_half": "must change chunk1 pooled BF16",
            "top512_cutoff": "covered separately by model-free 520-candidate fixture",
            "causal_mask_before_topk": "covered separately by model-free future-higher-score fixture",
        },
        "outputs": outmeta,
        "oracle_independence": {
            "producer": "tools/make_v5_csa_indexer_real_fixture.py independent Python source equations",
            "shares_waste_runtime_helpers": False,
            "hadamard": "direct Walsh matrix (-1)^popcount(row&col), not C butterfly",
        },
        "non_claims": [
            "short real fixture has two compressed candidates and therefore does not itself exercise the 512 cutoff",
            "main 512-wide ratio-4 attention compressor is not included",
            "final CSA-selected sparse attention is not included",
            "does not complete Gate E/V5",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True); f.write("\n")

    print("Gate E ratio-4 CSA real indexer fixture built")
    print("active:", provenance["input_recipe"]["active_tokens"])
    print("index q first8:", " ".join(outmeta["index_q"]["first8_hex"]))
    print("index comp chunk1 first8:", " ".join(f"0x{x:04x}" for x in comp_kv[INDEX_DIM:INDEX_DIM+8]))
    print("scores:", [f"0x{x:04x}" for x in scores], [qref.bf16_to_f32(x) for x in scores])
    print("topk:", topk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
