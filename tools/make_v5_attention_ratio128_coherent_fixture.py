#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build one coherent layer-3 ratio-128 attention fixture from real 0731 bytes.

A single 256-token structural input has exactly two non-zero tokens: positions
0 and 255. The same input drives:

* head-0 Q at position 255;
* local KV at positions 0 and 255 (all other local rows are exactly zero);
* both 128-token compressor chunks;
* row-255 local+compressed sparse attention;
* inverse compressed YaRN on the sparse-attention output.

Expected tensors are produced only by independent Python source equations. The
bounded raw attention checkpoint slices are committed with the fixture so the
full scalar C replay is offline and reproducible. Compressor checkpoint tensors
reuse the already-frozen real ratio-128 compressor fixture.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile

import make_v2_real_fixture as qref
import make_v5_attention_history_real_fixture as hist
import make_v5_compressor_ratio128_fixture as comp

REV = hist.REV
SEQLEN = 256
HIDDEN = hist.HIDDEN
HEAD = hist.HEAD
QUERY_POS = 255
ACTIVE_POS = (0, 255)

RAW_NAMES = {
    "wq_a": "wq-a.fp8.bin",
    "wq_a_scale": "wq-a-scale.e8m0.bin",
    "q_norm": "q-norm.bf16.bin",
    "wq_b": "wq-b-head0.fp8.bin",
    "wq_b_scale": "wq-b-scale-head0.e8m0.bin",
    "wkv": "wkv.fp8.bin",
    "wkv_scale": "wkv-scale.e8m0.bin",
    "kv_norm": "kv-norm.bf16.bin",
    "attn_sink": "attn-sink-head0.f32.bin",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_u16(path, xs):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(xs)}H", *xs))


def write_i32(path, xs):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(xs)}i", *xs))


def build_recipe(anchor):
    recipe = []
    for pos in ACTIVE_POS:
        lane, bits = hist.sparse_token_bits(anchor, pos)
        recipe.append({
            "position": pos,
            "lane": lane,
            "bf16_hex": f"0x{bits:04x}",
            "bits": bits,
        })
    return recipe


def derive_compressor(recipe, compressor_dir):
    wkv = hist.read_u16(os.path.join(compressor_dir, "wkv.bf16.bin"))
    wgate = hist.read_u16(os.path.join(compressor_dir, "wgate.bf16.bin"))
    ape = hist.read_f32(os.path.join(compressor_dir, "ape.f32.bin"))
    norm = [qref.bf16_to_f32(b) for b in hist.read_u16(
        os.path.join(compressor_dir, "norm.bf16.bin"))]
    if len(wkv) != 512 * 4096 or len(wgate) != 512 * 4096:
        raise ValueError("frozen compressor weight shape drifted")

    final = []
    diagnostics = {"pooled": [], "norm": [], "rope": []}
    for chunk, item in enumerate(recipe):
        # chunk 0 active token is local index 0; chunk 1 uses global position
        # 255, therefore local index 127 inside the second 128-token chunk.
        active_t = item["position"] - chunk * comp.RATIO
        if not (0 <= active_t < comp.RATIO):
            raise ValueError("coherent active token is outside its compressor chunk")
        xv = qref.bf16_to_f32(item["bits"])
        kv_active = comp.single_product_rows(wkv, item["lane"], xv)
        gate_active = comp.single_product_rows(wgate, item["lane"], xv)
        probs = comp.softmax_active(gate_active, ape, active_t)
        pooled = [comp.f32(kv_active[d] * probs[d]) for d in range(comp.KV)]
        pooled_bits = [qref.bf16_bits(v) for v in pooled]
        norm_bits = comp.learned_norm(pooled_bits, norm)
        rope_bits = comp.rope(norm_bits, chunk * comp.RATIO)
        final_bits = comp.qat(rope_bits)
        diagnostics["pooled"].extend(pooled_bits)
        diagnostics["norm"].extend(norm_bits)
        diagnostics["rope"].extend(rope_bits)
        final.extend(final_bits)
    return final, diagnostics


def derive_attention(recipe, raw_paths):
    by_pos = {x["position"]: x for x in recipe}
    wq_a = open(raw_paths["wq_a"], "rb").read()
    wq_a_scale = open(raw_paths["wq_a_scale"], "rb").read()
    q_norm = hist.read_u16(raw_paths["q_norm"])
    wq_b = open(raw_paths["wq_b"], "rb").read()
    wq_b_scale = open(raw_paths["wq_b_scale"], "rb").read()
    wkv = open(raw_paths["wkv"], "rb").read()
    wkv_scale = open(raw_paths["wkv_scale"], "rb").read()
    kv_norm = hist.read_u16(raw_paths["kv_norm"])
    sink = hist.read_f32(raw_paths["attn_sink"])[0]

    local = [0] * (SEQLEN * HEAD)
    active_local = []
    for pos in ACTIVE_POS:
        item = by_pos[pos]
        kv0 = hist.quantized_single_product(
            wkv, wkv_scale, HEAD, HIDDEN, item["lane"], item["bits"])
        kv1 = hist.learned_rms(kv0, kv_norm)
        kv2 = hist.rope(kv1, pos)
        kv3 = hist.qat_nope(kv2)
        local[pos * HEAD:(pos + 1) * HEAD] = kv3
        active_local.extend(kv3)

    item = by_pos[QUERY_POS]
    qa0 = hist.quantized_single_product(
        wq_a, wq_a_scale, hist.Q_RANK, HIDDEN, item["lane"], item["bits"])
    qa1 = hist.learned_rms(qa0, q_norm)
    qb = hist.quantized_dense_rows(
        wq_b, wq_b_scale, HEAD, hist.Q_RANK, qa1)
    qn = hist.head_norm(qb)
    q = hist.rope(qn, QUERY_POS)
    return q, local, active_local, sink


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compressor_dir = os.path.join(
        repo, "tests", "fixtures", "deepseek_v4", "v5_compressor_ratio128_real")
    old_history_dir = os.path.join(
        repo, "tests", "fixtures", "deepseek_v4", "v5_attention_history_ratio128_real")
    anchor_path = os.path.join(
        repo, "tests", "fixtures", "deepseek_v4", "v2_wq_a_real", "input.bf16.bin")
    for required in (
        os.path.join(compressor_dir, "provenance.json"),
        os.path.join(compressor_dir, "wkv.bf16.bin"),
        os.path.join(compressor_dir, "wgate.bf16.bin"),
        os.path.join(compressor_dir, "ape.f32.bin"),
        os.path.join(compressor_dir, "norm.bf16.bin"),
        os.path.join(old_history_dir, "provenance.json"),
        anchor_path,
    ):
        if not os.path.exists(required):
            raise ValueError(f"required frozen dependency missing: {required}")

    with open(os.path.join(compressor_dir, "provenance.json"), encoding="utf-8") as f:
        compressor_prov = json.load(f)
    if compressor_prov.get("revision") != REV:
        raise ValueError("compressor dependency revision drifted")

    anchor = hist.read_u16(anchor_path)
    if len(anchor) != HIDDEN:
        raise ValueError("Gate-C input anchor shape drifted")
    recipe = build_recipe(anchor)

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tensor_meta = {}
    raw_paths = {}
    with tempfile.TemporaryDirectory(prefix="v5-coherent-fetch-") as td:
        for key in hist.TENSORS:
            src, meta = hist.fetch(args.snapshot, key, td, args.token)
            dst = os.path.join(args.out_dir, RAW_NAMES[key])
            shutil.copyfile(src, dst)
            raw_paths[key] = dst
            meta = dict(meta)
            meta["file"] = RAW_NAMES[key]
            tensor_meta[key] = meta

        q, local, active_local, sink = derive_attention(recipe, raw_paths)

    compressed, comp_diag = derive_compressor(recipe, compressor_dir)
    attn_pre, idxs = hist.sparse_attention(q, local, compressed, sink)
    attn_post = hist.rope(attn_pre, QUERY_POS, inverse=True)

    # Load-bearing mutations: the new result must depend on the coherent
    # compressor output and on the inverse compressed-YaRN stage.
    old_compressed = hist.read_u16(os.path.join(compressor_dir, "compressed-kv.bf16.bin"))
    mutated_pre, _ = hist.sparse_attention(q, local, old_compressed, sink)
    if mutated_pre == attn_pre:
        raise AssertionError("replacing coherent compressed KV with unrelated frozen KV was invisible")
    if attn_post == attn_pre:
        raise AssertionError("inverse compressed YaRN was invisible")
    pair20 = hist.NOPE + 2 * 20
    if attn_post[pair20:pair20 + 2] == attn_pre[pair20:pair20 + 2]:
        raise AssertionError("inverse compressed YaRN pair20 did not separate at BF16")

    outputs = {
        "q": ("q-pos255-head0.bf16.bin", q, [HEAD]),
        "active_local_kv": ("local-kv-active-0-255.bf16.bin", active_local, [2, HEAD]),
        "compressed_kv": ("compressed-kv-coherent.bf16.bin", compressed, [2, HEAD]),
        "attn_pre_inverse": ("attn-pre-inverse.bf16.bin", attn_pre, [HEAD]),
        "attn_post_inverse": ("attn-post-inverse.bf16.bin", attn_post, [HEAD]),
    }
    output_meta = {}
    for key, (name, bits, shape) in outputs.items():
        path = os.path.join(args.out_dir, name)
        write_u16(path, bits)
        output_meta[key] = {
            "file": name,
            "shape": shape,
            "dtype": "BF16",
            "sha256": sha256_file(path),
            "first8_hex": [f"0x{x:04x}" for x in bits[:8]],
        }
    output_meta["attn_post_inverse"]["pair20_hex"] = [
        f"0x{x:04x}" for x in attn_post[pair20:pair20 + 2]]
    output_meta["attn_pre_inverse"]["pair20_hex"] = [
        f"0x{x:04x}" for x in attn_pre[pair20:pair20 + 2]]

    idx_path = os.path.join(args.out_dir, "topk-row255.i32.bin")
    write_i32(idx_path, idxs)
    output_meta["topk"] = {
        "file": os.path.basename(idx_path),
        "shape": [len(idxs)],
        "dtype": "I32",
        "sha256": sha256_file(idx_path),
        "first8": idxs[:8],
        "last8": idxs[-8:],
    }

    recipe_public = [
        {"position": x["position"], "lane": x["lane"], "bf16_hex": x["bf16_hex"]}
        for x in recipe
    ]
    provenance = {
        "schema_version": 1,
        "gate": "Gate E / V5 coherent same-input ratio-128 attention sub-seam",
        "evidence_state": "CHECKPOINT-WEIGHTS + SAME-STRUCTURAL-INPUT + SOURCE-ORACLE",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REV,
        "layer": 3,
        "compress_ratio": 128,
        "query_position": QUERY_POS,
        "input_recipe": {
            "seqlen": SEQLEN,
            "hidden": HIDDEN,
            "active_tokens": recipe_public,
            "all_other_tokens": "exact BF16 zero",
            "anchor": "tests/fixtures/deepseek_v4/v2_wq_a_real/input.bf16.bin",
            "anchor_sha256": sha256_file(anchor_path),
        },
        "checkpoint_tensors": tensor_meta,
        "compressor_dependency": {
            "fixture": "tests/fixtures/deepseek_v4/v5_compressor_ratio128_real",
            "revision": compressor_prov["revision"],
            "wkv_sha256": sha256_file(os.path.join(compressor_dir, "wkv.bf16.bin")),
            "wgate_sha256": sha256_file(os.path.join(compressor_dir, "wgate.bf16.bin")),
            "ape_sha256": sha256_file(os.path.join(compressor_dir, "ape.f32.bin")),
            "norm_sha256": sha256_file(os.path.join(compressor_dir, "norm.bf16.bin")),
        },
        "source_contract": {
            "q": "same token255 -> real wq_a -> learned RMSNorm -> real head0 wq_b -> direct BF16 head norm -> compressed YaRN position255",
            "local_kv": "same active tokens -> real wkv -> learned RMSNorm -> compressed YaRN by absolute position -> K64 E4M3 QAT first448; zero inputs remain zero",
            "compressor": "same 256-token input -> BF16 wkv/wgate, per-dimension APE softmax pooling, BF16 boundary, learned norm, chunk-position compressed YaRN, K64 QAT",
            "history": "row255 local 128..255 then compressed 256,257; sink sparse attention",
            "inverse": "source inverse compressed YaRN at query position255 before output projection",
        },
        "compressor_diagnostics": {
            "pooled_sha256": hashlib.sha256(struct.pack(f"<{len(comp_diag['pooled'])}H", *comp_diag["pooled"])).hexdigest(),
            "norm_sha256": hashlib.sha256(struct.pack(f"<{len(comp_diag['norm'])}H", *comp_diag["norm"])).hexdigest(),
            "rope_sha256": hashlib.sha256(struct.pack(f"<{len(comp_diag['rope'])}H", *comp_diag["rope"])).hexdigest(),
        },
        "outputs": output_meta,
        "mutations": {
            "replace_coherent_compressed_kv_with_previous_unrelated_fixture": "must change pre-inverse attention",
            "omit_inverse_compressed_yarn": "must change output",
            "inverse_pair20": "must separate at BF16",
        },
        "oracle_independence": {
            "producer": "tools/make_v5_attention_ratio128_coherent_fixture.py using Python source equations",
            "shares_waste_runtime_helpers": False,
        },
        "non_claims": [
            "one head and one query row only",
            "does not attach the already-proven grouped output projection",
            "does not prove ratio-4 CSA/indexer",
            "does not complete Gate E/V5",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Gate E coherent ratio-128 fixture built")
    print("active tokens:", recipe_public)
    print("compressed first8:", " ".join(output_meta["compressed_kv"]["first8_hex"]))
    print("pre-inverse first8:", " ".join(output_meta["attn_pre_inverse"]["first8_hex"]))
    print("post-inverse first8:", " ".join(output_meta["attn_post_inverse"]["first8_hex"]))
    print("pair20 pre/post:", output_meta["attn_pre_inverse"]["pair20_hex"], output_meta["attn_post_inverse"]["pair20_hex"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
