#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a coherent real layer-2 ratio-4 CSA attention fixture.

This fixture consumes the already-frozen real CSA Indexer result as an immutable
selection dependency and proves the remaining main-attention seam on the same
8-token structural input:

* main head-0 Q from the real layer-2 q-rank;
* local KV from the real quantized wkv path;
* the 512-wide ratio-4 overlapping main compressor;
* local window indices followed by the real Indexer top-k [compressed entries];
* sink sparse attention and inverse compressed YaRN.

The structural input and q-rank are dependency-hashed from
`v5_csa_indexer_real`, so Indexer and main attention are tied to one input
recipe without duplicating the 17 MB Indexer payload. Expected values use
independent Python source equations only; no WASTE C runtime helper is invoked.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys

import fetch_hf_tensor_slice as slice_fetch
import make_v2_real_fixture as qref
import make_v5_attention_history_real_fixture as hist
import make_v5_csa_indexer_real_fixture as csaidx

REV = csaidx.REV
SEQLEN = 8
WINDOW = 128
HIDDEN = 4096
Q_RANK = 1024
HEAD = 512
QUERY_POS = 7
RATIO = 4

TENSORS = {
    "wq_b": ("layers.2.attn.wq_b.weight", "0:512", "F8_E4M3", [32768,1024], "main-wq-b-head0.fp8.bin"),
    "wq_b_scale": ("layers.2.attn.wq_b.scale", "0:4", "F8_E8M0", [256,8], "main-wq-b-scale-head0.e8m0.bin"),
    "wkv": ("layers.2.attn.wkv.weight", None, "F8_E4M3", [512,4096], "main-wkv.fp8.bin"),
    "wkv_scale": ("layers.2.attn.wkv.scale", None, "F8_E8M0", [4,32], "main-wkv-scale.e8m0.bin"),
    "kv_norm": ("layers.2.attn.kv_norm.weight", None, "BF16", [512], "main-kv-norm.bf16.bin"),
    "attn_sink": ("layers.2.attn.attn_sink", "0:1", "F32", [64], "attn-sink-head0.f32.bin"),
    "comp_wkv": ("layers.2.attn.compressor.wkv.weight", None, "BF16", [1024,4096], "main-comp-wkv.bf16.bin"),
    "comp_wgate": ("layers.2.attn.compressor.wgate.weight", None, "BF16", [1024,4096], "main-comp-wgate.bf16.bin"),
    "comp_ape": ("layers.2.attn.compressor.ape", None, "F32", [4,1024], "main-comp-ape.f32.bin"),
    "comp_norm": ("layers.2.attn.compressor.norm.weight", None, "BF16", [512], "main-comp-norm.bf16.bin"),
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


def read_i32(path):
    data = open(path, "rb").read()
    return list(struct.unpack(f"<{len(data)//4}i", data))


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
        "rows": [info["row_start"], info["row_end"]],
        "absolute_range": [info["absolute_start"], info["absolute_end"]],
        "payload_bytes": info["payload_bytes"],
        "payload_sha256": info["payload_sha256"],
        "file": filename,
    }


def quantized_dense_rows_stable(weight, weight_scales, rows, k, input_bits):
    qx, act_scales = hist.quantize_dense(input_bits, 128)
    kb_count = k // 128

    def one(r, reverse=False):
        acc = f32(0.0)
        blocks = range(kb_count-1, -1, -1) if reverse else range(kb_count)
        for kb in blocks:
            part = f32(0.0)
            lanes = range(127, -1, -1) if reverse else range(128)
            base = kb * 128
            for lane in lanes:
                a = qref.e4m3_float(qx[base+lane])
                w = qref.e4m3_float(weight[r*k + base+lane])
                part = f32(part + f32(a*w))
            ws = qref.e8m0_float(weight_scales[(r//128)*kb_count + kb])
            acc = f32(acc + f32(part * f32(act_scales[kb] * ws)))
        return qref.bf16_bits(acc)

    out = []
    for r in range(rows):
        a = one(r, False); b = one(r, True)
        if a != b:
            raise ValueError(f"main wq_b reduction order reaches BF16 at row {r}: 0x{a:04x} vs 0x{b:04x}")
        out.append(a)
    return out


def overlap_pool_generic(kv, score, ape, d, reverse=False, ignore_overlap=False):
    two_d = 2*d
    order = range(7, -1, -1) if reverse else range(8)
    out = []
    for c in range(2):
        for j in range(d):
            logits = [0.0] * 8
            values = [0.0] * 8
            valid = [True] * 8
            for t in range(4):
                if c == 0 or ignore_overlap:
                    valid[t] = False; logits[t] = -math.inf
                else:
                    p = ((c-1)*4+t)*two_d+j
                    logits[t] = f32(score[p] + ape[t*two_d+j]); values[t] = kv[p]
                p = (c*4+t)*two_d+d+j
                logits[4+t] = f32(score[p] + ape[t*two_d+d+j]); values[4+t] = kv[p]
            m = max(z for z in logits if math.isfinite(z))
            den = f32(0.0); num = f32(0.0)
            for t in order:
                if not valid[t] or not math.isfinite(logits[t]):
                    continue
                e = f32(math.exp(f32(logits[t]-m)))
                den = f32(den + e)
                num = f32(num + f32(values[t] * e))
            out.append(qref.bf16_bits(f32(num / den)))
    return out


def main_compressor(recipe, wkv_bits, wgate_bits, ape, norm_bits):
    rows = 1024
    kv = [0.0] * (SEQLEN * rows)
    score = [0.0] * (SEQLEN * rows)
    for item in recipe:
        kvrow = csaidx.single_product_rows(wkv_bits, item["lane"], item["bits"], rows, HIDDEN)
        scrow = csaidx.single_product_rows(wgate_bits, item["lane"], item["bits"], rows, HIDDEN)
        base = item["position"] * rows
        kv[base:base+rows] = kvrow; score[base:base+rows] = scrow

    pooled = overlap_pool_generic(kv, score, ape, HEAD, False, False)
    pooled_rev = overlap_pool_generic(kv, score, ape, HEAD, True, False)
    if pooled != pooled_rev:
        first = next(i for i,(a,b) in enumerate(zip(pooled,pooled_rev)) if a != b)
        raise ValueError(f"main overlap reduction order reaches BF16 at {first}")
    wrong_pool = overlap_pool_generic(kv, score, ape, HEAD, False, True)
    if pooled[HEAD:2*HEAD] == wrong_pool[HEAD:2*HEAD]:
        raise ValueError("real main compressor failed to witness overlap")

    norms=[]; ropes=[]; finals=[]; wrong_finals=[]
    for c in range(2):
        p = pooled[c*HEAD:(c+1)*HEAD]
        wp = wrong_pool[c*HEAD:(c+1)*HEAD]
        n = csaidx.learned_rms_stable(p, norm_bits)
        wn = csaidx.learned_rms_stable(wp, norm_bits)
        r = hist.rope(n, c*RATIO)
        wr = hist.rope(wn, c*RATIO)
        f = hist.qat_nope(r)
        wf = hist.qat_nope(wr)
        norms.extend(n); ropes.extend(r); finals.extend(f); wrong_finals.extend(wf)
    if finals[HEAD:2*HEAD] == wrong_finals[HEAD:2*HEAD]:
        raise ValueError("ignore-overlap mutation vanished after main compressor norm/RoPE/QAT")
    return pooled, norms, ropes, finals, wrong_finals


def sparse_attention(q_bits, local_bits, comp_bits, idxs, sink):
    q = [qref.bf16_to_f32(b) for b in q_bits]
    local = [[qref.bf16_to_f32(b) for b in local_bits[t*HEAD:(t+1)*HEAD]] for t in range(SEQLEN)]
    comp = [[qref.bf16_to_f32(b) for b in comp_bits[t*HEAD:(t+1)*HEAD]] for t in range(2)]
    kv = local + comp
    scale = f32(HEAD ** -0.5)
    best = -math.inf
    den = f32(0.0)
    acc = [f32(0.0)] * HEAD
    for base in range(0, len(idxs), 64):
        block = idxs[base:base+64]
        scores = []
        new_best = best
        for idx in block:
            if idx < 0:
                scores.append(-math.inf); continue
            dot = f32(0.0)
            for a,b in zip(q,kv[idx]):
                dot = f32(dot + f32(a*b))
            score = f32(dot * scale)
            scores.append(score)
            if score > new_best: new_best = score
        rescale = f32(math.exp(best-new_best)) if math.isfinite(best) else f32(0.0)
        den = f32(den * rescale)
        acc = [f32(v * rescale) for v in acc]
        for idx,score in zip(block,scores):
            if not math.isfinite(score): continue
            e = f32(math.exp(score-new_best))
            den = f32(den + e)
            eb = qref.bf16_to_f32(qref.bf16_bits(e))
            for k in range(HEAD):
                acc[k] = f32(acc[k] + f32(eb * kv[idx][k]))
        best = new_best
    den = f32(den + f32(math.exp(f32(sink-best))))
    return [qref.bf16_bits(f32(v/den)) for v in acc]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(repo, "tests", "fixtures", "deepseek_v4", "v5_csa_indexer_real")
    prov_path = os.path.join(index_dir, "provenance.json")
    with open(prov_path, encoding="utf-8") as f:
        index_prov = json.load(f)
    if index_prov.get("revision") != REV or index_prov.get("query_position") != QUERY_POS:
        raise ValueError("real CSA indexer dependency drifted")
    recipe = []
    for item in index_prov["input_recipe"]["active_tokens"]:
        recipe.append({"position": int(item["position"]), "lane": int(item["lane"]),
                       "bits": int(item["bf16_hex"],16), "bf16_hex": item["bf16_hex"]})
    if [x["position"] for x in recipe] != [3,7]:
        raise ValueError("CSA indexer input recipe drifted")
    qr = read_u16(os.path.join(index_dir, "qr-pos7.bf16.bin"))
    index_topk = read_i32(os.path.join(index_dir, "topk-row7.i32.bin"))
    if len(qr) != Q_RANK or len(index_topk) != 2 or any(i not in (8,9) for i in index_topk):
        raise ValueError("CSA indexer output dependency drifted")

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    paths={}; tensor_meta={}
    for key in TENSORS:
        paths[key], tensor_meta[key] = fetch(args.snapshot,key,args.out_dir,args.token)

    wq_b = open(paths["wq_b"],"rb").read()
    wq_b_scale = open(paths["wq_b_scale"],"rb").read()
    wkv = open(paths["wkv"],"rb").read()
    wkv_scale = open(paths["wkv_scale"],"rb").read()
    kv_norm = read_u16(paths["kv_norm"])
    sink = read_f32(paths["attn_sink"])[0]
    comp_wkv = read_u16(paths["comp_wkv"])
    comp_wgate = read_u16(paths["comp_wgate"])
    comp_ape = read_f32(paths["comp_ape"])
    comp_norm = read_u16(paths["comp_norm"])

    q0 = quantized_dense_rows_stable(wq_b,wq_b_scale,HEAD,Q_RANK,qr)
    qn = hist.head_norm(q0)
    q = hist.rope(qn,QUERY_POS)

    local = [0] * (SEQLEN*HEAD)
    active_local=[]
    for item in recipe:
        kv0 = hist.quantized_single_product(wkv,wkv_scale,HEAD,HIDDEN,item["lane"],item["bits"])
        kv1 = csaidx.learned_rms_stable(kv0,kv_norm)
        kv2 = hist.rope(kv1,item["position"])
        kv3 = hist.qat_nope(kv2)
        pos=item["position"]
        local[pos*HEAD:(pos+1)*HEAD]=kv3
        active_local.extend(kv3)

    pooled,norms,ropes,comp,wrong_comp = main_compressor(recipe,comp_wkv,comp_wgate,comp_ape,comp_norm)
    combined_topk = list(range(SEQLEN)) + index_topk
    pre = sparse_attention(q,local,comp,combined_topk,sink)
    post = hist.rope(pre,QUERY_POS,inverse=True)

    wrong_pre = sparse_attention(q,local,wrong_comp,combined_topk,sink)
    if wrong_pre == pre:
        raise ValueError("ignore-overlap main compressor mutation did not change CSA attention")
    window_only = sparse_attention(q,local,comp,list(range(SEQLEN)),sink)
    if window_only == pre:
        raise ValueError("dropping indexer-selected compressed entries did not change CSA attention")
    if post == pre:
        raise ValueError("inverse compressed YaRN was invisible in CSA attention")

    outputs = {
        "q": ("q-head0-pos7.bf16.bin",q,[HEAD]),
        "active_local": ("local-kv-active-3-7.bf16.bin",active_local,[2,HEAD]),
        "comp_pooled": ("main-comp-pooled.bf16.bin",pooled,[2,HEAD]),
        "comp_norm": ("main-comp-post-norm.bf16.bin",norms,[2,HEAD]),
        "comp_rope": ("main-comp-post-rope.bf16.bin",ropes,[2,HEAD]),
        "comp_kv": ("main-comp-kv.bf16.bin",comp,[2,HEAD]),
        "attn_pre": ("attn-pre-inverse.bf16.bin",pre,[HEAD]),
        "attn_post": ("attn-post-inverse.bf16.bin",post,[HEAD]),
    }
    outmeta={}
    for key,(name,bits,shape) in outputs.items():
        path=os.path.join(args.out_dir,name); write_u16(path,bits)
        outmeta[key]={"file":name,"shape":shape,"dtype":"BF16","sha256":sha256_file(path),
                      "first8_hex":[f"0x{x:04x}" for x in bits[:8]]}
    topk_path=os.path.join(args.out_dir,"combined-topk-row7.i32.bin"); write_i32(topk_path,combined_topk)
    outmeta["topk"]={"file":os.path.basename(topk_path),"shape":[len(combined_topk)],"dtype":"I32",
                      "sha256":sha256_file(topk_path),"values":combined_topk}
    pair20=448+40
    outmeta["attn_pre"]["pair20_hex"]=[f"0x{x:04x}" for x in pre[pair20:pair20+2]]
    outmeta["attn_post"]["pair20_hex"]=[f"0x{x:04x}" for x in post[pair20:pair20+2]]

    provenance={
        "schema_version":1,
        "gate":"Gate E / V5 coherent ratio-4 CSA attention sub-seam",
        "evidence_state":"CHECKPOINT-WEIGHTS + SAME-STRUCTURAL-INPUT + FROZEN-REAL-INDEXER + SOURCE-ORACLE",
        "model":"deepseek-ai/DeepSeek-V4-Flash-0731","revision":REV,"layer":2,"compress_ratio":4,
        "query_position":QUERY_POS,
        "input_recipe":{"seqlen":SEQLEN,"active_tokens":[{"position":x["position"],"lane":x["lane"],"bf16_hex":x["bf16_hex"]} for x in recipe],"all_other_tokens":"exact BF16 zero"},
        "indexer_dependency":{"fixture":"tests/fixtures/deepseek_v4/v5_csa_indexer_real","provenance_sha256":sha256_file(prov_path),"qr_sha256":sha256_file(os.path.join(index_dir,"qr-pos7.bf16.bin")),"topk_sha256":sha256_file(os.path.join(index_dir,"topk-row7.i32.bin")),"topk_values":index_topk},
        "checkpoint_tensors":tensor_meta,
        "source_contract":{"q":"real main head0 wq_b on frozen real q-rank -> direct BF16 head norm -> compressed YaRN pos7","local_kv":"same active tokens -> real wkv/kv_norm -> compressed YaRN -> K64 FP8 simulation","main_compressor":"real BF16 1024x4096 wkv/wgate + F32 APE -> previous-first/current-second overlap -> BF16 -> norm -> compressed YaRN -> K64 FP8 simulation first448","selection":"window row7 [0..7] followed by frozen real indexer top-k [8/9]","attention":"BF16 q/kv sink sparse attention -> inverse compressed YaRN"},
        "stability_guards":{"main_wq_b":"forward/reverse K128 and lane reductions must agree for all 512 BF16 head0 outputs","main_overlap":"forward/reverse eight-slot softmax reductions must agree at BF16"},
        "mutations":{"ignore_previous_overlap_half":"must change final sparse attention","drop_indexer_compressed_entries":"must change final sparse attention","omit_inverse_compressed_yarn":"must change final output"},
        "outputs":outmeta,
        "oracle_independence":{"producer":"tools/make_v5_csa_attention_real_fixture.py independent Python source equations","shares_waste_runtime_helpers":False},
        "non_claims":["one main attention head and one query row only","shared grouped output projection was proved separately and is not repeated","does not yet prove a full transformer layer"],
    }
    with open(os.path.join(args.out_dir,"provenance.json"),"w",encoding="utf-8") as f:
        json.dump(provenance,f,indent=2,sort_keys=True); f.write("\n")

    print("Gate E coherent ratio-4 CSA attention fixture built")
    print("indexer topk:",index_topk,"combined:",combined_topk)
    print("main comp chunk1 first8:"," ".join(f"0x{x:04x}" for x in comp[HEAD:HEAD+8]))
    print("attn pre first8:"," ".join(outmeta["attn_pre"]["first8_hex"]))
    print("attn post first8:"," ".join(outmeta["attn_post"]["first8_hex"]))
    print("pair20:",outmeta["attn_pre"]["pair20_hex"],"->",outmeta["attn_post"]["pair20_hex"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
