#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a real two-chunk layer-3 ratio-128 compressor fixture.

The input has two 128-token chunks. Exactly one token and one hidden lane are
non-zero in each chunk. This still exercises all 512 compressor dimensions,
per-dimension APE/gate softmax over all 128 positions, learned RMSNorm,
position-128 compressed YaRN, and K64 QAT while eliminating wkv/wgate reduction
ambiguity: every non-zero linear output is exactly one BF16*BF16 product.
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

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SEQLEN = 256
RATIO = 128
HIDDEN = 4096
KV = 512
ROPE = 64
NOPE = 448
QAT = 64
ORIGINAL = 65536
BASE = 160000.0
FACTOR = 16.0
BETA_FAST = 32.0
BETA_SLOW = 1.0

TENSORS = {
    "wkv": ("layers.3.attn.compressor.wkv.weight", None, "BF16", [512,4096], "wkv.bf16.bin"),
    "wgate": ("layers.3.attn.compressor.wgate.weight", None, "BF16", [512,4096], "wgate.bf16.bin"),
    "ape": ("layers.3.attn.compressor.ape", None, "F32", [128,512], "ape.f32.bin"),
    "norm": ("layers.3.attn.compressor.norm.weight", None, "F32", [512], "norm.f32.bin"),
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
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//2}H", data))


def read_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}f", data))


def write_u16(path, xs):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(xs)}H", *xs))


def correction_dim(rotations, dim=ROPE):
    return dim * math.log(ORIGINAL / (rotations * 2 * math.pi)) / (2 * math.log(BASE))


def yarn_freq(pair):
    freq = 1.0 / (BASE ** ((2*pair)/ROPE))
    low = max(0.0, math.floor(correction_dim(BETA_FAST)))
    high = min(ROPE-1.0, math.ceil(correction_dim(BETA_SLOW)))
    if low == high:
        high += 0.001
    ramp = min(1.0, max(0.0, (pair-low)/(high-low)))
    smooth = 1.0-ramp
    return f32(freq/FACTOR*(1.0-smooth) + freq*smooth)


def rope(bits, position):
    vals = [qref.bf16_to_f32(b) for b in bits]
    for pair in range(ROPE//2):
        i = NOPE + 2*pair
        angle = f32(position * yarn_freq(pair))
        c, s = f32(math.cos(angle)), f32(math.sin(angle))
        re, im = vals[i], vals[i+1]
        vals[i] = qref.bf16_to_f32(qref.bf16_bits(f32(f32(re*c) - f32(im*s))))
        vals[i+1] = qref.bf16_to_f32(qref.bf16_bits(f32(f32(re*s) + f32(im*c))))
    return [qref.bf16_bits(v) for v in vals]


def qat(bits):
    vals = [qref.bf16_to_f32(b) for b in bits]
    for base in range(0, NOPE, QAT):
        block = vals[base:base+QAT]
        scale = qref.next_pow2_scale(max(abs(v) for v in block))
        for j, v in enumerate(block):
            z = max(-448.0, min(448.0, f32(v/scale)))
            code = qref.e4m3_encode(z)
            dq = f32(qref.e4m3_float(code) * scale)
            vals[base+j] = qref.bf16_to_f32(qref.bf16_bits(dq))
    return [qref.bf16_bits(v) for v in vals]


def build_input(repo_root, out_dir):
    anchor = os.path.join(repo_root, "tests", "fixtures", "deepseek_v4", "v2_wq_a_real", "input.bf16.bin")
    source = read_u16(anchor)
    if len(source) != HIDDEN:
        raise ValueError("Gate-C anchor input shape drifted")
    bits = [0] * (SEQLEN * HIDDEN)
    active = [
        {"token": 0, "lane": 17, "source_lane": 17, "bits": source[17]},
        {"token": 128, "lane": 2053, "source_lane": 2053, "bits": source[2053] ^ 0x8000},
    ]
    for a in active:
        bits[a["token"]*HIDDEN+a["lane"]] = a["bits"]
        a["bf16_hex"] = f"0x{a['bits']:04x}"
    path = os.path.join(out_dir, "input.bf16.bin")
    write_u16(path, bits)
    return active, {"file":"input.bf16.bin","shape":[SEQLEN,HIDDEN],"dtype":"BF16","sha256":sha256_file(path),"active":active}


def fetch(snapshot, key, out_dir, token):
    name, rows, dtype, shape, filename = TENSORS[key]
    path = os.path.join(out_dir, filename)
    info = slice_fetch.fetch_slice(snapshot, name, rows, path, token=token)
    if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
        raise ValueError(f"{name}: got {info['revision']} {info['dtype']} {info['shape']}, expected {REV} {dtype} {shape}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return path, {"name":name,"shard":info["shard"],"dtype":dtype,"shape":shape,"absolute_range":[info["absolute_start"],info["absolute_end"]],"payload_bytes":info["payload_bytes"],"payload_sha256":info["payload_sha256"],"file":filename}


def single_product_rows(weight_bits, lane, x):
    out = []
    for r in range(KV):
        w = qref.bf16_to_f32(weight_bits[r*HIDDEN+lane])
        out.append(f32(x*w))
    return out


def softmax_active(score_active, ape, active_t):
    out = []
    for d in range(KV):
        logits = [ape[t*KV+d] for t in range(RATIO)]
        logits[active_t] = f32(logits[active_t] + score_active[d])
        m = max(logits)
        den = f32(0.0)
        active_e = None
        for t, z in enumerate(logits):
            e = f32(math.exp(f32(z-m)))
            den = f32(den+e)
            if t == active_t:
                active_e = e
        out.append(f32(active_e/den))
    return out


def learned_norm(bits, norm):
    x = [qref.bf16_to_f32(b) for b in bits]
    sumsq = f32(0.0)
    for v in x:
        sumsq = f32(sumsq + f32(v*v))
    mean = f32(sumsq / KV)
    inv = f32(1.0 / math.sqrt(f32(mean + 1e-6)))
    return [qref.bf16_bits(f32(f32(v*inv)*norm[i])) for i,v in enumerate(x)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    active, input_meta = build_input(repo_root, args.out_dir)
    paths, tmeta = {}, {}
    for key in TENSORS:
        paths[key], tmeta[key] = fetch(args.snapshot, key, args.out_dir, args.token)

    wkv = read_u16(paths["wkv"]); wgate = read_u16(paths["wgate"])
    ape = read_f32(paths["ape"]); norm = read_f32(paths["norm"])
    pooled_all=[]; norm_all=[]; rope_all=[]; final_all=[]
    for chunk, a in enumerate(active):
        xv = qref.bf16_to_f32(a["bits"])
        kv_active = single_product_rows(wkv, a["lane"], xv)
        gate_active = single_product_rows(wgate, a["lane"], xv)
        probs = softmax_active(gate_active, ape, 0)
        pooled = [f32(kv_active[d]*probs[d]) for d in range(KV)]
        pooled_bits = [qref.bf16_bits(v) for v in pooled]
        norm_bits = learned_norm(pooled_bits, norm)
        rope_bits = rope(norm_bits, chunk*RATIO)
        final_bits = qat(rope_bits)
        pooled_all += pooled_bits; norm_all += norm_bits; rope_all += rope_bits; final_all += final_bits

    files = {
        "pooled_before_norm":"pooled-before-norm.bf16.bin",
        "post_norm":"post-norm.bf16.bin",
        "post_yarn_rope":"post-yarn-rope.bf16.bin",
        "compressed_kv":"compressed-kv.bf16.bin",
    }
    values = [pooled_all,norm_all,rope_all,final_all]
    outputs={}
    for (key,fn),bits in zip(files.items(),values):
        path=os.path.join(args.out_dir,fn); write_u16(path,bits)
        outputs[key]={"file":fn,"shape":[2,512],"dtype":"BF16","sha256":sha256_file(path),"chunk0_first8_hex":[f"0x{x:04x}" for x in bits[:8]],"chunk1_first8_hex":[f"0x{x:04x}" for x in bits[512:520]]}
    outputs["post_yarn_rope"]["chunk1_rope_tail8_hex"]=[f"0x{x:04x}" for x in rope_all[512+504:512+512]]

    prov={
        "schema_version":1,"gate":"Gate E / V5 ratio-128 compressor sub-seam",
        "evidence_state":"CHECKPOINT-WEIGHTS + STRUCTURAL-SPARSE-INPUT + SOURCE-ORACLE",
        "model":"deepseek-ai/DeepSeek-V4-Flash-0731","revision":REV,"layer":3,"compress_ratio":128,
        "input":input_meta,"tensors":tmeta,
        "source_contract":{"projection":"wkv/wgate checkpoint BF16 loaded into f32 Linear; sparse active token gives one product/output row","pool":"per-output-dimension f32 softmax(score+ape) over 128 tokens; f32 weighted pool then explicit BF16 cast","norm":"learned RMSNorm f32 math/F32 weight -> BF16","rope":"compressed YaRN base160000 factor16 beta_fast32 beta_slow1, chunk positions 0 and 128","qat":"finite E4M3 quant/dequant first448 in K64 blocks, BF16 result"},
        "oracle_independence":{"producer":"tools/make_v5_compressor_ratio128_fixture.py independent Python source equations","shares_waste_runtime_helpers":False,"reduction_design":"one nonzero hidden lane/token eliminates wkv/wgate reduction ambiguity"},
        "outputs":outputs,
        "non_claims":["does not execute official GPU kernels","does not yet include compressed-history attention scores","does not prove ratio-4 CSA/indexer","therefore does not complete Gate E/V5"]}
    with open(os.path.join(args.out_dir,"provenance.json"),"w",encoding="utf-8") as f:
        json.dump(prov,f,indent=2,sort_keys=True); f.write("\n")
    print("Gate E ratio-128 compressor fixture built")
    print("pooled chunk0:"," ".join(outputs["pooled_before_norm"]["chunk0_first8_hex"]))
    print("rope chunk1 tail:"," ".join(outputs["post_yarn_rope"]["chunk1_rope_tail8_hex"]))
    print("final chunk1:"," ".join(outputs["compressed_kv"]["chunk1_first8_hex"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
