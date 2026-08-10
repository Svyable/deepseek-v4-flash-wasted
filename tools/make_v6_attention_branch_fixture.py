#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build Gate H/V6 full layer-3 one-token attention branch evidence.

The exact frozen layer-3 HC-attn-pre BF16 state is the input. Independent Python
source equations derive all 64 attention-head outputs. The scalar C attention
core must match those 64x512 BF16 values exactly. Only after that comparison is
green, the separately mutation-pinned full 8-group output primitive evaluates
real wo_a/wo_b and freezes the complete 4096-value branch.

Position 0 is intentional: this is a full-branch composition proof, not a second
YaRN proof. Gate E retains responsibility for compressed-history and nonzero
compressed-YaRN angles (including pair 20).
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import fetch_hf_tensor_slice as slice_fetch
import make_v2_real_fixture as qref
import make_v5_attention_history_real_fixture as hist

REV = hist.REV
HIDDEN = 4096
Q_RANK = 1024
HEAD = 512
N_HEADS = 64
GROUPS = 8
GROUP_IN = 4096
GROUP_RANK = 1024
OUT = 4096

TENSORS = {
    "wq_a": ("layers.3.attn.wq_a.weight", None, "F8_E4M3", [1024,4096]),
    "wq_a_scale": ("layers.3.attn.wq_a.scale", None, "F8_E8M0", [8,32]),
    "q_norm": ("layers.3.attn.q_norm.weight", None, "BF16", [1024]),
    "wq_b": ("layers.3.attn.wq_b.weight", None, "F8_E4M3", [32768,1024]),
    "wq_b_scale": ("layers.3.attn.wq_b.scale", None, "F8_E8M0", [256,8]),
    "wkv": ("layers.3.attn.wkv.weight", None, "F8_E4M3", [512,4096]),
    "wkv_scale": ("layers.3.attn.wkv.scale", None, "F8_E8M0", [4,32]),
    "kv_norm": ("layers.3.attn.kv_norm.weight", None, "BF16", [512]),
    "attn_sink": ("layers.3.attn.attn_sink", None, "F32", [64]),
    "wo_a": ("layers.3.attn.wo_a.weight", None, "F8_E4M3", [8192,4096]),
    "wo_a_scale": ("layers.3.attn.wo_a.scale", None, "F8_E8M0", [64,32]),
    "wo_b": ("layers.3.attn.wo_b.weight", None, "F8_E4M3", [4096,8192]),
    "wo_b_scale": ("layers.3.attn.wo_b.scale", None, "F8_E8M0", [32,64]),
}


def sha(path):
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
        raise ValueError(f"{path}: non-f32 bytes")
    return list(struct.unpack(f"<{len(data)//4}f", data))


def write_u16(path, xs):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(xs)}H", *xs))


def fetch(snapshot, key, td, token):
    name, rows, dtype, shape = TENSORS[key]
    path = os.path.join(td, key + ".bin")
    info = slice_fetch.fetch_slice(snapshot, name, rows, path, token=token)
    if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
        raise ValueError(f"{name}: got {info['revision']} {info['dtype']} {info['shape']}; expected {REV} {dtype} {shape}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return path, {
        "name": name, "shard": info["shard"], "dtype": dtype, "shape": shape,
        "rows": [info["row_start"], info["row_end"]],
        "absolute_range": [info["absolute_start"], info["absolute_end"]],
        "payload_bytes": info["payload_bytes"], "payload_sha256": info["payload_sha256"],
    }


def sparse_quantized_linear(weight, scales, out_rows, k, input_bits):
    """Official K128 FP8 linear specialized to a sparse BF16 input."""
    nz = [i for i, b in enumerate(input_bits) if (b & 0x7FFF) != 0]
    vals = {i: qref.bf16_to_f32(input_bits[i]) for i in nz}
    kb_count = k // 128
    act_scales = []
    qx = {}
    for kb in range(kb_count):
        cols = [i for i in nz if i // 128 == kb]
        amax = max((abs(vals[i]) for i in cols), default=0.0)
        s = qref.next_pow2_scale(amax)
        act_scales.append(s)
        for i in cols:
            qx[i] = qref.e4m3_encode(hist.f32(vals[i] / s))
    out = []
    for r in range(out_rows):
        acc = hist.f32(0.0)
        row = r * k
        for kb in range(kb_count):
            part = hist.f32(0.0)
            for i in nz:
                if i // 128 != kb:
                    continue
                part = hist.f32(part + hist.f32(
                    qref.e4m3_float(qx[i]) * qref.e4m3_float(weight[row + i])))
            ws = qref.e8m0_float(scales[(r // 128) * kb_count + kb])
            acc = hist.f32(acc + hist.f32(part * hist.f32(act_scales[kb] * ws)))
        out.append(qref.bf16_bits(acc))
    return out



def one_row_online_attention(q_bits, kv_bits, sink):
    """One valid KV row through the real online-softmax operation order.

    Algebraically this looks like ``kv / (1 + exp(sink-score))``, but the
    source kernel first accumulates ``BF16(exp(0)) * kv`` into a +0 F32
    accumulator. That operation is observably different for a -0 KV lane:
    +0 + -0 becomes +0 under round-to-nearest. Keep the operational order;
    do not replace this with direct division.
    """
    if len(q_bits) != HEAD or len(kv_bits) != HEAD:
        raise ValueError("one-row attention requires one full 512-value head")
    qf = [qref.bf16_to_f32(b) for b in q_bits]
    kvf = [qref.bf16_to_f32(b) for b in kv_bits]
    scale = hist.f32(HEAD ** -0.5)
    dot = hist.f32(0.0)
    for a, b in zip(qf, kvf):
        dot = hist.f32(dot + hist.f32(a * b))
    score = hist.f32(dot * scale)

    best = score
    denom = hist.f32(0.0)
    acc = [hist.f32(0.0)] * HEAD
    e = hist.f32(math.exp(hist.f32(score - best)))
    denom = hist.f32(denom + e)
    e_bf16 = qref.bf16_to_f32(qref.bf16_bits(e))
    for k, v in enumerate(kvf):
        acc[k] = hist.f32(acc[k] + hist.f32(e_bf16 * v))

    denom = hist.f32(denom + hist.f32(math.exp(hist.f32(sink - best))))
    return [qref.bf16_bits(hist.f32(v / denom)) for v in acc]


def one_token_heads(x_bits, paths):
    wq_a = open(paths["wq_a"], "rb").read()
    wq_as = open(paths["wq_a_scale"], "rb").read()
    q_norm = read_u16(paths["q_norm"])
    wq_b = open(paths["wq_b"], "rb").read()
    wq_bs = open(paths["wq_b_scale"], "rb").read()
    wkv = open(paths["wkv"], "rb").read()
    wkv_s = open(paths["wkv_scale"], "rb").read()
    kv_norm = read_u16(paths["kv_norm"])
    sinks = read_f32(paths["attn_sink"])

    qa0 = sparse_quantized_linear(wq_a, wq_as, Q_RANK, HIDDEN, x_bits)
    qa1 = hist.learned_rms(qa0, q_norm)
    qb = hist.quantized_dense_rows(wq_b, wq_bs, N_HEADS * HEAD, Q_RANK, qa1)

    kv0 = sparse_quantized_linear(wkv, wkv_s, HEAD, HIDDEN, x_bits)
    kv1 = hist.learned_rms(kv0, kv_norm)
    # position 0 => compressed YaRN angle exactly zero. QAT still applies to KV.
    kv = hist.qat_nope(kv1)

    heads = []
    q_all = []
    for h in range(N_HEADS):
        qn = hist.head_norm(qb[h*HEAD:(h+1)*HEAD])
        q_all.extend(qn)
        heads.extend(one_row_online_attention(qn, kv, sinks[h]))
    return q_all, kv, heads, sinks


def compiler():
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and shutil.which(name):
            return name
    return None


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)


def as_u8(data):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def validate_and_project(repo, x_bits, paths, want_q, want_kv, want_heads):
    cc = compiler()
    if not cc:
        raise RuntimeError("no C compiler for Gate H attention acquisition")
    with tempfile.TemporaryDirectory(prefix="v6-attn-lib-") as td:
        lib = os.path.join(td, "libv6attn.so")
        cmd = [cc, "-std=c11", "-O2", "-shared", "-fPIC", "-I", os.path.join(repo, "src"),
               os.path.join(repo, "src", "deepseek_v4_attention_ref.c"),
               os.path.join(repo, "src", "deepseek_v4_attention_output_ref.c"),
               os.path.join(repo, "src", "quant", "deepseek_v4_linear_ref.c"),
               os.path.join(repo, "src", "quant", "fp4_e2m1.c"),
               os.path.join(repo, "src", "quant", "fp8_e4m3.c"), "-lm", "-o", lib]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stdout + built.stderr)
        so = ctypes.CDLL(lib)

        core = so.waste_ds_v4_attention_ratio0_prefill_ref
        core.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float,
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float)]
        core.restype = ctypes.c_int

        x = [qref.bf16_to_f32(b) for b in x_bits]
        qnw = [qref.bf16_to_f32(b) for b in read_u16(paths["q_norm"])]
        kvnw = [qref.bf16_to_f32(b) for b in read_u16(paths["kv_norm"])]
        sinks = read_f32(paths["attn_sink"])
        raw = {k: open(paths[k], "rb").read() for k in
               ("wq_a","wq_a_scale","wq_b","wq_b_scale","wkv","wkv_scale")}
        qout = (ctypes.c_float * (N_HEADS * HEAD))()
        kvout = (ctypes.c_float * HEAD)()
        hout = (ctypes.c_float * (N_HEADS * HEAD))()
        rc = core(as_cf(x), 1, N_HEADS,
                  as_u8(raw["wq_a"]), as_u8(raw["wq_a_scale"]), as_cf(qnw),
                  as_u8(raw["wq_b"]), as_u8(raw["wq_b_scale"]),
                  as_u8(raw["wkv"]), as_u8(raw["wkv_scale"]), as_cf(kvnw), as_cf(sinks),
                  ctypes.c_float(1e-6), ctypes.c_float(160000.0), qout, kvout, hout)
        if rc:
            raise RuntimeError(f"C attention core returned {rc}")
        got_q = [qref.bf16_bits(v) for v in qout]
        got_kv = [qref.bf16_bits(v) for v in kvout]
        got_h = [qref.bf16_bits(v) for v in hout]
        if got_q != want_q:
            i = next(i for i,(a,b) in enumerate(zip(got_q,want_q)) if a != b)
            raise ValueError(f"C/Python Q mismatch at {i}: {got_q[i]:04x}!={want_q[i]:04x}")
        if got_kv != want_kv:
            i = next(i for i,(a,b) in enumerate(zip(got_kv,want_kv)) if a != b)
            raise ValueError(f"C/Python KV mismatch at {i}: {got_kv[i]:04x}!={want_kv[i]:04x}")
        if got_h != want_heads:
            i = next(i for i,(a,b) in enumerate(zip(got_h,want_heads)) if a != b)
            raise ValueError(f"C/Python head mismatch at {i}: {got_h[i]:04x}!={want_heads[i]:04x}")

        full = so.waste_ds_v4_attention_output_full_ref
        full.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                         ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        full.restype = ctypes.c_int
        wo = {k: open(paths[k], "rb").read() for k in ("wo_a","wo_a_scale","wo_b","wo_b_scale")}
        hf = [qref.bf16_to_f32(b) for b in want_heads]
        lat = (ctypes.c_float * (GROUPS * GROUP_RANK))()
        out = (ctypes.c_float * OUT)()
        rc = full(as_cf(hf), as_u8(wo["wo_a"]), as_u8(wo["wo_a_scale"]),
                  as_u8(wo["wo_b"]), as_u8(wo["wo_b_scale"]), OUT, lat, out)
        if rc:
            raise RuntimeError(f"C full output returned {rc}")
        return [qref.bf16_bits(v) for v in lat], [qref.bf16_bits(v) for v in out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hc_dir = os.path.join(repo, "tests", "fixtures", "deepseek_v4", "v6_hc_composition_real")
    x_path = os.path.join(hc_dir, "attn-pre.bf16.bin")
    hc_prov_path = os.path.join(hc_dir, "provenance.json")
    if not os.path.isfile(x_path) or not os.path.isfile(hc_prov_path):
        raise ValueError("real Gate-H HC fixture must be frozen first")
    hp = json.load(open(hc_prov_path, encoding="utf-8"))
    if hp.get("revision") != REV or hp.get("layer") != 3:
        raise ValueError("Gate-H HC dependency provenance drifted")
    x_bits = read_u16(x_path)
    if len(x_bits) != HIDDEN:
        raise ValueError("HC attn-pre shape drifted")
    if not any((b & 0x7FFF) != 0 for b in x_bits):
        raise ValueError("HC attn-pre unexpectedly all zero")

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v6-attn-weights-") as td:
        paths = {}; meta = {}
        for key in TENSORS:
            paths[key], meta[key] = fetch(args.snapshot, key, td, args.token)
        q, kv, heads, sinks = one_token_heads(x_bits, paths)
        lat, branch = validate_and_project(repo, x_bits, paths, q, kv, heads)

    # Binding mutation: old unrelated Gate-E attention head bytes cannot stand in
    # for the exact HC-pre-derived heads of this fixture.
    old = os.path.join(repo, "tests", "fixtures", "deepseek_v4",
                       "v5_attention_ratio128_coherent_real", "attn-post-inverse.bf16.bin")
    if os.path.isfile(old):
        old_bits = read_u16(old)
        if heads[:HEAD] == old_bits:
            raise ValueError("unrelated old head fixture equals Gate-H head0")

    outputs = {
        "q": ("q-all64.bf16.bin", q, [N_HEADS, HEAD]),
        "kv": ("kv-local.bf16.bin", kv, [HEAD]),
        "heads": ("attention-heads-all64.bf16.bin", heads, [N_HEADS, HEAD]),
        "group_latent": ("group-latent-all8.bf16.bin", lat, [GROUPS, GROUP_RANK]),
        "branch": ("attention-branch.bf16.bin", branch, [OUT]),
    }
    om = {}
    for key,(name,bits,shape) in outputs.items():
        path = os.path.join(args.out_dir, name)
        write_u16(path, bits)
        om[key] = {"file": name, "dtype": "BF16", "shape": shape, "sha256": sha(path),
                   "first8_hex": [f"0x{x:04x}" for x in bits[:8]]}

    prov = {
        "schema_version": 1,
        "gate": "Gate H/V6 full layer-3 attention branch",
        "evidence_state": "REAL-HC-BOUND + INDEPENDENT-64-HEAD-ORACLE + PROVEN-FULL-OUTPUT-PRIMITIVE",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731", "revision": REV, "layer": 3,
        "position": 0, "n_heads": N_HEADS, "head_dim": HEAD,
        "input_dependency": {"file": "tests/fixtures/deepseek_v4/v6_hc_composition_real/attn-pre.bf16.bin",
                             "sha256": sha(x_path), "nonzero_count": sum((b & 0x7FFF) != 0 for b in x_bits)},
        "checkpoint_tensors": meta,
        "outputs": om,
        "source_contract": {
            "core": "HC-attn-pre BF16 -> wq_a -> learned q RMS -> all 64 wq_b heads -> BF16 head norm; shared wkv -> learned KV RMS -> K64 QAT; one-token sink attention",
            "position0": "compressed YaRN and inverse compressed YaRN angles are exactly zero; Gate E owns nonzero-angle proof",
            "output": "64x512 -> 8 groups x 4096 -> BF16-dequantized wo_a [8x1024x4096] -> complete 8192 latent -> quantized wo_b -> 4096 BF16 branch",
        },
        "oracle_independence": {
            "attention_heads": "independent Python source equations; C core must match all 32768 BF16 values exactly",
            "output_projection": "Gate-H full 8-group C primitive after model-free all-group mutation preflight; Gate-E separately proves real group arithmetic",
        },
        "mutations": {"unrelated_old_head0": "must not equal HC-bound head0"},
        "non_claims": ["position 0 does not re-prove nonzero compressed-YaRN angles or compressed history",
                       "full output projection is composed from an already independently tested primitive rather than re-oracled here"],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True); f.write("\n")
    print("PASS generated Gate H full layer-3 attention branch")
    print("head0 first8:", " ".join(om["heads"]["first8_hex"]))
    print("branch first8:", " ".join(om["branch"]["first8_hex"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
