#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Offline replay of V7's real layer-3 -> layer-4 route-discovery fixture.

Large layer-4 attention weights are acquisition-only. The permanent contract
therefore starts from the exact frozen Gate-H layer-3 output, recomputes both
layer-4 HyperConnection transitions around the frozen real attention branch,
and recomputes the learned router from committed real gate weight/bias bytes.

This keeps the multi-layer binding load-bearing without pretending the compact
fixture can re-run a 100+ MB attention acquisition offline.
"""

import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))
import test_v6_ffn_route_real as chain  # noqa: E402

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
FIX = os.path.join(BASE, "v7_layer4_route_real")
L3FIX = os.path.join(BASE, "v6_moe_branch_real")
SURFACE = os.path.join(REPO, "reference", "deepseek-v4-flash-0731.layer4-v7.json")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24
EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 2:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*2}")
    return list(struct.unpack(f"<{count}H", data))


def read_u32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*4}")
    return list(struct.unpack(f"<{count}I", data))


def read_f32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*4}")
    return list(struct.unpack(f"<{count}f", data))


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)


def oracle_hc_pre_y_f32(x, pre):
    """Independent HC-pre equation with the model's sequential F32 reduction."""
    out = []
    for d in range(DIM):
        acc = chain.f32(0.0)
        for j in range(HC):
            acc = chain.f32(acc + chain.f32(pre[j] * x[j * DIM + d]))
        out.append(acc)
    return out


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.isfile(prov_path):
        print("SKIP: v7_layer4_route_real fixture is not frozen yet")
        return 77
    p = json.load(open(prov_path, encoding="utf-8"))
    if p.get("revision") != REV or p.get("layer") != 4 or \
       p.get("structural_class") != "ratio4-learned":
        print("FAIL: V7 layer-4 fixture identity drifted")
        return 1

    surface = json.load(open(SURFACE, encoding="utf-8"))
    if surface.get("revision") != REV or surface.get("layer") != 4 or \
       surface.get("structural_class") != "ratio4-learned" or \
       surface.get("config_contract", {}).get("compress_ratio") != 4:
        print("FAIL: layer-4 checkpoint surface is not the frozen ratio4 target")
        return 1
    dep_surface = p.get("checkpoint_surface_dependency") or {}
    if sha(SURFACE) != dep_surface.get("sha256"):
        print("FAIL: route fixture was built against a different layer-4 surface")
        return 1

    l3 = json.load(open(os.path.join(L3FIX, "provenance.json"), encoding="utf-8"))
    l3_final_meta = l3.get("outputs", {}).get("layer3_final", {})
    l3_final = os.path.join(L3FIX, l3_final_meta.get("file", ""))
    input_meta = p.get("outputs", {}).get("input", {})
    input_path = os.path.join(FIX, input_meta.get("file", ""))
    if not os.path.isfile(l3_final) or not os.path.isfile(input_path):
        print("FAIL: layer3 output / layer4 input is missing")
        return 1
    if sha(l3_final) != l3_final_meta.get("sha256") or \
       sha(input_path) != input_meta.get("sha256") or \
       sha(input_path) != sha(l3_final):
        print("FAIL: layer3 output is not byte-identical to layer4 input")
        return 1
    if not (p.get("input_dependency") or {}).get("byte_identical"):
        print("FAIL: provenance does not state exact output->input chaining")
        return 1

    outputs = p.get("outputs") or {}
    counts = {
        "attn_pre": DIM,
        "q": 64 * 512,
        "kv": 512,
        "heads": 64 * 512,
        "group_latent": 8 * 1024,
        "attention_branch": DIM,
        "after_attn": FLAT,
        "ffn_pre": DIM,
    }
    paths = {}
    for key, count in counts.items():
        meta = outputs.get(key) or {}
        path = os.path.join(FIX, meta.get("file", ""))
        if not os.path.isfile(path) or sha(path) != meta.get("sha256"):
            print(f"FAIL: {key} output missing or SHA mismatch")
            return 1
        read_u16(path, count)
        paths[key] = path

    trunk = p.get("checkpoint_tensors", {}).get("hyperconnection_and_router", {})
    file_map = {
        "attn_fn": ("attn-fn.f32.bin", MIX * FLAT),
        "attn_scale": ("attn-scale.f32.bin", 3),
        "attn_base": ("attn-base.f32.bin", MIX),
        "ffn_fn": ("ffn-fn.f32.bin", MIX * FLAT),
        "ffn_scale": ("ffn-scale.f32.bin", 3),
        "ffn_base": ("ffn-base.f32.bin", MIX),
        "gate_bias": ("gate-bias.f32.bin", EXPERTS),
    }
    fparams = {}
    for key, (filename, count) in file_map.items():
        path = os.path.join(FIX, filename)
        meta = trunk.get(key) or {}
        if not os.path.isfile(path) or sha(path) != meta.get("payload_sha256"):
            print(f"FAIL: frozen layer4 {key} missing or checkpoint SHA mismatch")
            return 1
        fparams[key] = read_f32(path, count)
    gate_path = os.path.join(FIX, "gate-weight.bf16.bin")
    gate_meta = trunk.get("gate_weight") or {}
    if not os.path.isfile(gate_path) or sha(gate_path) != gate_meta.get("payload_sha256"):
        print("FAIL: frozen layer4 gate weight missing or checkpoint SHA mismatch")
        return 1
    gate_bits = read_u16(gate_path, EXPERTS * DIM)

    ids_path = os.path.join(FIX, p.get("router", {}).get("ids_file", ""))
    weights_path = os.path.join(FIX, p.get("router", {}).get("weights_file", ""))
    if sha(ids_path) != p["router"]["ids_sha256"] or \
       sha(weights_path) != p["router"]["weights_sha256"]:
        print("FAIL: router output file SHA mismatch")
        return 1
    want_ids = read_u32(ids_path, TOPK)
    want_weights = read_f32(weights_path, TOPK)

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77
    with tempfile.TemporaryDirectory(prefix="v7-l4-route-") as td:
        lib = os.path.join(td, "libstate.so")
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", "-shared", "-fPIC", "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"),
               os.path.join(REPO, "src", "deepseek_v4_router_ref.c"),
               "-lm", "-o", lib]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile V7 layer4 route replay")
            return 1
        so = ctypes.CDLL(lib)
        pre = so.waste_ds_v4_hc_pre_ref
        pre.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint,
                        ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float)]
        pre.restype = ctypes.c_int
        post = so.waste_ds_v4_hc_post_ref
        post.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                         ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        post.restype = ctypes.c_int
        score = so.waste_ds_v4_router_score_ref
        score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float),
                          ctypes.POINTER(ctypes.c_float)]
        score.restype = ctypes.c_int
        learned = so.waste_ds_v4_router_learned_ref
        learned.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        learned.restype = ctypes.c_int

        def c_pre(x, fn, scale, base):
            y = (ctypes.c_float * DIM)()
            po = (ctypes.c_float * HC)()
            co = (ctypes.c_float * (HC * HC))()
            rc = pre(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base),
                     ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6),
                     y, None, None, po, co)
            if rc:
                raise RuntimeError("hc_pre rejected V7 replay")
            return list(y), list(po), [list(co[i*HC:(i+1)*HC]) for i in range(HC)]

        def c_post(branch, residual, po, co):
            y = (ctypes.c_float * FLAT)()
            rc = post(as_cf(branch), as_cf(residual), DIM, as_cf(po),
                      as_cf([v for row in co for v in row]), y)
            if rc:
                raise RuntimeError("hc_post rejected V7 replay")
            return list(y)

        residual = [chain.bf16_to_f32(v) for v in read_u16(input_path, FLAT)]
        attn_pre_f, apo, aco = c_pre(
            residual, fparams["attn_fn"], fparams["attn_scale"], fparams["attn_base"])
        attn_pre_bits, _ = chain.cast_bf16(attn_pre_f)
        if attn_pre_bits != read_u16(paths["attn_pre"], DIM):
            print("FAIL: exact layer3 output does not reproduce frozen layer4 attn_pre")
            return 1

        # Independent Python HC-pre must land at the same BF16 boundary.
        opre, _, _ = chain.oracle_hc_params(
            residual, fparams["attn_fn"], fparams["attn_scale"], fparams["attn_base"])
        opre = [chain.f32(v) for v in opre]
        if chain.cast_bf16(oracle_hc_pre_y_f32(residual, opre))[0] != attn_pre_bits:
            print("FAIL: layer4 attention hc_pre disagrees with independent oracle")
            return 1

        branch = [chain.bf16_to_f32(v) for v in read_u16(paths["attention_branch"], DIM)]
        after_f = c_post(branch, residual, apo, aco)
        after_bits, after = chain.cast_bf16(after_f)
        if after_bits != read_u16(paths["after_attn"], FLAT):
            print("FAIL: layer4 attention hc_post no longer matches frozen state")
            return 1

        ffn_pre_f, _, _ = c_pre(
            after, fparams["ffn_fn"], fparams["ffn_scale"], fparams["ffn_base"])
        ffn_pre_bits, ffn_pre = chain.cast_bf16(ffn_pre_f)
        if ffn_pre_bits != read_u16(paths["ffn_pre"], DIM):
            print("FAIL: layer4 FFN hc_pre no longer matches frozen state")
            return 1
        fpre, _, _ = chain.oracle_hc_params(
            after, fparams["ffn_fn"], fparams["ffn_scale"], fparams["ffn_base"])
        fpre = [chain.f32(v) for v in fpre]
        if chain.cast_bf16(oracle_hc_pre_y_f32(after, fpre))[0] != ffn_pre_bits:
            print("FAIL: layer4 FFN hc_pre disagrees with independent oracle")
            return 1

        wb = (ctypes.c_uint16 * len(gate_bits))(*gate_bits)
        raw = (ctypes.c_float * EXPERTS)()
        scores = (ctypes.c_float * EXPERTS)()
        if score(as_cf(ffn_pre), DIM, wb, raw, scores):
            print("FAIL: scalar router score rejected frozen layer4 ffn_pre")
            return 1
        ids = (ctypes.c_uint32 * TOPK)()
        weights = (ctypes.c_float * TOPK)()
        selection = (ctypes.c_float * EXPERTS)()
        if learned(scores, as_cf(fparams["gate_bias"]), ctypes.c_float(ROUTE_SCALE),
                   ids, weights, selection):
            print("FAIL: scalar learned router rejected layer4 scores")
            return 1
        got_ids = list(ids)
        got_weights = list(weights)
        if got_ids != want_ids:
            print(f"FAIL: layer4 route IDs changed: {got_ids} != {want_ids}")
            return 1
        if [chain.f32_bits(x) for x in got_weights] != [chain.f32_bits(x) for x in want_weights]:
            print("FAIL: layer4 exact scalar route weights changed")
            return 1

        py_ids, py_weights, _, _, margin = chain.oracle_router(
            ffn_pre, gate_bits, fparams["gate_bias"])
        if py_ids != got_ids or max(abs(a-b) for a,b in zip(py_weights, got_weights)) > 1e-5:
            print("FAIL: independent layer4 router no longer supports scalar selection")
            return 1
        if not margin > 1e-4:
            print("FAIL: layer4 top-k boundary is no longer stable")
            return 1

        # Multi-layer binding mutation: one BF16-bit change in the prior layer
        # output must not reproduce the exact layer4 hc_pre boundary.
        bad_bits = read_u16(input_path, FLAT)
        bad_bits[0] ^= 0x0001
        bad = [chain.bf16_to_f32(v) for v in bad_bits]
        bad_pre_f, _, _ = c_pre(
            bad, fparams["attn_fn"], fparams["attn_scale"], fparams["attn_base"])
        if chain.cast_bf16(bad_pre_f)[0] == attn_pre_bits:
            print("FAIL: one-bit prior-layer mutation is invisible at layer4 hc_pre")
            return 1

    attn_check = p.get("attention_crosscheck") or {}
    if attn_check.get("python_vs_c_exact_head_bf16") != 64 * 512:
        print("FAIL: acquisition provenance lacks full 64-head exact cross-check")
        return 1
    nonclaims = " ".join(p.get("non_claims") or [])
    if "compressed history" not in nonclaims or "Gate E" not in nonclaims:
        print("FAIL: ratio4 position-0 evidence boundary is not explicit")
        return 1

    print("PASS V7 real chained layer4 route replay")
    print("layer3 output SHA == layer4 input SHA:", sha(input_path))
    print("selected experts:", want_ids)
    print("route weights:", " ".join(f"{w:.9f}" for w in want_weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
