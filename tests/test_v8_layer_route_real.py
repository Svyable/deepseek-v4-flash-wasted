#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Permanent no-network replay for a generic V8 learned-layer route fixture."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import test_v6_ffn_route_real as chain  # noqa: E402
import deepseek_v4_hc_oracle as hc_oracle  # noqa: E402

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24
EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5
DEFAULT = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v8_layer5_route_real")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16(path, count):
    return chain.read_u16(path, count)


def read_f32(path, count):
    return chain.read_f32(path, count)


def read_u32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*4}")
    import struct
    return list(struct.unpack(f"<{count}I", data))


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", nargs="?", default=DEFAULT)
    args = ap.parse_args(argv)
    fixture = os.path.abspath(args.fixture)
    prov_path = os.path.join(fixture, "provenance.json")
    if not os.path.isfile(prov_path):
        print(f"SKIP: generic V8 route fixture not frozen: {fixture}")
        return 77

    try:
        p = json.load(open(prov_path, encoding="utf-8"))
        if p.get("revision") != REV or p.get("gate") != "Gate I/V8 generic one-position layer through router":
            raise AssertionError("generic route identity/revision drifted")
        layer = p.get("layer")
        if not isinstance(layer, int) or not 3 <= layer < 43:
            raise AssertionError("generic route layer identity invalid")
        if p.get("structural_class") not in {"ratio0-learned", "ratio4-learned", "ratio128-learned"}:
            raise AssertionError("generic route structural class invalid")

        dep = p.get("input_dependency") or {}
        source = os.path.join(REPO, dep.get("source_file", ""))
        inp = os.path.join(fixture, dep.get("fixture_file", ""))
        if not os.path.isfile(source) or not os.path.isfile(inp):
            raise AssertionError("generic route input/source missing")
        if sha(source) != dep.get("source_sha256") or sha(inp) != dep.get("fixture_sha256") or \
           sha(source) != sha(inp) or dep.get("byte_identical") is not True:
            raise AssertionError("generic route input is not byte-identical to prior output")

        sdep = p.get("checkpoint_surface_dependency") or {}
        surface = os.path.join(REPO, sdep.get("file", ""))
        if not os.path.isfile(surface) or sha(surface) != sdep.get("sha256"):
            raise AssertionError("generic route header-surface dependency drifted")
        sf = json.load(open(surface, encoding="utf-8"))
        if sf.get("layer") != layer or sf.get("revision") != REV or \
           sf.get("structural_class") != p.get("structural_class"):
            raise AssertionError("generic route surface identity disagrees with provenance")

        outputs = p.get("outputs") or {}
        counts = {
            "input": FLAT,
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
            m = outputs.get(key) or {}
            path = os.path.join(fixture, m.get("file", ""))
            if not os.path.isfile(path) or sha(path) != m.get("sha256"):
                raise AssertionError(f"{key}: missing/SHA mismatch")
            read_u16(path, count)
            paths[key] = path

        trunk = (p.get("checkpoint_tensors") or {}).get("hyperconnection_and_router") or {}
        params = {}
        for key, filename, count in (
            ("attn_fn", "attn-fn.f32.bin", MIX * FLAT),
            ("attn_scale", "attn-scale.f32.bin", 3),
            ("attn_base", "attn-base.f32.bin", MIX),
            ("ffn_fn", "ffn-fn.f32.bin", MIX * FLAT),
            ("ffn_scale", "ffn-scale.f32.bin", 3),
            ("ffn_base", "ffn-base.f32.bin", MIX),
            ("gate_bias", "gate-bias.f32.bin", EXPERTS),
        ):
            path = os.path.join(fixture, filename)
            m = trunk.get(key) or {}
            if not os.path.isfile(path) or sha(path) != m.get("payload_sha256"):
                raise AssertionError(f"{key}: frozen checkpoint payload SHA drifted")
            params[key] = read_f32(path, count)
        gate_path = os.path.join(fixture, "gate-weight.bf16.bin")
        if sha(gate_path) != (trunk.get("gate_weight") or {}).get("payload_sha256"):
            raise AssertionError("gate weight payload SHA drifted")
        gate_bits = read_u16(gate_path, EXPERTS * DIM)

        router = p.get("router") or {}
        ids_path = os.path.join(fixture, router.get("ids_file", ""))
        weights_path = os.path.join(fixture, router.get("weights_file", ""))
        if sha(ids_path) != router.get("ids_sha256") or sha(weights_path) != router.get("weights_sha256"):
            raise AssertionError("route output SHA drifted")
        want_ids = read_u32(ids_path, TOPK)
        want_weights = read_f32(weights_path, TOPK)
        if want_ids != router.get("ids_topk_order"):
            raise AssertionError("route ID bytes disagree with provenance")
        if [chain.f32_bits(x) for x in want_weights] != [chain.f32_bits(x) for x in router.get("runtime_weights_f32", [])]:
            raise AssertionError("route weight bytes disagree with provenance")

        cc = compiler()
        if not cc:
            print("SKIP: no C compiler available")
            return 77
        with tempfile.TemporaryDirectory(prefix=f"v8-layer{layer}-route-replay-") as td:
            lib = os.path.join(td, "libstate.so")
            cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-ffp-contract=off", "-shared", "-fPIC", "-I", os.path.join(REPO, "src"),
                   os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"),
                   os.path.join(REPO, "src", "deepseek_v4_router_ref.c"), "-lm", "-o", lib]
            built = subprocess.run(cmd, capture_output=True, text=True)
            if built.returncode:
                raise AssertionError(built.stdout + built.stderr)
            so = ctypes.CDLL(lib)
            pre = so.waste_ds_v4_hc_pre_ref
            pre.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint,
                            ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
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
                y = (ctypes.c_float * DIM)(); po = (ctypes.c_float * HC)(); co = (ctypes.c_float * (HC*HC))()
                rc = pre(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base), ctypes.c_float(1e-6), 20,
                         ctypes.c_float(1e-6), y, None, None, po, co)
                if rc: raise AssertionError("hc_pre rejected replay")
                return list(y), list(po), [list(co[i*HC:(i+1)*HC]) for i in range(HC)]

            def c_post(branch, residual, po, co):
                y = (ctypes.c_float * FLAT)()
                rc = post(as_cf(branch), as_cf(residual), DIM, as_cf(po),
                          as_cf([v for row in co for v in row]), y)
                if rc: raise AssertionError("hc_post rejected replay")
                return list(y)

            residual = [chain.bf16_to_f32(x) for x in read_u16(paths["input"], FLAT)]
            apf, apo, aco = c_pre(residual, params["attn_fn"], params["attn_scale"], params["attn_base"])
            ap_bits, _ = chain.cast_bf16(apf)
            if ap_bits != read_u16(paths["attn_pre"], DIM):
                raise AssertionError("generic replay attn_pre changed")
            opre, _, _, _ = hc_oracle.params(residual, params["attn_fn"], params["attn_scale"], params["attn_base"], dim=DIM)
            if chain.cast_bf16(hc_oracle.pre_y(residual, [chain.f32(x) for x in opre], dim=DIM))[0] != ap_bits:
                raise AssertionError("independent HC attn_pre no longer agrees")

            branch = [chain.bf16_to_f32(x) for x in read_u16(paths["attention_branch"], DIM)]
            after_f = c_post(branch, residual, apo, aco)
            after_bits, after = chain.cast_bf16(after_f)
            if after_bits != read_u16(paths["after_attn"], FLAT):
                raise AssertionError("generic replay after_attn changed")
            fpf, _, _ = c_pre(after, params["ffn_fn"], params["ffn_scale"], params["ffn_base"])
            fp_bits, fp = chain.cast_bf16(fpf)
            if fp_bits != read_u16(paths["ffn_pre"], DIM):
                raise AssertionError("generic replay ffn_pre changed")

            wb = (ctypes.c_uint16 * len(gate_bits))(*gate_bits)
            raw = (ctypes.c_float * EXPERTS)(); scores = (ctypes.c_float * EXPERTS)()
            if score(as_cf(fp), DIM, wb, raw, scores):
                raise AssertionError("router score rejected replay")
            ids = (ctypes.c_uint32 * TOPK)(); weights = (ctypes.c_float * TOPK)(); selection = (ctypes.c_float * EXPERTS)()
            if learned(scores, as_cf(params["gate_bias"]), ctypes.c_float(ROUTE_SCALE), ids, weights, selection):
                raise AssertionError("learned router rejected replay")
            got_ids, got_weights = list(ids), list(weights)
            if got_ids != want_ids:
                raise AssertionError(f"route IDs changed {got_ids} != {want_ids}")
            if [chain.f32_bits(x) for x in got_weights] != [chain.f32_bits(x) for x in want_weights]:
                raise AssertionError("exact scalar route weights changed")

            py_ids, py_weights, _, _, margin = chain.oracle_router(fp, gate_bits, params["gate_bias"])
            if py_ids != got_ids or max(abs(a-b) for a,b in zip(py_weights, got_weights)) > 1e-5:
                raise AssertionError("independent Python router no longer supports scalar route")
            if not margin > 1e-4:
                raise AssertionError("route top-k boundary became unstable")

        ac = p.get("attention_crosscheck") or {}
        if ac.get("python_vs_c_exact_head_bf16") != 64 * 512:
            raise AssertionError("acquisition lacks full attention head cross-check")
        nonclaims = " ".join(p.get("non_claims") or [])
        if "no completed compressed history" not in nonclaims or "Gate E" not in nonclaims:
            raise AssertionError("compressed-history evidence boundary missing")

        print(f"PASS V8 generic layer {layer} route replay")
        print("structural class:", p["structural_class"])
        print("input sha256:", dep["source_sha256"])
        print("selected experts:", want_ids)
        print("route weights:", " ".join(f"{x:.9f}" for x in want_weights))
        return 0
    except Exception as exc:
        print("FAIL:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
