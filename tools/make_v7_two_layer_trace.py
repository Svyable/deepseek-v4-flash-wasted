#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the first real consecutive two-layer V7 localization trace offline.

No checkpoint access occurs here. All branch outputs and router decisions are
already frozen by earlier gates. This tool binds those real boundaries across
layers 3 and 4 while recomputing HyperConnection composition two ways:

* ``expected`` uses the independent F32 Python HC oracle; and
* ``runtime`` uses WASTE's C scalar HC reference.

Both paths must reproduce every frozen HC boundary exactly before a trace is
written. Attention/MoE/router arithmetic is intentionally reused from its
separate checkpoint-backed fixtures rather than re-proved here. V7's purpose is
consecutive-layer composition and first-divergence localization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import make_v7_layer4_route_fixture as l4route  # noqa: E402
import v7_hc_oracle as hc_oracle  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24
TOPK = 6

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
L3HC = os.path.join(BASE, "v6_hc_composition_real")
L3ATT = os.path.join(BASE, "v6_attention_branch_real")
L3MOE = os.path.join(BASE, "v6_moe_branch_real")
L4ROUTE = os.path.join(BASE, "v7_layer4_route_real")
L4FULL = os.path.join(BASE, "v7_layer4_full_real")

BOUNDARIES = (
    ("input", "BF16", [HC, DIM]),
    ("attn_pre", "BF16", [DIM]),
    ("after_attn", "BF16", [HC, DIM]),
    ("ffn_pre", "BF16", [DIM]),
    ("router_ids", "U32", [TOPK]),
    ("moe_branch", "BF16", [DIM]),
    ("output", "BF16", [HC, DIM]),
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_u16(path: str, count: int) -> list[int]:
    data = open(path, "rb").read()
    if len(data) != count * 2:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 2}")
    return list(struct.unpack(f"<{count}H", data))


def read_u32(path: str, count: int) -> list[int]:
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}I", data))


def read_f32(path: str, count: int) -> list[float]:
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}f", data))


def write_u16(path: str, values: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_u32(path: str, values: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def checked_file(root: str, meta: dict, label: str, count: int,
                 reader=read_u16):
    path = os.path.join(root, meta.get("file", ""))
    if not os.path.isfile(path) or sha256_file(path) != meta.get("sha256"):
        raise ValueError(f"{label}: file missing or SHA mismatch")
    return path, reader(path, count)


def f32_param(root: str, params: dict, key: str, count: int) -> list[float]:
    meta = params.get(key) or {}
    path = os.path.join(root, meta.get("file", ""))
    want = meta.get("payload_sha256", meta.get("sha256"))
    if not os.path.isfile(path) or (want and sha256_file(path) != want):
        raise ValueError(f"{key}: parameter missing or SHA mismatch")
    return read_f32(path, count)


def bf16_floats(bits: list[int]) -> list[float]:
    return [chain.bf16_to_f32(v) for v in bits]


def exact_bits(label: str, got: list[int], want: list[int]) -> None:
    if got == want:
        return
    i = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
    raise ValueError(f"{label} mismatch at {i}: {got[i]:04x}!={want[i]:04x}")


def exact_f32(label: str, got: list[float], want: list[float]) -> None:
    gb = [chain.f32_bits(v) for v in got]
    wb = [chain.f32_bits(v) for v in want]
    if gb == wb:
        return
    i = next(i for i, (a, b) in enumerate(zip(gb, wb)) if a != b)
    raise ValueError(f"{label} F32 mismatch at {i}: {gb[i]:08x}!={wb[i]:08x}")


def load_frozen_sources() -> dict:
    prov_paths = {
        "l3_hc": os.path.join(L3HC, "provenance.json"),
        "l3_att": os.path.join(L3ATT, "provenance.json"),
        "l3_moe": os.path.join(L3MOE, "provenance.json"),
        "l4_route": os.path.join(L4ROUTE, "provenance.json"),
        "l4_full": os.path.join(L4FULL, "provenance.json"),
    }
    for label, path in prov_paths.items():
        if not os.path.isfile(path):
            raise ValueError(f"{label}: frozen provenance missing")
    prov = {k: load_json(v) for k, v in prov_paths.items()}
    for label, p in prov.items():
        if p.get("model") != MODEL or p.get("revision") != REV:
            raise ValueError(f"{label}: model/revision drifted")

    hcp, hcs = prov["l3_hc"]["parameters"], prov["l3_hc"]["states"]
    l3_input_path, l3_input = checked_file(
        L3HC, {"file": hcs["residual"]["file"], "sha256": hcs["residual"]["sha256"]},
        "layer3 input", FLAT)
    l3_attn_pre_path, l3_attn_pre = checked_file(
        L3HC, {"file": hcs["attn_pre"]["file"], "sha256": hcs["attn_pre"]["sha256"]},
        "layer3 attn_pre", DIM)
    l3_att_meta = prov["l3_att"]["outputs"]["branch"]
    l3_att_branch_path, l3_att_branch = checked_file(
        L3ATT, l3_att_meta, "layer3 real attention branch", DIM)
    l3_moe_dep = prov["l3_moe"]["input_dependency"]
    l3_ffn_path, l3_ffn_pre = checked_file(
        L3MOE, l3_moe_dep, "layer3 ffn_pre", DIM)
    l3_ids_meta = prov["l3_moe"]["outputs"]["ids"]
    l3_ids_path, l3_ids = checked_file(
        L3MOE, l3_ids_meta, "layer3 router ids", TOPK, reader=read_u32)
    l3_moe_meta = prov["l3_moe"]["outputs"]["moe_branch"]
    l3_moe_path, l3_moe = checked_file(L3MOE, l3_moe_meta, "layer3 MoE branch", DIM)
    l3_out_meta = prov["l3_moe"]["outputs"]["layer3_final"]
    l3_out_path, l3_output = checked_file(L3MOE, l3_out_meta, "layer3 output", FLAT)

    l3_params = {
        "attn_fn": f32_param(L3HC, hcp, "attn_fn", MIX * FLAT),
        "attn_scale": f32_param(L3HC, hcp, "attn_scale", 3),
        "attn_base": f32_param(L3HC, hcp, "attn_base", MIX),
        "ffn_fn": f32_param(L3HC, hcp, "ffn_fn", MIX * FLAT),
        "ffn_scale": f32_param(L3HC, hcp, "ffn_scale", 3),
        "ffn_base": f32_param(L3HC, hcp, "ffn_base", MIX),
    }

    l4p = prov["l4_route"]
    l4_outputs = l4p["outputs"]
    l4_input_path, l4_input = checked_file(L4ROUTE, l4_outputs["input"], "layer4 input", FLAT)
    l4_attn_pre_path, l4_attn_pre = checked_file(
        L4ROUTE, l4_outputs["attn_pre"], "layer4 attn_pre", DIM)
    l4_att_path, l4_att_branch = checked_file(
        L4ROUTE, l4_outputs["attention_branch"], "layer4 real attention branch", DIM)
    l4_after_path, l4_after = checked_file(
        L4ROUTE, l4_outputs["after_attn"], "layer4 after_attn", FLAT)
    l4_ffn_path, l4_ffn_pre = checked_file(
        L4ROUTE, l4_outputs["ffn_pre"], "layer4 ffn_pre", DIM)
    l4_router = l4p["router"]
    l4_ids_path = os.path.join(L4ROUTE, l4_router["ids_file"])
    if sha256_file(l4_ids_path) != l4_router["ids_sha256"]:
        raise ValueError("layer4 router ids SHA drifted")
    l4_ids = read_u32(l4_ids_path, TOPK)
    l4_full_out = prov["l4_full"]["outputs"]
    l4_moe_path, l4_moe = checked_file(
        L4FULL, l4_full_out["moe_branch"], "layer4 MoE branch", DIM)
    l4_out_path, l4_output = checked_file(
        L4FULL, l4_full_out["layer4_final"], "layer4 output", FLAT)

    l4_trunk = l4p["checkpoint_tensors"]["hyperconnection_and_router"]
    def l4_param(key: str, count: int) -> list[float]:
        meta = l4_trunk[key]
        path = os.path.join(L4ROUTE, key.replace("_", "-") + ".f32.bin")
        if not os.path.isfile(path) or sha256_file(path) != meta["payload_sha256"]:
            raise ValueError(f"layer4 {key}: compact parameter SHA drifted")
        return read_f32(path, count)

    l4_params = {
        "attn_fn": l4_param("attn_fn", MIX * FLAT),
        "attn_scale": l4_param("attn_scale", 3),
        "attn_base": l4_param("attn_base", MIX),
        "ffn_fn": l4_param("ffn_fn", MIX * FLAT),
        "ffn_scale": l4_param("ffn_scale", 3),
        "ffn_base": l4_param("ffn_base", MIX),
    }

    if l3_output != l4_input:
        raise ValueError("frozen layer3 output is not byte-identical to frozen layer4 input")

    return {
        "prov_paths": prov_paths,
        "l3": {
            "input": l3_input, "attn_pre": l3_attn_pre,
            "attn_branch": l3_att_branch, "ffn_pre": l3_ffn_pre,
            "router_ids": l3_ids, "moe_branch": l3_moe, "output": l3_output,
            "params": l3_params,
        },
        "l4": {
            "input": l4_input, "attn_pre": l4_attn_pre,
            "attn_branch": l4_att_branch, "after_attn": l4_after,
            "ffn_pre": l4_ffn_pre, "router_ids": l4_ids,
            "moe_branch": l4_moe, "output": l4_output, "params": l4_params,
        },
    }


def independent_layer(frozen: dict) -> dict:
    residual = bf16_floats(frozen["input"])
    attn = bf16_floats(frozen["attn_branch"])
    moe = bf16_floats(frozen["moe_branch"])
    p = frozen["params"]

    apre, apost, acomb, _ = hc_oracle.params(
        residual, p["attn_fn"], p["attn_scale"], p["attn_base"], dim=DIM)
    attn_pre, _ = chain.cast_bf16(hc_oracle.pre_y(residual, apre, dim=DIM))
    exact_bits("independent attn_pre", attn_pre, frozen["attn_pre"])
    after_f = hc_oracle.post_y(attn, residual, apost, acomb, dim=DIM)
    after_bits, after = chain.cast_bf16(after_f)
    if "after_attn" in frozen:
        exact_bits("independent after_attn", after_bits, frozen["after_attn"])

    fpre, fpost, fcomb, _ = hc_oracle.params(
        after, p["ffn_fn"], p["ffn_scale"], p["ffn_base"], dim=DIM)
    ffn_pre, _ = chain.cast_bf16(hc_oracle.pre_y(after, fpre, dim=DIM))
    exact_bits("independent ffn_pre", ffn_pre, frozen["ffn_pre"])
    final_f = hc_oracle.post_y(moe, after, fpost, fcomb, dim=DIM)
    final_bits, _ = chain.cast_bf16(final_f)
    exact_bits("independent layer output", final_bits, frozen["output"])
    return {
        "input": frozen["input"], "attn_pre": attn_pre,
        "after_attn": after_bits, "ffn_pre": ffn_pre,
        "router_ids": frozen["router_ids"], "moe_branch": frozen["moe_branch"],
        "output": final_bits,
    }


def runtime_layer(frozen: dict, pre_fn, post_fn) -> dict:
    residual = bf16_floats(frozen["input"])
    attn = bf16_floats(frozen["attn_branch"])
    moe = bf16_floats(frozen["moe_branch"])
    p = frozen["params"]

    attn_pre_f, apost, acomb = l4route.c_pre(
        pre_fn, residual, p["attn_fn"], p["attn_scale"], p["attn_base"])
    attn_pre, _ = chain.cast_bf16(attn_pre_f)
    exact_bits("runtime attn_pre", attn_pre, frozen["attn_pre"])
    after_f = l4route.c_post(post_fn, attn, residual, apost, acomb)
    after_bits, after = chain.cast_bf16(after_f)
    if "after_attn" in frozen:
        exact_bits("runtime after_attn", after_bits, frozen["after_attn"])

    ffn_pre_f, fpost, fcomb = l4route.c_pre(
        pre_fn, after, p["ffn_fn"], p["ffn_scale"], p["ffn_base"])
    ffn_pre, _ = chain.cast_bf16(ffn_pre_f)
    exact_bits("runtime ffn_pre", ffn_pre, frozen["ffn_pre"])
    final_f = l4route.c_post(post_fn, moe, after, fpost, fcomb)
    final_bits, _ = chain.cast_bf16(final_f)
    exact_bits("runtime layer output", final_bits, frozen["output"])
    return {
        "input": frozen["input"], "attn_pre": attn_pre,
        "after_attn": after_bits, "ffn_pre": ffn_pre,
        "router_ids": frozen["router_ids"], "moe_branch": frozen["moe_branch"],
        "output": final_bits,
    }


def write_trace(root: str, name: str, layers: list[tuple[int, str, dict]]) -> str:
    os.makedirs(root, exist_ok=True)
    raw_layers = []
    for number, structural_class, values in layers:
        layer_dir = os.path.join(root, f"layer{number}")
        os.makedirs(layer_dir, exist_ok=True)
        raw_boundaries = []
        for boundary, dtype, shape in BOUNDARIES:
            ext = "u32.bin" if dtype == "U32" else "bf16.bin"
            rel = os.path.join(f"layer{number}", f"{boundary}.{ext}")
            path = os.path.join(root, rel)
            if dtype == "U32":
                write_u32(path, values[boundary])
            else:
                write_u16(path, values[boundary])
            raw_boundaries.append({
                "name": boundary,
                "file": rel.replace(os.sep, "/"),
                "dtype": dtype,
                "shape": shape,
                "sha256": sha256_file(path),
            })
        raw_layers.append({
            "layer": number,
            "structural_class": structural_class,
            "boundaries": raw_boundaries,
        })

    manifest = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REV,
        "trace_name": name,
        "layers": raw_layers,
    }
    path = os.path.join(root, "trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="output fixture directory")
    args = ap.parse_args(argv)

    src = load_frozen_sources()
    expected_l3 = independent_layer(src["l3"])
    expected_l4 = independent_layer(src["l4"])

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v7-two-layer-") as td:
        pre_fn, post_fn, _, _ = l4route.build_state_lib(td)
        runtime_l3 = runtime_layer(src["l3"], pre_fn, post_fn)
        runtime_l4 = runtime_layer(src["l4"], pre_fn, post_fn)

    # The chained endpoint is the V7 invariant, not an incidental copy.
    for label, layers in (("expected", (expected_l3, expected_l4)),
                          ("runtime", (runtime_l3, runtime_l4))):
        exact_bits(f"{label} layer3-output/layer4-input chain",
                   layers[0]["output"], layers[1]["input"])

    expected_manifest = write_trace(
        os.path.join(args.out_dir, "expected"), "real-layer3-to-layer4-expected",
        [(3, "ratio128-learned", expected_l3),
         (4, "ratio4-learned", expected_l4)])
    runtime_manifest = write_trace(
        os.path.join(args.out_dir, "runtime"), "real-layer3-to-layer4-runtime",
        [(3, "ratio128-learned", runtime_l3),
         (4, "ratio4-learned", runtime_l4)])

    provenance = {
        "schema_version": 1,
        "gate": "V7 first real consecutive two-layer localization trace",
        "evidence_state": "END-TO-END-VERIFIED-TWO-LAYER-SCALAR-BOUNDARIES",
        "model": MODEL,
        "revision": REV,
        "layers": [3, 4],
        "structural_classes": ["ratio128-learned", "ratio4-learned"],
        "boundary_order": [b[0] for b in BOUNDARIES],
        "expected_manifest": os.path.relpath(expected_manifest, args.out_dir).replace(os.sep, "/"),
        "runtime_manifest": os.path.relpath(runtime_manifest, args.out_dir).replace(os.sep, "/"),
        "source_provenance_sha256": {
            label: sha256_file(path) for label, path in src["prov_paths"].items()
        },
        "composition_paths": {
            "expected_hyperconnection": "independent Python F32 oracle using system libm; no WASTE source linkage",
            "runtime_hyperconnection": "src/deepseek_v4_mhc_ref.c through the scalar C reference",
            "attention_moe_router": "reuse already-frozen real checkpoint-backed boundaries; primitive arithmetic remains owned by Gate E/F/H and the complete layer-4 fixture",
        },
        "chain": {
            "layer3_output_sha256": sha256_file(os.path.join(args.out_dir, "expected", "layer3", "output.bf16.bin")),
            "layer4_input_sha256": sha256_file(os.path.join(args.out_dir, "expected", "layer4", "input.bf16.bin")),
            "byte_identical": True,
        },
        "non_claims": [
            "two consecutive layers only, not all 43 transformer layers",
            "position 0 does not re-prove non-empty ratio-4 compressor/indexer history; Gate E owns that primitive evidence",
            "no final-model logits, generation, converted-container/cache identity, RAM or throughput claim",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    print("PASS built real V7 layer3->layer4 expected/runtime trace")
    print("chain sha256:", provenance["chain"]["layer3_output_sha256"])
    print("layer4 final sha256:", sha256_file(os.path.join(args.out_dir, "expected", "layer4", "output.bf16.bin")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
