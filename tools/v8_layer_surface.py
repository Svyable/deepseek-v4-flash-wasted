#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Measure one DeepSeek-V4 main-layer surface from immutable headers.

This is the generic header gate for V8 propagation. It derives the layer's
compression ratio and routing mode from config/checkpoint truth, summarizes all
main-layer tensor families, and refuses unresolved/unclassified/malformed expert
records. It reads safetensors headers only; no tensor payload is materialized.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import inventory  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
ALLOWED = {
    "attention", "compressor", "csa_indexer", "mhc", "norm",
    "routed_expert", "shared_expert", "router", "router_hash",
}


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk(value, path + (str(key),))
    elif isinstance(obj, list):
        yield path, obj


def find_compress_ratios(raw: dict, n_layers: int) -> tuple[list[int], str]:
    """Find the unique compression schedule and return its main-stack prefix.

    The 0731 config may carry trailing speculative-path entries after the 43
    main decoder layers. Those entries must not be mistaken for extra main
    layers, but they also must not make the real schedule undiscoverable.
    """
    candidates = []
    allowed = {0, 4, 128}
    for path, value in _walk(raw):
        key = path[-1].lower() if path else ""
        if "compress" not in key or "ratio" not in key:
            continue
        if len(value) < n_layers or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
            continue
        if any(x not in allowed for x in value[:n_layers]):
            continue
        candidates.append((value, ".".join(path)))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one compression-ratio list with >= {n_layers} entries, found "
            f"{[(p, len(v)) for v, p in candidates]}")
    ratios, path = candidates[0]
    return list(ratios[:n_layers]), path


def structural_class(ratio: int, learned: bool) -> str:
    return f"ratio{ratio}-{'learned' if learned else 'hash'}"


def group_rows(rows: list[dict]) -> dict:
    groups = {}
    for subsystem in sorted(ALLOWED):
        members = [r for r in rows if r["subsystem"] == subsystem]
        dtype_counts = Counter(r.get("dtype") for r in members)
        groups[subsystem] = {
            "tensor_count": len(members),
            "payload_bytes": sum(r.get("stored_bytes") or 0 for r in members),
            "dtype_counts": dict(sorted((k, v) for k, v in dtype_counts.items() if k is not None)),
            "names": [r["name"] for r in members[:24]],
        }
        if len(members) > 24:
            groups[subsystem]["names_sample_truncated"] = True
    return groups


def check_experts(rows: list[dict], cfg: dict) -> dict:
    by_expert = defaultdict(lambda: defaultdict(lambda: {"weight": 0, "scale": 0}))
    for row in rows:
        if row["subsystem"] != "routed_expert":
            continue
        expert = row.get("expert")
        matrix = row.get("matrix")
        if expert is None or matrix not in {"w1", "w2", "w3"}:
            raise ValueError(f"malformed routed-expert name: {row['name']}")
        by_expert[expert][matrix]["scale" if row.get("is_scale") else "weight"] += 1
    want = int(cfg["n_routed"])
    if set(by_expert) != set(range(want)):
        raise ValueError(
            f"routed expert IDs drifted: have {len(by_expert)}, expected exactly 0..{want - 1}")
    for expert, matrices in sorted(by_expert.items()):
        for matrix in ("w1", "w2", "w3"):
            slot = matrices.get(matrix)
            if slot != {"weight": 1, "scale": 1}:
                raise ValueError(f"expert {expert} {matrix}: {slot}, expected one weight + one scale")
    return {
        "expert_count": want,
        "tensors_per_expert": 6,
        "all_have_w1_w2_w3_plus_scales": True,
    }


def check_shared(rows: list[dict], cfg: dict) -> dict:
    shared = [r for r in rows if r["subsystem"] == "shared_expert"]
    n_shared = int(cfg.get("n_shared") or 0)
    if n_shared == 0:
        if shared:
            raise ValueError("shared-expert tensors present but config says zero shared experts")
        return {"configured_shared_experts": 0, "tensor_count": 0, "complete": True}
    if n_shared != 1:
        raise ValueError(f"generic surface does not yet support n_shared={n_shared}")
    names = {r["name"].rsplit(".", 2)[-2] + "." + r["name"].rsplit(".", 1)[-1] for r in shared}
    want = {f"{m}.{suffix}" for m in ("w1", "w2", "w3") for suffix in ("weight", "scale")}
    if names != want:
        raise ValueError(f"shared expert surface {sorted(names)} != {sorted(want)}")
    return {"configured_shared_experts": 1, "tensor_count": 6, "complete": True}


def check_structure(layer: int, ratio: int, learned: bool, groups: dict, rows: list[dict], cfg: dict) -> dict:
    count = lambda name: groups[name]["tensor_count"]
    if count("attention") == 0:
        raise ValueError(f"layer {layer}: no attention tensors")
    if count("mhc") != 6:
        raise ValueError(f"layer {layer}: mHC tensor count {count('mhc')} != 6")
    if count("norm") == 0:
        raise ValueError(f"layer {layer}: no norm tensors")

    if ratio == 0:
        if count("compressor") or count("csa_indexer"):
            raise ValueError("ratio0 layer unexpectedly exposes compressor/indexer tensors")
    elif ratio == 4:
        if count("compressor") == 0 or count("csa_indexer") == 0:
            raise ValueError("ratio4 layer requires compressor and CSA indexer surfaces")
    elif ratio == 128:
        if count("compressor") == 0 or count("csa_indexer") != 0:
            raise ValueError("ratio128 layer requires compressor and no sparse CSA indexer")
    else:
        raise ValueError(f"unsupported ratio {ratio}")

    hash_count = count("router_hash")
    router_count = count("router")
    if learned:
        if hash_count:
            raise ValueError(f"learned-routing layer {layer} unexpectedly has hash mapping")
        router_names = {r["name"] for r in rows if r["subsystem"] == "router"}
        want = {f"layers.{layer}.ffn.gate.weight", f"layers.{layer}.ffn.gate.bias"}
        if router_names != want:
            raise ValueError(f"learned router surface {sorted(router_names)} != {sorted(want)}")
    else:
        if hash_count == 0:
            raise ValueError(f"hash-routing layer {layer} lacks token->expert mapping")
        if router_count == 0:
            raise ValueError(f"hash-routing layer {layer} lacks score weight surface")

    expert = check_experts(rows, cfg)
    shared = check_shared(rows, cfg)
    return {
        "attention_present": True,
        "compression_contract": f"ratio{ratio}",
        "routing_contract": "learned" if learned else "hash",
        "mhc_two_parameter_sets": True,
        "routed_experts": expert,
        "shared_expert": shared,
    }


def build_surface(snapshot: str, layer: int) -> dict:
    idx, cfg, all_rows = inventory.build(snapshot)
    if cfg is None:
        raise ValueError("config.json is required")
    if not idx.have_headers:
        raise ValueError(f"header snapshot incomplete: {idx.missing_shards[:3]}")
    n_layers = cfg.get("layers")
    if not isinstance(n_layers, int) or not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} outside main stack 0..{(n_layers or 0) - 1}")
    if cfg.get("hidden") != 4096 or cfg.get("n_routed") != 256 or \
       cfg.get("top_k") != 6 or cfg.get("moe_inter") != 2048:
        raise ValueError("core 0731 config contract drifted")

    ratios, ratios_path = find_compress_ratios(cfg["_raw"], n_layers)
    ratio = ratios[layer]
    n_hash = cfg.get("n_hash")
    if not isinstance(n_hash, int) or n_hash < 0:
        raise ValueError("n_hash layer count is missing")
    learned = layer >= n_hash

    rows = [r for r in all_rows if r.get("module") == "main" and r.get("layer") == layer]
    if not rows:
        raise ValueError(f"layer {layer}: no main checkpoint tensors")
    bad = [r["name"] for r in rows if not r.get("resolved") or r.get("subsystem") not in ALLOWED]
    if bad:
        raise ValueError(f"layer {layer}: unresolved/unclassified rows: {bad[:5]}")
    prefix = f"layers.{layer}."
    if any(not r["name"].startswith(prefix) for r in rows):
        raise ValueError("layer selector admitted a tensor outside its namespace")

    groups = group_rows(rows)
    checks = check_structure(layer, ratio, learned, groups, rows, cfg)
    dtype_counts = Counter(r["dtype"] for r in rows)
    total_bytes = sum(r["stored_bytes"] for r in rows)
    surface = {
        "schema_version": 1,
        "gate": "Gate I/V8 generic layer header surface",
        "evidence_state": "CHECKPOINT-VERIFIED-HEADERS",
        "model": MODEL,
        "revision": REVISION,
        "layer": layer,
        "structural_class": structural_class(ratio, learned),
        "config_contract": {
            "num_hidden_layers": n_layers,
            "hidden_size": cfg["hidden"],
            "n_routed_experts": cfg["n_routed"],
            "n_shared_experts": cfg["n_shared"],
            "num_experts_per_tok": cfg["top_k"],
            "moe_intermediate_size": cfg["moe_inter"],
            "hash_layers": n_hash,
            "compress_ratio": ratio,
            "compress_ratios_path": ratios_path,
        },
        "checks": checks,
        "groups": groups,
        "surface": {
            "tensor_count": len(rows),
            "payload_bytes": total_bytes,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "subsystem_counts": {k: groups[k]["tensor_count"] for k in sorted(groups) if groups[k]["tensor_count"]},
            "subsystem_bytes": {k: groups[k]["payload_bytes"] for k in sorted(groups) if groups[k]["tensor_count"]},
        },
        "tensors": rows,
        "non_claims": [
            "header-only per-layer surface; no tensor payload arithmetic is claimed",
            "this does not prove execution from the prior layer state",
            "structural class is derived from immutable config plus measured tensor families",
        ],
    }
    return surface


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("layer", type=int)
    ap.add_argument("out")
    args = ap.parse_args(argv)
    surface = build_surface(args.snapshot, args.layer)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(surface, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"PASS V8 layer {args.layer} header surface: {surface['structural_class']}, "
        f"{surface['surface']['tensor_count']} tensors, {surface['surface']['payload_bytes']} bytes")
    for name, group in surface["groups"].items():
        if group["tensor_count"]:
            print(f"  {name}: {group['tensor_count']} tensors / {group['payload_bytes']} bytes")
    print("HEADER ONLY — NO LAYER EXECUTION CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
