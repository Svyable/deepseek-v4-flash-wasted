#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Freeze Gate-I/V8 final-head checkpoint and pinned-source surface.

This gate is intentionally header/source-only. It proves the exact five
checkpoint records and exact 0731 source bodies that define the final
HyperConnection collapse -> RMSNorm -> vocabulary head path. It does not read
any tensor payload or claim final hidden/logit parity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8_probe_surface as probe  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SOURCE_SHA256 = "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"
SHARD = "model-00045-of-00048.safetensors"

EXPECTED_TENSORS = {
    "hc_head_base": ("F32", [4], [0, 16]),
    "hc_head_fn": ("F32", [4, 16384], [16, 262160]),
    "hc_head_scale": ("F32", [1], [262160, 262164]),
    "head.weight": ("BF16", [129280, 4096], [262164, 1059323924]),
    "norm.weight": ("BF16", [4096], [1059323924, 1059332116]),
}

EXPECTED_METHODS = {
    "RMSNorm.__init__": "82702085674d2331aba475cfb0e2d540d7102ac74d74b16f62245a94358a970d",
    "RMSNorm.forward": "8038d3c6f16015ff357534366da3aac1c0fd06d737ffa61d93548ea663748d07",
    "Block.__init__": "9ff975e2aea5177c7cadeb77eea43997c1924c090fd5326d0a3963a1e9a61cd9",
    "Block.hc_head": "1ed85047f151da017a1d1bab28996b0db7e5bb3427756fefc2f9fd0969613c24",
    "ParallelHead.__init__": "e0648d7c821c0570aa86e3fa69d095c6184464458fd56366040b7c269fb242c8",
    "ParallelHead.forward": "a320818b11d65670b4be856e220810f3011bfee48a1c6fdc5540daa643908d8e",
    "Transformer.__init__": "697feecd1a78b72db4a2236aa758ca712ad974dcf32f2ff7d3ce069bf6b62755",
    "Transformer.forward": "e054cad6ca3ea41755850aa0dfae2b49cc43141ffdf922e2a39fbad01396c667",
}

SEMANTIC_CONTRACT = {
    "order": ["hc_head", "rms_norm", "vocab_head"],
    "hc_head": {
        "input_logical_shape": [4, 4096],
        "flatten_to_f32": True,
        "rms_stat_over": 16384,
        "mixes": "F.linear(flat_f32, hc_head_fn) * rsqrt(mean(square(flat_f32)) + norm_eps)",
        "weights": "sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps",
        "collapse": "sum(weights[j] * x[j], j=0..3)",
        "output_cast": "input dtype (BF16 for the checkpoint path)",
    },
    "rms_norm": {
        "compute_dtype": "F32",
        "stat_axis": -1,
        "formula": "x * rsqrt(mean(x^2) + eps)",
        "weight": "norm.weight",
        "output_cast": "input dtype",
    },
    "vocab_head": {
        "input_cast": "F32",
        "weight": "head.weight",
        "checkpoint_weight_dtype": "BF16",
        "logical_weight_shape": [129280, 4096],
        "output_dtype": "F32",
    },
}


def payload_bytes(dtype: str, shape: list[int]) -> int:
    size = {"BF16": 2, "F32": 4}[dtype]
    n = 1
    for dim in shape:
        n *= dim
    return n * size


def _method_map(methods: list[dict]) -> dict[str, dict]:
    out = {}
    for item in methods:
        symbol = item.get("symbol")
        if symbol in out:
            raise ValueError(f"duplicate source symbol {symbol}")
        out[symbol] = item
    return out


def validate_surface(surface: dict) -> None:
    if surface.get("schema_version") != 1:
        raise ValueError("unsupported V8 surface schema")
    if surface.get("gate") != "Gate I / V8 final-head surface":
        raise ValueError("unexpected V8 gate identity")
    if surface.get("evidence_state") != "CHECKPOINT-HEADER-AND-PINNED-SOURCE-VERIFIED":
        raise ValueError("V8 surface evidence state drifted")
    if surface.get("model") != MODEL or surface.get("revision") != REVISION:
        raise ValueError("V8 model/revision drifted")

    tensors = surface.get("checkpoint_tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(EXPECTED_TENSORS):
        raise ValueError("V8 tensor set drifted")
    for name, (dtype, shape, offsets) in EXPECTED_TENSORS.items():
        item = tensors[name]
        if item.get("name") != name or item.get("shard") != SHARD:
            raise ValueError(f"{name}: name/shard drifted")
        if item.get("dtype") != dtype or item.get("shape") != shape:
            raise ValueError(f"{name}: dtype/shape drifted")
        if item.get("data_offsets") != offsets:
            raise ValueError(f"{name}: checkpoint offsets drifted")
        if item.get("payload_bytes") != payload_bytes(dtype, shape):
            raise ValueError(f"{name}: payload-byte geometry drifted")

    source = surface.get("official_source") or {}
    if source.get("path") != "inference/model.py" or source.get("sha256") != SOURCE_SHA256:
        raise ValueError("pinned inference/model.py identity drifted")
    methods = source.get("load_bearing_methods")
    if not isinstance(methods, dict) or set(methods) != set(EXPECTED_METHODS):
        raise ValueError("V8 source-method set drifted")
    for symbol, digest in EXPECTED_METHODS.items():
        item = methods[symbol]
        if item.get("source_sha256") != digest:
            raise ValueError(f"{symbol}: pinned source-body hash drifted")
        if not isinstance(item.get("lineno"), int) or not isinstance(item.get("end_lineno"), int):
            raise ValueError(f"{symbol}: source line provenance missing")

    if surface.get("semantic_contract") != SEMANTIC_CONTRACT:
        raise ValueError("V8 semantic contract drifted")
    nonclaims = " ".join(surface.get("non_claims") or [])
    for required in ("no tensor payload", "no final hidden", "no logits", "no 43-layer"):
        if required not in nonclaims:
            raise ValueError(f"V8 non-claim missing: {required}")

    header_sha = surface.get("checkpoint_header_stub_sha256")
    if not isinstance(header_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", header_sha):
        raise ValueError("V8 shard-header provenance SHA is invalid")


def build_surface(snapshot: str, source_path: str) -> dict:
    candidates, manifest = probe.checkpoint_surface(snapshot)
    if manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
        raise ValueError("header snapshot model/revision drifted")

    by_name = {item["name"]: item for item in candidates}
    # The broad root discovery currently contains exactly the five Gate-I
    # records plus embed.weight. Treat any additional head/norm/hc candidate as
    # a review event instead of silently ignoring it.
    if set(by_name) != set(EXPECTED_TENSORS) | {"embed.weight"}:
        raise ValueError(f"unexpected root final-head candidates: {sorted(by_name)}")

    tensors = {}
    for name, (dtype, shape, offsets) in EXPECTED_TENSORS.items():
        item = deepcopy(by_name[name])
        if item["shard"] != SHARD or item["dtype"] != dtype or \
           item["shape"] != shape or item["data_offsets"] != offsets:
            raise ValueError(f"{name}: measured checkpoint record disagrees with frozen contract")
        item["payload_bytes"] = payload_bytes(dtype, shape)
        tensors[name] = item

    shard_meta = (manifest.get("shards") or {}).get(SHARD) or {}
    header_sha = shard_meta.get("header_stub_sha256")
    if not isinstance(header_sha, str):
        raise ValueError("shard-45 header provenance is missing")

    methods, source_sha = probe.source_methods(source_path)
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"pinned source SHA {source_sha} != expected {SOURCE_SHA256}")
    available = _method_map(methods)
    selected = {}
    for symbol, digest in EXPECTED_METHODS.items():
        item = available.get(symbol)
        if not item or item.get("source_sha256") != digest:
            raise ValueError(f"{symbol}: pinned source body disagrees with frozen contract")
        selected[symbol] = {
            "lineno": item["lineno"],
            "end_lineno": item["end_lineno"],
            "source_sha256": item["source_sha256"],
        }

    surface = {
        "schema_version": 1,
        "gate": "Gate I / V8 final-head surface",
        "evidence_state": "CHECKPOINT-HEADER-AND-PINNED-SOURCE-VERIFIED",
        "model": MODEL,
        "revision": REVISION,
        "checkpoint_header_stub_sha256": header_sha,
        "checkpoint_tensors": tensors,
        "official_source": {
            "path": "inference/model.py",
            "sha256": source_sha,
            "load_bearing_methods": selected,
            "note": "0731 ParallelHead has __init__ + forward; do not substitute current-main get_logits/hc_head refactors.",
        },
        "semantic_contract": deepcopy(SEMANTIC_CONTRACT),
        "non_claims": [
            "no tensor payload was read by this gate",
            "no final hidden state was evaluated",
            "no logits were evaluated",
            "no 43-layer forward was evaluated",
            "V8 numerical parity is not passed by this surface artifact",
        ],
    }
    validate_surface(surface)
    return surface


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="header-only immutable checkpoint snapshot")
    ap.add_argument("source", help="exact pinned 0731 inference/model.py")
    ap.add_argument("out", help="output reference JSON")
    args = ap.parse_args(argv)
    surface = build_surface(args.snapshot, args.source)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(surface, f, indent=2, sort_keys=True)
        f.write("\n")
    print("PASS V8 final-head header/source surface")
    print("tensors:", ", ".join(surface["checkpoint_tensors"]))
    print("source sha256:", surface["official_source"]["sha256"])
    print("NO PAYLOAD / NO LOGIT CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
