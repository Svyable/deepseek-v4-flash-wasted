#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free contracts for the generic V8 per-layer header classifier."""

from __future__ import annotations

import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import v8_layer_surface as layer  # noqa: E402


def groups(**counts):
    out = {}
    for name in sorted(layer.ALLOWED):
        out[name] = {"tensor_count": counts.get(name, 0), "payload_bytes": 0,
                     "dtype_counts": {}, "names": []}
    return out


def learned_rows(n=4):
    return [
        {"name": f"layers.{n}.ffn.gate.weight", "subsystem": "router"},
        {"name": f"layers.{n}.ffn.gate.bias", "subsystem": "router"},
    ]


def expert_rows(n=4):
    rows = []
    for e in range(2):
        for m in ("w1", "w2", "w3"):
            rows.append({"name": f"layers.{n}.ffn.experts.{e}.{m}.weight",
                         "subsystem": "routed_expert", "expert": e,
                         "matrix": m, "is_scale": False})
            rows.append({"name": f"layers.{n}.ffn.experts.{e}.{m}.scale",
                         "subsystem": "routed_expert", "expert": e,
                         "matrix": m, "is_scale": True})
    return rows


def shared_rows(n=4):
    return [
        {"name": f"layers.{n}.ffn.shared_experts.{m}.{suffix}",
         "subsystem": "shared_expert"}
        for m in ("w1", "w2", "w3") for suffix in ("weight", "scale")
    ]


def main():
    # Discovery must be semantic rather than path-hardcoded. The main-stack
    # prefix is accepted even when the real config carries trailing speculative
    # entries; a second plausible schedule remains an ambiguity and fails.
    raw = {"model": {"attention": {"compress_ratios": [0, 4, 128, 4, 128, 4]}}}
    ratios, path = layer.find_compress_ratios(raw, 4)
    if ratios != [0, 4, 128, 4] or path != "model.attention.compress_ratios":
        raise AssertionError((ratios, path))
    bad = copy.deepcopy(raw)
    bad["other_compress_ratio"] = [0, 4, 128, 4]
    try:
        layer.find_compress_ratios(bad, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous compression-ratio lists were accepted")

    cfg = {"n_routed": 2, "n_shared": 1}
    base_rows = expert_rows() + shared_rows() + learned_rows()

    # ratio4 learned: compressor + sparse indexer both load-bearing.
    g = groups(attention=11, compressor=4, csa_indexer=7, mhc=6, norm=4,
               routed_expert=12, shared_expert=6, router=2)
    result = layer.check_structure(4, 4, True, g, base_rows, cfg)
    if result["routing_contract"] != "learned" or result["compression_contract"] != "ratio4":
        raise AssertionError(result)

    # ratio0 must refuse accidentally inherited compressed-attention tensors.
    try:
        layer.check_structure(4, 0, True, g, base_rows, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("ratio0 accepted compressor/indexer tensors")

    # ratio128 is compressed dense history, not sparse CSA selection.
    g128 = groups(attention=11, compressor=4, mhc=6, norm=4,
                  routed_expert=12, shared_expert=6, router=2)
    layer.check_structure(4, 128, True, g128, base_rows, cfg)
    g128["csa_indexer"]["tensor_count"] = 1
    try:
        layer.check_structure(4, 128, True, g128, base_rows, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("ratio128 accepted a sparse CSA indexer")

    # Learned routing must be exactly gate.weight + gate.bias and no hash map.
    wrong_router = expert_rows() + shared_rows() + [
        {"name": "layers.4.ffn.gate.weight", "subsystem": "router"}]
    try:
        layer.check_structure(4, 4, True, g, wrong_router, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("learned router without correction bias was accepted")

    # Expert completeness is exact, not merely a tensor-count check.
    broken = expert_rows()[:-1]
    try:
        layer.check_experts(broken, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("routed expert missing one scale tensor was accepted")

    print("PASS V8 generic layer surface model-free contracts")
    print("mutations refused: speculative-tail ambiguity, ratio0 compression, ratio128 indexer, router bias, expert scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
