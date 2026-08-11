#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Permanent no-network regression for the measured V8 layer-5 header surface."""

from __future__ import annotations

import copy
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import v8_layer_contract as contract  # noqa: E402

PATH = os.path.join(REPO, "reference", "deepseek-v4-flash-0731.layer5-v8.json")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIGEST = "1ba66b4f1461805c2a86f37bba280aac183e51daf99cf9db6b7d3f4ce8cc7064"


def main():
    try:
        s = json.load(open(PATH, encoding="utf-8"))
        if s.get("schema_version") != 1 or s.get("revision") != REV or s.get("layer") != 5:
            raise AssertionError("layer-5 compact surface identity drifted")
        if s.get("structural_class") != "ratio128-learned":
            raise AssertionError("layer-5 structural class drifted")
        cfg = s.get("config_contract") or {}
        expected_cfg = {
            "compress_ratio": 128, "compress_ratios_path": "compress_ratios",
            "hash_layers": 3, "hidden_size": 4096, "moe_intermediate_size": 2048,
            "n_routed_experts": 256, "n_shared_experts": 1,
            "num_experts_per_tok": 6, "num_hidden_layers": 43,
        }
        if cfg != expected_cfg:
            raise AssertionError(f"layer-5 config contract drifted: {cfg}")
        surface = s.get("surface") or {}
        if surface.get("tensor_count") != 1569 or surface.get("payload_bytes") != 3568596312:
            raise AssertionError("layer-5 tensor count/payload bytes drifted")
        if surface.get("dtype_counts") != {
            "BF16": 8, "F32": 9, "F8_E4M3": 8, "F8_E8M0": 776, "I8": 768}:
            raise AssertionError("layer-5 dtype counts drifted")
        if surface.get("subsystem_counts") != {
            "attention": 11, "compressor": 4, "mhc": 6, "norm": 4,
            "routed_expert": 1536, "router": 2, "shared_expert": 6}:
            raise AssertionError("layer-5 subsystem counts drifted")
        if "csa_indexer" in surface.get("subsystem_counts", {}):
            raise AssertionError("ratio128 layer unexpectedly gained a CSA indexer")
        if s.get("tensor_contract_sha256") != DIGEST:
            raise AssertionError("layer-5 full tensor-record digest drifted")
        checks = s.get("checks") or {}
        if checks.get("compression_contract") != "ratio128" or \
           checks.get("routing_contract") != "learned" or \
           (checks.get("routed_experts") or {}).get("expert_count") != 256 or \
           (checks.get("shared_expert") or {}).get("complete") is not True:
            raise AssertionError("layer-5 structural checks drifted")
        prov = s.get("provenance") or {}
        if prov.get("source_workflow_run") != 31449308083 or \
           prov.get("source_head_sha") != "536af281da6f960248f3ed2f5e5186a0ff1f933f" or \
           prov.get("full_surface_artifact_sha256") != "4a3bb7af57666a968ccfd4f6e98744737da424b6c2d70ba96c7b18dcfdc5e3b3":
            raise AssertionError("layer-5 surface acquisition provenance drifted")

        # Digest helper must be order-independent but record-sensitive.
        fake = {"tensors": [
            {"name": "b", "dtype": "BF16", "shape": [2]},
            {"name": "a", "dtype": "F32", "shape": [1]},
        ]}
        d1 = contract.tensor_contract_sha256(fake)
        d2 = contract.tensor_contract_sha256({"tensors": list(reversed(fake["tensors"]))})
        if d1 != d2:
            raise AssertionError("tensor contract digest depends on row order")
        bad = copy.deepcopy(fake)
        bad["tensors"][0]["shape"] = [3]
        if contract.tensor_contract_sha256(bad) == d1:
            raise AssertionError("tensor contract digest ignored record mutation")

        print("PASS V8 layer-5 immutable header contract: ratio128-learned")
        print("1569 tensors / 3568596312 checkpoint payload bytes")
        print("tensor contract sha256:", DIGEST)
        print("HEADER ONLY — NO LAYER-5 EXECUTION CLAIM")
        return 0
    except Exception as exc:
        print("FAIL:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
