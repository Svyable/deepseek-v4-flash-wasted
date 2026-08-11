#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Permanent no-network Gate-I/V8 final-head surface regression."""

from __future__ import annotations

import copy
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import make_v8_head_surface as gate  # noqa: E402

SURFACE = os.path.join(
    REPO, "reference", "deepseek-v4-flash-0731.v8-head.json")

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SHARD = "model-00045-of-00048.safetensors"
SOURCE_SHA = "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"

# Independent literal copy of the measured 0731 header surface. Do not import
# gate.EXPECTED_TENSORS here: the fixture must not share the convention it
# proves, especially after PR #3's shared-packer mutation lesson.
EXPECTED_TENSORS = {
    "hc_head_base": ("F32", [4], [0, 16], 16),
    "hc_head_fn": ("F32", [4, 16384], [16, 262160], 262144),
    "hc_head_scale": ("F32", [1], [262160, 262164], 4),
    "head.weight": ("BF16", [129280, 4096], [262164, 1059323924], 1059061760),
    "norm.weight": ("BF16", [4096], [1059323924, 1059332116], 8192),
}

EXPECTED_METHOD_HASHES = {
    "RMSNorm.__init__": "82702085674d2331aba475cfb0e2d540d7102ac74d74b16f62245a94358a970d",
    "RMSNorm.forward": "8038d3c6f16015ff357534366da3aac1c0fd06d737ffa61d93548ea663748d07",
    "Block.__init__": "9ff975e2aea5177c7cadeb77eea43997c1924c090fd5326d0a3963a1e9a61cd9",
    "Block.hc_head": "1ed85047f151da017a1d1bab28996b0db7e5bb3427756fefc2f9fd0969613c24",
    "ParallelHead.__init__": "e0648d7c821c0570aa86e3fa69d095c6184464458fd56366040b7c269fb242c8",
    "ParallelHead.forward": "a320818b11d65670b4be856e220810f3011bfee48a1c6fdc5540daa643908d8e",
    "Transformer.__init__": "697feecd1a78b72db4a2236aa758ca712ad974dcf32f2ff7d3ce069bf6b62755",
    "Transformer.forward": "e054cad6ca3ea41755850aa0dfae2b49cc43141ffdf922e2a39fbad01396c667",
}


def refused(surface: dict) -> bool:
    try:
        gate.validate_surface(surface)
    except (ValueError, KeyError, TypeError):
        return True
    return False


def main() -> int:
    if not os.path.isfile(SURFACE):
        print("SKIP: frozen Gate-I/V8 head surface is not present")
        return 77

    try:
        surface = json.load(open(SURFACE, encoding="utf-8"))
        if surface.get("model") != MODEL or surface.get("revision") != REVISION:
            raise AssertionError("V8 model/revision drifted")
        if surface.get("evidence_state") != "CHECKPOINT-HEADER-AND-PINNED-SOURCE-VERIFIED":
            raise AssertionError("V8 evidence state drifted")
        if not re.fullmatch(r"[0-9a-f]{64}", surface.get("checkpoint_header_stub_sha256", "")):
            raise AssertionError("V8 shard-45 header SHA is missing")

        tensors = surface.get("checkpoint_tensors") or {}
        if set(tensors) != set(EXPECTED_TENSORS):
            raise AssertionError(f"V8 tensor set drifted: {sorted(tensors)}")
        for name, (dtype, shape, offsets, nbytes) in EXPECTED_TENSORS.items():
            item = tensors[name]
            got = (
                item.get("dtype"), item.get("shape"), item.get("data_offsets"),
                item.get("payload_bytes"), item.get("shard"), item.get("name"),
            )
            want = (dtype, shape, offsets, nbytes, SHARD, name)
            if got != want:
                raise AssertionError(f"{name}: {got!r} != {want!r}")

        source = surface.get("official_source") or {}
        if source.get("path") != "inference/model.py" or source.get("sha256") != SOURCE_SHA:
            raise AssertionError("pinned 0731 source identity drifted")
        methods = source.get("load_bearing_methods") or {}
        if set(methods) != set(EXPECTED_METHOD_HASHES):
            raise AssertionError("V8 load-bearing source method set drifted")
        for symbol, digest in EXPECTED_METHOD_HASHES.items():
            if methods[symbol].get("source_sha256") != digest:
                raise AssertionError(f"{symbol}: source-body hash drifted")

        semantic = surface.get("semantic_contract") or {}
        if semantic.get("order") != ["hc_head", "rms_norm", "vocab_head"]:
            raise AssertionError("final-head operation order drifted")
        if semantic.get("hc_head", {}).get("input_logical_shape") != [4, 4096]:
            raise AssertionError("hc_head input geometry drifted")
        if semantic.get("hc_head", {}).get("flatten_to_f32") is not True:
            raise AssertionError("hc_head F32 flatten contract drifted")
        if semantic.get("rms_norm", {}).get("compute_dtype") != "F32":
            raise AssertionError("final RMSNorm compute dtype drifted")
        if semantic.get("rms_norm", {}).get("output_cast") != "input dtype":
            raise AssertionError("final RMSNorm cast contract drifted")
        if semantic.get("vocab_head", {}).get("input_cast") != "F32" or \
           semantic.get("vocab_head", {}).get("output_dtype") != "F32":
            raise AssertionError("vocabulary-head F32 contract drifted")

        # The generator's validator is mutation-tested separately from the
        # independent literals above. Each fault is a silent/plausible failure
        # mode that the permanent contract must refuse.
        bad = copy.deepcopy(surface)
        bad["checkpoint_tensors"]["head.weight"]["shape"] = [129280, 4095]
        if not refused(bad):
            raise AssertionError("mutated head shape was accepted")

        bad = copy.deepcopy(surface)
        bad["checkpoint_tensors"]["norm.weight"]["data_offsets"][0] -= 2
        if not refused(bad):
            raise AssertionError("mutated norm offset was accepted")

        bad = copy.deepcopy(surface)
        bad["official_source"]["load_bearing_methods"]["Block.hc_head"]["source_sha256"] = "0" * 64
        if not refused(bad):
            raise AssertionError("mutated Block.hc_head source hash was accepted")

        bad = copy.deepcopy(surface)
        bad["semantic_contract"]["order"] = ["rms_norm", "hc_head", "vocab_head"]
        if not refused(bad):
            raise AssertionError("mutated final-head operation order was accepted")

        gate.validate_surface(surface)
        print("PASS Gate I/V8 final-head header/source surface")
        print("checkpoint records: 5, all on shard 45")
        print("source sha256:", SOURCE_SHA)
        print("mutations refused: head-shape, norm-offset, hc_head-source, operation-order")
        print("NO PAYLOAD / NO LOGIT CLAIM")
        return 0
    except Exception as exc:
        print("FAIL:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
