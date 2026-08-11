#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Permanent no-network replay of the real-state Gate-I head primitive."""

from __future__ import annotations

import hashlib
import json
import os
import struct

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
FIX = os.path.join(BASE, "v8_head_primitive_real")
V7 = os.path.join(BASE, "v7_layer4_full_real")
SURFACE = os.path.join(REPO, "reference", "deepseek-v4-flash-0731.v8-head.json")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
INPUT_SHA = "f32bec5a013bc5a0da98a6ac940dd4946fe0dad877714ffb6811c177837cf749"
ROW_IDS = list(range(0, 8)) + list(range(128796, 128804)) + list(range(129272, 129280))
PAYLOAD_BYTES = 466964


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_exact(path, size):
    data = open(path, "rb").read()
    if len(data) != size:
        raise AssertionError(f"{path}: {len(data)} bytes, expected {size}")
    return data


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.isfile(prov_path):
        print("SKIP: frozen V8 real-state head primitive is not present")
        return 77

    try:
        p = json.load(open(prov_path, encoding="utf-8"))
        if p.get("schema_version") != 1 or p.get("revision") != REV:
            raise AssertionError("V8 primitive schema/revision drifted")
        if p.get("gate") != "Gate I/V8 real-state final-head primitive" or \
           p.get("evidence_state") != "CHECKPOINT-BOUND-REAL-STATE-HEAD-PRIMITIVE":
            raise AssertionError("V8 primitive gate/evidence identity drifted")

        v7 = json.load(open(os.path.join(V7, "provenance.json"), encoding="utf-8"))
        v7_meta = (v7.get("outputs") or {}).get("layer4_final") or {}
        v7_path = os.path.join(V7, v7_meta.get("file", ""))
        inp = p.get("input") or {}
        fixture_input = os.path.join(FIX, inp.get("fixture_file", ""))
        if v7_meta.get("sha256") != INPUT_SHA or sha(v7_path) != INPUT_SHA or \
           inp.get("source_sha256") != INPUT_SHA or inp.get("fixture_sha256") != INPUT_SHA or \
           sha(fixture_input) != INPUT_SHA:
            raise AssertionError("V8 primitive is not bound to the frozen real layer-4 state")
        if inp.get("shape") != [4, 4096] or inp.get("dtype") != "BF16" or \
           inp.get("is_true_final_model_hidden") is not False:
            raise AssertionError("V8 primitive input evidence boundary drifted")

        dep = p.get("surface_dependency") or {}
        if dep.get("file") != "reference/deepseek-v4-flash-0731.v8-head.json" or \
           dep.get("sha256") != sha(SURFACE):
            raise AssertionError("V8 primitive surface dependency drifted")

        cfg = p.get("effective_inference_config") or {}
        expected_cfg = {
            "vocab_size": 129280,
            "dim": 4096,
            "n_layers": 43,
            "hc_mult": 4,
            "norm_eps": 1e-6,
            "hc_eps": 1e-6,
        }
        for key, value in expected_cfg.items():
            if cfg.get(key) != value:
                raise AssertionError(f"effective config {key} drifted: {cfg.get(key)!r}")
        if cfg.get("norm_eps_source") != "ModelArgs default" or \
           cfg.get("hc_eps_source") != "ModelArgs default":
            raise AssertionError("0731 inference epsilon provenance drifted")

        cp = p.get("checkpoint_payload") or {}
        if cp.get("total_bytes_fetched") != PAYLOAD_BYTES or \
           cp.get("raw_payload_committed") is not False:
            raise AssertionError("bounded checkpoint payload contract drifted")
        if cp.get("selected_row_ids") != ROW_IDS:
            raise AssertionError("selected vocabulary-row contract drifted")
        if set((cp.get("full_parameters") or {})) != {"fn", "base", "scale", "norm"}:
            raise AssertionError("full small-parameter set drifted")
        slices = cp.get("head_row_slices") or []
        if [x.get("rows") for x in slices] != [[0, 8], [128796, 128804], [129272, 129280]]:
            raise AssertionError("bounded head-row slices drifted")
        if any(x.get("tensor") != "head.weight" for x in slices):
            raise AssertionError("head-row tensor identity drifted")

        outputs = p.get("outputs") or {}
        contracts = {
            "hc_head": ("BF16", [4096], 4096 * 2),
            "hc_mixes": ("F32", [4], 4 * 4),
            "hc_pre": ("F32", [4], 4 * 4),
            "final_norm": ("BF16", [4096], 4096 * 2),
            "row_ids": ("U32", [24], 24 * 4),
            "selected_logits": ("F32", [24], 24 * 4),
        }
        for name, (dtype, shape, size) in contracts.items():
            meta = outputs.get(name) or {}
            path = os.path.join(FIX, meta.get("file", ""))
            if meta.get("dtype") != dtype or meta.get("shape") != shape:
                raise AssertionError(f"{name}: dtype/shape drifted")
            read_exact(path, size)
            if sha(path) != meta.get("sha256"):
                raise AssertionError(f"{name}: SHA drifted")

        ids_meta = outputs["row_ids"]
        ids_raw = read_exact(os.path.join(FIX, ids_meta["file"]), 24 * 4)
        ids = list(struct.unpack("<24I", ids_raw))
        if ids != ROW_IDS:
            raise AssertionError("selected-row output bytes drifted")
        logit_meta = outputs["selected_logits"]
        logits_raw = read_exact(os.path.join(FIX, logit_meta["file"]), 24 * 4)
        logit_bits = list(struct.unpack("<24I", logits_raw))
        if [f"0x{x:08x}" for x in logit_bits] != logit_meta.get("f32_bits"):
            raise AssertionError("selected-logit raw F32 bits disagree with provenance")

        verify = p.get("acquisition_verification") or {}
        expected_counts = {
            "independent_python_vs_waste_hc_raw_exact_f32": 4096,
            "independent_python_vs_waste_hc_mixes_exact_f32": 4,
            "independent_python_vs_waste_hc_pre_exact_f32": 4,
            "independent_python_vs_waste_hc_bf16_cast_exact": 4096,
            "independent_python_vs_waste_rms_raw_exact_f32": 4096,
            "independent_python_vs_waste_rms_bf16_cast_exact": 4096,
            "independent_python_vs_waste_selected_logits_exact_f32": 24,
        }
        if verify != expected_counts:
            raise AssertionError(f"V8 primitive acquisition verification drifted: {verify}")

        nonclaims = " ".join(p.get("non_claims") or [])
        for phrase in ("not the true layer-42/final", "not final-model logits",
                       "does not execute the remaining transformer layers",
                       "does not fetch or validate the full 1.059 GB"):
            if phrase not in nonclaims:
                raise AssertionError(f"V8 primitive non-claim missing: {phrase}")

        print("PASS Gate I/V8 real-state head primitive replay")
        print("input layer4 sha256:", INPUT_SHA)
        print("checkpoint payload bytes:", PAYLOAD_BYTES)
        print("selected rows:", ROW_IDS)
        print("PRIMITIVE ONLY — NOT FINAL-MODEL LOGITS / V8 NOT PASSED")
        return 0
    except Exception as exc:
        print("FAIL:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
