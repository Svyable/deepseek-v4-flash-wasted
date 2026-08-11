#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay and adversarially test the real layer-3 -> layer-4 V7 trace."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import v7_localize as loc  # noqa: E402
from v7_parent_freshness import canonical_layer3_sha  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v7_two_layer_real")
EXPECTED = os.path.join(FIX, "expected", "trace.json")
RUNTIME = os.path.join(FIX, "runtime", "trace.json")
PROV = os.path.join(FIX, "provenance.json")
FINAL_SHA = "d875657a6ce99540e050a5fbd1590a977116c3073ea2d3a8ff141d54d83ecbc4"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mutate_bf16(trace_root: str, manifest: dict, layer: int, boundary: str,
                 index: int, xor_mask: int = 1) -> None:
    item = next(x for x in manifest["layers"] if x["layer"] == layer)
    b = next(x for x in item["boundaries"] if x["name"] == boundary)
    if b["dtype"] != "BF16":
        raise AssertionError("mutation helper is BF16-only")
    path = os.path.join(trace_root, b["file"])
    data = bytearray(open(path, "rb").read())
    off = index * 2
    old = struct.unpack_from("<H", data, off)[0]
    struct.pack_into("<H", data, off, old ^ xor_mask)
    with open(path, "wb") as f:
        f.write(data)
    b["sha256"] = sha(path)


def write_manifest(root: str, manifest: dict) -> str:
    path = os.path.join(root, "trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def main():
    # Resolve the current parent endpoint before deciding whether the downstream
    # fixture exists. A corrected Gate-H parent therefore invalidates stale V7
    # assumptions even while the two-layer acquisition rung is still absent.
    chain_sha = canonical_layer3_sha(REPO)

    if not (os.path.isfile(EXPECTED) and os.path.isfile(RUNTIME) and os.path.isfile(PROV)):
        print("SKIP: frozen real V7 two-layer trace is not present")
        print("current Gate-H chain sha256:", chain_sha)
        return 77

    try:
        expected = loc.load_trace(EXPECTED)
        runtime = loc.load_trace(RUNTIME)
        result = loc.compare(expected, runtime)
        if result.get("status") != "match" or result.get("layers") != [3, 4] or \
           result.get("boundaries_compared") != 14:
            raise AssertionError(f"real two-layer trace did not match exactly: {result}")

        prov = json.load(open(PROV, encoding="utf-8"))
        if prov.get("evidence_state") != "END-TO-END-VERIFIED-TWO-LAYER-SCALAR-BOUNDARIES":
            raise AssertionError("two-layer evidence state drifted")
        if prov.get("layers") != [3, 4] or \
           prov.get("structural_classes") != ["ratio128-learned", "ratio4-learned"]:
            raise AssertionError("two-layer identity/structural classes drifted")
        chain = prov.get("chain") or {}
        if chain.get("byte_identical") is not True or \
           chain.get("layer3_output_sha256") != chain_sha or \
           chain.get("layer4_input_sha256") != chain_sha:
            raise AssertionError(f"layer3->4 chain provenance drifted: {chain}")

        e_l3_out = next(b for b in expected.layers[0].boundaries if b.name == "output")
        e_l4_in = next(b for b in expected.layers[1].boundaries if b.name == "input")
        e_l4_out = next(b for b in expected.layers[1].boundaries if b.name == "output")
        if e_l3_out.sha256 != chain_sha or e_l4_in.sha256 != chain_sha:
            raise AssertionError("manifest does not pin the canonical layer3->4 chain SHA")
        if e_l4_out.sha256 != FINAL_SHA:
            raise AssertionError(f"layer4 final SHA drifted: {e_l4_out.sha256}")

        # Mutation 1 stays a *valid runtime chain*: change layer-3 output and
        # layer-4 input identically, then update both manifest SHAs. The first
        # divergence must still be layer 3 / output, never the later input.
        with tempfile.TemporaryDirectory(prefix="v7-real-trace-chain-mutation-") as td:
            root = os.path.join(td, "runtime")
            shutil.copytree(os.path.join(FIX, "runtime"), root)
            manifest = json.load(open(os.path.join(root, "trace.json"), encoding="utf-8"))
            mutate_bf16(root, manifest, 3, "output", 17)
            mutate_bf16(root, manifest, 4, "input", 17)
            l3 = next(x for x in manifest["layers"] if x["layer"] == 3)
            l4 = next(x for x in manifest["layers"] if x["layer"] == 4)
            l3_sha = next(x for x in l3["boundaries"] if x["name"] == "output")["sha256"]
            l4_in = next(x for x in l4["boundaries"] if x["name"] == "input")
            l4_in["sha256"] = l3_sha
            # Both files were changed by the same operation from identical
            # bytes, so the chain must remain identical after mutation.
            if sha(os.path.join(root, next(x for x in l3["boundaries"] if x["name"] == "output")["file"])) != \
               sha(os.path.join(root, l4_in["file"])):
                raise AssertionError("chain-preserving mutation produced unequal files")
            mutated_path = write_manifest(root, manifest)
            mutated = loc.load_trace(mutated_path)
            diff = loc.compare(expected, mutated)
            if diff.get("status") != "diverged" or diff.get("layer") != 3 or \
               diff.get("boundary") != "output" or diff.get("index") != 17:
                raise AssertionError(f"chain mutation localized incorrectly: {diff}")

        # Mutation 2 changes only the terminal layer-4 output and keeps the
        # manifest internally valid. The localizer must walk all earlier real
        # boundaries before identifying that exact late seam.
        with tempfile.TemporaryDirectory(prefix="v7-real-trace-late-mutation-") as td:
            root = os.path.join(td, "runtime")
            shutil.copytree(os.path.join(FIX, "runtime"), root)
            manifest = json.load(open(os.path.join(root, "trace.json"), encoding="utf-8"))
            mutate_bf16(root, manifest, 4, "output", 23)
            mutated_path = write_manifest(root, manifest)
            mutated = loc.load_trace(mutated_path)
            diff = loc.compare(expected, mutated)
            if diff.get("status") != "diverged" or diff.get("layer") != 4 or \
               diff.get("boundary") != "output" or diff.get("index") != 23:
                raise AssertionError(f"late mutation localized incorrectly: {diff}")

        print("PASS V7 real two-layer trace: 14 boundaries exact, earliest-divergence mutations localized")
        print("layers: 3 ratio128-learned -> 4 ratio4-learned")
        print("chain sha256:", chain_sha)
        print("layer4 final sha256:", FINAL_SHA)
        return 0
    except Exception as exc:
        print("FAIL:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
