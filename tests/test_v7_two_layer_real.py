#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay and adversarially test the real layer-3 -> layer-4 V7 trace."""

from __future__ import annotations

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
from v7_parent_freshness import (  # noqa: E402
    assert_two_layer_chain,
    canonical_layer3_sha,
)

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
FIX = os.path.join(BASE, "v7_two_layer_real")
EXPECTED = os.path.join(FIX, "expected", "trace.json")
RUNTIME = os.path.join(FIX, "runtime", "trace.json")
PROV = os.path.join(FIX, "provenance.json")
SOURCE_PROVENANCE = {
    "l3_hc": os.path.join(BASE, "v6_hc_composition_real", "provenance.json"),
    "l3_att": os.path.join(BASE, "v6_attention_branch_real", "provenance.json"),
    "l3_moe": os.path.join(BASE, "v6_moe_branch_real", "provenance.json"),
    "l4_route": os.path.join(BASE, "v7_layer4_route_real", "provenance.json"),
    "l4_full": os.path.join(BASE, "v7_layer4_full_real", "provenance.json"),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_layer4_final_sha() -> str:
    path = SOURCE_PROVENANCE["l4_full"]
    if not os.path.isfile(path):
        raise AssertionError("complete layer-4 provenance is missing")
    prov = json.load(open(path, encoding="utf-8"))
    meta = (prov.get("outputs") or {}).get("layer4_final") or {}
    if meta.get("dtype") != "BF16" or meta.get("shape") != [4, 4096]:
        raise AssertionError("complete layer-4 final geometry drifted")
    final_path = os.path.join(os.path.dirname(path), meta.get("file", ""))
    if not os.path.isfile(final_path):
        raise AssertionError("complete layer-4 final file is missing")
    actual = sha(final_path)
    if actual != meta.get("sha256"):
        raise AssertionError(
            f"complete layer-4 final SHA {actual} != provenance {meta.get('sha256')}"
        )
    return actual


def assert_source_provenance(prov: dict) -> None:
    declared = prov.get("source_provenance_sha256") or {}
    if set(declared) != set(SOURCE_PROVENANCE):
        raise AssertionError(
            f"two-layer source provenance keys drifted: {sorted(declared)}"
        )
    for label, path in SOURCE_PROVENANCE.items():
        if not os.path.isfile(path):
            raise AssertionError(f"two-layer source provenance missing: {label}")
        actual = sha(path)
        if declared.get(label) != actual:
            raise AssertionError(
                f"two-layer source {label} SHA {declared.get(label)!r} != current {actual}"
            )


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
    try:
        chain_sha = canonical_layer3_sha(REPO)
    except Exception as exc:
        print("FAIL:", exc)
        return 1

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
        assert_two_layer_chain(prov, chain_sha, label=PROV)
        assert_source_provenance(prov)
        final_sha = current_layer4_final_sha()

        e_l3_out = next(b for b in expected.layers[0].boundaries if b.name == "output")
        e_l4_in = next(b for b in expected.layers[1].boundaries if b.name == "input")
        e_l4_out = next(b for b in expected.layers[1].boundaries if b.name == "output")
        if e_l3_out.sha256 != chain_sha or e_l4_in.sha256 != chain_sha:
            raise AssertionError("manifest does not pin the canonical layer3->4 chain SHA")
        if e_l4_out.sha256 != final_sha:
            raise AssertionError(
                f"layer4 final SHA {e_l4_out.sha256} != current full fixture {final_sha}"
            )

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
        print("layer4 final sha256:", final_sha)
        return 0
    except Exception as exc:
        print("FAIL:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
