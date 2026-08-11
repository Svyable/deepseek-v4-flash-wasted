#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Bind V7 evidence to the current canonical Gate-H layer-3 endpoint.

The canonical chain input is not a copied constant. It is derived from Gate H's
current provenance and independently checked against the frozen BF16 file. V7
fixtures may then assert their input dependency against that value. A corrected
parent therefore makes downstream evidence stale automatically instead of
requiring someone to remember which checkpoint hash to update.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

OLD_REFUSED_SHA = "0e65c4ecb328d5067f2274e724f9f46e4a13218e7fdeb706f7d1a465c0ee4761"
PARENT_PROVENANCE = os.path.join(
    "tests", "fixtures", "deepseek_v4", "v6_moe_branch_real", "provenance.json"
)


class V7ParentFreshnessError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise V7ParentFreshnessError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V7ParentFreshnessError(f"{path}: expected a JSON object")
    return value


def canonical_layer3_sha(repo: str) -> str:
    """Return Gate H's current endpoint after proving provenance and file agree."""
    provenance_path = os.path.join(repo, PARENT_PROVENANCE)
    provenance = _load(provenance_path)
    correction = provenance.get("hc_precision_correction") or {}
    outputs = provenance.get("outputs") or {}
    final = outputs.get("layer3_final") or {}
    corrected = correction.get("corrected_layer3_final_sha256")
    declared = final.get("sha256")
    old = correction.get("old_layer3_final_sha256")
    if not isinstance(corrected, str) or len(corrected) != 64:
        raise V7ParentFreshnessError("Gate H corrected layer-3 SHA is missing")
    if declared != corrected:
        raise V7ParentFreshnessError(
            f"Gate H output SHA {declared!r} != corrected endpoint {corrected!r}"
        )
    if old != OLD_REFUSED_SHA or old == corrected:
        raise V7ParentFreshnessError("Gate H refused/corrected endpoint contract drifted")
    relative = final.get("file")
    if not isinstance(relative, str) or not relative:
        raise V7ParentFreshnessError("Gate H final output file is missing")
    final_path = os.path.join(os.path.dirname(provenance_path), relative)
    actual = sha256_file(final_path)
    if actual != corrected:
        raise V7ParentFreshnessError(
            f"Gate H final file SHA {actual} != provenance {corrected}"
        )
    return corrected


def assert_input_dependency(
    repo: str,
    provenance_path: str,
    canonical_sha: str,
) -> None:
    """Require one downstream fixture to consume the current Gate-H endpoint."""
    absolute = (
        provenance_path
        if os.path.isabs(provenance_path)
        else os.path.join(repo, provenance_path)
    )
    provenance = _load(absolute)
    dependency = provenance.get("input_dependency")
    if not isinstance(dependency, dict):
        raise V7ParentFreshnessError(
            f"{provenance_path}: missing input_dependency contract"
        )
    declared = dependency.get("fixture_sha256")
    if declared != canonical_sha:
        raise V7ParentFreshnessError(
            f"{provenance_path}: input SHA {declared!r} != current Gate H {canonical_sha}"
        )
    fixture_file = dependency.get("fixture_file")
    if isinstance(fixture_file, str) and fixture_file:
        fixture_path = os.path.join(os.path.dirname(absolute), fixture_file)
        if not os.path.isfile(fixture_path):
            raise V7ParentFreshnessError(
                f"{provenance_path}: missing chained input file {fixture_file}"
            )
        actual = sha256_file(fixture_path)
        if actual != canonical_sha:
            raise V7ParentFreshnessError(
                f"{provenance_path}: chained input file SHA {actual} != {canonical_sha}"
            )
    source_file = dependency.get("source_file")
    if isinstance(source_file, str) and source_file:
        source_path = os.path.join(repo, source_file)
        if not os.path.isfile(source_path):
            raise V7ParentFreshnessError(
                f"{provenance_path}: missing parent source file {source_file}"
            )
        actual = sha256_file(source_path)
        if actual != canonical_sha:
            raise V7ParentFreshnessError(
                f"{provenance_path}: parent source SHA {actual} != {canonical_sha}"
            )


def assert_two_layer_chain(
    provenance: dict[str, Any],
    canonical_sha: str,
    *,
    label: str,
) -> None:
    chain = provenance.get("chain")
    if not isinstance(chain, dict):
        raise V7ParentFreshnessError(f"{label}: missing chain contract")
    if chain.get("byte_identical") is not True:
        raise V7ParentFreshnessError(f"{label}: chain is not declared byte-identical")
    for key in ("layer3_output_sha256", "layer4_input_sha256"):
        if chain.get(key) != canonical_sha:
            raise V7ParentFreshnessError(
                f"{label}: {key}={chain.get(key)!r} != current Gate H {canonical_sha}"
            )
