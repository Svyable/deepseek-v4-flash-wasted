#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Compact/digest immutable generic V8 per-layer header surfaces."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy


def tensor_contract_sha256(surface: dict) -> str:
    rows = surface.get("tensors")
    if not isinstance(rows, list) or not rows:
        raise ValueError("generic layer surface has no tensor records")
    names = [r.get("name") for r in rows]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("generic layer surface has an invalid tensor name")
    if len(set(names)) != len(names):
        raise ValueError("generic layer surface has duplicate tensor names")
    ordered = sorted(rows, key=lambda r: r["name"])
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def compact_surface(surface: dict) -> dict:
    required = (
        "schema_version", "gate", "evidence_state", "model", "revision",
        "layer", "structural_class", "config_contract", "checks", "groups",
        "surface", "non_claims",
    )
    missing = [key for key in required if key not in surface]
    if missing:
        raise ValueError(f"generic layer surface missing {missing}")
    groups = {}
    for name, group in sorted(surface["groups"].items()):
        if group.get("tensor_count"):
            groups[name] = {
                "tensor_count": group["tensor_count"],
                "payload_bytes": group["payload_bytes"],
                "dtype_counts": deepcopy(group["dtype_counts"]),
            }
    return {
        "schema_version": 1,
        "gate": "Gate I/V8 compact generic layer header contract",
        "evidence_state": surface["evidence_state"],
        "model": surface["model"],
        "revision": surface["revision"],
        "layer": surface["layer"],
        "structural_class": surface["structural_class"],
        "config_contract": deepcopy(surface["config_contract"]),
        "checks": deepcopy(surface["checks"]),
        "groups": groups,
        "surface": deepcopy(surface["surface"]),
        "tensor_contract_sha256": tensor_contract_sha256(surface),
        "non_claims": deepcopy(surface["non_claims"]),
    }


def validate_against_full(compact: dict, full: dict) -> None:
    want = compact_surface(full)
    if compact != want:
        raise ValueError("compact layer contract does not match measured full surface")
