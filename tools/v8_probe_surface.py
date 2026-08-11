#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Probe Gate-I final-head checkpoint/source surface without tensor payloads.

This is intentionally exploratory. It consumes only a header-only safetensors
snapshot plus the exact pinned `inference/model.py`, then records candidate
root-level final-head tensors and all methods on the source classes responsible
for final normalization/head evaluation. It does not assert final tensor names
or promote V8; the next tranche converts the observed result into a fail-closed
frozen contract.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
from typing import Any


TARGET_CLASSES = ("RMSNorm", "ParallelHead", "Transformer")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > (100 << 20):
            raise ValueError(f"{path}: implausible header size {n}")
        blob = f.read(n)
        if len(blob) != n:
            raise ValueError(f"{path}: truncated safetensors JSON header")
        if f.read(1):
            raise ValueError(f"{path}: expected a header-only stub, found payload bytes")
    obj = json.loads(blob)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: header is not an object")
    return obj


def candidate(name: str) -> bool:
    # Deliberately broad root-level discovery. We do not assume the final norm
    # or HC-head spelling; we merely keep names plausibly related to embedding,
    # final normalization/head, and global HyperConnection head parameters.
    rootish = not name.startswith("layers.") and not name.startswith("mtp.")
    if not rootish:
        return False
    low = name.lower()
    return any(part in low for part in ("embed", "head", "norm", "hc_"))


def checkpoint_surface(snapshot: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = os.path.join(snapshot, "reference_headers.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("header_only") or not manifest.get("immutable_revision"):
        raise ValueError("snapshot provenance is not immutable header-only evidence")

    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard in sorted(manifest.get("shards", {})):
        path = os.path.join(snapshot, shard)
        header = load_header(path)
        for name, meta in header.items():
            if name == "__metadata__" or not candidate(name):
                continue
            if name in seen:
                raise ValueError(f"duplicate tensor name across headers: {name}")
            seen.add(name)
            if not isinstance(meta, dict):
                raise ValueError(f"{name}: metadata is not an object")
            found.append({
                "name": name,
                "shard": shard,
                "dtype": meta.get("dtype"),
                "shape": meta.get("shape"),
                "data_offsets": meta.get("data_offsets"),
            })
    found.sort(key=lambda x: x["name"])
    return found, manifest


def source_methods(source_path: str) -> tuple[list[dict[str, Any]], str]:
    raw = open(source_path, "rb").read()
    text = raw.decode("utf-8")
    tree = ast.parse(text, filename=source_path)
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    out = []
    for cls_name in TARGET_CLASSES:
        cls = classes.get(cls_name)
        if cls is None:
            raise ValueError(f"pinned source lacks class {cls_name}")
        methods = [
            n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not methods:
            raise ValueError(f"pinned source class {cls_name} has no methods")
        for method in methods:
            segment = ast.get_source_segment(text, method)
            if not segment:
                raise ValueError(f"could not recover source for {cls_name}.{method.name}")
            out.append({
                "symbol": f"{cls_name}.{method.name}",
                "lineno": method.lineno,
                "end_lineno": method.end_lineno,
                "source_sha256": sha256_bytes(segment.encode()),
                "source": segment,
            })
    return out, sha256_bytes(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="header-only checkpoint snapshot")
    ap.add_argument("source", help="pinned official inference/model.py")
    ap.add_argument("out", help="JSON probe output")
    args = ap.parse_args(argv)

    tensors, manifest = checkpoint_surface(args.snapshot)
    print("V8 root candidate tensors:")
    for item in tensors:
        print(f"  {item['name']}: {item['dtype']} {item['shape']} @ {item['shard']}")

    methods, source_sha = source_methods(args.source)
    result = {
        "schema_version": 1,
        "evidence_state": "EXPLORATORY-HEADER-AND-PINNED-SOURCE",
        "gate": "Gate I / V8 surface probe",
        "model": manifest.get("model"),
        "revision": manifest.get("revision"),
        "checkpoint_headers_sha256": sha256_bytes(
            open(os.path.join(args.snapshot, "reference_headers.json"), "rb").read()),
        "candidate_root_tensors": tensors,
        "official_source": {
            "path": "inference/model.py",
            "sha256": source_sha,
            "methods": methods,
        },
        "non_claims": [
            "exploratory discovery only; exact Gate-I tensor contract is not yet frozen",
            "no tensor payload was read",
            "no final hidden state, norm output, logits, or 43-layer forward was evaluated",
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    print("V8 pinned source methods:")
    for item in methods:
        print(f"  {item['symbol']}: lines {item['lineno']}-{item['end_lineno']} sha256={item['source_sha256']}")
    print(f"official inference/model.py sha256={source_sha}")
    print("PROBE ONLY — no V8 numerical gate promoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
