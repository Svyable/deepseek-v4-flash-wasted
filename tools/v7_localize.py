#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Compare two chained DeepSeek layer-boundary traces and stop at first drift.

V7's job is localization, not tolerance. A trace is valid only when:

* every boundary file exists and matches its declared SHA-256/byte geometry;
* layers are consecutive;
* each layer starts at `input` and ends at `output`;
* one layer's exact `output` is the next layer's exact `input`;
* expected and actual traces describe the same model/revision/layers/classes and
  the same ordered boundary contracts.

Only after both traces pass those integrity checks are values compared. The
first differing boundary is reported with the first differing element and raw
bit pattern. This prevents a late mismatch from hiding an earlier one and
prevents a corrupted manifest from being misreported as a numerical drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from typing import Any

SCHEMA = 1
SUPPORTED_DTYPES = {
    "BF16": (2, "H"),
    "F32": (4, "I"),  # compare raw IEEE bits, never Python float equality
    "U32": (4, "I"),
    "I32": (4, "i"),
    "I64": (8, "q"),
    "U8": (1, "B"),
}


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class Boundary:
    name: str
    path: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    nbytes: int


@dataclass(frozen=True)
class Layer:
    number: int
    structural_class: str
    boundaries: tuple[Boundary, ...]


@dataclass(frozen=True)
class Trace:
    manifest_path: str
    root: str
    model: str
    revision: str
    name: str
    layers: tuple[Layer, ...]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def product(shape: tuple[int, ...]) -> int:
    p = 1
    for n in shape:
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise TraceError(f"invalid shape dimension {n!r}")
        p *= n
    return p


def parse_boundary(root: str, layer: int, raw: Any) -> Boundary:
    if not isinstance(raw, dict):
        raise TraceError(f"layer {layer}: boundary must be an object")
    name = raw.get("name")
    rel = raw.get("file")
    dtype = raw.get("dtype")
    shape_raw = raw.get("shape")
    digest = raw.get("sha256")
    if not isinstance(name, str) or not name:
        raise TraceError(f"layer {layer}: boundary has invalid name")
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        raise TraceError(f"layer {layer} {name}: file must be a relative path")
    norm = os.path.normpath(rel)
    if norm == ".." or norm.startswith(".." + os.sep):
        raise TraceError(f"layer {layer} {name}: file escapes trace root")
    if dtype not in SUPPORTED_DTYPES:
        raise TraceError(f"layer {layer} {name}: unsupported dtype {dtype!r}")
    if not isinstance(shape_raw, list) or not shape_raw:
        raise TraceError(f"layer {layer} {name}: shape must be a non-empty list")
    shape = tuple(shape_raw)
    count = product(shape)
    item_bytes = SUPPORTED_DTYPES[dtype][0]
    want_bytes = count * item_bytes
    if not isinstance(digest, str) or len(digest) != 64:
        raise TraceError(f"layer {layer} {name}: invalid SHA-256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise TraceError(f"layer {layer} {name}: invalid SHA-256") from exc
    path = os.path.join(root, norm)
    if not os.path.isfile(path):
        raise TraceError(f"layer {layer} {name}: missing file {rel}")
    got_bytes = os.path.getsize(path)
    if got_bytes != want_bytes:
        raise TraceError(
            f"layer {layer} {name}: {got_bytes} bytes, expected {want_bytes} for {dtype}{list(shape)}")
    got_sha = sha256_file(path)
    if got_sha != digest:
        raise TraceError(
            f"layer {layer} {name}: file SHA {got_sha} != manifest {digest}")
    return Boundary(name, path, dtype, shape, digest, want_bytes)


def load_trace(manifest_path: str) -> Trace:
    manifest_path = os.path.abspath(manifest_path)
    root = os.path.dirname(manifest_path)
    try:
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot read trace manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TraceError("trace manifest must be an object")
    if raw.get("schema_version") != SCHEMA:
        raise TraceError(f"unsupported schema_version {raw.get('schema_version')!r}")
    model = raw.get("model")
    revision = raw.get("revision")
    name = raw.get("trace_name")
    if not isinstance(model, str) or not model:
        raise TraceError("trace model must be a non-empty string")
    if not isinstance(revision, str) or not revision:
        raise TraceError("trace revision must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise TraceError("trace_name must be a non-empty string")
    layers_raw = raw.get("layers")
    if not isinstance(layers_raw, list) or len(layers_raw) < 2:
        raise TraceError("V7 trace must contain at least two consecutive layers")

    layers: list[Layer] = []
    previous_number = None
    previous_output: Boundary | None = None
    for item in layers_raw:
        if not isinstance(item, dict):
            raise TraceError("layer entry must be an object")
        number = item.get("layer")
        structural_class = item.get("structural_class")
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise TraceError(f"invalid layer number {number!r}")
        if previous_number is not None and number != previous_number + 1:
            raise TraceError(
                f"layers must be consecutive: got {previous_number} then {number}")
        if not isinstance(structural_class, str) or not structural_class:
            raise TraceError(f"layer {number}: structural_class must be non-empty")
        boundaries_raw = item.get("boundaries")
        if not isinstance(boundaries_raw, list) or len(boundaries_raw) < 2:
            raise TraceError(f"layer {number}: need at least input/output boundaries")
        boundaries = tuple(parse_boundary(root, number, b) for b in boundaries_raw)
        names = [b.name for b in boundaries]
        if len(set(names)) != len(names):
            raise TraceError(f"layer {number}: duplicate boundary name")
        if names[0] != "input" or names[-1] != "output":
            raise TraceError(
                f"layer {number}: boundary order must start input and end output")
        current_input = boundaries[0]
        current_output = boundaries[-1]
        if current_input.dtype != "BF16" or current_output.dtype != "BF16":
            raise TraceError(f"layer {number}: input/output must be BF16")
        if current_input.shape != current_output.shape:
            raise TraceError(
                f"layer {number}: input shape {current_input.shape} != output {current_output.shape}")
        if previous_output is not None:
            if previous_output.dtype != current_input.dtype or previous_output.shape != current_input.shape:
                raise TraceError(
                    f"layer {number}: input geometry does not match previous output")
            if previous_output.sha256 != current_input.sha256:
                raise TraceError(
                    f"layer {number}: chain break; previous output SHA "
                    f"{previous_output.sha256} != input SHA {current_input.sha256}")
        layers.append(Layer(number, structural_class, boundaries))
        previous_number = number
        previous_output = current_output

    return Trace(manifest_path, root, model, revision, name, tuple(layers))


def ensure_compatible(expected: Trace, actual: Trace) -> None:
    if expected.model != actual.model:
        raise TraceError(f"model mismatch: {expected.model!r} != {actual.model!r}")
    if expected.revision != actual.revision:
        raise TraceError(
            f"revision mismatch: {expected.revision!r} != {actual.revision!r}")
    if len(expected.layers) != len(actual.layers):
        raise TraceError(
            f"layer-count mismatch: {len(expected.layers)} != {len(actual.layers)}")
    for e_layer, a_layer in zip(expected.layers, actual.layers):
        if e_layer.number != a_layer.number:
            raise TraceError(
                f"layer identity mismatch: {e_layer.number} != {a_layer.number}")
        if e_layer.structural_class != a_layer.structural_class:
            raise TraceError(
                f"layer {e_layer.number}: structural class mismatch: "
                f"{e_layer.structural_class!r} != {a_layer.structural_class!r}")
        if len(e_layer.boundaries) != len(a_layer.boundaries):
            raise TraceError(f"layer {e_layer.number}: boundary-count mismatch")
        for eb, ab in zip(e_layer.boundaries, a_layer.boundaries):
            if (eb.name, eb.dtype, eb.shape) != (ab.name, ab.dtype, ab.shape):
                raise TraceError(
                    f"layer {e_layer.number}: boundary contract mismatch: "
                    f"{(eb.name, eb.dtype, eb.shape)} != {(ab.name, ab.dtype, ab.shape)}")


def first_element_difference(expected: Boundary, actual: Boundary) -> dict[str, Any]:
    item_bytes, fmt = SUPPORTED_DTYPES[expected.dtype]
    with open(expected.path, "rb") as ef, open(actual.path, "rb") as af:
        index = 0
        chunk_items = max(1, (1 << 20) // item_bytes)
        while True:
            eraw = ef.read(chunk_items * item_bytes)
            araw = af.read(chunk_items * item_bytes)
            if not eraw and not araw:
                break
            if len(eraw) != len(araw):
                raise TraceError("compatible boundary files changed size during comparison")
            if eraw == araw:
                index += len(eraw) // item_bytes
                continue
            for off in range(0, len(eraw), item_bytes):
                e = eraw[off:off + item_bytes]
                a = araw[off:off + item_bytes]
                if e != a:
                    e_val = struct.unpack("<" + fmt, e)[0]
                    a_val = struct.unpack("<" + fmt, a)[0]
                    return {
                        "index": index + off // item_bytes,
                        "expected_value": e_val,
                        "actual_value": a_val,
                        "expected_hex": e.hex(),
                        "actual_hex": a.hex(),
                    }
            index += len(eraw) // item_bytes
    raise TraceError("boundary SHA differs but no differing element was found")


def compare(expected: Trace, actual: Trace) -> dict[str, Any]:
    ensure_compatible(expected, actual)
    for e_layer, a_layer in zip(expected.layers, actual.layers):
        for eb, ab in zip(e_layer.boundaries, a_layer.boundaries):
            if eb.sha256 != ab.sha256:
                detail = first_element_difference(eb, ab)
                return {
                    "status": "diverged",
                    "layer": e_layer.number,
                    "structural_class": e_layer.structural_class,
                    "boundary": eb.name,
                    "dtype": eb.dtype,
                    "shape": list(eb.shape),
                    "expected_sha256": eb.sha256,
                    "actual_sha256": ab.sha256,
                    **detail,
                }
    return {
        "status": "match",
        "layers": [layer.number for layer in expected.layers],
        "boundaries_compared": sum(len(layer.boundaries) for layer in expected.layers),
        "model": expected.model,
        "revision": expected.revision,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("expected", help="expected trace manifest JSON")
    ap.add_argument("actual", help="actual trace manifest JSON")
    ap.add_argument("--json", action="store_true", help="emit one JSON object")
    args = ap.parse_args(argv)
    try:
        expected = load_trace(args.expected)
        actual = load_trace(args.actual)
        result = compare(expected, actual)
    except TraceError as exc:
        result = {"status": "invalid", "error": str(exc)}
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"INVALID TRACE: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif result["status"] == "match":
        print(
            f"MATCH {len(result['layers'])} layers / "
            f"{result['boundaries_compared']} boundaries")
    else:
        print(
            "FIRST DIVERGENCE "
            f"layer={result['layer']} class={result['structural_class']} "
            f"boundary={result['boundary']} index={result['index']} "
            f"expected=0x{result['expected_hex']} actual=0x{result['actual_hex']}")
    return 0 if result["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
