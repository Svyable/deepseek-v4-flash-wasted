#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Fetch a bounded contiguous row slice from one safetensors tensor.

This is the payload companion to ``fetch_hf_headers.py``. It deliberately does
*not* provide an arbitrary whole-checkpoint downloader. Given a local
header-only snapshot, it resolves one indexed tensor to its shard and fetches
only the requested row range with HTTP Range. A server that ignores Range is
refused by the shared transport helper.

Example (first eight rows of the Gate-C projection):

  python3 tools/fetch_hf_tensor_slice.py \
    reference/deepseek-v4-flash-0731 \
    layers.0.attn.wq_a.weight --rows 0:8 \
    --out /tmp/wq-a-rows0-8.bin

The sidecar ``<out>.json`` records revision, shard, exact absolute byte range,
tensor-relative offsets, dtype/shape, slice bounds, byte count and SHA-256.
"""

import argparse
import hashlib
import json
import os
import struct
import sys

import fetch_hf_headers as hf


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class SliceError(RuntimeError):
    pass


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SliceError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SliceError(f"{path}: expected JSON object")
    return value


def _read_header_stub(path):
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise SliceError(f"{path}: truncated safetensors prefix")
            (hlen,) = struct.unpack("<Q", raw)
            if hlen == 0 or hlen > hf.MAX_HEADER_BYTES:
                raise SliceError(f"{path}: implausible safetensors header {hlen}")
            blob = f.read(hlen)
    except OSError as exc:
        raise SliceError(f"{path}: {exc}") from exc
    if len(blob) != hlen:
        raise SliceError(f"{path}: truncated safetensors JSON header")
    try:
        header = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SliceError(f"{path}: invalid safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise SliceError(f"{path}: safetensors header is not an object")
    return hlen, header


def _parse_rows(spec, rows):
    if spec is None:
        return 0, rows
    if ":" not in spec:
        raise SliceError("--rows must use START:END (END exclusive)")
    left, right = spec.split(":", 1)
    if not left or not right:
        raise SliceError("--rows requires both START and END")
    try:
        start, end = int(left), int(right)
    except ValueError as exc:
        raise SliceError("--rows bounds must be integers") from exc
    if start < 0 or end <= start or end > rows:
        raise SliceError(f"invalid rows {start}:{end} for tensor with {rows} rows")
    return start, end


def resolve_slice(snapshot, tensor, rows_spec=None):
    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    provenance_path = os.path.join(snapshot, "reference_headers.json")
    index = _load_json(index_path)
    provenance = _load_json(provenance_path)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or tensor not in weight_map:
        raise SliceError(f"tensor not present in index: {tensor}")
    shard = weight_map[tensor]
    stub = os.path.join(snapshot, shard)
    hlen, header = _read_header_stub(stub)
    meta = header.get(tensor)
    if not isinstance(meta, dict):
        raise SliceError(f"{shard}: indexed tensor missing from header: {tensor}")

    dtype = meta.get("dtype")
    shape = meta.get("shape")
    offsets = meta.get("data_offsets")
    width = DTYPE_BYTES.get(dtype)
    if width is None:
        raise SliceError(f"unsupported dtype for row slicing: {dtype}")
    if not isinstance(shape, list) or not shape or not all(
            isinstance(d, int) and d > 0 for d in shape):
        raise SliceError(f"{tensor}: invalid shape {shape!r}")
    if (not isinstance(offsets, list) or len(offsets) != 2 or
            not all(isinstance(v, int) for v in offsets)):
        raise SliceError(f"{tensor}: invalid data_offsets {offsets!r}")

    beg, end = offsets
    elements_per_row = 1
    for dim in shape[1:]:
        elements_per_row *= dim
    row_bytes = elements_per_row * width
    expected = shape[0] * row_bytes
    if end - beg != expected:
        raise SliceError(
            f"{tensor}: stored bytes {end - beg} != dense row-major bytes {expected}")

    row_start, row_end = _parse_rows(rows_spec, shape[0])
    relative_start = beg + row_start * row_bytes
    relative_end = beg + row_end * row_bytes  # exclusive
    data_base = 8 + hlen
    absolute_start = data_base + relative_start
    absolute_end = data_base + relative_end - 1  # HTTP inclusive

    model = provenance.get("model")
    revision = provenance.get("revision")
    endpoint = provenance.get("endpoint", hf.DEFAULT_ENDPOINT)
    hf.validate_revision(revision)
    if not isinstance(model, str) or not model:
        raise SliceError(f"{provenance_path}: missing model")

    return {
        "model": model,
        "revision": revision,
        "endpoint": endpoint,
        "tensor": tensor,
        "shard": shard,
        "dtype": dtype,
        "shape": shape,
        "tensor_data_offsets": offsets,
        "header_json_bytes": hlen,
        "row_bytes": row_bytes,
        "row_start": row_start,
        "row_end": row_end,
        "relative_start": relative_start,
        "relative_end": relative_end,
        "absolute_start": absolute_start,
        "absolute_end": absolute_end,
    }


def fetch_slice(snapshot, tensor, rows_spec, out_path, *, token=None,
                opener=hf.urlopen, timeout=60):
    info = resolve_slice(snapshot, tensor, rows_spec)
    url = hf.resolve_url(
        info["model"], info["revision"], info["shard"], info["endpoint"])
    try:
        data, remote_size = hf.fetch_range(
            url,
            info["absolute_start"],
            info["absolute_end"],
            token=token,
            opener=opener,
            timeout=timeout,
        )
    except hf.FetchError as exc:
        raise SliceError(str(exc)) from exc

    want = info["absolute_end"] - info["absolute_start"] + 1
    if len(data) != want:
        raise SliceError(f"fetched {len(data)} bytes, expected {want}")

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

    sidecar = dict(info)
    sidecar.update({
        "schema_version": 1,
        "remote_file_bytes": remote_size,
        "payload_bytes": len(data),
        "payload_sha256": hashlib.sha256(data).hexdigest(),
        "output_file": os.path.basename(out_path),
    })
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")
    return sidecar


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="header-only snapshot directory")
    ap.add_argument("tensor", help="exact tensor name from the safetensors index")
    ap.add_argument("--rows", help="row slice START:END (END exclusive); default all")
    ap.add_argument("--out", required=True, help="output raw payload path")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="optional Hugging Face token; defaults to HF_TOKEN")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args(argv)

    try:
        info = fetch_slice(
            args.snapshot, args.tensor, args.rows, args.out,
            token=args.token, timeout=args.timeout)
    except SliceError as exc:
        print(f"tensor-slice: {exc}", file=sys.stderr)
        return 2

    print(
        f"fetched {info['tensor']} rows {info['row_start']}:{info['row_end']} "
        f"from {info['shard']}: {info['payload_bytes']} bytes")
    print(f"sha256 {info['payload_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
