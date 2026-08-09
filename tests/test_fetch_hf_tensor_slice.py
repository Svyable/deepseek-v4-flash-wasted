#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Offline tests for tools/fetch_hf_tensor_slice.py."""

import io
import json
import os
import struct
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import fetch_hf_tensor_slice as target  # noqa: E402


class Response:
    def __init__(self, status, data, content_range=None):
        self.status = status
        self._stream = io.BytesIO(data)
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self.headers["Content-Length"] = str(len(data))

    def read(self, n=-1):
        return self._stream.read(n)

    def close(self):
        pass


def make_snapshot(root):
    os.makedirs(root, exist_ok=True)
    tensor = "layers.0.demo.weight"
    shard = "model-00001-of-00001.safetensors"
    payload = bytes(range(16))
    header = {
        tensor: {
            "dtype": "I8",
            "shape": [4, 4],
            "data_offsets": [0, len(payload)],
        }
    }
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    prefix = struct.pack("<Q", len(blob)) + blob
    with open(os.path.join(root, shard), "wb") as f:
        f.write(prefix)  # header-only stub on purpose
    with open(os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"weight_map": {tensor: shard}}, f)
    with open(os.path.join(root, "reference_headers.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "owner/model",
            "revision": "0" * 40,
            "endpoint": "https://example.invalid",
        }, f)
    return tensor, shard, prefix + payload, payload


def opener_for(remote, ignore_range=False):
    def opener(req, timeout=None):
        del timeout
        header = req.headers.get("Range") or req.headers.get("range")
        if ignore_range:
            return Response(200, remote)
        if not header or not header.startswith("bytes="):
            raise AssertionError(f"missing Range request: {header!r}")
        start_s, end_s = header[6:].split("-", 1)
        start, end = int(start_s), int(end_s)
        data = remote[start:end + 1]
        return Response(206, data, f"bytes {start}-{end}/{len(remote)}")
    return opener


def main():
    with tempfile.TemporaryDirectory(prefix="tensor-slice-") as td:
        snapshot = os.path.join(td, "snapshot")
        tensor, shard, remote, payload = make_snapshot(snapshot)
        out = os.path.join(td, "rows.bin")

        info = target.fetch_slice(
            snapshot, tensor, "1:3", out, opener=opener_for(remote))
        with open(out, "rb") as f:
            got = f.read()
        if got != payload[4:12]:
            print(f"FAIL: wrong row bytes {got!r} != {payload[4:12]!r}")
            return 1
        if info["shard"] != shard or info["row_bytes"] != 4:
            print("FAIL: row/shard metadata mismatch")
            return 1
        if info["payload_bytes"] != 8:
            print("FAIL: payload byte count mismatch")
            return 1

        with open(out + ".json", encoding="utf-8") as f:
            sidecar = json.load(f)
        if sidecar["row_start"] != 1 or sidecar["row_end"] != 3:
            print("FAIL: sidecar slice bounds mismatch")
            return 1

        try:
            target.fetch_slice(
                snapshot, tensor, "1:3", os.path.join(td, "ignored.bin"),
                opener=opener_for(remote, ignore_range=True))
        except target.SliceError as exc:
            if "ignored Range" not in str(exc):
                print(f"FAIL: unexpected ignored-Range error: {exc}")
                return 1
        else:
            print("FAIL: server ignoring Range was accepted")
            return 1

        for bad in ("-1:1", "1:1", "0:5", "1", ":2", "2:"):
            try:
                target.resolve_slice(snapshot, tensor, bad)
            except target.SliceError:
                pass
            else:
                print(f"FAIL: invalid row slice accepted: {bad}")
                return 1

    print("PASS fail-closed safetensors tensor-row Range slicing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
