#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Offline tests for tools/fetch_hf_headers.py.

The fake Hub returns ordinary metadata files and exact HTTP 206 Range bodies
for a synthetic safetensors shard. Nothing reaches the network. The test
checks the failure modes that matter for Gate A/V0 acquisition: refusing a
mutable revision by default, refusing a server that ignores Range, refusing
index/header disagreement, and reusing a valid immutable header stub.
"""

import json
import os
import shutil
import struct
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import fetch_hf_headers as fh  # noqa: E402
import inventory  # noqa: E402

fails = 0


def ck(cond, msg):
    global fails
    if not cond:
        fails += 1
        print(f"  FAIL {msg}")
    return cond


class FakeResponse:
    def __init__(self, body, status, headers):
        self.body = body
        self.status = status
        self.headers = headers
        self.pos = 0

    def read(self, n=-1):
        if n < 0:
            n = len(self.body) - self.pos
        out = self.body[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def close(self):
        pass


class FakeHub:
    def __init__(self, files, ignore_range=False):
        self.files = files
        self.ignore_range = ignore_range
        self.calls = []

    def __call__(self, req, timeout=None):
        del timeout
        range_header = req.get_header("Range")
        self.calls.append((req.full_url, range_header))
        path = req.full_url.split("/resolve/", 1)[1].split("/", 1)[1]
        data = self.files[path]
        if range_header and not self.ignore_range:
            start, end = map(int, range_header[6:].split("-"))
            body = data[start:end + 1]
            return FakeResponse(
                body,
                206,
                {
                    "Content-Range": f"bytes {start}-{end}/{len(data)}",
                    "Content-Length": str(len(body)),
                },
            )
        return FakeResponse(data, 200, {"Content-Length": str(len(data))})


def make_shard(header, payload_bytes=64):
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    return struct.pack("<Q", len(blob)) + blob + b"X" * payload_bytes


def main():
    revision = "a" * 40
    config = {"num_hidden_layers": 1, "hidden_size": 4}
    header = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [2, 4],
            "data_offsets": [0, 16],
        },
        "model.norm.weight": {
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [16, 24],
        },
    }
    index = {
        "metadata": {"total_size": 24},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
            "model.norm.weight": "model-00001-of-00001.safetensors",
        },
    }
    files = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors.index.json": json.dumps(index).encode(),
        "model-00001-of-00001.safetensors": make_shard(header, 80),
    }

    tmp = tempfile.mkdtemp(prefix="hf-header-test-")
    try:
        hub = FakeHub(files)
        manifest = fh.fetch_reference(
            fh.DEFAULT_MODEL, revision, tmp, opener=hub, token="secret-token")

        shard_path = os.path.join(tmp, "model-00001-of-00001.safetensors")
        ck(os.path.getsize(shard_path) < len(files["model-00001-of-00001.safetensors"]),
           "local shard is a header-only stub, not the remote payload")

        stub, parsed = fh._read_stub(shard_path)
        ck("model.norm.weight" in parsed,
           "header-only stub remains valid safetensors metadata")
        ck(manifest["immutable_revision"] and manifest["header_only"],
           "manifest records immutable/header-only provenance")
        ck("secret-token" not in json.dumps(manifest),
           "authentication token is never serialized")

        # The existing inventory parser must consume the stub directly. This
        # is the architectural point of the fetcher: no second metadata model.
        inv = inventory.HeaderOnlyIndex(tmp)
        ck(inv.have_headers, "inventory sees all fetched header stubs as headers")
        ck(inv.meta("model.embed_tokens.weight") == ("BF16", [2, 4], 16),
           "inventory reads dtype/shape/bytes from the fetched stub")

        initial_range_calls = sum(
            1 for url, rng in hub.calls
            if "model-00001-of-00001.safetensors" in url and rng)
        fh.fetch_reference(fh.DEFAULT_MODEL, revision, tmp, opener=hub)
        later_range_calls = sum(
            1 for url, rng in hub.calls
            if "model-00001-of-00001.safetensors" in url and rng)
        ck(later_range_calls == initial_range_calls,
           "valid immutable shard stub is reused without new Range requests")

        try:
            fh.validate_revision("main")
            ck(False, "mutable revision is refused by default")
        except fh.FetchError:
            pass

        # If a proxy/server strips Range, accepting HTTP 200 would risk
        # downloading a multi-gigabyte shard. Refusal is mandatory.
        try:
            fh.fetch_safetensors_header(
                fh.resolve_url("x/y", revision,
                               "model-00001-of-00001.safetensors"),
                opener=FakeHub(files, ignore_range=True))
            ck(False, "a server that ignores Range is refused")
        except fh.FetchError:
            pass

        # The index is authoritative about which tensors a shard must expose.
        bad_header = dict(header)
        bad_header.pop("model.norm.weight")
        bad_files = dict(files)
        bad_files["model-00001-of-00001.safetensors"] = make_shard(bad_header, 80)
        bad_dir = tempfile.mkdtemp(prefix="hf-header-bad-")
        try:
            try:
                fh.fetch_reference(
                    fh.DEFAULT_MODEL, revision, bad_dir, opener=FakeHub(bad_files))
                ck(False, "index/header tensor disagreement is refused")
            except fh.FetchError:
                pass
        finally:
            shutil.rmtree(bad_dir, ignore_errors=True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("HF HEADER FETCH FAILED" if fails else "HF HEADER FETCH OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
