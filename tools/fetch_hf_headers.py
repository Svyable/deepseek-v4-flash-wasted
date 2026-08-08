#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Fetch Hugging Face safetensors headers without downloading tensor payloads.

The output directory looks like a normal Hugging Face snapshot to
``tools/inventory.py``: it contains ``config.json``,
``model.safetensors.index.json`` and one file per shard. The shard files are
intentional *header-only stubs*: the original 8-byte safetensors header length
followed by the original JSON header, with no tensor data.

Hugging Face documents this exact two-Range-request metadata technique. Keeping
the stub in the native safetensors framing means the existing inventory parser
remains the only implementation of checkpoint byte/shape accounting.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import struct
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_ENDPOINT = "https://huggingface.co"
MAX_HEADER_BYTES = 100 << 20
MAX_METADATA_BYTES = 64 << 20
SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class FetchError(RuntimeError):
    pass


def validate_revision(revision, allow_mutable=False):
    if not revision:
        raise FetchError("revision is required")
    if not allow_mutable and not SHA40_RE.fullmatch(revision):
        raise FetchError(
            "revision must be a full 40-hex commit SHA; pass "
            "--allow-mutable-revision only for disposable exploration")
    return revision


def resolve_url(model, revision, path, endpoint=DEFAULT_ENDPOINT):
    model = quote(model.strip("/"), safe="/")
    revision = quote(revision, safe="")
    path = quote(path.lstrip("/"), safe="/")
    return f"{endpoint.rstrip('/')}/{model}/resolve/{revision}/{path}"


def _request(url, token=None, range_header=None):
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "deepseek-v4-flash-wasted-header-fetch/1",
    }
    if range_header:
        headers["Range"] = range_header
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return Request(url, headers=headers, method="GET")


def _status(resp):
    status = getattr(resp, "status", None)
    if status is None and hasattr(resp, "getcode"):
        status = resp.getcode()
    return status


def _header(resp, name):
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    if hasattr(headers, "get"):
        return headers.get(name)
    return None


def _open(opener, request, timeout):
    try:
        return opener(request, timeout=timeout)
    except TypeError:
        return opener(request)


def _close(resp):
    close = getattr(resp, "close", None)
    if close:
        close()


def fetch_small(url, *, token=None, opener=urlopen, timeout=60,
                max_bytes=MAX_METADATA_BYTES):
    req = _request(url, token=token)
    try:
        resp = _open(opener, req, timeout)
    except (HTTPError, URLError, OSError) as e:
        raise FetchError(f"{url}: {e}") from e
    try:
        status = _status(resp)
        if status not in (200, 206, None):
            raise FetchError(f"{url}: unexpected HTTP status {status}")
        cl = _header(resp, "Content-Length")
        if cl is not None:
            try:
                if int(cl) > max_bytes:
                    raise FetchError(
                        f"{url}: metadata is {cl} bytes, limit is {max_bytes}")
            except ValueError:
                pass
        data = resp.read(max_bytes + 1)
    finally:
        _close(resp)
    if len(data) > max_bytes:
        raise FetchError(f"{url}: metadata exceeds {max_bytes} bytes")
    return data


def _parse_content_range(value):
    if not value:
        return None
    m = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip())
    if not m:
        raise FetchError(f"malformed Content-Range: {value!r}")
    start, end = int(m.group(1)), int(m.group(2))
    total = None if m.group(3) == "*" else int(m.group(3))
    return start, end, total


def fetch_range(url, start, end, *, token=None, opener=urlopen, timeout=60):
    if start < 0 or end < start:
        raise FetchError(f"invalid byte range {start}-{end}")
    want = end - start + 1
    req = _request(url, token=token, range_header=f"bytes={start}-{end}")
    try:
        resp = _open(opener, req, timeout)
    except (HTTPError, URLError, OSError) as e:
        raise FetchError(f"{url} range {start}-{end}: {e}") from e
    try:
        status = _status(resp)
        if status != 206:
            raise FetchError(
                f"{url}: server ignored Range bytes={start}-{end} "
                f"(HTTP {status}); refusing a possible full-shard download")
        cr = _parse_content_range(_header(resp, "Content-Range"))
        if cr is not None:
            got_start, got_end, total = cr
            if got_start != start or got_end != end:
                raise FetchError(
                    f"{url}: Content-Range returned {got_start}-{got_end}, "
                    f"wanted {start}-{end}")
        else:
            total = None
        data = resp.read(want + 1)
    finally:
        _close(resp)
    if len(data) != want:
        raise FetchError(
            f"{url}: short/long range body {len(data)} bytes, wanted {want}")
    return data, total


def _validate_header_object(header, source):
    if not isinstance(header, dict):
        raise FetchError(f"{source}: safetensors header is not an object")
    tensors = 0
    max_end = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        tensors += 1
        if not isinstance(meta, dict):
            raise FetchError(f"{source}: {name}: tensor metadata is not an object")
        if not all(k in meta for k in ("dtype", "shape", "data_offsets")):
            raise FetchError(f"{source}: {name}: incomplete tensor metadata")
        offsets = meta["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(v, int) for v in offsets)):
            raise FetchError(f"{source}: {name}: invalid data_offsets")
        beg, end = offsets
        if beg < 0 or end < beg:
            raise FetchError(f"{source}: {name}: invalid data_offsets {offsets}")
        max_end = max(max_end, end)
    return tensors, max_end


def fetch_safetensors_header(url, *, token=None, opener=urlopen, timeout=60):
    raw_len, total = fetch_range(
        url, 0, 7, token=token, opener=opener, timeout=timeout)
    (header_len,) = struct.unpack("<Q", raw_len)
    if header_len == 0 or header_len > MAX_HEADER_BYTES:
        raise FetchError(f"{url}: implausible header length {header_len}")
    raw_header, total2 = fetch_range(
        url, 8, 7 + header_len, token=token, opener=opener, timeout=timeout)
    if total is None:
        total = total2
    elif total2 is not None and total2 != total:
        raise FetchError(f"{url}: remote size changed between Range requests")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise FetchError(f"{url}: invalid safetensors JSON header: {e}") from e
    tensor_count, max_end = _validate_header_object(header, url)
    if total is not None and 8 + header_len + max_end > total:
        raise FetchError(
            f"{url}: header describes data through byte "
            f"{8 + header_len + max_end}, remote file is {total} bytes")
    return raw_len + raw_header, header, total, tensor_count


def _atomic_write(path, data):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _load_json_bytes(data, source):
    try:
        obj = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise FetchError(f"{source}: invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise FetchError(f"{source}: expected a JSON object")
    return obj


def _read_stub(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise FetchError(f"{path}: truncated local header stub")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len == 0 or header_len > MAX_HEADER_BYTES:
            raise FetchError(f"{path}: invalid local header length {header_len}")
        blob = f.read(header_len)
        if len(blob) != header_len:
            raise FetchError(f"{path}: truncated local JSON header")
    try:
        header = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise FetchError(f"{path}: invalid local JSON header: {e}") from e
    _validate_header_object(header, path)
    return raw + blob, header


def fetch_reference(model, revision, out_dir, *, endpoint=DEFAULT_ENDPOINT,
                    token=None, opener=urlopen, timeout=60,
                    allow_mutable_revision=False, force=False, progress=None):
    validate_revision(revision, allow_mutable=allow_mutable_revision)
    os.makedirs(out_dir, exist_ok=True)

    def note(msg):
        if progress:
            progress(msg)

    config_url = resolve_url(model, revision, "config.json", endpoint)
    index_url = resolve_url(
        model, revision, "model.safetensors.index.json", endpoint)
    config_bytes = fetch_small(
        config_url, token=token, opener=opener, timeout=timeout)
    index_bytes = fetch_small(
        index_url, token=token, opener=opener, timeout=timeout)
    _load_json_bytes(config_bytes, config_url)
    index = _load_json_bytes(index_bytes, index_url)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise FetchError(f"{index_url}: missing/non-object weight_map")
    if not all(isinstance(k, str) and isinstance(v, str)
               for k, v in weight_map.items()):
        raise FetchError(f"{index_url}: weight_map must map strings to strings")

    _atomic_write(os.path.join(out_dir, "config.json"), config_bytes)
    _atomic_write(
        os.path.join(out_dir, "model.safetensors.index.json"), index_bytes)

    expected_by_shard = {}
    for tensor, shard in weight_map.items():
        expected_by_shard.setdefault(shard, set()).add(tensor)

    shard_manifest = {}
    for shard in sorted(expected_by_shard):
        local = os.path.join(out_dir, shard)
        expected = expected_by_shard[shard]
        stub = header = None
        reused = False
        if not force and os.path.exists(local):
            try:
                stub, header = _read_stub(local)
                names = set(header) - {"__metadata__"}
                if expected.issubset(names):
                    reused = True
                    note(f"reuse {shard}")
                else:
                    stub = header = None
            except FetchError:
                stub = header = None

        remote_size = None
        if stub is None:
            url = resolve_url(model, revision, shard, endpoint)
            note(f"fetch {shard} header")
            stub, header, remote_size, _ = fetch_safetensors_header(
                url, token=token, opener=opener, timeout=timeout)
            names = set(header) - {"__metadata__"}
            missing = sorted(expected - names)
            if missing:
                raise FetchError(
                    f"{shard}: {len(missing)} index tensor(s) missing from "
                    f"header, first: {missing[:3]}")
            _atomic_write(local, stub)
        tensor_count, max_end = _validate_header_object(header, local)
        shard_manifest[shard] = {
            "header_stub_sha256": _sha256(stub),
            "header_stub_bytes": len(stub),
            "header_json_bytes": len(stub) - 8,
            "tensor_count": tensor_count,
            "indexed_tensor_count": len(expected),
            "max_data_offset": max_end,
            "remote_file_bytes": remote_size,
            "reused": reused,
        }

    manifest = {
        "schema_version": 1,
        "model": model,
        "revision": revision,
        "immutable_revision": bool(SHA40_RE.fullmatch(revision)),
        "endpoint": endpoint.rstrip("/"),
        "header_only": True,
        "retrieved_at_utc": _dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat(),
        "config_sha256": _sha256(config_bytes),
        "index_sha256": _sha256(index_bytes),
        "index_metadata": index.get("metadata", {}),
        "shards": shard_manifest,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(os.path.join(out_dir, "reference_headers.json"), manifest_bytes)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fetch only Hugging Face config/index and safetensors "
                    "headers for Gate A/V0 inventory.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--revision", required=True,
                    help="full immutable 40-hex Hugging Face commit SHA")
    ap.add_argument("--out", default="reference/deepseek-v4-flash-0731",
                    help="header-only snapshot directory")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="Hugging Face endpoint; mainly useful for tests/mirrors")
    ap.add_argument("--token-env", default="HF_TOKEN",
                    help="environment variable containing an optional Hub token")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--force", action="store_true",
                    help="refetch shard headers even when a valid local stub exists")
    ap.add_argument("--allow-mutable-revision", action="store_true",
                    help="allow branch/tag/short revisions for disposable exploration")
    args = ap.parse_args(argv)

    token = os.environ.get(args.token_env) if args.token_env else None
    try:
        manifest = fetch_reference(
            args.model, args.revision, args.out,
            endpoint=args.endpoint,
            token=token,
            timeout=args.timeout,
            allow_mutable_revision=args.allow_mutable_revision,
            force=args.force,
            progress=lambda s: print(f"  {s}"),
        )
    except FetchError as e:
        print(f"fetch_hf_headers: {e}", file=sys.stderr)
        return 2

    n = len(manifest["shards"])
    print(f"\nheader-only snapshot ready: {args.out}")
    print(f"  model:    {manifest['model']}")
    print(f"  revision: {manifest['revision']}")
    print(f"  shards:   {n}")
    print("\nnext:")
    print(f"  python3 tools/inventory.py {args.out} --strict --by-layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
