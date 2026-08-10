#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free contract tests for tools/v7_localize.py.

These tests make "first divergent layer" load-bearing before real multi-layer
checkpoint acquisition. A comparator that reports the last mismatch, accepts a
broken output->input chain, or trusts stale manifest hashes must fail here.
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "v7_localize.py")
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_values(path, dtype, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = {"BF16": "H", "F32": "I", "U32": "I"}[dtype]
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}{fmt}", *values))


def add_boundary(root, layer, name, dtype, shape, values):
    rel = f"layer-{layer}/{name}.bin"
    path = os.path.join(root, rel)
    write_values(path, dtype, values)
    return {"name": name, "file": rel, "dtype": dtype, "shape": shape, "sha256": sha(path)}


def state(seed):
    return [((seed * 257 + i * 37) & 0x7FFF) | (0x8000 if i % 5 == 1 else 0)
            for i in range(32)]


def vector(seed):
    return [((seed * 131 + i * 19) & 0x7FFF) | (0x8000 if i % 3 == 2 else 0)
            for i in range(8)]


def build_trace(root):
    # Layer 3 output is *exactly* layer 4 input. Everything else is distinct.
    l3_in = state(3)
    l3_out = state(4)
    l4_out = state(5)
    layers = []
    for layer, cls, inp, out in (
        (3, "ratio128-learned", l3_in, l3_out),
        (4, "ratio4-learned", l3_out, l4_out),
    ):
        boundaries = [
            add_boundary(root, layer, "input", "BF16", [4, 8], inp),
            add_boundary(root, layer, "attn_pre", "BF16", [8], vector(layer * 10 + 1)),
            add_boundary(root, layer, "after_attn", "BF16", [4, 8], state(layer * 10 + 2)),
            add_boundary(root, layer, "ffn_pre", "BF16", [8], vector(layer * 10 + 3)),
            add_boundary(root, layer, "router_ids", "U32", [6], [layer + i * 7 for i in range(6)]),
            add_boundary(root, layer, "moe_branch", "BF16", [8], vector(layer * 10 + 4)),
            add_boundary(root, layer, "output", "BF16", [4, 8], out),
        ]
        layers.append({"layer": layer, "structural_class": cls, "boundaries": boundaries})
    manifest = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REV,
        "trace_name": "synthetic-v7-two-layer",
        "layers": layers,
    }
    path = os.path.join(root, "trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def boundary(manifest, layer, name):
    for item in manifest["layers"]:
        if item["layer"] == layer:
            for b in item["boundaries"]:
                if b["name"] == name:
                    return b
    raise KeyError((layer, name))


def mutate(root, manifest_path, layer, name, index, xor, update_sha=True):
    m = load_manifest(manifest_path)
    b = boundary(m, layer, name)
    path = os.path.join(root, b["file"])
    data = bytearray(open(path, "rb").read())
    width = {"BF16": 2, "U32": 4, "F32": 4}[b["dtype"]]
    off = index * width
    data[off] ^= xor
    with open(path, "wb") as f:
        f.write(data)
    if update_sha:
        b["sha256"] = sha(path)
        save_manifest(manifest_path, m)


def run(expected, actual):
    p = subprocess.run([sys.executable, TOOL, expected, actual, "--json"],
                       capture_output=True, text=True)
    text = p.stdout.strip()
    if not text:
        raise AssertionError(f"localizer produced no JSON: rc={p.returncode} stderr={p.stderr}")
    return p.returncode, json.loads(text)


def fresh_pair(td, suffix):
    exp = os.path.join(td, suffix + "-expected")
    act = os.path.join(td, suffix + "-actual")
    os.makedirs(exp)
    exp_manifest = build_trace(exp)
    shutil.copytree(exp, act)
    return exp, exp_manifest, act, os.path.join(act, "trace.json")


def main():
    with tempfile.TemporaryDirectory(prefix="v7-localize-") as td:
        exp, em, act, am = fresh_pair(td, "match")
        rc, result = run(em, am)
        if rc != 0 or result.get("status") != "match" or result.get("layers") != [3, 4]:
            print("FAIL: identical chained traces did not match", result)
            return 1
        if result.get("boundaries_compared") != 14:
            print("FAIL: not all 14 boundaries were compared", result)
            return 1

        exp, em, act, am = fresh_pair(td, "late")
        mutate(act, am, 4, "ffn_pre", 5, 0x01)
        rc, result = run(em, am)
        if rc != 1 or (result.get("layer"), result.get("boundary"), result.get("index")) != (4, "ffn_pre", 5):
            print("FAIL: late divergence was not localized exactly", result)
            return 1

        exp, em, act, am = fresh_pair(td, "earliest")
        mutate(act, am, 4, "ffn_pre", 5, 0x01)
        mutate(act, am, 3, "after_attn", 17, 0x01)
        rc, result = run(em, am)
        if rc != 1 or (result.get("layer"), result.get("boundary"), result.get("index")) != (3, "after_attn", 17):
            print("FAIL: comparator did not stop at earliest divergence", result)
            return 1

        exp, em, act, am = fresh_pair(td, "integrity")
        mutate(act, am, 4, "moe_branch", 2, 0x01, update_sha=False)
        rc, result = run(em, am)
        if rc != 2 or result.get("status") != "invalid" or "file SHA" not in result.get("error", ""):
            print("FAIL: stale manifest hash was treated as numerical drift", result)
            return 1

        exp, em, act, am = fresh_pair(td, "chain")
        mutate(act, am, 4, "input", 0, 0x01, update_sha=True)
        rc, result = run(em, am)
        if rc != 2 or result.get("status") != "invalid" or "chain break" not in result.get("error", ""):
            print("FAIL: broken output->next-input chain was accepted", result)
            return 1

        exp, em, act, am = fresh_pair(td, "class")
        m = load_manifest(am)
        m["layers"][1]["structural_class"] = "ratio128-learned"
        save_manifest(am, m)
        rc, result = run(em, am)
        if rc != 2 or result.get("status") != "invalid" or "structural class mismatch" not in result.get("error", ""):
            print("FAIL: incompatible structural class was compared", result)
            return 1

    print("PASS V7 first-divergence localizer: integrity, chaining, compatibility and earliest-boundary semantics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
