#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Offline replay for the frozen real layer-3 Gate H MoE branch.

The raw ~105 MB of checkpoint expert data is intentionally not committed.
Acquisition provenance records exact 4096-BF16 agreement between the standalone
oracle and WASTE for six routed experts plus the shared expert. This permanent
replay protects the compact boundary: input digest, routing identity, all frozen
outputs, official ascending-expert-ID accumulation, and branch visibility.
"""

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v6_moe_branch_real")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
IDS = [255, 30, 99, 40, 44, 238]
OUT = 4096
TOPK = 6


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 2:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 2}")
    return list(struct.unpack(f"<{count}H", data))


def read_u32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}I", data))


def bits_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def bf16_float(u):
    return bits_f32((u & 0xFFFF) << 16)


def f32(v):
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", f32(v)))[0]


def bf16_bits(v):
    u = f32_bits(v)
    exp, frac = u & 0x7F800000, u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def combine(ids, routed, shared):
    y = [f32(0.0)] * OUT
    for slot in sorted(range(TOPK), key=lambda i: ids[i]):
        row = routed[slot]
        for d in range(OUT):
            y[d] = f32(y[d] + bf16_float(row[d]))
    for d in range(OUT):
        y[d] = f32(y[d] + bf16_float(shared[d]))
    return [bf16_bits(v) for v in y]


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.isfile(prov_path):
        print("SKIP: v6_moe_branch_real fixture is not frozen yet")
        return 77
    p = json.load(open(prov_path, encoding="utf-8"))
    if p.get("revision") != REV or p.get("layer") != 3:
        print("FAIL: MoE fixture is not pinned layer-3 0731 evidence")
        return 1

    dep = p.get("input_dependency") or {}
    input_path = os.path.join(FIX, dep.get("file", ""))
    if not os.path.isfile(input_path) or sha(input_path) != dep.get("sha256"):
        print("FAIL: MoE branch input dependency is missing or changed")
        return 1
    if len(read_u16(input_path, OUT)) != OUT:
        print("FAIL: ffn_pre input shape drifted")
        return 1

    outputs = p.get("outputs") or {}
    required = {
        "routed": TOPK * OUT,
        "shared": OUT,
        "moe_branch": OUT,
        "layer3_final": 4 * OUT,
    }
    paths = {}
    for key, count in required.items():
        meta = outputs.get(key) or {}
        path = os.path.join(FIX, meta.get("file", ""))
        if not os.path.isfile(path) or sha(path) != meta.get("sha256"):
            print(f"FAIL: frozen {key} output missing or SHA mismatch")
            return 1
        if len(read_u16(path, count)) != count:
            print(f"FAIL: frozen {key} output shape drifted")
            return 1
        paths[key] = path

    ids_meta = outputs.get("ids") or {}
    ids_path = os.path.join(FIX, ids_meta.get("file", ""))
    if not os.path.isfile(ids_path) or sha(ids_path) != ids_meta.get("sha256"):
        print("FAIL: routed ID file missing or SHA mismatch")
        return 1
    ids = read_u32(ids_path, TOPK)
    if ids != IDS or p.get("router", {}).get("ids_topk_order") != IDS:
        print(f"FAIL: frozen real route changed: {ids}")
        return 1

    verify = p.get("acquisition_verification") or {}
    if verify.get("routed_exact_bf16_values") != TOPK * OUT or \
       verify.get("shared_exact_bf16_values") != OUT or \
       verify.get("total_expert_exact_bf16_values") != (TOPK + 1) * OUT:
        print("FAIL: acquisition provenance does not record all 7x4096 exact comparisons")
        return 1
    if verify.get("raw_checkpoint_weights_committed") is not False:
        print("FAIL: fixture provenance no longer states raw expert weights are transient")
        return 1
    routed_meta = p.get("routed_experts") or []
    if [x.get("expert") for x in routed_meta] != IDS or \
       any(x.get("oracle_vs_waste_exact_bf16") != OUT for x in routed_meta):
        print("FAIL: per-expert acquisition evidence is incomplete")
        return 1
    if (p.get("shared_expert") or {}).get("oracle_vs_waste_exact_bf16") != OUT:
        print("FAIL: shared-expert acquisition evidence is incomplete")
        return 1

    routed_flat = read_u16(paths["routed"], TOPK * OUT)
    routed = [routed_flat[i * OUT:(i + 1) * OUT] for i in range(TOPK)]
    shared = read_u16(paths["shared"], OUT)
    want = read_u16(paths["moe_branch"], OUT)
    if combine(ids, routed, shared) != want:
        print("FAIL: independent ascending-ID combination disagrees with frozen MoE branch")
        return 1

    # All seven branches must remain load-bearing at the BF16 branch boundary.
    for slot, expert in enumerate(ids):
        muted = [list(row) for row in routed]
        muted[slot] = [0] * OUT
        if combine(ids, muted, shared) == want:
            print(f"FAIL: dropping routed expert {expert} is invisible")
            return 1
    if combine(ids, routed, [0] * OUT) == want:
        print("FAIL: dropping the shared expert is invisible")
        return 1

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77
    with tempfile.TemporaryDirectory(prefix="v6-moe-replay-") as td:
        exe = os.path.join(td, "check")
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "tools", "v6_moe_runtime_check.c"),
               os.path.join(REPO, "src", "deepseek_v4_expert_ref.c"),
               os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
               os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
               os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
               "-lm", "-o", exe]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile Gate H MoE combine replay")
            return 1
        r = subprocess.run([exe, "combine", ids_path, paths["routed"],
                            paths["shared"], paths["moe_branch"]],
                           capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout + r.stderr)
            print("FAIL: WASTE full-width MoE combine disagrees with frozen branch")
            return 1

    weights = p.get("router", {}).get("weights_f32") or []
    if len(weights) != TOPK or not math.isclose(sum(weights), 1.5, rel_tol=0, abs_tol=2e-6):
        print("FAIL: route weights no longer sum to the 1.5 routed scaling")
        return 1

    print("PASS Gate H/V6 frozen real MoE branch: 6 routed + shared -> 4096 BF16")
    print("experts:", ids)
    print("moe first8:", " ".join(f"0x{x:04x}" for x in want[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
