#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Finalize Gate H's layer-3 expected state at the real F32 HC boundary.

The independent Sinkhorn equations are intentionally written in Python and keep
more precision than the checkpoint runtime. `post` and `comb` are float32 model
tensors at the hc_post boundary. Usually the extra Python precision disappears
at the final BF16 cast; the first full real MoE branch found one value exactly
on that boundary. This finalizer makes the boundary explicit instead of adding
a tolerance: round post/comb to F32, execute the same independently ordered
hc_post equation, and rewrite the frozen expected BF16 state/provenance.
"""

import argparse
import hashlib
import json
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import make_v6_moe_branch_fixture as fixture  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402


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


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture_dir")
    args = ap.parse_args(argv)

    prov_path = os.path.join(args.fixture_dir, "provenance.json")
    p = json.load(open(prov_path, encoding="utf-8"))
    if p.get("revision") != fixture.REV or p.get("layer") != fixture.LAYER:
        raise ValueError("refusing to finalize a fixture other than pinned 0731 layer 3")

    state = fixture.derive_real_ffn_input()
    branch_meta = p["outputs"]["moe_branch"]
    branch_path = os.path.join(args.fixture_dir, branch_meta["file"])
    if sha(branch_path) != branch_meta["sha256"]:
        raise ValueError("MoE branch changed before final HC finalization")
    branch = [chain.bf16_to_f32(v) for v in read_u16(branch_path, fixture.OUT)]

    # Explicit model boundary: Sinkhorn outputs are F32 tensors before hc_post.
    post = [chain.f32(v) for v in state["ffn_post"]]
    comb = [[chain.f32(v) for v in row] for row in state["ffn_comb"]]
    final_f = chain.oracle_hc_post(branch, state["after_attn"], post, comb)
    final_bits, _ = chain.cast_bf16(final_f)

    meta = p["outputs"]["layer3_final"]
    final_path = os.path.join(args.fixture_dir, meta["file"])
    before = read_u16(final_path, chain.HC * fixture.OUT)
    changed = [i for i, (a, b) in enumerate(zip(before, final_bits)) if a != b]
    write_u16(final_path, final_bits)
    meta["sha256"] = sha(final_path)
    meta["first8_hex"] = [f"0x{x:04x}" for x in final_bits[:8]]
    p["final_hc_boundary"] = {
        "post_dtype": "F32",
        "comb_dtype": "F32",
        "oracle_rule": "independent Sinkhorn post/comb are rounded to F32 before hc_post; final state then casts to BF16",
        "values_changed_vs_unrounded_python_oracle": len(changed),
        "first_changed_indices": changed[:16],
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Gate H final HC F32 boundary pinned; changed {len(changed)} / {len(final_bits)} BF16 values")
    if changed:
        print("first changed indices:", changed[:16])
    print("final first8:", " ".join(f"0x{x:04x}" for x in final_bits[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
