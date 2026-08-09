#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a compact real layer-3 Gate F MoE combination fixture.

The expected-value producer is deliberately independent from src/: a standalone
C oracle in tools/v4_moe_oracle.c has its own FP4/FP8 decoders, activation
quantization, SwiGLU and linear loops. Before that fast oracle is trusted for
other routed experts, expert 2 is required to match the already-frozen exact
Fraction-based Python fixture bit-for-bit.

All six selected routed experts are fetched/evaluated, but only their compact
8-output BF16 results and source hashes are frozen. The distinct shared FP8
expert keeps full w1/w3 plus eight w2 rows so ordinary offline tests can replay
that separate arithmetic path permanently.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile

import fetch_hf_tensor_slice as slice_fetch
import make_v4_routed_expert_fixture as exact

LAYER = 3
INPUT = 4096
INTER = 2048
OUT_ROWS = 8
ROUTER_FIXTURE = os.path.join("tests", "fixtures", "deepseek_v4", "v3_router_real")
EXACT_EXPERT_FIXTURE = os.path.join("tests", "fixtures", "deepseek_v4", "v4_routed_expert_real")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_u32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[exact.f32(v) for v in values]))


def read_u16(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 2:
        raise ValueError(f"{path}: odd BF16 byte count")
    return list(struct.unpack(f"<{len(data)//2}H", data))


def read_router(repo_root):
    root = os.path.join(repo_root, ROUTER_FIXTURE)
    with open(os.path.join(root, "provenance.json"), encoding="utf-8") as f:
        p = json.load(f)
    ids = list(p["learned"]["ids"])
    weights = list(p["learned"]["weights"])
    if len(ids) != 6 or len(set(ids)) != 6 or len(weights) != 6:
        raise ValueError("router fixture is not a unique top-6 fixture")
    with open(os.path.join(root, "input.bf16.bin"), "rb") as f:
        raw_input = f.read()
    if len(raw_input) != INPUT * 2:
        raise ValueError("router input is not one BF16 hidden vector")
    return p, ids, [exact.f32(v) for v in weights], raw_input


def fetch_bytes(snapshot, tensor, rows, path, token):
    info = slice_fetch.fetch_slice(snapshot, tensor, rows, path, token=token)
    with open(path, "rb") as f:
        data = f.read()
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return info, data


def build_oracle(repo_root, work):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("no C compiler available for independent Gate F fixture oracle")
    src = os.path.join(repo_root, "tools", "v4_moe_oracle.c")
    exe = os.path.join(work, "v4_moe_oracle")
    cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           "-ffp-contract=off", src, "-lm", "-o", exe]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("independent Gate F oracle compile failed:\n" + p.stdout + p.stderr)
    return exe


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("fixture oracle failed:\n$ " + " ".join(cmd) + "\n" + p.stdout + p.stderr)


def routed_out(snapshot, expert, route_weight, input_path, work, token, oracle_exe, repo_root):
    base = f"layers.{LAYER}.ffn.experts.{expert}"
    names = {
        "w1": base + ".w1.weight", "s1": base + ".w1.scale",
        "w3": base + ".w3.weight", "s3": base + ".w3.scale",
        "w2": base + ".w2.weight", "s2": base + ".w2.scale",
    }
    paths = {key: os.path.join(work, key + ".bin") for key in names}
    infos, data = {}, {}
    for key in ("w1", "s1", "w3", "s3"):
        infos[key], data[key] = fetch_bytes(snapshot, names[key], None, paths[key], token)
    infos["w2"], data["w2"] = fetch_bytes(snapshot, names["w2"], f"0:{OUT_ROWS}", paths["w2"], token)
    infos["s2"], data["s2"] = fetch_bytes(snapshot, names["s2"], f"0:{OUT_ROWS}", paths["s2"], token)

    expected = {
        "w1": ("I8", [INTER, INPUT // 2]),
        "s1": ("F8_E8M0", [INTER, INPUT // 32]),
        "w3": ("I8", [INTER, INPUT // 2]),
        "s3": ("F8_E8M0", [INTER, INPUT // 32]),
        "w2": ("I8", [4096, INTER // 2]),
        "s2": ("F8_E8M0", [4096, INTER // 32]),
    }
    for key, (dtype, shape) in expected.items():
        if infos[key]["dtype"] != dtype or infos[key]["shape"] != shape:
            raise ValueError(
                f"expert {expert} {key}: got {infos[key]['dtype']} {infos[key]['shape']}, "
                f"expected {dtype} {shape}")

    out_path = os.path.join(work, "out8.bf16.bin")
    # v4_moe_oracle intentionally accepts one trailing ignored argument. It
    # makes the fixed CLI shape obvious and keeps accidental argc drift fatal.
    run([oracle_exe, "routed", input_path, paths["w1"], paths["s1"],
         paths["w3"], paths["s3"], paths["w2"], paths["s2"],
         repr(route_weight), out_path, "unused"])
    out_bits = read_u16(out_path)
    if len(out_bits) != OUT_ROWS:
        raise ValueError(f"expert {expert}: oracle returned {len(out_bits)} rows")

    if expert == 2:
        exact_bits = read_u16(os.path.join(repo_root, EXACT_EXPERT_FIXTURE, "expected-out8.bf16.bin"))
        if out_bits != exact_bits:
            raise ValueError(
                "fast independent oracle disagrees with the exact Fraction-based expert-2 anchor: "
                f"{[hex(x) for x in out_bits]} != {[hex(x) for x in exact_bits]}")

    return out_bits, {
        "expert": expert,
        "route_weight": route_weight,
        "output_bf16_hex": [f"0x{x:04x}" for x in out_bits],
        "oracle_anchor": "exact v4_routed_expert_real" if expert == 2 else "fast independent oracle cross-validated on expert 2",
        "tensors": {k: {
            "name": names[k], "shard": infos[k]["shard"],
            "dtype": infos[k]["dtype"], "shape": infos[k]["shape"],
            "payload_sha256": hashlib.sha256(data[k]).hexdigest(),
        } for k in names},
    }


def build_shared(snapshot, input_path, out_dir, token, oracle_exe):
    base = f"layers.{LAYER}.ffn.shared_experts"
    names = {
        "w1": base + ".w1.weight", "s1": base + ".w1.scale",
        "w3": base + ".w3.weight", "s3": base + ".w3.scale",
        "w2": base + ".w2.weight", "s2": base + ".w2.scale",
    }
    paths = {
        "w1": os.path.join(out_dir, "shared-w1.fp8.bin"),
        "s1": os.path.join(out_dir, "shared-w1-scale.e8m0.bin"),
        "w3": os.path.join(out_dir, "shared-w3.fp8.bin"),
        "s3": os.path.join(out_dir, "shared-w3-scale.e8m0.bin"),
        "w2": os.path.join(out_dir, "shared-w2-rows0-8.fp8.bin"),
        "s2": os.path.join(out_dir, "shared-w2-scale-row0.e8m0.bin"),
    }
    infos, data = {}, {}
    for key in ("w1", "s1", "w3", "s3"):
        infos[key], data[key] = fetch_bytes(snapshot, names[key], None, paths[key], token)
    infos["w2"], data["w2"] = fetch_bytes(snapshot, names["w2"], f"0:{OUT_ROWS}", paths["w2"], token)
    infos["s2"], data["s2"] = fetch_bytes(snapshot, names["s2"], "0:1", paths["s2"], token)

    expected = {
        "w1": ("F8_E4M3", [INTER, INPUT]), "s1": ("F8_E8M0", [16, 32]),
        "w3": ("F8_E4M3", [INTER, INPUT]), "s3": ("F8_E8M0", [16, 32]),
        "w2": ("F8_E4M3", [4096, INTER]), "s2": ("F8_E8M0", [32, 16]),
    }
    for key, (dtype, shape) in expected.items():
        if infos[key]["dtype"] != dtype or infos[key]["shape"] != shape:
            raise ValueError(
                f"shared {key}: got {infos[key]['dtype']} {infos[key]['shape']}, expected {dtype} {shape}")

    gate_path = os.path.join(out_dir, "shared-gate-clamped.f32.bin")
    up_path = os.path.join(out_dir, "shared-up-clamped.f32.bin")
    hidden_path = os.path.join(out_dir, "shared-hidden.bf16.bin")
    out_path = os.path.join(out_dir, "shared-out8.bf16.bin")
    run([oracle_exe, "shared", input_path, paths["w1"], paths["s1"],
         paths["w3"], paths["s3"], paths["w2"], paths["s2"],
         gate_path, up_path, hidden_path, out_path, "unused"])
    out_bits = read_u16(out_path)
    if len(out_bits) != OUT_ROWS:
        raise ValueError("shared oracle output length mismatch")

    return out_bits, {
        "output_bf16_hex": [f"0x{x:04x}" for x in out_bits],
        "hidden_sha256": sha256_file(hidden_path),
        "oracle": "standalone tools/v4_moe_oracle.c; no src/ linkage",
        "tensors": {k: {
            "name": names[k], "shard": infos[k]["shard"],
            "dtype": infos[k]["dtype"], "shape": infos[k]["shape"],
            "frozen_file": os.path.basename(paths[k]),
            "payload_sha256": hashlib.sha256(data[k]).hexdigest(),
        } for k in names},
    }


def combine(ids, routed_bits, shared_bits, order):
    y = [exact.f32(0.0) for _ in range(OUT_ROWS)]
    slots = list(range(len(ids)))
    if order == "expert-id":
        slots.sort(key=lambda i: ids[i])
    for slot in slots:
        vals = [exact.bf16_to_f32(b) for b in routed_bits[slot]]
        for d in range(OUT_ROWS):
            y[d] = exact.f32(y[d] + vals[d])
    shared = [exact.bf16_to_f32(b) for b in shared_bits]
    for d in range(OUT_ROWS):
        y[d] = exact.f32(y[d] + shared[d])
    return [exact.bf16_bits(v) for v in y]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    router_p, ids, weights, raw_input = read_router(repo_root)
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    input_path = os.path.join(args.out_dir, "input.bf16.bin")
    with open(input_path, "wb") as f:
        f.write(raw_input)

    routed_bits, routed_meta = [], []
    with tempfile.TemporaryDirectory(prefix="gate-f-fast-oracle-") as td:
        oracle_exe = build_oracle(repo_root, td)
        for slot, (expert, weight) in enumerate(zip(ids, weights)):
            work = os.path.join(td, f"e{expert}")
            os.makedirs(work)
            bits, meta = routed_out(
                args.snapshot, expert, weight, input_path, work,
                args.token, oracle_exe, repo_root)
            routed_bits.append(bits)
            routed_meta.append(meta)
            print(f"routed {slot+1}/6 expert {expert}: " + " ".join(meta["output_bf16_hex"]))

        shared_bits, shared_meta = build_shared(
            args.snapshot, input_path, args.out_dir, args.token, oracle_exe)

    combined = combine(ids, routed_bits, shared_bits, "expert-id")
    topk_combined = combine(ids, routed_bits, shared_bits, "topk")

    write_u32(os.path.join(args.out_dir, "routed-ids.u32.bin"), ids)
    write_f32(os.path.join(args.out_dir, "routed-weights.f32.bin"), weights)
    write_u16(os.path.join(args.out_dir, "routed-out8.bf16.bin"), [b for row in routed_bits for b in row])
    write_u16(os.path.join(args.out_dir, "combined-out8.bf16.bin"), combined)

    provenance = {
        "schema_version": 2,
        "gate": "Gate F / V4 complete MoE primitive",
        "fixture_class": "F3 pinned official source + real 0731 checkpoint",
        "evidence_state": "CHECKPOINT-VERIFIED_SOURCE-ORACLE",
        "model": router_p["model"], "revision": router_p["revision"],
        "layer": LAYER, "input_sha256": hashlib.sha256(raw_input).hexdigest(),
        "router": {"ids_topk_order": ids, "weights": weights},
        "oracle_independence": {
            "producer": "tools/v4_moe_oracle.c",
            "shares_runtime_helpers": False,
            "routed_anchor": "expert 2 must exactly match the earlier Fraction-based Python fixture before remaining experts are accepted",
        },
        "official_accumulation": {
            "routed_loop_order": sorted(ids),
            "shared_added_after_routed": True,
            "accumulator": "float32 zeros_like",
            "final_cast": "BF16",
            "source": "inference/model.py MoE.forward",
        },
        "routed_experts": routed_meta,
        "shared_expert": shared_meta,
        "expected": {
            "routed_file": "routed-out8.bf16.bin",
            "shared_file": "shared-out8.bf16.bin",
            "combined_file": "combined-out8.bf16.bin",
            "combined_bf16_hex": [f"0x{x:04x}" for x in combined],
            "topk_order_bf16_hex": [f"0x{x:04x}" for x in topk_combined],
            "order_mutation_visible_at_bf16": combined != topk_combined,
        },
        "storage_policy": "All six routed experts are fetched and independently evaluated during fixture generation. Only their compact BF16 outputs/hashes are frozen; full routed expert 3/2 remains a separate exact fixture. Full shared w1/w3 plus eight w2 rows are frozen because shared FP8 is a distinct arithmetic path.",
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    print("shared output:", " ".join(shared_meta["output_bf16_hex"]))
    print("combined output:", " ".join(provenance["expected"]["combined_bf16_hex"]))
    print("expert-id order:", sorted(ids))
    print("order mutation visible at BF16:", provenance["expected"]["order_mutation_visible_at_bf16"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
