#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the full real layer-3 Gate H MoE branch from bounded 0731 slices.

The expensive question has already been answered offline: the real layer-3
FFN input routes to experts [255,30,99,40,44,238]. This generator therefore
fetches exactly those six routed records plus the one shared expert, evaluates
all 4096 output rows twice, and freezes only compact outputs/provenance.

Expected values come from a 4096-row specialization of the standalone Gate-F
oracle (`tools/v4_moe_oracle.c`), which links no WASTE source. The same raw
checkpoint slices are then replayed through `tools/v6_moe_runtime_check.c`,
which does link the WASTE scalar refs. Nothing freezes unless all 7 * 4096
BF16 expert outputs agree exactly.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import fetch_hf_tensor_slice as slice_fetch  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
LAYER = 3
INPUT = 4096
INTER = 2048
OUT = 4096
TOPK = 6
EXPECTED_IDS = [255, 30, 99, 40, 44, 238]

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
HCFIX = os.path.join(BASE, "v6_hc_composition_real")
ATTFIX = os.path.join(BASE, "v6_attention_branch_real")
RTFIX = os.path.join(BASE, "v3_router_real")


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
        f.write(struct.pack(f"<{len(values)}f", *[chain.f32(v) for v in values]))


def read_u16(path):
    data = open(path, "rb").read()
    if len(data) % 2:
        raise ValueError(f"{path}: odd BF16 byte count")
    return list(struct.unpack(f"<{len(data)//2}H", data))


def run(cmd, *, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(
            "command failed:\n$ " + " ".join(cmd) + "\n" + p.stdout + p.stderr)
    return p


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError("no C compiler available for Gate H acquisition")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def derive_real_ffn_input():
    """Independent Python composition through real layer-3 ffn_pre + router."""
    hc = load_json(os.path.join(HCFIX, "provenance.json"))
    att = load_json(os.path.join(ATTFIX, "provenance.json"))
    rt = load_json(os.path.join(RTFIX, "provenance.json"))
    for label, prov in (("hc", hc), ("attention", att), ("router", rt)):
        if prov.get("revision") != REV:
            raise ValueError(f"{label} dependency revision drifted")

    params = hc["parameters"]
    states = hc["states"]
    attn_pre_path = os.path.join(HCFIX, states["attn_pre"]["file"])
    dep = att.get("input_dependency") or {}
    if sha256_file(attn_pre_path) != dep.get("sha256"):
        raise ValueError("real attention branch is not bound to this attn_pre")

    def f32_param(key, count):
        meta = params[key]
        path = os.path.join(HCFIX, meta["file"])
        want = meta.get("payload_sha256", meta.get("sha256"))
        if want and sha256_file(path) != want:
            raise ValueError(f"HC parameter SHA drifted: {key}")
        return chain.read_f32(path, count)

    residual = [chain.bf16_to_f32(v) for v in chain.read_u16(
        os.path.join(HCFIX, states["residual"]["file"]), chain.FLAT)]
    branch_meta = att["outputs"]["branch"]
    branch_path = os.path.join(ATTFIX, branch_meta["file"])
    if sha256_file(branch_path) != branch_meta["sha256"]:
        raise ValueError("real attention branch SHA drifted")
    attention_branch = [chain.bf16_to_f32(v) for v in
                        chain.read_u16(branch_path, INPUT)]

    attn_fn = f32_param("attn_fn", chain.MIX * chain.FLAT)
    attn_base = f32_param("attn_base", chain.MIX)
    attn_scale = f32_param("attn_scale", 3)
    ffn_fn = f32_param("ffn_fn", chain.MIX * chain.FLAT)
    ffn_base = f32_param("ffn_base", chain.MIX)
    ffn_scale = f32_param("ffn_scale", 3)

    apre, apost, acomb = chain.oracle_hc_params(
        residual, attn_fn, attn_scale, attn_base)
    attn_pre_bits, _ = chain.cast_bf16(chain.oracle_hc_pre_y(residual, apre))
    if attn_pre_bits != chain.read_u16(attn_pre_path, INPUT):
        raise ValueError("independent hc_attn_pre no longer matches frozen input")

    after_f = chain.oracle_hc_post(attention_branch, residual, apost, acomb)
    after_bits, after = chain.cast_bf16(after_f)
    fpre, fpost, fcomb = chain.oracle_hc_params(after, ffn_fn, ffn_scale, ffn_base)
    ffn_pre_bits, ffn_pre = chain.cast_bf16(chain.oracle_hc_pre_y(after, fpre))

    gate_path = os.path.join(RTFIX, "layer3-gate-weight.bf16.bin")
    bias_path = os.path.join(RTFIX, "layer3-gate-bias.f32.bin")
    gate_bits = chain.read_u16(gate_path, chain.EXPERTS * INPUT)
    bias = chain.read_f32(bias_path, chain.EXPERTS)
    ids, weights64, _, _, margin = chain.oracle_router(ffn_pre, gate_bits, bias)
    weights = [chain.f32(v) for v in weights64]
    if ids != EXPECTED_IDS:
        raise ValueError(f"real routing drifted: {ids} != {EXPECTED_IDS}")
    if not margin > 1e-4:
        raise ValueError(f"top-k boundary became unstable: {margin}")

    return {
        "hc": hc,
        "attention": att,
        "router": rt,
        "attention_branch_path": branch_path,
        "after_attn_bits": after_bits,
        "after_attn": after,
        "ffn_pre_bits": ffn_pre_bits,
        "ffn_pre": ffn_pre,
        "ffn_post": fpost,
        "ffn_comb": fcomb,
        "ids": ids,
        "weights": weights,
        "margin": margin,
    }


def tensor_meta(info):
    return {
        "name": info["tensor"],
        "shard": info["shard"],
        "dtype": info["dtype"],
        "shape": info["shape"],
        "rows": [info["row_start"], info["row_end"]],
        "absolute_http_range_inclusive": [info["absolute_start"], info["absolute_end"]],
        "payload_bytes": info["payload_bytes"],
        "payload_sha256": info["payload_sha256"],
    }


def fetch(snapshot, tensor, path, token):
    info = slice_fetch.fetch_slice(snapshot, tensor, None, path, token=token)
    if info["revision"] != REV:
        raise ValueError(f"{tensor}: revision {info['revision']} != {REV}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return info


def validate_routed_shapes(expert, infos):
    expected = {
        "w1": ("I8", [INTER, INPUT // 2]),
        "s1": ("F8_E8M0", [INTER, INPUT // 32]),
        "w3": ("I8", [INTER, INPUT // 2]),
        "s3": ("F8_E8M0", [INTER, INPUT // 32]),
        "w2": ("I8", [OUT, INTER // 2]),
        "s2": ("F8_E8M0", [OUT, INTER // 32]),
    }
    for key, (dtype, shape) in expected.items():
        got = infos[key]
        if got["dtype"] != dtype or got["shape"] != shape:
            raise ValueError(
                f"expert {expert} {key}: {got['dtype']} {got['shape']} != {dtype} {shape}")


def validate_shared_shapes(infos):
    expected = {
        "w1": ("F8_E4M3", [INTER, INPUT]),
        "s1": ("F8_E8M0", [INTER // 128, INPUT // 128]),
        "w3": ("F8_E4M3", [INTER, INPUT]),
        "s3": ("F8_E8M0", [INTER // 128, INPUT // 128]),
        "w2": ("F8_E4M3", [OUT, INTER]),
        "s2": ("F8_E8M0", [OUT // 128, INTER // 128]),
    }
    for key, (dtype, shape) in expected.items():
        got = infos[key]
        if got["dtype"] != dtype or got["shape"] != shape:
            raise ValueError(f"shared {key}: {got['dtype']} {got['shape']} != {dtype} {shape}")


def build_tools(work):
    cc = compiler()
    gate_f_source = os.path.join(REPO, "tools", "v4_moe_oracle.c")
    source = open(gate_f_source, encoding="utf-8").read()
    old_define = "#define OUT_ROWS 8u"
    old_shared_scale = "uint8_t *s2 = read_exact(argv[8], (INTER/128));"
    new_define = f"#define OUT_ROWS {OUT}u"
    new_shared_scale = (
        "uint8_t *s2 = read_exact(argv[8], "
        "((OUT_ROWS + 127u) / 128u) * (INTER/128u));")
    if source.count(old_define) != 1 or source.count(old_shared_scale) != 1:
        raise ValueError("Gate-F standalone oracle source changed; specialization no longer applies exactly")
    specialized = source.replace(old_define, new_define).replace(old_shared_scale, new_shared_scale)
    oracle_src = os.path.join(work, "v6_moe_oracle.c")
    with open(oracle_src, "w", encoding="utf-8") as f:
        f.write(specialized)
    oracle = os.path.join(work, "v6_moe_oracle")
    run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
         "-ffp-contract=off", oracle_src, "-lm", "-o", oracle])

    checker = os.path.join(work, "v6_moe_runtime_check")
    run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
         "-ffp-contract=off", "-I", os.path.join(REPO, "src"),
         os.path.join(REPO, "tools", "v6_moe_runtime_check.c"),
         os.path.join(REPO, "src", "deepseek_v4_expert_ref.c"),
         os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
         os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
         os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
         "-lm", "-o", checker])
    return oracle, checker, {
        "base_source": "tools/v4_moe_oracle.c",
        "base_source_sha256": sha256_file(gate_f_source),
        "specialization": [
            f"OUT_ROWS: 8 -> {OUT}",
            "shared w2 E8M0 bytes: one scale-grid row -> ceil(OUT_ROWS/128) rows",
        ],
        "specialized_source_sha256": sha256_file(oracle_src),
        "shares_waste_runtime_helpers": False,
        "runtime_checker": "tools/v6_moe_runtime_check.c",
        "runtime_checker_sha256": sha256_file(os.path.join(REPO, "tools", "v6_moe_runtime_check.c")),
    }


def independent_combine(ids, routed_bits, shared_bits):
    y = [chain.f32(0.0)] * OUT
    for slot in sorted(range(TOPK), key=lambda i: ids[i]):
        row = routed_bits[slot]
        for d in range(OUT):
            y[d] = chain.f32(y[d] + chain.bf16_to_f32(row[d]))
    for d in range(OUT):
        y[d] = chain.f32(y[d] + chain.bf16_to_f32(shared_bits[d]))
    return [chain.bf16_bits(v) for v in y]


def output_meta(path, shape):
    bits = read_u16(path)
    return {
        "file": os.path.basename(path),
        "dtype": "BF16",
        "shape": shape,
        "sha256": sha256_file(path),
        "first8_hex": [f"0x{x:04x}" for x in bits[:8]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="header-only pinned checkpoint snapshot")
    ap.add_argument("out_dir", help="fixture output directory")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    state = derive_real_ffn_input()
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    input_path = os.path.join(args.out_dir, "ffn-pre.bf16.bin")
    ids_path = os.path.join(args.out_dir, "routed-ids.u32.bin")
    weights_path = os.path.join(args.out_dir, "routed-weights.f32.bin")
    write_u16(input_path, state["ffn_pre_bits"])
    write_u32(ids_path, state["ids"])
    write_f32(weights_path, state["weights"])

    routed_bits = []
    routed_meta = []
    shared_bits = None
    shared_meta = None

    with tempfile.TemporaryDirectory(prefix="gate-h-moe-") as work:
        oracle, checker, oracle_meta = build_tools(work)

        for slot, (expert, route_weight) in enumerate(zip(state["ids"], state["weights"])):
            d = os.path.join(work, f"expert-{expert}")
            os.makedirs(d)
            base = f"layers.{LAYER}.ffn.experts.{expert}"
            names = {
                "w1": base + ".w1.weight", "s1": base + ".w1.scale",
                "w3": base + ".w3.weight", "s3": base + ".w3.scale",
                "w2": base + ".w2.weight", "s2": base + ".w2.scale",
            }
            paths = {k: os.path.join(d, k + ".bin") for k in names}
            infos = {k: fetch(args.snapshot, names[k], paths[k], args.token) for k in names}
            validate_routed_shapes(expert, infos)
            expected = os.path.join(d, "expected.bf16.bin")
            run([oracle, "routed", input_path, paths["w1"], paths["s1"],
                 paths["w3"], paths["s3"], paths["w2"], paths["s2"],
                 repr(route_weight), expected, "unused"])
            run([checker, "routed", input_path, paths["w1"], paths["s1"],
                 paths["w3"], paths["s3"], paths["w2"], paths["s2"],
                 repr(route_weight), expected])
            bits = read_u16(expected)
            if len(bits) != OUT:
                raise ValueError(f"expert {expert}: oracle emitted {len(bits)} outputs")
            routed_bits.append(bits)
            routed_meta.append({
                "slot": slot,
                "expert": expert,
                "route_weight_f32": route_weight,
                "output_sha256": sha256_file(expected),
                "output_first8_hex": [f"0x{x:04x}" for x in bits[:8]],
                "oracle_vs_waste_exact_bf16": OUT,
                "checkpoint_tensors": {k: tensor_meta(infos[k]) for k in names},
            })
            print(f"routed {slot + 1}/{TOPK} expert {expert}: exact {OUT} BF16")

        d = os.path.join(work, "shared")
        os.makedirs(d)
        base = f"layers.{LAYER}.ffn.shared_experts"
        names = {
            "w1": base + ".w1.weight", "s1": base + ".w1.scale",
            "w3": base + ".w3.weight", "s3": base + ".w3.scale",
            "w2": base + ".w2.weight", "s2": base + ".w2.scale",
        }
        paths = {k: os.path.join(d, k + ".bin") for k in names}
        infos = {k: fetch(args.snapshot, names[k], paths[k], args.token) for k in names}
        validate_shared_shapes(infos)
        gate_diag = os.path.join(d, "gate.f32.bin")
        up_diag = os.path.join(d, "up.f32.bin")
        hidden_diag = os.path.join(d, "hidden.bf16.bin")
        expected = os.path.join(d, "expected.bf16.bin")
        run([oracle, "shared", input_path, paths["w1"], paths["s1"],
             paths["w3"], paths["s3"], paths["w2"], paths["s2"],
             gate_diag, up_diag, hidden_diag, expected, "unused"])
        run([checker, "shared", input_path, paths["w1"], paths["s1"],
             paths["w3"], paths["s3"], paths["w2"], paths["s2"], expected])
        shared_bits = read_u16(expected)
        if len(shared_bits) != OUT:
            raise ValueError(f"shared expert: oracle emitted {len(shared_bits)} outputs")
        shared_meta = {
            "output_sha256": sha256_file(expected),
            "output_first8_hex": [f"0x{x:04x}" for x in shared_bits[:8]],
            "hidden_sha256": sha256_file(hidden_diag),
            "oracle_vs_waste_exact_bf16": OUT,
            "checkpoint_tensors": {k: tensor_meta(infos[k]) for k in names},
        }
        print(f"shared expert: exact {OUT} BF16")

        routed_path = os.path.join(args.out_dir, "routed-out-all6.bf16.bin")
        shared_path = os.path.join(args.out_dir, "shared-out.bf16.bin")
        combined_path = os.path.join(args.out_dir, "moe-branch.bf16.bin")
        write_u16(routed_path, [v for row in routed_bits for v in row])
        write_u16(shared_path, shared_bits)
        combined = independent_combine(state["ids"], routed_bits, shared_bits)
        write_u16(combined_path, combined)
        run([checker, "combine", ids_path, routed_path, shared_path, combined_path])

        # Every selected branch and the shared branch must be visible in the
        # final BF16 vector. This prevents a successful acquisition from
        # quietly freezing a dead branch that happened to route but contributed
        # nothing at the final model boundary.
        mutation_visibility = {}
        for slot, expert in enumerate(state["ids"]):
            muted = [list(row) for row in routed_bits]
            muted[slot] = [0] * OUT
            other = independent_combine(state["ids"], muted, shared_bits)
            diff = sum(a != b for a, b in zip(other, combined))
            if diff == 0:
                raise ValueError(f"dropping routed expert {expert} is BF16-invisible")
            mutation_visibility[f"drop_expert_{expert}"] = diff
        other = independent_combine(state["ids"], routed_bits, [0] * OUT)
        shared_diff = sum(a != b for a, b in zip(other, combined))
        if shared_diff == 0:
            raise ValueError("dropping the shared expert is BF16-invisible")
        mutation_visibility["drop_shared_expert"] = shared_diff

        # Final real layer state, still from independent Python HC equations.
        branch_f32 = [chain.bf16_to_f32(b) for b in combined]
        final_f = chain.oracle_hc_post(
            branch_f32, state["after_attn"], state["ffn_post"], state["ffn_comb"])
        final_bits, _ = chain.cast_bf16(final_f)
        final_path = os.path.join(args.out_dir, "expected-layer3-final.bf16.bin")
        write_u16(final_path, final_bits)

        provenance = {
            "schema_version": 1,
            "gate": "Gate H/V6 full real layer-3 MoE branch and final composition fixture",
            "evidence_state": "REAL-FFN-PRE-BOUND + 7x4096 INDEPENDENT-ORACLE/WASTE-EXACT",
            "model": MODEL,
            "revision": REV,
            "layer": LAYER,
            "input_dependency": {
                "file": "ffn-pre.bf16.bin",
                "sha256": sha256_file(input_path),
                "shape": [INPUT],
                "dtype": "BF16",
                "derived_from": {
                    "hc_fixture": "tests/fixtures/deepseek_v4/v6_hc_composition_real",
                    "hc_provenance_sha256": sha256_file(os.path.join(HCFIX, "provenance.json")),
                    "attention_fixture": "tests/fixtures/deepseek_v4/v6_attention_branch_real",
                    "attention_branch_sha256": sha256_file(state["attention_branch_path"]),
                },
            },
            "router": {
                "ids_topk_order": state["ids"],
                "ids_ascending_accumulation_order": sorted(state["ids"]),
                "weights_f32": state["weights"],
                "topk_boundary_margin": state["margin"],
                "source_contract": "selection by score+bias; weights from unbiased sqrt(softplus(score)), normalized then x1.5",
            },
            "routed_experts": routed_meta,
            "shared_expert": shared_meta,
            "oracle_independence": oracle_meta,
            "acquisition_verification": {
                "routed_exact_bf16_values": TOPK * OUT,
                "shared_exact_bf16_values": OUT,
                "total_expert_exact_bf16_values": (TOPK + 1) * OUT,
                "raw_checkpoint_weights_committed": False,
                "raw_checkpoint_weights_lifetime": "temporary acquisition workspace only",
            },
            "outputs": {
                "routed": output_meta(routed_path, [TOPK, OUT]),
                "shared": output_meta(shared_path, [OUT]),
                "moe_branch": output_meta(combined_path, [OUT]),
                "layer3_final": output_meta(final_path, [chain.HC, INPUT]),
                "ids": {"file": os.path.basename(ids_path), "sha256": sha256_file(ids_path), "shape": [TOPK], "dtype": "U32"},
                "weights": {"file": os.path.basename(weights_path), "sha256": sha256_file(weights_path), "shape": [TOPK], "dtype": "F32"},
            },
            "mutations": mutation_visibility,
            "non_claims": [
                "one layer-3 token/state transition only; no full-model logits",
                "raw expert checkpoint records are verified during acquisition but intentionally not committed",
                "converted WASTE container/cache identity remains a separate systems gate",
            ],
        }
        with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, sort_keys=True)
            f.write("\n")

    print("Gate H real MoE branch fixture built")
    print("experts:", state["ids"])
    print("moe first8:", " ".join(f"0x{x:04x}" for x in combined[:8]))
    print("final first8:", " ".join(f"0x{x:04x}" for x in final_bits[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
