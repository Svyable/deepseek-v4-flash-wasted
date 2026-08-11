#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Generic DeepSeek-V4 one-position layer continuation through learned routing.

This tool generalizes only orchestration. Arithmetic stays in the already
mutation-pinned V7 helpers: HyperConnection, one-token attention/output
projection, and learned routing. The layer namespace, prior-state provenance,
and immutable header contract are inputs.

At position 0 there is no completed compressed history for ratio4/ratio128
layers. This route fixture therefore fetches only the resident/base attention
surface needed for the actual one-token branch; Gate E remains the proof of
non-empty compressor/indexer history semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import fetch_hf_tensor_slice as slice_fetch  # noqa: E402
import make_v6_attention_branch_fixture as attn  # noqa: E402
import make_v7_layer4_route_fixture as v7  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402
import deepseek_v4_hc_oracle as hc_oracle  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24
EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5


def load_prior(provenance_path: str, output_key: str):
    prov = json.load(open(provenance_path, encoding="utf-8"))
    if prov.get("revision") != REV:
        raise ValueError("prior-state revision drifted")
    meta = (prov.get("outputs") or {}).get(output_key) or {}
    root = os.path.dirname(provenance_path)
    path = os.path.join(root, meta.get("file", ""))
    if meta.get("dtype") != "BF16" or meta.get("shape") != [HC, DIM]:
        raise ValueError(f"prior {output_key}: expected BF16 [4,4096]")
    if not os.path.isfile(path) or v7.sha(path) != meta.get("sha256"):
        raise ValueError(f"prior {output_key}: missing or SHA mismatch")
    return prov, meta, path


def load_surface(path: str, layer: int):
    surface = json.load(open(path, encoding="utf-8"))
    if surface.get("revision") != REV or surface.get("layer") != layer:
        raise ValueError("layer surface identity drifted")
    cls = surface.get("structural_class")
    if cls not in {"ratio0-learned", "ratio4-learned", "ratio128-learned"}:
        raise ValueError(f"generic learned route does not support {cls!r}")
    cfg = surface.get("config_contract") or {}
    if cfg.get("hidden_size") != DIM or cfg.get("n_routed_experts") != EXPERTS or \
       cfg.get("num_experts_per_tok") != TOPK or cfg.get("moe_intermediate_size") != 2048:
        raise ValueError("layer surface core config drifted")
    if (surface.get("checks") or {}).get("routing_contract") != "learned":
        raise ValueError("generic route requires learned routing")
    return surface


def attention_tensors(layer: int):
    prefix = f"layers.{layer}."
    return {
        key: (prefix + name.split(".", 2)[2], rows, dtype, shape)
        for key, (name, rows, dtype, shape) in attn.TENSORS.items()
    }


def trunk_tensors(layer: int):
    p = f"layers.{layer}"
    return {
        "attn_fn": (f"{p}.hc_attn_fn", "F32", [24, 16384], "attn-fn.f32.bin"),
        "attn_scale": (f"{p}.hc_attn_scale", "F32", [3], "attn-scale.f32.bin"),
        "attn_base": (f"{p}.hc_attn_base", "F32", [24], "attn-base.f32.bin"),
        "ffn_fn": (f"{p}.hc_ffn_fn", "F32", [24, 16384], "ffn-fn.f32.bin"),
        "ffn_scale": (f"{p}.hc_ffn_scale", "F32", [3], "ffn-scale.f32.bin"),
        "ffn_base": (f"{p}.hc_ffn_base", "F32", [24], "ffn-base.f32.bin"),
        "gate_weight": (f"{p}.ffn.gate.weight", "BF16", [256, 4096], "gate-weight.bf16.bin"),
        "gate_bias": (f"{p}.ffn.gate.bias", "F32", [256], "gate-bias.f32.bin"),
    }


def fetch_named(snapshot, tensor, dtype, shape, path, token):
    info = slice_fetch.fetch_slice(snapshot, tensor, None, path, token=token)
    if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
        raise ValueError(
            f"{tensor}: got {info['revision']} {info['dtype']} {info['shape']}; "
            f"expected {REV} {dtype} {shape}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("layer", type=int)
    ap.add_argument("surface")
    ap.add_argument("prior_provenance")
    ap.add_argument("prior_output_key")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    if args.layer < 3 or args.layer >= 43:
        raise ValueError("generic learned-route layer must be in main learned stack 3..42")
    surface = load_surface(args.surface, args.layer)
    prior, prior_meta, prior_path = load_prior(args.prior_provenance, args.prior_output_key)
    input_sha = prior_meta["sha256"]
    residual_bits = v7.read_u16(prior_path, FLAT)
    residual = [chain.bf16_to_f32(x) for x in residual_bits]

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    input_path = os.path.join(args.out_dir, "input.bf16.bin")
    shutil.copyfile(prior_path, input_path)

    trunk_paths = {}
    trunk_meta = {}
    attention_meta = {}
    with tempfile.TemporaryDirectory(prefix=f"v8-layer{args.layer}-route-") as work:
        for key, (tensor, dtype, shape, filename) in trunk_tensors(args.layer).items():
            dest = os.path.join(args.out_dir, filename)
            info = fetch_named(args.snapshot, tensor, dtype, shape, dest, args.token)
            trunk_paths[key] = dest
            trunk_meta[key] = v7.meta(info)

        attention_paths = {}
        for key, (tensor, rows, dtype, shape) in attention_tensors(args.layer).items():
            dest = os.path.join(work, "attn-" + key + ".bin")
            info = slice_fetch.fetch_slice(args.snapshot, tensor, rows, dest, token=args.token)
            if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
                raise ValueError(
                    f"{tensor}: got {info['revision']} {info['dtype']} {info['shape']}; "
                    f"expected {REV} {dtype} {shape}")
            side = dest + ".json"
            if os.path.exists(side):
                os.remove(side)
            attention_paths[key] = dest
            attention_meta[key] = v7.meta(info)

        attn_fn = v7.read_f32(trunk_paths["attn_fn"], MIX * FLAT)
        attn_scale = v7.read_f32(trunk_paths["attn_scale"], 3)
        attn_base = v7.read_f32(trunk_paths["attn_base"], MIX)
        ffn_fn = v7.read_f32(trunk_paths["ffn_fn"], MIX * FLAT)
        ffn_scale = v7.read_f32(trunk_paths["ffn_scale"], 3)
        ffn_base = v7.read_f32(trunk_paths["ffn_base"], MIX)
        gate_bits = v7.read_u16(trunk_paths["gate_weight"], EXPERTS * DIM)
        gate_bias = v7.read_f32(trunk_paths["gate_bias"], EXPERTS)

        pre_fn, post_fn, score_fn, learned_fn = v7.build_state_lib(work)
        attn_pre_f, attn_post, attn_comb = v7.c_pre(
            pre_fn, residual, attn_fn, attn_scale, attn_base)
        attn_pre_bits, attn_pre = chain.cast_bf16(attn_pre_f)

        o_pre, o_post, o_comb, _ = hc_oracle.params(
            residual, attn_fn, attn_scale, attn_base, dim=DIM)
        o_pre = [chain.f32(x) for x in o_pre]
        o_post = [chain.f32(x) for x in o_post]
        o_comb = [[chain.f32(x) for x in row] for row in o_comb]
        if chain.cast_bf16(hc_oracle.pre_y(residual, o_pre, dim=DIM))[0] != attn_pre_bits:
            raise ValueError(f"layer {args.layer}: independent HC-attn-pre disagrees at BF16")
        c_bits = [chain.f32_bits(x) for x in attn_post] + [
            chain.f32_bits(x) for row in attn_comb for x in row]
        o_bits = [chain.f32_bits(x) for x in o_post] + [
            chain.f32_bits(x) for row in o_comb for x in row]
        if c_bits != o_bits:
            raise ValueError(f"layer {args.layer}: independent HC-attn post/comb F32 mismatch")

        q_bits, kv_bits, head_bits, _ = attn.one_token_heads(attn_pre_bits, attention_paths)
        latent_bits, branch_bits = attn.validate_and_project(
            REPO, attn_pre_bits, attention_paths, q_bits, kv_bits, head_bits)
        branch = [chain.bf16_to_f32(x) for x in branch_bits]

        after_f = v7.c_post(post_fn, branch, residual, attn_post, attn_comb)
        after_bits, after = chain.cast_bf16(after_f)
        ffn_pre_f, ffn_post, ffn_comb = v7.c_pre(
            pre_fn, after, ffn_fn, ffn_scale, ffn_base)
        ffn_pre_bits, ffn_pre = chain.cast_bf16(ffn_pre_f)

        fo_pre, fo_post, fo_comb, _ = hc_oracle.params(
            after, ffn_fn, ffn_scale, ffn_base, dim=DIM)
        fo_pre = [chain.f32(x) for x in fo_pre]
        fo_post = [chain.f32(x) for x in fo_post]
        fo_comb = [[chain.f32(x) for x in row] for row in fo_comb]
        if chain.cast_bf16(hc_oracle.pre_y(after, fo_pre, dim=DIM))[0] != ffn_pre_bits:
            raise ValueError(f"layer {args.layer}: independent HC-FFN-pre disagrees at BF16")
        c_bits = [chain.f32_bits(x) for x in ffn_post] + [
            chain.f32_bits(x) for row in ffn_comb for x in row]
        o_bits = [chain.f32_bits(x) for x in fo_post] + [
            chain.f32_bits(x) for row in fo_comb for x in row]
        if c_bits != o_bits:
            raise ValueError(f"layer {args.layer}: independent HC-FFN post/comb F32 mismatch")

        ids, weights, c_scores, c_selection = v7.c_route(
            score_fn, learned_fn, ffn_pre, gate_bits, gate_bias)
        py_ids, py_weights, py_scores, py_selection, margin = chain.oracle_router(
            ffn_pre, gate_bits, gate_bias)
        if ids != py_ids:
            raise ValueError(f"layer {args.layer}: router IDs C={ids} Python={py_ids}")
        weight_delta = max(abs(a - b) for a, b in zip(weights, py_weights))
        if weight_delta > 1e-5:
            raise ValueError(f"layer {args.layer}: route weight delta {weight_delta} > 1e-5")
        if not margin > 1e-4:
            raise ValueError(f"layer {args.layer}: top-k boundary {margin} is unstable")
        bias_observation = v7.observe_correction_bias(c_scores, gate_bias, c_selection, ids)
        py_expected = [chain.f32(py_scores[e] + gate_bias[e]) for e in range(EXPERTS)]
        if any(chain.f32_bits(a) != chain.f32_bits(b)
               for a, b in zip(py_selection, py_expected)):
            raise ValueError("independent Python router did not apply correction bias as score+bias")

    outputs = {
        "input": ("input.bf16.bin", residual_bits, [HC, DIM]),
        "attn_pre": ("attn-pre.bf16.bin", attn_pre_bits, [DIM]),
        "q": ("q-all64.bf16.bin", q_bits, [attn.N_HEADS, attn.HEAD]),
        "kv": ("kv-local.bf16.bin", kv_bits, [attn.HEAD]),
        "heads": ("attention-heads-all64.bf16.bin", head_bits, [attn.N_HEADS, attn.HEAD]),
        "group_latent": ("group-latent-all8.bf16.bin", latent_bits, [attn.GROUPS, attn.GROUP_RANK]),
        "attention_branch": ("attention-branch.bf16.bin", branch_bits, [DIM]),
        "after_attn": ("after-attn.bf16.bin", after_bits, [HC, DIM]),
        "ffn_pre": ("ffn-pre.bf16.bin", ffn_pre_bits, [DIM]),
    }
    out_meta = {}
    for key, (filename, bits, shape) in outputs.items():
        path = os.path.join(args.out_dir, filename)
        if key != "input":
            v7.write_u16(path, bits)
        out_meta[key] = {
            "file": filename, "dtype": "BF16", "shape": shape,
            "sha256": v7.sha(path),
            "first8_hex": [f"0x{x:04x}" for x in bits[:8]],
        }

    ids_path = os.path.join(args.out_dir, "router-ids.u32.bin")
    weights_path = os.path.join(args.out_dir, "router-weights.f32.bin")
    v7.write_u32(ids_path, ids)
    v7.write_f32(weights_path, weights)

    prov = {
        "schema_version": 1,
        "gate": "Gate I/V8 generic one-position layer through router",
        "evidence_state": "CHECKPOINT-BOUND-GENERIC-LAYER-THROUGH-ROUTER",
        "model": MODEL,
        "revision": REV,
        "layer": args.layer,
        "structural_class": surface["structural_class"],
        "position": 0,
        "input_dependency": {
            "provenance_file": os.path.relpath(args.prior_provenance, REPO),
            "output_key": args.prior_output_key,
            "source_file": os.path.relpath(prior_path, REPO),
            "source_sha256": input_sha,
            "fixture_file": "input.bf16.bin",
            "fixture_sha256": v7.sha(input_path),
            "byte_identical": v7.sha(input_path) == input_sha,
        },
        "checkpoint_surface_dependency": {
            "file": os.path.relpath(args.surface, REPO),
            "sha256": v7.sha(args.surface),
            "structural_class": surface["structural_class"],
            "compress_ratio": surface["config_contract"]["compress_ratio"],
        },
        "checkpoint_tensors": {
            "hyperconnection_and_router": trunk_meta,
            "attention": attention_meta,
        },
        "outputs": out_meta,
        "router": {
            "ids_topk_order": ids,
            "ids_ascending_expert_order": sorted(ids),
            "runtime_weights_f32": weights,
            "independent_python_weights": py_weights,
            "runtime_vs_python_max_abs": weight_delta,
            "topk_boundary_margin": margin,
            "selection_without_correction_bias": bias_observation["unbiased_ids"],
            "correction_bias": bias_observation,
            "ids_runtime_vs_python_exact": True,
            "ids_file": "router-ids.u32.bin", "ids_sha256": v7.sha(ids_path),
            "weights_file": "router-weights.f32.bin", "weights_sha256": v7.sha(weights_path),
        },
        "hc_crosschecks": {
            "attn_post_comb_exact_f32": True,
            "ffn_post_comb_exact_f32": True,
            "pre_boundaries": "independent Python and C exact at BF16",
        },
        "attention_crosscheck": {
            "python_vs_c_exact_head_bf16": attn.N_HEADS * attn.HEAD,
            "full_output_projection": "all-eight-group scalar primitive mutation-pinned in Gate H",
        },
        "next_acquisition": {
            "layer": args.layer,
            "routed_experts": sorted(ids),
            "shared_expert": True,
        },
        "non_claims": [
            "position 0 has no completed compressed history; this fixture does not re-prove non-empty ratio4/ratio128 compressor history semantics",
            "Gate E owns non-empty compressed-history attention semantics",
            "no selected expert payload is fetched or executed by this route step",
            "a route fixture alone does not prove the complete layer output",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"PASS V8 generic layer {args.layer} route: {surface['structural_class']}")
    print("input sha256:", input_sha)
    print("selected experts:", ids)
    print("route weights:", " ".join(f"{x:.9f}" for x in weights))
    print("top-k boundary margin:", margin)
    print("next fetch:", ",".join(str(x) for x in sorted(ids)), "+ shared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
