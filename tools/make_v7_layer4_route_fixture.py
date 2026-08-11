#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the first real V7 chained layer-4 boundary fixture.

Input is the exact frozen Gate-H layer-3 final BF16 [4,4096] state. This tool
fetches only resident layer-4 trunk tensors needed to reach the learned router:

    layer3 output == layer4 input
      -> layer4 hc_attn_pre
      -> full 64-head checkpoint attention at position 0
      -> layer4 hc_attn_post
      -> layer4 hc_ffn_pre
      -> layer4 learned router

The complete Gate-H attention producer is reused for q/kv/head and all-eight-
group output-projection arithmetic. Its Python head equations must agree with
the scalar C attention core for all 32,768 BF16 head values before the branch
is frozen. Layer-4 compressor/indexer payload is deliberately not fetched here:
at this one-token position-0 composition seam there is no compressed history
to select. Gate E remains the proof of non-empty ratio-4 CSA/indexer semantics.

This fixture ends at router IDs/weights on purpose. No routed expert payload is
fetched until this cheap checkpoint-backed step has fixed the exact six IDs.
"""

from __future__ import annotations

import argparse
import ctypes
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
import make_v6_attention_branch_fixture as attn  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402
import deepseek_v4_hc_oracle as hc_oracle  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
LAYER = 4
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24
EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
L3FIX = os.path.join(BASE, "v6_moe_branch_real")
SURFACE = os.path.join(REPO, "reference", "deepseek-v4-flash-0731.layer4-v7.json")

ATTN_TENSORS = {
    key: (name.replace("layers.3.", "layers.4."), rows, dtype, shape)
    for key, (name, rows, dtype, shape) in attn.TENSORS.items()
}
TRUNK_TENSORS = {
    "attn_fn": ("layers.4.hc_attn_fn", "F32", [24, 16384], "attn-fn.f32.bin"),
    "attn_scale": ("layers.4.hc_attn_scale", "F32", [3], "attn-scale.f32.bin"),
    "attn_base": ("layers.4.hc_attn_base", "F32", [24], "attn-base.f32.bin"),
    "ffn_fn": ("layers.4.hc_ffn_fn", "F32", [24, 16384], "ffn-fn.f32.bin"),
    "ffn_scale": ("layers.4.hc_ffn_scale", "F32", [3], "ffn-scale.f32.bin"),
    "ffn_base": ("layers.4.hc_ffn_base", "F32", [24], "ffn-base.f32.bin"),
    "gate_weight": ("layers.4.ffn.gate.weight", "BF16", [256, 4096], "gate-weight.bf16.bin"),
    "gate_bias": ("layers.4.ffn.gate.bias", "F32", [256], "gate-bias.f32.bin"),
}


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_u16(path: str, count: int | None = None) -> list[int]:
    data = open(path, "rb").read()
    if len(data) % 2:
        raise ValueError(f"{path}: odd BF16 byte count")
    values = list(struct.unpack(f"<{len(data)//2}H", data))
    if count is not None and len(values) != count:
        raise ValueError(f"{path}: {len(values)} BF16, expected {count}")
    return values


def read_f32(path: str, count: int | None = None) -> list[float]:
    data = open(path, "rb").read()
    if len(data) % 4:
        raise ValueError(f"{path}: non-F32 byte count")
    values = list(struct.unpack(f"<{len(data)//4}f", data))
    if count is not None and len(values) != count:
        raise ValueError(f"{path}: {len(values)} F32, expected {count}")
    return values


def write_u16(path: str, values: list[int]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_u32(path: str, values: list[int]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def write_f32(path: str, values: list[float]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[chain.f32(v) for v in values]))


def meta(info: dict) -> dict:
    return {
        "name": info["tensor"],
        "shard": info["shard"],
        "dtype": info["dtype"],
        "shape": info["shape"],
        "rows": [info["row_start"], info["row_end"]],
        "absolute_range": [info["absolute_start"], info["absolute_end"]],
        "payload_bytes": info["payload_bytes"],
        "payload_sha256": info["payload_sha256"],
    }


def fetch_named(snapshot: str, tensor: str, dtype: str, shape: list[int],
                path: str, token: str | None) -> dict:
    info = slice_fetch.fetch_slice(snapshot, tensor, None, path, token=token)
    if info["revision"] != REV or info["dtype"] != dtype or info["shape"] != shape:
        raise ValueError(
            f"{tensor}: got {info['revision']} {info['dtype']} {info['shape']}; "
            f"expected {REV} {dtype} {shape}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return info


def compiler() -> str:
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError("no C compiler available for V7 layer-4 acquisition")


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)



def build_state_lib(work: str):
    cc = compiler()
    lib = os.path.join(work, "libv7state.so")
    cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           "-ffp-contract=off", "-shared", "-fPIC", "-I", os.path.join(REPO, "src"),
           os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"),
           os.path.join(REPO, "src", "deepseek_v4_router_ref.c"),
           "-lm", "-o", lib]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("V7 state library compile failed:\n" + p.stdout + p.stderr)
    so = ctypes.CDLL(lib)
    pre = so.waste_ds_v4_hc_pre_ref
    pre.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint,
                    ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float)]
    pre.restype = ctypes.c_int
    post = so.waste_ds_v4_hc_post_ref
    post.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                     ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    post.restype = ctypes.c_int
    score = so.waste_ds_v4_router_score_ref
    score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                      ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float),
                      ctypes.POINTER(ctypes.c_float)]
    score.restype = ctypes.c_int
    learned = so.waste_ds_v4_router_learned_ref
    learned.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    learned.restype = ctypes.c_int
    return pre, post, score, learned


def c_pre(pre_fn, x, fn, scale, base):
    y = (ctypes.c_float * DIM)()
    po = (ctypes.c_float * HC)()
    co = (ctypes.c_float * (HC * HC))()
    rc = pre_fn(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base),
                ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6),
                y, None, None, po, co)
    if rc:
        raise RuntimeError("C layer-4 hc_pre rejected the state")
    return list(y), list(po), [list(co[j * HC:(j + 1) * HC]) for j in range(HC)]


def c_post(post_fn, branch, residual, post, comb):
    out = (ctypes.c_float * FLAT)()
    flat_comb = [v for row in comb for v in row]
    rc = post_fn(as_cf(branch), as_cf(residual), DIM, as_cf(post), as_cf(flat_comb), out)
    if rc:
        raise RuntimeError("C layer-4 hc_post rejected the state")
    return list(out)


def c_route(score_fn, learned_fn, x, gate_bits, bias):
    wb = (ctypes.c_uint16 * len(gate_bits))(*gate_bits)
    raw = (ctypes.c_float * EXPERTS)()
    scores = (ctypes.c_float * EXPERTS)()
    if score_fn(as_cf(x), DIM, wb, raw, scores):
        raise RuntimeError("C layer-4 router score rejected ffn_pre")
    ids = (ctypes.c_uint32 * TOPK)()
    weights = (ctypes.c_float * TOPK)()
    selection = (ctypes.c_float * EXPERTS)()
    if learned_fn(scores, as_cf(bias), ctypes.c_float(ROUTE_SCALE),
                  ids, weights, selection):
        raise RuntimeError("C layer-4 learned router rejected scores")
    return list(ids), list(weights), list(scores), list(selection)



def observe_correction_bias(scores, bias, selection, biased_ids):
    """Validate the mandatory bias operation and record its input-specific effect.

    A correction bias is semantically load-bearing because learned routing selects
    from ``score + bias``. It does *not* follow that every input must select a
    different top-k set than unbiased scores would. Treat that as an observation,
    never as a gate.
    """
    if not (len(scores) == len(bias) == len(selection) == EXPERTS):
        raise ValueError("router bias observation geometry mismatch")
    expected = [chain.f32(scores[e] + bias[e]) for e in range(EXPERTS)]
    for e, (got, want) in enumerate(zip(selection, expected)):
        if chain.f32_bits(got) != chain.f32_bits(want):
            raise ValueError(
                f"router correction-bias selection score mismatch at expert {e}: "
                f"{chain.f32_bits(got):08x}!={chain.f32_bits(want):08x}")
    unbiased_ids = sorted(range(EXPERTS), key=lambda e: (-scores[e], e))[:TOPK]
    nonzero = sum((chain.f32_bits(v) & 0x7fffffff) != 0 for v in bias)
    changed_scores = sum(
        chain.f32_bits(selection[e]) != chain.f32_bits(scores[e])
        for e in range(EXPERTS))
    return {
        "biased_ids": list(biased_ids),
        "unbiased_ids": unbiased_ids,
        "changes_topk_on_this_input": list(biased_ids) != unbiased_ids,
        "nonzero_f32_count": nonzero,
        "changed_selection_score_f32_count": changed_scores,
        "selection_scores_equal_score_plus_bias_f32": True,
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    if not os.path.isfile(SURFACE):
        raise ValueError("frozen layer-4 V7 header surface is required first")
    surface = json.load(open(SURFACE, encoding="utf-8"))
    if surface.get("revision") != REV or surface.get("layer") != LAYER or \
       surface.get("structural_class") != "ratio4-learned":
        raise ValueError("layer-4 header surface is not canonical ratio4-learned 0731")

    l3_prov_path = os.path.join(L3FIX, "provenance.json")
    if not os.path.isfile(l3_prov_path):
        raise ValueError("Gate-H complete layer-3 fixture is required")
    l3 = json.load(open(l3_prov_path, encoding="utf-8"))
    final_meta = l3.get("outputs", {}).get("layer3_final", {})
    l3_final = os.path.join(L3FIX, final_meta.get("file", ""))
    if not os.path.isfile(l3_final) or sha(l3_final) != final_meta.get("sha256"):
        raise ValueError("Gate-H final layer-3 state missing or SHA mismatch")
    if final_meta.get("shape") != [HC, DIM] or final_meta.get("dtype") != "BF16":
        raise ValueError("Gate-H final state geometry drifted")
    residual_bits = read_u16(l3_final, FLAT)
    residual = [chain.bf16_to_f32(v) for v in residual_bits]

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    input_path = os.path.join(args.out_dir, "input-layer3-final.bf16.bin")
    shutil.copyfile(l3_final, input_path)

    trunk_paths = {}
    trunk_meta = {}
    attention_meta = {}
    with tempfile.TemporaryDirectory(prefix="v7-layer4-") as work:
        # HC/router tensors are compact enough to freeze for permanent offline
        # chaining; large attention weights remain transient acquisition bytes.
        for key, (tensor, dtype, shape, filename) in TRUNK_TENSORS.items():
            dest = os.path.join(args.out_dir, filename)
            info = fetch_named(args.snapshot, tensor, dtype, shape, dest, args.token)
            trunk_paths[key] = dest
            trunk_meta[key] = meta(info)

        attention_paths = {}
        for key, (tensor, rows, dtype, shape) in ATTN_TENSORS.items():
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
            attention_meta[key] = meta(info)

        attn_fn = read_f32(trunk_paths["attn_fn"], MIX * FLAT)
        attn_scale = read_f32(trunk_paths["attn_scale"], 3)
        attn_base = read_f32(trunk_paths["attn_base"], MIX)
        ffn_fn = read_f32(trunk_paths["ffn_fn"], MIX * FLAT)
        ffn_scale = read_f32(trunk_paths["ffn_scale"], 3)
        ffn_base = read_f32(trunk_paths["ffn_base"], MIX)
        gate_bits = read_u16(trunk_paths["gate_weight"], EXPERTS * DIM)
        gate_bias = read_f32(trunk_paths["gate_bias"], EXPERTS)

        pre_fn, post_fn, score_fn, learned_fn = build_state_lib(work)
        attn_pre_f, attn_post, attn_comb = c_pre(
            pre_fn, residual, attn_fn, attn_scale, attn_base)
        attn_pre_bits, attn_pre = chain.cast_bf16(attn_pre_f)

        # Independent HC equations must land on the same BF16 branch input.
        o_pre, o_post, o_comb, _ = hc_oracle.params(
            residual, attn_fn, attn_scale, attn_base, dim=DIM)
        o_pre = [chain.f32(v) for v in o_pre]
        o_post = [chain.f32(v) for v in o_post]
        o_comb = [[chain.f32(v) for v in row] for row in o_comb]
        o_attn_pre_bits, _ = chain.cast_bf16(
            hc_oracle.pre_y(residual, o_pre, dim=DIM))
        if o_attn_pre_bits != attn_pre_bits:
            i = next(i for i, (a, b) in enumerate(zip(o_attn_pre_bits, attn_pre_bits)) if a != b)
            raise ValueError(
                f"layer4 HC-attn-pre Python/C BF16 mismatch at {i}: "
                f"{o_attn_pre_bits[i]:04x}!={attn_pre_bits[i]:04x}")
        attn_c_bits = [chain.f32_bits(v) for v in attn_post] + [
            chain.f32_bits(v) for row in attn_comb for v in row]
        attn_o_bits = [chain.f32_bits(v) for v in o_post] + [
            chain.f32_bits(v) for row in o_comb for v in row]
        if attn_c_bits != attn_o_bits:
            i = next(i for i, (a, b) in enumerate(zip(attn_c_bits, attn_o_bits)) if a != b)
            raise ValueError(
                f"layer4 HC-attn post/comb F32 mismatch at {i}: "
                f"{attn_c_bits[i]:08x}!={attn_o_bits[i]:08x}")

        q_bits, kv_bits, head_bits, _ = attn.one_token_heads(
            attn_pre_bits, attention_paths)
        latent_bits, branch_bits = attn.validate_and_project(
            REPO, attn_pre_bits, attention_paths, q_bits, kv_bits, head_bits)
        branch = [chain.bf16_to_f32(v) for v in branch_bits]

        after_attn_f = c_post(post_fn, branch, residual, attn_post, attn_comb)
        after_attn_bits, after_attn = chain.cast_bf16(after_attn_f)
        ffn_pre_f, ffn_post, ffn_comb = c_pre(
            pre_fn, after_attn, ffn_fn, ffn_scale, ffn_base)
        ffn_pre_bits, ffn_pre = chain.cast_bf16(ffn_pre_f)

        fo_pre, fo_post, fo_comb, _ = hc_oracle.params(
            after_attn, ffn_fn, ffn_scale, ffn_base, dim=DIM)
        fo_pre = [chain.f32(v) for v in fo_pre]
        fo_post = [chain.f32(v) for v in fo_post]
        fo_comb = [[chain.f32(v) for v in row] for row in fo_comb]
        if chain.cast_bf16(hc_oracle.pre_y(after_attn, fo_pre, dim=DIM))[0] != ffn_pre_bits:
            raise ValueError("layer4 HC-FFN-pre disagrees with independent oracle at BF16")
        ffn_c_bits = [chain.f32_bits(v) for v in ffn_post] + [
            chain.f32_bits(v) for row in ffn_comb for v in row]
        ffn_o_bits = [chain.f32_bits(v) for v in fo_post] + [
            chain.f32_bits(v) for row in fo_comb for v in row]
        if ffn_c_bits != ffn_o_bits:
            i = next(i for i, (a, b) in enumerate(zip(ffn_c_bits, ffn_o_bits)) if a != b)
            raise ValueError(
                f"layer4 HC-FFN post/comb F32 mismatch at {i}: "
                f"{ffn_c_bits[i]:08x}!={ffn_o_bits[i]:08x}")

        ids, weights, c_scores, c_selection = c_route(
            score_fn, learned_fn, ffn_pre, gate_bits, gate_bias)
        py_ids, py_weights, py_scores, py_selection, margin = chain.oracle_router(
            ffn_pre, gate_bits, gate_bias)
        if ids != py_ids:
            raise ValueError(f"layer4 router IDs C={ids} Python={py_ids}")
        weight_delta = max(abs(a - b) for a, b in zip(weights, py_weights))
        if weight_delta > 1e-5:
            raise ValueError(f"layer4 route weight delta {weight_delta} > 1e-5")
        if not margin > 1e-4:
            raise ValueError(f"layer4 top-k boundary {margin} is not stable enough for acquisition")

        bias_observation = observe_correction_bias(
            c_scores, gate_bias, c_selection, ids)
        # Independent Python selection must implement the same semantic
        # operation; numerical route weights remain a separate tolerance-bound
        # transcendental cross-check, while exact C weights are frozen below.
        py_expected_selection = [chain.f32(py_scores[e] + gate_bias[e])
                                 for e in range(EXPERTS)]
        if any(chain.f32_bits(a) != chain.f32_bits(b)
               for a, b in zip(py_selection, py_expected_selection)):
            raise ValueError("independent Python router did not apply correction bias as score+bias")

    outputs = {
        "input": ("input-layer3-final.bf16.bin", residual_bits, [HC, DIM]),
        "attn_pre": ("attn-pre.bf16.bin", attn_pre_bits, [DIM]),
        "q": ("q-all64.bf16.bin", q_bits, [attn.N_HEADS, attn.HEAD]),
        "kv": ("kv-local.bf16.bin", kv_bits, [attn.HEAD]),
        "heads": ("attention-heads-all64.bf16.bin", head_bits, [attn.N_HEADS, attn.HEAD]),
        "group_latent": ("group-latent-all8.bf16.bin", latent_bits, [attn.GROUPS, attn.GROUP_RANK]),
        "attention_branch": ("attention-branch.bf16.bin", branch_bits, [DIM]),
        "after_attn": ("after-attn.bf16.bin", after_attn_bits, [HC, DIM]),
        "ffn_pre": ("ffn-pre.bf16.bin", ffn_pre_bits, [DIM]),
    }
    out_meta = {}
    for key, (filename, bits, shape) in outputs.items():
        path = os.path.join(args.out_dir, filename)
        if key != "input":
            write_u16(path, bits)
        out_meta[key] = {
            "file": filename,
            "dtype": "BF16",
            "shape": shape,
            "sha256": sha(path),
            "first8_hex": [f"0x{x:04x}" for x in bits[:8]],
        }

    ids_path = os.path.join(args.out_dir, "router-ids.u32.bin")
    weights_path = os.path.join(args.out_dir, "router-weights.f32.bin")
    write_u32(ids_path, ids)
    write_f32(weights_path, weights)

    prov = {
        "schema_version": 1,
        "gate": "V7 layer-3 -> layer-4 chained attention/route discovery",
        "evidence_state": "CHECKPOINT-BOUND-LAYER4-THROUGH-ROUTER",
        "model": MODEL,
        "revision": REV,
        "layer": LAYER,
        "structural_class": "ratio4-learned",
        "position": 0,
        "input_dependency": {
            "producer": "Gate H/V6 complete real layer-3 final state",
            "source_file": "tests/fixtures/deepseek_v4/v6_moe_branch_real/" + final_meta["file"],
            "source_sha256": final_meta["sha256"],
            "fixture_file": "input-layer3-final.bf16.bin",
            "fixture_sha256": sha(input_path),
            "byte_identical": sha(input_path) == final_meta["sha256"],
        },
        "checkpoint_surface_dependency": {
            "file": "reference/deepseek-v4-flash-0731.layer4-v7.json",
            "sha256": sha(SURFACE),
            "compress_ratio": 4,
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
            "ids_file": "router-ids.u32.bin",
            "ids_sha256": sha(ids_path),
            "weights_file": "router-weights.f32.bin",
            "weights_sha256": sha(weights_path),
        },
        "hc_crosschecks": {
            "attn_post_comb_exact_f32": True,
            "ffn_post_comb_exact_f32": True,
            "pre_boundaries": "independent Python and C exact at BF16",
        },
        "attention_crosscheck": {
            "python_vs_c_exact_head_bf16": attn.N_HEADS * attn.HEAD,
            "full_output_projection": "all-eight-group scalar primitive previously mutation-pinned in Gate H",
        },
        "next_acquisition": {
            "layer": LAYER,
            "routed_experts": sorted(ids),
            "shared_expert": True,
            "reason": "only these experts are selected by the exact chained layer-4 FFN-pre state",
        },
        "non_claims": [
            "position 0 has no compressed history; this fixture does not re-prove non-empty ratio-4 compressor/indexer selection",
            "Gate E owns non-empty ratio-4 CSA/indexer arithmetic and sparse-attention semantics",
            "no layer-4 expert payload is fetched or executed here",
            "V7 is not complete until the selected experts, layer-4 final state, and a chained two-layer trace are frozen",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
        f.write("\n")

    print("PASS V7 real layer-4 route discovery")
    print("layer3 output == layer4 input:", prov["input_dependency"]["byte_identical"])
    print("attn_pre first8:", " ".join(out_meta["attn_pre"]["first8_hex"]))
    print("attention branch first8:", " ".join(out_meta["attention_branch"]["first8_hex"]))
    print("ffn_pre first8:", " ".join(out_meta["ffn_pre"]["first8_hex"]))
    print("selected experts:", ids)
    print("route weights:", " ".join(f"{w:.9f}" for w in weights))
    print("top-k boundary margin:", f"{margin:.9g}")
    print("fetch next:", ",".join(str(i) for i in sorted(ids)), "+ shared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
