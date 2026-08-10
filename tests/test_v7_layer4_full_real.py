#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the complete real layer-4 V7 transition without network access."""

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import make_v6_moe_branch_fixture as gateh  # noqa: E402
import make_v7_layer4_moe_fixture as full  # noqa: E402
import make_v7_layer4_route_fixture as route  # noqa: E402
import v7_hc_oracle as hc_oracle  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v7_layer4_full_real")


def fail(msg):
    print("FAIL:", msg)
    return 1


def check_output(prov, key, count, shape):
    meta = (prov.get("outputs") or {}).get(key) or {}
    path = os.path.join(FIX, meta.get("file", ""))
    if meta.get("dtype") != "BF16" or meta.get("shape") != shape:
        raise ValueError(f"{key}: metadata geometry drifted")
    if not os.path.isfile(path) or full.sha256_file(path) != meta.get("sha256"):
        raise ValueError(f"{key}: file missing or SHA mismatch")
    bits = gateh.read_u16(path)
    if len(bits) != count:
        raise ValueError(f"{key}: {len(bits)} BF16 values, expected {count}")
    return path, bits


def main():
    prov_path = os.path.join(FIX, "provenance.json")
    if not os.path.isfile(prov_path):
        print("SKIP: frozen complete layer-4 V7 fixture not present")
        return 77

    try:
        prov = json.load(open(prov_path, encoding="utf-8"))
        if prov.get("model") != full.MODEL or prov.get("revision") != full.REV or \
           prov.get("layer") != full.LAYER or prov.get("structural_class") != "ratio4-learned":
            raise ValueError("fixture identity drifted")
        if prov.get("evidence_state") != "CHECKPOINT-BOUND-LAYER4-COMPLETE-SCALAR":
            raise ValueError("fixture evidence state drifted")

        route_state = full.checked_route_state()
        dep = prov.get("route_dependency") or {}
        if dep.get("provenance_sha256") != full.sha256_file(route_state["provenance_path"]):
            raise ValueError("full layer-4 fixture is not bound to current frozen route provenance")
        if dep.get("ffn_pre_sha256") != full.sha256_file(route_state["ffn_path"]) or \
           dep.get("after_attn_sha256") != full.sha256_file(route_state["after_path"]):
            raise ValueError("full layer-4 fixture route-state dependency drifted")
        if dep.get("ids_topk_order") != route_state["ids"] or \
           dep.get("ids_ascending_expert_order") != sorted(route_state["ids"]):
            raise ValueError("full layer-4 fixture expert IDs drifted from route fixture")
        if [chain.f32_bits(v) for v in dep.get("weights_f32", [])] != \
           [chain.f32_bits(v) for v in route_state["weights"]]:
            raise ValueError("full layer-4 fixture route weights drifted")

        ffn_path, ffn_bits = check_output(prov, "ffn_pre", full.INPUT, [full.INPUT])
        routed_path, routed_flat = check_output(
            prov, "routed", full.TOPK * full.OUT, [full.TOPK, full.OUT])
        shared_path, shared_bits = check_output(prov, "shared", full.OUT, [full.OUT])
        combined_path, combined = check_output(prov, "moe_branch", full.OUT, [full.OUT])
        final_path, final_bits = check_output(prov, "layer4_final", full.FLAT, [full.HC, full.INPUT])

        if ffn_bits != route_state["ffn_bits"]:
            raise ValueError("committed full layer-4 FFN input is not byte-identical to frozen route")

        ids_meta = (prov.get("outputs") or {}).get("ids") or {}
        weights_meta = (prov.get("outputs") or {}).get("weights") or {}
        ids_path = os.path.join(FIX, ids_meta.get("file", ""))
        weights_path = os.path.join(FIX, weights_meta.get("file", ""))
        if not os.path.isfile(ids_path) or full.sha256_file(ids_path) != ids_meta.get("sha256"):
            raise ValueError("committed layer-4 IDs missing or SHA mismatch")
        if not os.path.isfile(weights_path) or full.sha256_file(weights_path) != weights_meta.get("sha256"):
            raise ValueError("committed layer-4 weights missing or SHA mismatch")
        ids = full.read_u32(ids_path, full.TOPK)
        weights = full.read_f32(weights_path, full.TOPK)
        if ids != route_state["ids"] or \
           [chain.f32_bits(v) for v in weights] != [chain.f32_bits(v) for v in route_state["weights"]]:
            raise ValueError("committed layer-4 router bytes drifted from route fixture")

        routed = [routed_flat[i * full.OUT:(i + 1) * full.OUT]
                  for i in range(full.TOPK)]
        want_combined = gateh.independent_combine(ids, routed, shared_bits)
        if want_combined != combined:
            i = next(i for i, (a, b) in enumerate(zip(want_combined, combined)) if a != b)
            raise ValueError(
                f"offline layer-4 MoE combine mismatch at {i}: "
                f"{want_combined[i]:04x}!={combined[i]:04x}")

        acq = prov.get("acquisition_verification") or {}
        if acq.get("routed_exact_bf16_values") != full.TOPK * full.OUT or \
           acq.get("shared_exact_bf16_values") != full.OUT or \
           acq.get("total_expert_exact_bf16_values") != (full.TOPK + 1) * full.OUT or \
           acq.get("raw_checkpoint_weights_committed") is not False:
            raise ValueError("expert acquisition evidence counts/provenance drifted")
        if acq.get("ffn_hc_post_comb_exact_f32") is not True or \
           acq.get("final_hc_exact_f32_values") != full.FLAT:
            raise ValueError("final HC exact-F32 evidence drifted")

        mutations = prov.get("mutations") or {}
        for slot, expert in enumerate(ids):
            muted = [list(row) for row in routed]
            muted[slot] = [0] * full.OUT
            other = gateh.independent_combine(ids, muted, shared_bits)
            diff = sum(a != b for a, b in zip(other, combined))
            if diff <= 0 or mutations.get(f"drop_expert_{expert}") != diff:
                raise ValueError(f"drop-expert mutation visibility drifted for {expert}: {diff}")
        other = gateh.independent_combine(ids, routed, [0] * full.OUT)
        shared_diff = sum(a != b for a, b in zip(other, combined))
        if shared_diff <= 0 or mutations.get("drop_shared_expert") != shared_diff:
            raise ValueError(f"drop-shared mutation visibility drifted: {shared_diff}")

        after = [chain.bf16_to_f32(b) for b in route_state["after_bits"]]
        branch = [chain.bf16_to_f32(b) for b in combined]
        with tempfile.TemporaryDirectory(prefix="v7-layer4-full-replay-") as work:
            pre_fn, post_fn, _, _ = route.build_state_lib(work)
            c_pre_f, c_post, c_comb = route.c_pre(
                pre_fn, after, route_state["params"]["ffn_fn"],
                route_state["params"]["ffn_scale"], route_state["params"]["ffn_base"])
            c_pre_bits, _ = chain.cast_bf16(c_pre_f)
            if c_pre_bits != ffn_bits:
                i = next(i for i, (a, b) in enumerate(zip(c_pre_bits, ffn_bits)) if a != b)
                raise ValueError(f"C FFN-pre replay mismatch at {i}: {c_pre_bits[i]:04x}!={ffn_bits[i]:04x}")

            o_pre, o_post, o_comb, _ = hc_oracle.params(
                after, route_state["params"]["ffn_fn"], route_state["params"]["ffn_scale"],
                route_state["params"]["ffn_base"], dim=full.INPUT)
            o_pre_bits, _ = chain.cast_bf16(hc_oracle.pre_y(after, o_pre, dim=full.INPUT))
            if o_pre_bits != ffn_bits:
                i = next(i for i, (a, b) in enumerate(zip(o_pre_bits, ffn_bits)) if a != b)
                raise ValueError(f"oracle FFN-pre replay mismatch at {i}: {o_pre_bits[i]:04x}!={ffn_bits[i]:04x}")

            c_param_bits = [chain.f32_bits(v) for v in c_post] + [
                chain.f32_bits(v) for row in c_comb for v in row]
            o_param_bits = [chain.f32_bits(v) for v in o_post] + [
                chain.f32_bits(v) for row in o_comb for v in row]
            if c_param_bits != o_param_bits:
                i = next(i for i, (a, b) in enumerate(zip(c_param_bits, o_param_bits)) if a != b)
                raise ValueError(f"FFN HC coefficient F32 mismatch at {i}: {c_param_bits[i]:08x}!={o_param_bits[i]:08x}")

            c_final = route.c_post(post_fn, branch, after, c_post, c_comb)
            o_final = hc_oracle.post_y(branch, after, o_post, o_comb, dim=full.INPUT)
            c_f32 = [chain.f32_bits(v) for v in c_final]
            o_f32 = [chain.f32_bits(v) for v in o_final]
            if c_f32 != o_f32:
                i = next(i for i, (a, b) in enumerate(zip(c_f32, o_f32)) if a != b)
                raise ValueError(f"final HC F32 replay mismatch at {i}: {c_f32[i]:08x}!={o_f32[i]:08x}")
            want_final, _ = chain.cast_bf16(c_final)
            if want_final != final_bits:
                i = next(i for i, (a, b) in enumerate(zip(want_final, final_bits)) if a != b)
                raise ValueError(f"layer4 final BF16 mismatch at {i}: {want_final[i]:04x}!={final_bits[i]:04x}")

        print("PASS V7 complete real layer-4 composition, exact scalar boundaries")
        print("experts:", ids)
        print("ffn-pre first8:", " ".join(f"0x{x:04x}" for x in ffn_bits[:8]))
        print("moe first8:", " ".join(f"0x{x:04x}" for x in combined[:8]))
        print("final first8:", " ".join(f"0x{x:04x}" for x in final_bits[:8]))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
