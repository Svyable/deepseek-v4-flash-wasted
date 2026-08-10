#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Acquire the bounded real layer-4 MoE branch and final V7 state.

The expensive routing question is already frozen in
``tests/fixtures/deepseek_v4/v7_layer4_route_real``. This generator therefore
never searches experts: it reads the exact route IDs/weights from that fixture,
fetches only the six selected routed FP4 records plus the shared FP8 expert,
and evaluates every 4096-output expert twice:

* standalone Gate-F oracle, specialized to all 4096 output rows and linking no
  WASTE source; and
* WASTE's scalar expert/runtime checker.

Nothing freezes unless all 7 * 4096 BF16 expert outputs agree exactly. The
combined MoE branch is then composed through layer 4's frozen FFN
HyperConnection, with independent F32 HC equations required to match the C
reference bit-for-bit before the final BF16 [4,4096] state is written.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import make_v6_moe_branch_fixture as gateh  # noqa: E402
import make_v7_layer4_route_fixture as route  # noqa: E402
import deepseek_v4_hc_oracle as hc_oracle  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
LAYER = 4
INPUT = 4096
INTER = 2048
OUT = 4096
TOPK = 6
HC = 4
FLAT = HC * INPUT

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
ROUTEFIX = os.path.join(BASE, "v7_layer4_route_real")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_u32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}I", data))


def read_f32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}f", data))


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def output_meta(path, shape):
    bits = gateh.read_u16(path)
    return {
        "file": os.path.basename(path),
        "dtype": "BF16",
        "shape": shape,
        "sha256": sha256_file(path),
        "first8_hex": [f"0x{x:04x}" for x in bits[:8]],
    }


def checked_route_state():
    prov_path = os.path.join(ROUTEFIX, "provenance.json")
    if not os.path.isfile(prov_path):
        raise ValueError("frozen V7 layer-4 route fixture is required first")
    prov = load_json(prov_path)
    if prov.get("model") != MODEL or prov.get("revision") != REV or prov.get("layer") != LAYER:
        raise ValueError("layer-4 route fixture identity drifted")
    if prov.get("structural_class") != "ratio4-learned":
        raise ValueError("layer-4 route fixture is not the frozen ratio4-learned target")
    hc = prov.get("hc_crosschecks") or {}
    if hc.get("attn_post_comb_exact_f32") is not True or \
       hc.get("ffn_post_comb_exact_f32") is not True:
        raise ValueError("layer-4 route fixture no longer carries exact-F32 HC evidence")

    router = prov.get("router") or {}
    ids_path = os.path.join(ROUTEFIX, router.get("ids_file", ""))
    weights_path = os.path.join(ROUTEFIX, router.get("weights_file", ""))
    if not os.path.isfile(ids_path) or sha256_file(ids_path) != router.get("ids_sha256"):
        raise ValueError("frozen layer-4 router IDs missing or SHA mismatch")
    if not os.path.isfile(weights_path) or sha256_file(weights_path) != router.get("weights_sha256"):
        raise ValueError("frozen layer-4 router weights missing or SHA mismatch")
    ids = read_u32(ids_path, TOPK)
    weights = read_f32(weights_path, TOPK)
    if ids != router.get("ids_topk_order"):
        raise ValueError(f"router ID binary/provenance mismatch: {ids}")
    if [chain.f32_bits(v) for v in weights] != \
       [chain.f32_bits(v) for v in router.get("runtime_weights_f32", [])]:
        raise ValueError("router weight binary/provenance F32 mismatch")

    next_acq = prov.get("next_acquisition") or {}
    want_sorted = sorted(ids)
    if next_acq.get("routed_experts") != want_sorted or next_acq.get("shared_expert") is not True:
        raise ValueError("frozen route acquisition set is not the exact selected expert set")

    ffn_meta = (prov.get("outputs") or {}).get("ffn_pre") or {}
    ffn_path = os.path.join(ROUTEFIX, ffn_meta.get("file", ""))
    if not os.path.isfile(ffn_path) or sha256_file(ffn_path) != ffn_meta.get("sha256"):
        raise ValueError("frozen layer-4 FFN-pre missing or SHA mismatch")
    if ffn_meta.get("dtype") != "BF16" or ffn_meta.get("shape") != [INPUT]:
        raise ValueError("frozen layer-4 FFN-pre geometry drifted")

    after_meta = (prov.get("outputs") or {}).get("after_attn") or {}
    after_path = os.path.join(ROUTEFIX, after_meta.get("file", ""))
    if not os.path.isfile(after_path) or sha256_file(after_path) != after_meta.get("sha256"):
        raise ValueError("frozen layer-4 after-attention state missing or SHA mismatch")
    if after_meta.get("dtype") != "BF16" or after_meta.get("shape") != [HC, INPUT]:
        raise ValueError("frozen layer-4 after-attention geometry drifted")

    trunk = (prov.get("checkpoint_tensors") or {}).get("hyperconnection_and_router") or {}
    params = {}
    for key, count in (("ffn_fn", route.MIX * route.FLAT),
                       ("ffn_scale", 3), ("ffn_base", route.MIX)):
        meta = trunk.get(key) or {}
        path = os.path.join(ROUTEFIX, f"{key.replace('_', '-')}.f32.bin")
        # Route fixture file naming is ffn-fn / ffn-scale / ffn-base.
        if not os.path.isfile(path):
            raise ValueError(f"frozen layer-4 {key} file missing")
        if sha256_file(path) != meta.get("payload_sha256"):
            raise ValueError(f"frozen layer-4 {key} payload SHA drifted")
        params[key] = read_f32(path, count)

    return {
        "provenance": prov,
        "provenance_path": prov_path,
        "ids": ids,
        "weights": weights,
        "ids_path": ids_path,
        "weights_path": weights_path,
        "ffn_path": ffn_path,
        "ffn_bits": gateh.read_u16(ffn_path),
        "after_path": after_path,
        "after_bits": gateh.read_u16(after_path),
        "params": params,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="header-only pinned 0731 checkpoint snapshot")
    ap.add_argument("out_dir", help="fixture output directory")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    state = checked_route_state()
    if len(state["ffn_bits"]) != INPUT or len(state["after_bits"]) != FLAT:
        raise ValueError("frozen route state byte geometry drifted")

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    input_path = os.path.join(args.out_dir, "ffn-pre.bf16.bin")
    ids_path = os.path.join(args.out_dir, "router-ids.u32.bin")
    weights_path = os.path.join(args.out_dir, "router-weights.f32.bin")
    shutil.copyfile(state["ffn_path"], input_path)
    shutil.copyfile(state["ids_path"], ids_path)
    shutil.copyfile(state["weights_path"], weights_path)

    routed_bits = []
    routed_meta = []
    shared_bits = None
    shared_meta = None

    with tempfile.TemporaryDirectory(prefix="v7-layer4-moe-") as work:
        oracle, checker, oracle_meta = gateh.build_tools(work)

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
            infos = {k: gateh.fetch(args.snapshot, names[k], paths[k], args.token)
                     for k in names}
            gateh.validate_routed_shapes(expert, infos)

            expected = os.path.join(d, "expected.bf16.bin")
            gateh.run([oracle, "routed", input_path,
                       paths["w1"], paths["s1"], paths["w3"], paths["s3"],
                       paths["w2"], paths["s2"], repr(route_weight), expected, "unused"])
            gateh.run([checker, "routed", input_path,
                       paths["w1"], paths["s1"], paths["w3"], paths["s3"],
                       paths["w2"], paths["s2"], repr(route_weight), expected])
            bits = gateh.read_u16(expected)
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
                "checkpoint_tensors": {k: gateh.tensor_meta(infos[k]) for k in names},
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
        infos = {k: gateh.fetch(args.snapshot, names[k], paths[k], args.token)
                 for k in names}
        gateh.validate_shared_shapes(infos)

        gate_diag = os.path.join(d, "gate.f32.bin")
        up_diag = os.path.join(d, "up.f32.bin")
        hidden_diag = os.path.join(d, "hidden.bf16.bin")
        expected = os.path.join(d, "expected.bf16.bin")
        gateh.run([oracle, "shared", input_path,
                   paths["w1"], paths["s1"], paths["w3"], paths["s3"],
                   paths["w2"], paths["s2"], gate_diag, up_diag, hidden_diag,
                   expected, "unused"])
        gateh.run([checker, "shared", input_path,
                   paths["w1"], paths["s1"], paths["w3"], paths["s3"],
                   paths["w2"], paths["s2"], expected])
        shared_bits = gateh.read_u16(expected)
        if len(shared_bits) != OUT:
            raise ValueError(f"shared expert: oracle emitted {len(shared_bits)} outputs")
        shared_meta = {
            "output_sha256": sha256_file(expected),
            "output_first8_hex": [f"0x{x:04x}" for x in shared_bits[:8]],
            "hidden_sha256": sha256_file(hidden_diag),
            "oracle_vs_waste_exact_bf16": OUT,
            "checkpoint_tensors": {k: gateh.tensor_meta(infos[k]) for k in names},
        }
        print(f"shared expert: exact {OUT} BF16")

        routed_path = os.path.join(args.out_dir, "routed-out-all6.bf16.bin")
        shared_path = os.path.join(args.out_dir, "shared-out.bf16.bin")
        combined_path = os.path.join(args.out_dir, "moe-branch.bf16.bin")
        write_u16(routed_path, [v for row in routed_bits for v in row])
        write_u16(shared_path, shared_bits)
        combined = gateh.independent_combine(state["ids"], routed_bits, shared_bits)
        write_u16(combined_path, combined)
        gateh.run([checker, "combine", ids_path, routed_path, shared_path, combined_path])

        mutation_visibility = {}
        for slot, expert in enumerate(state["ids"]):
            muted = [list(row) for row in routed_bits]
            muted[slot] = [0] * OUT
            other = gateh.independent_combine(state["ids"], muted, shared_bits)
            diff = sum(a != b for a, b in zip(other, combined))
            if diff == 0:
                raise ValueError(f"dropping routed expert {expert} is BF16-invisible")
            mutation_visibility[f"drop_expert_{expert}"] = diff
        other = gateh.independent_combine(state["ids"], routed_bits, [0] * OUT)
        shared_diff = sum(a != b for a, b in zip(other, combined))
        if shared_diff == 0:
            raise ValueError("dropping the shared expert is BF16-invisible")
        mutation_visibility["drop_shared_expert"] = shared_diff

        # Reconstruct the exact frozen FFN HC boundary before composing the
        # real MoE branch through hc_post.
        after = [chain.bf16_to_f32(b) for b in state["after_bits"]]
        ffn_pre_fn, ffn_post_fn, _, _ = route.build_state_lib(work)
        c_pre_f, c_post, c_comb = route.c_pre(
            ffn_pre_fn, after, state["params"]["ffn_fn"],
            state["params"]["ffn_scale"], state["params"]["ffn_base"])
        c_pre_bits, _ = chain.cast_bf16(c_pre_f)
        if c_pre_bits != state["ffn_bits"]:
            i = next(i for i, (a, b) in enumerate(zip(c_pre_bits, state["ffn_bits"])) if a != b)
            raise ValueError(
                f"layer4 FFN-pre reconstruction mismatch at {i}: "
                f"{c_pre_bits[i]:04x}!={state['ffn_bits'][i]:04x}")

        o_pre, o_post, o_comb, _ = hc_oracle.params(
            after, state["params"]["ffn_fn"], state["params"]["ffn_scale"],
            state["params"]["ffn_base"], dim=INPUT)
        o_pre_bits, _ = chain.cast_bf16(hc_oracle.pre_y(after, o_pre, dim=INPUT))
        if o_pre_bits != state["ffn_bits"]:
            i = next(i for i, (a, b) in enumerate(zip(o_pre_bits, state["ffn_bits"])) if a != b)
            raise ValueError(
                f"independent layer4 FFN-pre mismatch at {i}: "
                f"{o_pre_bits[i]:04x}!={state['ffn_bits'][i]:04x}")
        c_param_bits = [chain.f32_bits(v) for v in c_post] + [
            chain.f32_bits(v) for row in c_comb for v in row]
        o_param_bits = [chain.f32_bits(v) for v in o_post] + [
            chain.f32_bits(v) for row in o_comb for v in row]
        if c_param_bits != o_param_bits:
            i = next(i for i, (a, b) in enumerate(zip(c_param_bits, o_param_bits)) if a != b)
            raise ValueError(
                f"layer4 FFN post/comb F32 mismatch at {i}: "
                f"{c_param_bits[i]:08x}!={o_param_bits[i]:08x}")

        branch = [chain.bf16_to_f32(b) for b in combined]
        c_final = route.c_post(ffn_post_fn, branch, after, c_post, c_comb)
        o_final = hc_oracle.post_y(branch, after, o_post, o_comb, dim=INPUT)
        c_final_bits_f32 = [chain.f32_bits(v) for v in c_final]
        o_final_bits_f32 = [chain.f32_bits(v) for v in o_final]
        if c_final_bits_f32 != o_final_bits_f32:
            i = next(i for i, (a, b) in enumerate(zip(c_final_bits_f32, o_final_bits_f32)) if a != b)
            raise ValueError(
                f"layer4 final HC F32 mismatch at {i}: "
                f"{c_final_bits_f32[i]:08x}!={o_final_bits_f32[i]:08x}")
        final_bits, _ = chain.cast_bf16(c_final)
        final_path = os.path.join(args.out_dir, "layer4-final.bf16.bin")
        write_u16(final_path, final_bits)

        provenance = {
            "schema_version": 1,
            "gate": "V7 complete real layer-4 MoE branch and final composition",
            "evidence_state": "CHECKPOINT-BOUND-LAYER4-COMPLETE-SCALAR",
            "model": MODEL,
            "revision": REV,
            "layer": LAYER,
            "structural_class": "ratio4-learned",
            "route_dependency": {
                "fixture": "tests/fixtures/deepseek_v4/v7_layer4_route_real",
                "provenance_sha256": sha256_file(state["provenance_path"]),
                "ffn_pre_sha256": sha256_file(state["ffn_path"]),
                "after_attn_sha256": sha256_file(state["after_path"]),
                "ids_topk_order": state["ids"],
                "ids_ascending_expert_order": sorted(state["ids"]),
                "weights_f32": state["weights"],
                "topk_boundary_margin": (state["provenance"].get("router") or {}).get("topk_boundary_margin"),
            },
            "routed_experts": routed_meta,
            "shared_expert": shared_meta,
            "oracle_independence": oracle_meta,
            "acquisition_verification": {
                "routed_exact_bf16_values": TOPK * OUT,
                "shared_exact_bf16_values": OUT,
                "total_expert_exact_bf16_values": (TOPK + 1) * OUT,
                "final_hc_exact_f32_values": FLAT,
                "ffn_hc_post_comb_exact_f32": True,
                "raw_checkpoint_weights_committed": False,
                "raw_checkpoint_weights_lifetime": "temporary acquisition workspace only",
            },
            "outputs": {
                "ffn_pre": output_meta(input_path, [INPUT]),
                "routed": output_meta(routed_path, [TOPK, OUT]),
                "shared": output_meta(shared_path, [OUT]),
                "moe_branch": output_meta(combined_path, [OUT]),
                "layer4_final": output_meta(final_path, [HC, INPUT]),
                "ids": {"file": os.path.basename(ids_path), "sha256": sha256_file(ids_path),
                        "shape": [TOPK], "dtype": "U32"},
                "weights": {"file": os.path.basename(weights_path), "sha256": sha256_file(weights_path),
                            "shape": [TOPK], "dtype": "F32"},
            },
            "mutations": mutation_visibility,
            "non_claims": [
                "position 0 does not re-prove non-empty ratio-4 compressor/indexer selection; Gate E owns that primitive evidence",
                "this freezes one real layer-4 state transition chained from Gate H, not all 43 layers",
                "no final-model logits, generation, converted-container/cache identity, RAM or throughput claim",
            ],
        }
        with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, sort_keys=True)
            f.write("\n")

    print("PASS V7 complete real layer-4 MoE/final fixture built")
    print("experts:", state["ids"])
    print("moe first8:", " ".join(f"0x{x:04x}" for x in combined[:8]))
    print("final first8:", " ".join(f"0x{x:04x}" for x in final_bits[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
