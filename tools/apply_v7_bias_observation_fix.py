#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic patcher for the V7 correction-bias review fix."""

from pathlib import Path

GEN = Path("tools/make_v7_layer4_route_fixture.py")
REPLAY = Path("tests/test_v7_layer4_route_real.py")
RUN = Path("tests/run.sh")


def patch_generator():
    s = GEN.read_text()
    anchor = '''\ndef main(argv=None) -> int:\n'''
    helper = '''\n\ndef observe_correction_bias(scores, bias, selection, biased_ids):\n    """Validate the mandatory bias operation and record its input-specific effect.\n\n    A correction bias is semantically load-bearing because learned routing selects\n    from ``score + bias``. It does *not* follow that every input must select a\n    different top-k set than unbiased scores would. Treat that as an observation,\n    never as a gate.\n    """\n    if not (len(scores) == len(bias) == len(selection) == EXPERTS):\n        raise ValueError("router bias observation geometry mismatch")\n    expected = [chain.f32(scores[e] + bias[e]) for e in range(EXPERTS)]\n    for e, (got, want) in enumerate(zip(selection, expected)):\n        if chain.f32_bits(got) != chain.f32_bits(want):\n            raise ValueError(\n                f"router correction-bias selection score mismatch at expert {e}: "\n                f"{chain.f32_bits(got):08x}!={chain.f32_bits(want):08x}")\n    unbiased_ids = sorted(range(EXPERTS), key=lambda e: (-scores[e], e))[:TOPK]\n    nonzero = sum((chain.f32_bits(v) & 0x7fffffff) != 0 for v in bias)\n    changed_scores = sum(\n        chain.f32_bits(selection[e]) != chain.f32_bits(scores[e])\n        for e in range(EXPERTS))\n    return {\n        "biased_ids": list(biased_ids),\n        "unbiased_ids": unbiased_ids,\n        "changes_topk_on_this_input": list(biased_ids) != unbiased_ids,\n        "nonzero_f32_count": nonzero,\n        "changed_selection_score_f32_count": changed_scores,\n        "selection_scores_equal_score_plus_bias_f32": True,\n    }\n'''
    if "def observe_correction_bias(" not in s:
        if s.count(anchor) != 1:
            raise SystemExit(f"generator main anchor count={s.count(anchor)}")
        s = s.replace(anchor, helper + anchor, 1)

    old = '''        ids, weights, _, _ = c_route(\n            score_fn, learned_fn, ffn_pre, gate_bits, gate_bias)\n        py_ids, py_weights, py_scores, py_selection, margin = chain.oracle_router(\n            ffn_pre, gate_bits, gate_bias)\n        if ids != py_ids:\n            raise ValueError(f"layer4 router IDs C={ids} Python={py_ids}")\n        weight_delta = max(abs(a - b) for a, b in zip(weights, py_weights))\n        if weight_delta > 1e-5:\n            raise ValueError(f"layer4 route weight delta {weight_delta} > 1e-5")\n        if not margin > 1e-4:\n            raise ValueError(f"layer4 top-k boundary {margin} is not stable enough for acquisition")\n\n        unbiased_ids = sorted(range(EXPERTS), key=lambda e: (-py_scores[e], e))[:TOPK]\n        if unbiased_ids == ids:\n            # This is not mathematically impossible, but for this fixture we\n            # want correction bias to remain observably load-bearing.\n            raise ValueError("layer4 correction-bias selection is invisible on the chained input")\n'''
    new = '''        ids, weights, c_scores, c_selection = c_route(\n            score_fn, learned_fn, ffn_pre, gate_bits, gate_bias)\n        py_ids, py_weights, py_scores, py_selection, margin = chain.oracle_router(\n            ffn_pre, gate_bits, gate_bias)\n        if ids != py_ids:\n            raise ValueError(f"layer4 router IDs C={ids} Python={py_ids}")\n        weight_delta = max(abs(a - b) for a, b in zip(weights, py_weights))\n        if weight_delta > 1e-5:\n            raise ValueError(f"layer4 route weight delta {weight_delta} > 1e-5")\n        if not margin > 1e-4:\n            raise ValueError(f"layer4 top-k boundary {margin} is not stable enough for acquisition")\n\n        bias_observation = observe_correction_bias(\n            c_scores, gate_bias, c_selection, ids)\n        # Independent Python selection must implement the same semantic\n        # operation; numerical route weights remain a separate tolerance-bound\n        # transcendental cross-check, while exact C weights are frozen below.\n        py_expected_selection = [chain.f32(py_scores[e] + gate_bias[e])\n                                 for e in range(EXPERTS)]\n        if any(chain.f32_bits(a) != chain.f32_bits(b)\n               for a, b in zip(py_selection, py_expected_selection)):\n            raise ValueError("independent Python router did not apply correction bias as score+bias")\n'''
    if old in s:
        s = s.replace(old, new, 1)
    elif new not in s:
        raise SystemExit("generator router assertion anchor drifted")

    old_prov = '''            "selection_without_correction_bias": unbiased_ids,\n            "ids_runtime_vs_python_exact": True,\n'''
    new_prov = '''            "selection_without_correction_bias": bias_observation["unbiased_ids"],\n            "correction_bias": bias_observation,\n            "ids_runtime_vs_python_exact": True,\n'''
    if old_prov in s:
        s = s.replace(old_prov, new_prov, 1)
    elif new_prov not in s:
        raise SystemExit("generator bias provenance anchor drifted")
    if "correction-bias selection is invisible" in s:
        raise SystemExit("over-strong correction-bias assertion remains")
    GEN.write_text(s)


def patch_replay():
    s = REPLAY.read_text()
    old = '''        got_ids = list(ids)\n        got_weights = list(weights)\n        if got_ids != want_ids:\n            print(f"FAIL: layer4 route IDs changed: {got_ids} != {want_ids}")\n            return 1\n        if [chain.f32_bits(x) for x in got_weights] != [chain.f32_bits(x) for x in want_weights]:\n            print("FAIL: layer4 exact scalar route weights changed")\n            return 1\n\n        py_ids, py_weights, _, _, margin = chain.oracle_router(\n'''
    new = '''        got_ids = list(ids)\n        got_weights = list(weights)\n        got_scores = list(scores)\n        got_selection = list(selection)\n        if got_ids != want_ids:\n            print(f"FAIL: layer4 route IDs changed: {got_ids} != {want_ids}")\n            return 1\n        if [chain.f32_bits(x) for x in got_weights] != [chain.f32_bits(x) for x in want_weights]:\n            print("FAIL: layer4 exact scalar route weights changed")\n            return 1\n\n        # Correction bias must be applied exactly as F32(score + bias), but\n        # it is legal for this particular input to keep the same top-k IDs.\n        bias = fparams["gate_bias"]\n        expected_selection = [chain.f32(got_scores[e] + bias[e]) for e in range(EXPERTS)]\n        if any(chain.f32_bits(a) != chain.f32_bits(b)\n               for a, b in zip(got_selection, expected_selection)):\n            print("FAIL: layer4 scalar correction-bias selection scores drifted")\n            return 1\n        unbiased = sorted(range(EXPERTS), key=lambda e: (-got_scores[e], e))[:TOPK]\n        bias_meta = (p.get("router") or {}).get("correction_bias") or {}\n        changed_count = sum(chain.f32_bits(got_selection[e]) != chain.f32_bits(got_scores[e])\n                            for e in range(EXPERTS))\n        nonzero_count = sum((chain.f32_bits(v) & 0x7fffffff) != 0 for v in bias)\n        if bias_meta.get("unbiased_ids") != unbiased or \\\n           bias_meta.get("changes_topk_on_this_input") != (unbiased != got_ids) or \\\n           bias_meta.get("changed_selection_score_f32_count") != changed_count or \\\n           bias_meta.get("nonzero_f32_count") != nonzero_count or \\\n           bias_meta.get("selection_scores_equal_score_plus_bias_f32") is not True:\n            print("FAIL: layer4 correction-bias provenance drifted")\n            return 1\n\n        py_ids, py_weights, _, _, margin = chain.oracle_router(\n'''
    if old in s:
        s = s.replace(old, new, 1)
    elif new not in s:
        raise SystemExit("replay router anchor drifted")
    REPLAY.write_text(s)


def patch_run():
    s = RUN.read_text()
    anchor = '''deepseek_replay "V7 — one-row online-softmax signed-zero operation order" \\\n                "test_v7_one_row_attention_signed_zero.py"\n'''
    block = '''deepseek_replay "V7 — correction bias operation vs input-specific top-k outcome" \\\n                "test_v7_router_bias_observation.py"\n'''
    if block not in s:
        if s.count(anchor) != 1:
            raise SystemExit(f"tests/run bias anchor count={s.count(anchor)}")
        s = s.replace(anchor, anchor + block, 1)
    RUN.write_text(s)


def main():
    patch_generator()
    patch_replay()
    patch_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
