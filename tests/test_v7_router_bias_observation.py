#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Pin the V7 real-route correction-bias evidence boundary.

Correction bias is part of learned-router *selection*, but a nonzero bias need
not change the top-k set for every input. V7 must record whether it changes this
particular real route, not require that outcome. What is mandatory is the
operation itself: every scalar selection score equals F32(score + bias).
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import make_v7_layer4_route_fixture as route  # noqa: E402


def f32(x):
    return route.chain.f32(x)


def topk(values):
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:route.TOPK]


def observe(scores, bias):
    selection = [f32(a + b) for a, b in zip(scores, bias)]
    ids = topk(selection)
    return route.observe_correction_bias(scores, bias, selection, ids)


def main():
    scores = [f32(300.0 - i) for i in range(route.EXPERTS)]

    # Nonzero correction bias may perturb selection scores without changing
    # this input's already-wide top-6 ranking. That is legal and was the exact
    # real layer-4 condition that an over-strong acquisition assertion rejected.
    bias = [f32(0.0)] * route.EXPERTS
    bias[0] = f32(0.125)
    bias[100] = f32(-0.25)
    obs = observe(scores, bias)
    if obs["changes_topk_on_this_input"] is not False:
        raise AssertionError("small nonzero bias unexpectedly changed synthetic top-k")
    if obs["nonzero_f32_count"] != 2 or obs["changed_selection_score_f32_count"] != 2:
        raise AssertionError(f"bias observation counts drifted: {obs}")
    if obs["unbiased_ids"] != list(range(route.TOPK)):
        raise AssertionError(f"unbiased synthetic IDs drifted: {obs['unbiased_ids']}")

    # A different input/bias pattern may change the selected set; record that
    # outcome rather than requiring either true or false globally.
    bias2 = [f32(0.0)] * route.EXPERTS
    bias2[10] = f32(400.0)
    obs2 = observe(scores, bias2)
    if obs2["changes_topk_on_this_input"] is not True or 10 not in obs2["biased_ids"]:
        raise AssertionError(f"load-bearing bias change was not observed: {obs2}")

    # The actual mandatory invariant: runtime selection is F32(score + bias).
    selection = [f32(a + b) for a, b in zip(scores, bias)]
    selection[7] = f32(selection[7] + 1.0)
    try:
        route.observe_correction_bias(scores, bias, selection, topk(selection))
    except ValueError as exc:
        if "selection score" not in str(exc):
            raise
    else:
        raise AssertionError("mutated correction-bias selection score was accepted")

    print("PASS V7 correction bias: operation exact; top-k change is an observed outcome")
    print("unchanged-topk case=False changed-topk case=True mutated-selection=refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
