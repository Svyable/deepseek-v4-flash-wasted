#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Refuse V7 evidence chained from any superseded Gate-H endpoint."""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from v7_parent_freshness import (  # noqa: E402
    OLD_REFUSED_SHA,
    V7ParentFreshnessError,
    assert_input_dependency,
    assert_two_layer_chain,
    canonical_layer3_sha,
)

ROUTE_PROV = os.path.join(
    "tests", "fixtures", "deepseek_v4", "v7_layer4_route_real", "provenance.json"
)
FULL_PROV = os.path.join(
    "tests", "fixtures", "deepseek_v4", "v7_layer4_full_real", "provenance.json"
)
TWO_LAYER_PROV = os.path.join(
    "tests", "fixtures", "deepseek_v4", "v7_two_layer_real", "provenance.json"
)
TWO_LAYER_TEST = os.path.join(REPO, "tests", "test_v7_two_layer_real.py")


def load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    try:
        canonical = canonical_layer3_sha(REPO)

        # The already-frozen layer-4 route acquisition is authoritative V7
        # downstream evidence today, so it must always bind to current Gate H.
        assert_input_dependency(REPO, ROUTE_PROV, canonical)

        # These rungs are intentionally optional until reacquired. Once they
        # appear, freshness becomes mandatory automatically.
        checked = [ROUTE_PROV]
        for relative in (FULL_PROV,):
            if os.path.isfile(os.path.join(REPO, relative)):
                assert_input_dependency(REPO, relative, canonical)
                checked.append(relative)
        if os.path.isfile(os.path.join(REPO, TWO_LAYER_PROV)):
            provenance = load(os.path.join(REPO, TWO_LAYER_PROV))
            assert_two_layer_chain(
                provenance, canonical, label=TWO_LAYER_PROV
            )
            checked.append(TWO_LAYER_PROV)

        # Future two-layer evidence derives the parent SHA dynamically. The
        # specifically refused endpoint must never re-enter as a copied literal.
        text = open(TWO_LAYER_TEST, encoding="utf-8").read()
        if OLD_REFUSED_SHA in text:
            raise V7ParentFreshnessError(
                "dormant two-layer regression still embeds the refused Gate-H SHA"
            )

        print("PASS V7 parent freshness")
        print("canonical layer-3 sha256:", canonical)
        print("downstream provenance checked:", len(checked))
        for path in checked:
            print(" ", path)
        return 0
    except Exception as exc:
        print("FAIL:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
