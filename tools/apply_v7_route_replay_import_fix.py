#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic patcher for V7 route replay registration/import."""

from pathlib import Path

TEST = Path("tests/test_v7_layer4_route_real.py")
RUN = Path("tests/run.sh")


def main() -> int:
    s = TEST.read_text()
    old = '''REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nsys.path.insert(0, os.path.join(REPO, "tests"))\nimport test_v6_ffn_route_real as chain  # noqa: E402\nimport v7_hc_oracle as hc_oracle  # noqa: E402\n'''
    new = '''REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nsys.path.insert(0, os.path.join(REPO, "tools"))\nsys.path.insert(0, os.path.join(REPO, "tests"))\nimport test_v6_ffn_route_real as chain  # noqa: E402\nimport v7_hc_oracle as hc_oracle  # noqa: E402\n'''
    if old in s:
        s = s.replace(old, new, 1)
    elif new not in s:
        raise SystemExit("route replay import anchor drifted")
    TEST.write_text(s)

    r = RUN.read_text()
    anchor = '''deepseek_replay "V7 — fail-closed first-divergence layer trace localizer" \\\n                "test_v7_localize.py"\n'''
    block = '''deepseek_replay "V7 — real chained layer-3 output through layer-4 router" \\\n                "test_v7_layer4_route_real.py"\n'''
    if block not in r:
        if r.count(anchor) != 1:
            raise SystemExit(f"V7 localizer registration anchor count={r.count(anchor)}")
        r = r.replace(anchor, anchor + block, 1)
    RUN.write_text(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
