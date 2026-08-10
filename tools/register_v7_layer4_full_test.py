#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic registration helper for the V7 layer-4 full replay."""

from pathlib import Path

PATH = Path("tests/run.sh")


def main() -> int:
    s = PATH.read_text()
    anchor = '''deepseek_replay "V7 — real chained layer-3 output through layer-4 router" \\
                "test_v7_layer4_route_real.py"\n'''
    block = '''deepseek_replay "V7 — complete real layer-4 composition from frozen layer-3 output" \\
                "test_v7_layer4_full_real.py"\n'''
    if block not in s:
        if s.count(anchor) != 1:
            raise SystemExit(f"V7 route registration anchor count={s.count(anchor)}")
        s = s.replace(anchor, anchor + block, 1)
    PATH.write_text(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
