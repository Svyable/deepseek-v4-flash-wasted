#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""One-shot registration of Gate-I scalar and real-state primitive replays."""

from pathlib import Path


def main() -> int:
    path = Path("tests/run.sh")
    s = path.read_text()
    anchor = (
        'deepseek_replay "Gate I/V8 — immutable final-head header/source surface" \\\n'
        '                "test_v8_head_surface.py"\n'
    )
    if s.count(anchor) != 1:
        raise SystemExit(f"Gate-I surface anchor count {s.count(anchor)} != 1")
    addition = (
        '# Source-derived scalar primitive: independent Python and C must agree\n'
        '# exactly before any checkpoint row slice is trusted.\n'
        'deepseek_replay "Gate I/V8 — model-free scalar final-head primitive" \\\n'
        '                "test_v8_head_ref.py"\n'
        '# Bounded real-state fixture: uses V7 layer-4 final only as a test\n'
        '# vector and deliberately refuses any final-model-logit interpretation.\n'
        'deepseek_replay "Gate I/V8 — bounded real-state selected head rows" \\\n'
        '                "test_v8_head_primitive_real.py"\n'
    )
    path.write_text(s.replace(anchor, anchor + addition, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
