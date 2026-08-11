#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""One-shot registration of generic V8 layer-engine header/route regressions."""

from pathlib import Path


def main() -> int:
    path = Path("tests/run.sh")
    s = path.read_text()
    anchor = (
        'deepseek_replay "Gate I/V8 — bounded real-state selected head rows" \\\n'
        '                "test_v8_head_primitive_real.py"\n'
    )
    if s.count(anchor) != 1:
        raise SystemExit(f"Gate-I primitive anchor count {s.count(anchor)} != 1")
    addition = (
        '# Generic propagation starts by measuring each immutable layer surface;\n'
        '# layer 5 is the first new ratio128-learned continuation contract.\n'
        'deepseek_replay "Gate I/V8 — generic layer surface model-free contracts" \\\n'
        '                "test_v8_layer_surface.py"\n'
        'deepseek_replay "Gate I/V8 — immutable layer-5 ratio128 header contract" \\\n'
        '                "test_v8_layer5_surface.py"\n'
        '# The first executable generic continuation freezes only through learned\n'
        '# routing; selected expert payload remains a separate follow-on gate.\n'
        'deepseek_replay "Gate I/V8 — generic layer-5 route continuation" \\\n'
        '                "test_v8_layer_route_real.py"\n'
    )
    path.write_text(s.replace(anchor, anchor + addition, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
