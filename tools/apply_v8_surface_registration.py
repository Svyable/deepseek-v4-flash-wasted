#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""One-shot registration of the permanent Gate-I/V8 surface regression."""

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def main() -> int:
    path = Path("tests/run.sh")
    s = path.read_text()

    s = replace_once(
        s,
        "# Gate H/V6 — PARTIAL. Layer-3 composition. The model-free half pins the block\n",
        "# Gate H/V6 — PASSED at one complete real layer-3 transition. The model-free half pins the block\n",
        "Gate-H status comment",
    )

    anchor = (
        'deepseek_replay "V7 — real consecutive layer-3 to layer-4 trace" \\\n'
        '                "test_v7_two_layer_real.py"\n'
    )
    addition = (
        '\n# Gate I/V8 starts with an immutable header/source contract. This gate reads no\n'
        '# tensor payload and makes no logit claim; it prevents later numeric work from\n'
        '# silently targeting current-main source shapes or guessed final-head records.\n'
        'deepseek_replay "Gate I/V8 — immutable final-head header/source surface" \\\n'
        '                "test_v8_head_surface.py"\n'
    )
    s = replace_once(s, anchor, anchor + addition, "V8 registration anchor")
    path.write_text(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
