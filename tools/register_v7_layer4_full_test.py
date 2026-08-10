#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic registration/fix helper for the V7 layer-4 replay."""

from pathlib import Path

RUN = Path("tests/run.sh")
GEN = Path("tools/make_v7_layer4_moe_fixture.py")


def patch_run() -> None:
    s = RUN.read_text()
    anchor = '''deepseek_replay "V7 — real chained layer-3 output through layer-4 router" \\
                "test_v7_layer4_route_real.py"\n'''
    block = '''deepseek_replay "V7 — complete real layer-4 composition from frozen layer-3 output" \\
                "test_v7_layer4_full_real.py"\n'''
    if block not in s:
        if s.count(anchor) != 1:
            raise SystemExit(f"V7 route registration anchor count={s.count(anchor)}")
        s = s.replace(anchor, anchor + block, 1)
    RUN.write_text(s)


def patch_generator() -> None:
    s = GEN.read_text()
    old = '''    trunk = (prov.get("checkpoint_tensors") or {}).get("trunk") or {}\n'''
    new = '''    trunk = (prov.get("checkpoint_tensors") or {}).get("hyperconnection_and_router") or {}\n'''
    if old in s:
        s = s.replace(old, new, 1)
    elif new not in s:
        raise SystemExit("layer-4 route checkpoint metadata key drifted")
    GEN.write_text(s)


def main() -> int:
    patch_run()
    patch_generator()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
