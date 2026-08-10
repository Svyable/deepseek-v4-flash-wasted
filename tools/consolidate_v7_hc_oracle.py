#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic migration to the canonical DeepSeek V4 HC oracle."""

from pathlib import Path

RUN = Path("tests/run.sh")
OLD_IMPORT = "import v7_hc_oracle as hc_oracle  # noqa: E402"
NEW_IMPORT = "import deepseek_v4_hc_oracle as hc_oracle  # noqa: E402"


def patch_imports() -> list[str]:
    changed = []
    for root in (Path("tools"), Path("tests")):
        for p in sorted(root.rglob("*.py")):
            if p.name in {"v7_hc_oracle.py", "test_v7_hc_oracle_f32.py"}:
                continue
            s = p.read_text()
            if OLD_IMPORT in s:
                p.write_text(s.replace(OLD_IMPORT, NEW_IMPORT))
                changed.append(str(p))
    return changed


def patch_run() -> None:
    s = RUN.read_text()
    block = '''deepseek_replay "V7 — bit-exact independent F32 HyperConnection oracle" \\
                "test_v7_hc_oracle_f32.py"\n'''
    if block in s:
        s = s.replace(block, "", 1)
    elif "test_v7_hc_oracle_f32.py" in s:
        raise SystemExit("V7 duplicate HC test registration changed shape")
    RUN.write_text(s)


def main() -> int:
    changed = patch_imports()
    if not changed:
        raise SystemExit("no V7 consumers referenced the duplicate HC oracle")
    patch_run()
    print("canonicalized HC imports in:")
    for p in changed:
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
