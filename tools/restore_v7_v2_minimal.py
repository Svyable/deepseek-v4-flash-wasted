#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Restore the Gate-H V2 fixture generator and apply only V7's signed-zero fix.

Temporary maintainer-review helper. The validation workflow removes this file
once the minimal diff and complete suites are green.
"""

from pathlib import Path
import subprocess

BASE = "98604765d61c44c25888c32d4b7d95422a02caf7"
PATH = "tools/make_v2_real_fixture.py"


def main() -> int:
    base = subprocess.check_output(
        ["git", "show", f"{BASE}:{PATH}"], text=True)
    old = '''def e4m3_float(code):\n    return float(e4m3_fraction(code))\n'''
    new = '''def e4m3_float(code):\n    """Decode finite E4M3 while preserving the format's signed zero."""\n    # Fraction is exact for nonzero values but has no signed-zero state.\n    if (code & 0x7F) == 0:\n        return -0.0 if (code & 0x80) else 0.0\n    return float(e4m3_fraction(code))\n'''
    if base.count(old) != 1:
        raise SystemExit("Gate-H base e4m3_float anchor drifted")
    expected = base.replace(old, new, 1)
    Path(PATH).write_text(expected)

    # Fail closed: the resulting file must be exactly base + this one hunk.
    reread = Path(PATH).read_text()
    if reread != expected:
        raise SystemExit("minimal signed-zero restoration did not reproduce expected file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
