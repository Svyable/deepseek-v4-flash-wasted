#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay pinned 0731 quantization evidence through the scalar C seams.

The FP4 expected values come from a small F3 fixture derived from the pinned
official release source. After that source-convention check passes, the driver
runs the model-free Gate C/V2 scalar-linear preflight. The latter is a
source-derived arithmetic check only; it does not claim a real projection gate.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

import test_v2_linear_ref  # noqa: E402

FIXTURE = os.path.join(
    REPO, "tests", "fixtures", "deepseek_v4", "fp4_release_convention.json")


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    with open(FIXTURE, encoding="utf-8") as f:
        fx = json.load(f)

    revision = fx.get("model_revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in revision):
        print("FAIL: fixture revision is not an immutable 40-hex SHA")
        return 1

    case = fx["literal_case"]
    packed = int(case["packed_byte_hex"], 16)
    scale = int(case["ue8m0_scale_hex"], 16)
    expected = case["expected_logical_values"]
    if len(expected) != 2:
        print("FAIL: fixture must contain exactly two logical values")
        return 1

    program = f'''\
#include "fp4_e2m1.h"
#include <stdint.h>
#include <stdio.h>

int main(void) {{
    uint8_t weight[16] = {{0}};  /* one 32-column FP4 row */
    uint8_t scales[1] = {{{scale}}};
    weight[0] = {packed};

    float a = waste_fp4_at(weight, scales, 32, 0, 0);
    float b = waste_fp4_at(weight, scales, 32, 0, 1);
    if (a != {float(expected[0])!r}f || b != {float(expected[1])!r}f) {{
        fprintf(stderr, "got %.9g %.9g, want %.9g %.9g\\n",
                a, b, {float(expected[0])!r}, {float(expected[1])!r});
        return 1;
    }}
    return 0;
}}
'''

    with tempfile.TemporaryDirectory(prefix="release-fp4-") as td:
        probe = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe.exe" if os.name == "nt" else "probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write(program)

        cmd = [
            cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-I", os.path.join(REPO, "src", "quant"),
            probe,
            os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
            "-lm", "-o", exe,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            sys.stderr.write(built.stdout)
            sys.stderr.write(built.stderr)
            print("FAIL: could not compile release-convention probe")
            return 1

        ran = subprocess.run([exe], capture_output=True, text=True)
        if ran.returncode != 0:
            sys.stderr.write(ran.stdout)
            sys.stderr.write(ran.stderr)
            print("FAIL: scalar FP4 path disagrees with pinned 0731 convention fixture")
            return 1

    print("PASS pinned 0731 FP4 nibble order + E8M0 scale application")

    v2 = test_v2_linear_ref.main()
    if v2 != 0:
        if v2 == 77:
            print("SKIP Gate C scalar preflight: no compiler")
            return 77
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
