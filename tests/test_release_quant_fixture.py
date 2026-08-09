#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay pinned 0731 source/checkpoint evidence through scalar C seams.

This driver is reached by ``tests/test_inventory.py`` and therefore by
``make check``. It keeps source literals, acquisition mechanics, real
checkpoint fixtures, and model-semantic scalar references adjacent while
preserving their evidence boundaries.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
TOOLS = os.path.join(REPO, "tools")
for path in (TESTS, TOOLS):
    if path not in sys.path:
        sys.path.insert(0, path)

import inventory  # noqa: E402
import test_fetch_hf_tensor_slice  # noqa: E402
import test_v2_linear_ref  # noqa: E402
import test_v2_real_projection  # noqa: E402
import test_v3_mhc_scalar  # noqa: E402
import test_v3_mhc_real  # noqa: E402
import test_v3_router_scalar  # noqa: E402
import test_v3_router_real  # noqa: E402
import test_v4_routed_expert_real  # noqa: E402
import test_v4_moe_real  # noqa: E402

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


def check_release_router_names():
    """Pin source and real-export bootstrap-routing spellings."""
    cases = (
        "model.layers.0.mlp.gate.tid2eid",
        "model.layers.1.mlp.gate.tie2eid",
        "layers.0.ffn.gate.tid2eid",
    )
    for name in cases:
        row = inventory.classify(name, 43)
        if row["subsystem"] != "router_hash" or row["module"] != "main":
            print(
                f"FAIL: official bootstrap route name {name!r} classified as "
                f"{row['subsystem']}/{row['module']}")
            return 1
    print("PASS pinned 0731 bootstrap routing-name classification")
    return 0


def run_gate(name, fn):
    rc = fn()
    if rc == 77:
        print(f"SKIP {name}")
        return 77
    if rc != 0:
        print(f"FAIL {name}")
        return 1
    return 0


def main():
    if check_release_router_names() != 0:
        return 1

    # The payload transport is a separate seam from model arithmetic.
    if run_gate("bounded safetensors payload Range transport",
                test_fetch_hf_tensor_slice.main) != 0:
        return 1

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

    if run_gate("Gate C scalar linear preflight", test_v2_linear_ref.main) != 0:
        return 1
    if run_gate("Gate C / V2 real checkpoint projection",
                test_v2_real_projection.main) != 0:
        return 1
    print("PASS Gate C / V2 real checkpoint projection")

    if run_gate("Gate D model-free mHC invariants",
                test_v3_mhc_scalar.main) != 0:
        return 1
    if run_gate("Gate D / V3 real checkpoint mHC primitive",
                test_v3_mhc_real.main) != 0:
        return 1
    print("PASS Gate D / V3 real checkpoint mHC primitive")

    # Gate F is deliberately layered. A routing-only success is insufficient:
    # the ordinary suite permanently replays learned/hash routing, one full
    # routed FP4 expert, the resident shared FP8 expert, all six selected branch
    # outputs, and the official f32 accumulation/final-BF16 combination.
    if run_gate("Gate F model-free routing invariants",
                test_v3_router_scalar.main) != 0:
        return 1
    if run_gate("Gate F real learned + hash routing",
                test_v3_router_real.main) != 0:
        return 1
    if run_gate("Gate F real routed FP4 expert",
                test_v4_routed_expert_real.main) != 0:
        return 1
    if run_gate("Gate F real shared expert + complete MoE",
                test_v4_moe_real.main) != 0:
        return 1
    print("PASS Gate F / V4 complete routing + MoE")

    return 0


if __name__ == "__main__":
    sys.exit(main())
