#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Pin the shared FP8 expert across a 128-row w2 scale-grid boundary.

Gate F only needed eight shared-expert output rows and therefore supplied one
E8M0 w2 scale-grid row. Gate H needs all 4096 model outputs. This model-free
fixture is deliberately 129 rows: row 128 must consume the *second* scale-grid
row. It kills both the old fixture-era `out_rows <= 128` cap and the subtler
mutation that reuses scale row 0 after crossing the boundary.
"""

import ctypes
import math
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 128
INTER = 128
ROWS = 129
E4M3_ONE = 0x38
E8M0_ONE = 127
E8M0_TWO = 128


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


def make_weight(rows, cols):
    data = bytearray(rows * cols)
    for r in range(rows):
        data[r * cols] = E4M3_ONE
    return data


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    with tempfile.TemporaryDirectory(prefix="v6-shared-rows-") as td:
        lib = os.path.join(td, "libexpert.so")
        cmd = [
            cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
            "-I", os.path.join(REPO, "src"),
            os.path.join(REPO, "src", "deepseek_v4_expert_ref.c"),
            os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
            os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
            os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
            "-lm", "-o", lib,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile shared-expert boundary replay")
            return 1

        so = ctypes.CDLL(lib)
        fn = so.waste_ds_v4_shared_expert_ref
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        fn.restype = ctypes.c_int

        x = (ctypes.c_float * K)(*([1.0] + [0.0] * (K - 1)))
        w1b = make_weight(INTER, K)
        w3b = make_weight(INTER, K)
        w2b = make_weight(ROWS, INTER)
        w1 = (ctypes.c_uint8 * len(w1b)).from_buffer_copy(w1b)
        w3 = (ctypes.c_uint8 * len(w3b)).from_buffer_copy(w3b)
        w2 = (ctypes.c_uint8 * len(w2b)).from_buffer_copy(w2b)
        s1 = (ctypes.c_uint8 * 1)(E8M0_ONE)
        s3 = (ctypes.c_uint8 * 1)(E8M0_ONE)

        def run(scale1):
            # ceil(129/128) * (128/128) == two w2 scale bytes.
            s2 = (ctypes.c_uint8 * 2)(E8M0_ONE, scale1)
            out = (ctypes.c_float * ROWS)()
            rc = fn(x, K, INTER, w1, s1, w3, s3, ctypes.c_float(10.0),
                    w2, s2, ROWS, None, None, None, out)
            return rc, list(out)

        rc, out = run(E8M0_TWO)
        if rc:
            print("FAIL: shared expert rejected 129 output rows")
            return 1
        if not all(math.isfinite(v) for v in out):
            print("FAIL: shared expert produced non-finite output")
            return 1
        if len(set(out[:128])) != 1:
            print("FAIL: identical first-grid rows did not produce identical outputs")
            return 1
        if out[128] == out[127]:
            print("FAIL: second w2 E8M0 scale-grid row was invisible")
            return 1

        rc2, muted = run(E8M0_ONE)
        if rc2:
            print("FAIL: shared expert rejected the scale-row mutation")
            return 1
        if muted[128] != muted[127]:
            print("FAIL: equalizing the second scale row did not equalize row 128")
            return 1
        if muted[128] == out[128]:
            print("FAIL: changing the second scale row did not change row 128")
            return 1

    print("PASS Gate H/V6 shared FP8 expert crosses the 128-row w2 scale-grid boundary")
    print(f"row127={out[127]:.9g} row128(scale=2)={out[128]:.9g} "
          f"row128(scale=1)={muted[128]:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
