#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Require the independent V7 F32 HC oracle to match the C reference bitwise.

This is model-free and small, but it exercises the full dtype sequence that the
real layer-4 chain exposed: F32 residual norm, 24 mix dot-products, stable
sigmoid, row-softmax + 20 Sinkhorn pairs, HC-pre reduction, and HC-post. The
Python producer links no WASTE source; only this test loads the C implementation
for differential comparison.
"""

import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import v7_hc_oracle as oracle  # noqa: E402

DIM = 37
HC = 4
MIX = 24
FLAT = HC * DIM


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


def bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def as_cf(values):
    return (ctypes.c_float * len(values))(*values)


def same_bits(a, b, label):
    if len(a) != len(b):
        raise AssertionError(f"{label}: length mismatch")
    for i, (x, y) in enumerate(zip(a, b)):
        if bits(x) != bits(y):
            raise AssertionError(
                f"{label}[{i}]: 0x{bits(x):08x} != 0x{bits(y):08x} ({x} != {y})")


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    # Nontrivial deterministic F32 values; not symmetric and not tiny enough to
    # turn the Sinkhorn/HC reductions into vacuous zeros.
    x = [oracle.f32((((i * 43 + 17) % 211) - 105) / 37.0) for i in range(FLAT)]
    fn = [oracle.f32((((i * 29 + 11) % 257) - 128) / 613.0)
          for i in range(MIX * FLAT)]
    scale = [oracle.f32(0.73), oracle.f32(1.09), oracle.f32(0.61)]
    base = [oracle.f32((((i * 19 + 5) % 53) - 26) / 31.0) for i in range(MIX)]
    branch = [oracle.f32((((i * 17 + 3) % 101) - 50) / 23.0) for i in range(DIM)]

    py_pre, py_post, py_comb, py_mixes = oracle.params(
        x, fn, scale, base, dim=DIM)
    py_y = oracle.pre_y(x, py_pre, dim=DIM)
    py_after = oracle.post_y(branch, x, py_post, py_comb, dim=DIM)

    with tempfile.TemporaryDirectory(prefix="v7-hc-oracle-") as td:
        lib = os.path.join(td, "libhc.so")
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", "-shared", "-fPIC",
               "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"),
               "-lm", "-o", lib]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            sys.stderr.write(p.stdout + p.stderr)
            print("FAIL: could not compile HC differential reference")
            return 1
        so = ctypes.CDLL(lib)
        pre_fn = so.waste_ds_v4_hc_pre_ref
        pre_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint,
                           ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        pre_fn.restype = ctypes.c_int
        post_fn = so.waste_ds_v4_hc_post_ref
        post_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        post_fn.restype = ctypes.c_int

        c_y = (ctypes.c_float * DIM)()
        c_mixes = (ctypes.c_float * MIX)()
        c_pre = (ctypes.c_float * HC)()
        c_post = (ctypes.c_float * HC)()
        c_comb = (ctypes.c_float * (HC * HC))()
        rc = pre_fn(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base),
                    ctypes.c_float(1e-6), 20, ctypes.c_float(1e-6),
                    c_y, c_mixes, c_pre, c_post, c_comb)
        if rc:
            print("FAIL: C HC-pre rejected synthetic input")
            return 1

        same_bits(py_mixes, list(c_mixes), "mixes")
        same_bits(py_pre, list(c_pre), "pre")
        same_bits(py_post, list(c_post), "post")
        same_bits([v for row in py_comb for v in row], list(c_comb), "comb")
        same_bits(py_y, list(c_y), "hc_pre_y")

        c_after = (ctypes.c_float * FLAT)()
        rc = post_fn(as_cf(branch), as_cf(x), DIM, c_post, c_comb, c_after)
        if rc:
            print("FAIL: C HC-post rejected synthetic input")
            return 1
        same_bits(py_after, list(c_after), "hc_post_y")

    print("PASS V7 independent F32 HyperConnection oracle matches C bit-for-bit")
    print("checked: 24 mixes + 4 pre + 4 post + 16 comb + 37 pre-y + 148 post-y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
