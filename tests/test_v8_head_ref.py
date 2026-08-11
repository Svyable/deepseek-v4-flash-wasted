#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Independent model-free differential for the Gate-I scalar head primitive."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HC = 4
DIM = 37
FLAT = HC * DIM
EPS = 1e-6


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


def bf16_bits(x):
    u = f32_bits(x)
    exp = u & 0x7F800000
    frac = u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return u >> 16
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return u >> 16


def bf16_float(x):
    return f32(struct.unpack("<f", struct.pack("<I", (int(x) & 0xFFFF) << 16))[0])


_libm_name = ctypes.util.find_library("m")
_libm = ctypes.CDLL(_libm_name) if _libm_name else ctypes.CDLL(None)
_expf = _libm.expf
_expf.argtypes = [ctypes.c_float]
_expf.restype = ctypes.c_float
_sqrtf = _libm.sqrtf
_sqrtf.argtypes = [ctypes.c_float]
_sqrtf.restype = ctypes.c_float


def expf(x):
    return f32(_expf(ctypes.c_float(f32(x))))


def sqrtf(x):
    return f32(_sqrtf(ctypes.c_float(f32(x))))


def sigmoidf(x):
    x = f32(x)
    if x >= 0.0:
        z = expf(f32(-x))
        return f32(f32(1.0) / f32(f32(1.0) + z))
    z = expf(x)
    return f32(z / f32(f32(1.0) + z))


def py_hc_head(x, fn, scale, base):
    total = f32(0.0)
    for v in x:
        total = f32(total + f32(v * v))
    mean = f32(total / f32(float(FLAT)))
    rsqrt = f32(f32(1.0) / sqrtf(f32(mean + f32(EPS))))
    mixes = []
    pre = []
    for row in range(HC):
        acc = f32(0.0)
        off = row * FLAT
        for c in range(FLAT):
            acc = f32(acc + f32(fn[off + c] * x[c]))
        mix = f32(acc * rsqrt)
        affine = f32(f32(mix * scale[0]) + base[row])
        mixes.append(mix)
        pre.append(f32(sigmoidf(affine) + f32(EPS)))
    out = []
    for d in range(DIM):
        acc = f32(0.0)
        for stream in range(HC):
            acc = f32(acc + f32(pre[stream] * x[stream * DIM + d]))
        out.append(acc)
    return out, mixes, pre


def py_rms(x, weight):
    total = f32(0.0)
    for v in x:
        total = f32(total + f32(v * v))
    mean = f32(total / f32(float(DIM)))
    rsqrt = f32(f32(1.0) / sqrtf(f32(mean + f32(EPS))))
    return [f32(weight[d] * f32(x[d] * rsqrt)) for d in range(DIM)]


def py_dot(x, row):
    acc = f32(0.0)
    for a, b in zip(x, row):
        acc = f32(acc + f32(a * b))
    return acc


def build_lib(td):
    cc = os.environ.get("CC") or next(
        (shutil.which(x) for x in ("cc", "gcc", "clang") if shutil.which(x)), None)
    if not cc:
        return None
    out = os.path.join(td, "libv8head.so")
    cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           "-ffp-contract=off", "-shared", "-fPIC", "-I", os.path.join(REPO, "src"),
           os.path.join(REPO, "src", "deepseek_v4_head_ref.c"), "-lm", "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return ctypes.CDLL(out)


def arr(values):
    return (ctypes.c_float * len(values))(*[f32(x) for x in values])


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if f32_bits(x) != f32_bits(y):
            return i
    return None


def main():
    x = [f32((((i * 17) % 101) - 50) / 37.0) for i in range(FLAT)]
    # Keep weights small enough to avoid saturation while making every index
    # matter. This producer is independent of the C implementation.
    fn = [f32((((i * 13) % 97) - 48) / 509.0) for i in range(HC * FLAT)]
    scale = [f32(0.73)]
    base = [f32(-0.31), f32(0.12), f32(0.44), f32(-0.08)]
    norm_weight = [bf16_float(bf16_bits(0.85 + ((i * 7) % 19) / 53.0))
                   for i in range(DIM)]
    head_row = [bf16_float(bf16_bits((((i * 11) % 43) - 21) / 29.0))
                for i in range(DIM)]

    want_h, want_m, want_p = py_hc_head(x, fn, scale, base)
    hc_bf16 = [bf16_float(bf16_bits(v)) for v in want_h]
    want_norm = py_rms(hc_bf16, norm_weight)
    norm_bf16 = [bf16_float(bf16_bits(v)) for v in want_norm]
    want_logit = py_dot(norm_bf16, head_row)

    with tempfile.TemporaryDirectory(prefix="v8-head-ref-") as td:
        lib = build_lib(td)
        if lib is None:
            print("SKIP: no C compiler available")
            return 77

        hc = lib.waste_ds_v4_hc_head_ref
        hc.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float,
                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float)]
        hc.restype = ctypes.c_int
        rms = lib.waste_ds_v4_final_rmsnorm_ref
        rms.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_size_t, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
        rms.restype = ctypes.c_int
        dot = lib.waste_ds_v4_head_row_ref
        dot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
        dot.restype = ctypes.c_int

        got_h = (ctypes.c_float * DIM)()
        got_m = (ctypes.c_float * HC)()
        got_p = (ctypes.c_float * HC)()
        if hc(arr(x), DIM, arr(fn), arr(scale), arr(base), ctypes.c_float(EPS),
              ctypes.c_float(EPS), got_h, got_m, got_p):
            raise AssertionError("C hc_head rejected valid geometry")
        for label, got, want in (("hc output", list(got_h), want_h),
                                 ("hc mixes", list(got_m), want_m),
                                 ("hc pre", list(got_p), want_p)):
            idx = first_diff(got, want)
            if idx is not None:
                raise AssertionError(
                    f"{label} differs at {idx}: {f32_bits(got[idx]):08x}!={f32_bits(want[idx]):08x}")

        got_norm = (ctypes.c_float * DIM)()
        if rms(arr(hc_bf16), arr(norm_weight), DIM, ctypes.c_float(EPS), got_norm):
            raise AssertionError("C final RMSNorm rejected valid geometry")
        idx = first_diff(list(got_norm), want_norm)
        if idx is not None:
            raise AssertionError(
                f"RMSNorm differs at {idx}: {f32_bits(got_norm[idx]):08x}!={f32_bits(want_norm[idx]):08x}")

        got_logit = ctypes.c_float()
        if dot(arr(norm_bf16), arr(head_row), DIM, ctypes.byref(got_logit)):
            raise AssertionError("C head row rejected valid geometry")
        if f32_bits(got_logit.value) != f32_bits(want_logit):
            raise AssertionError(
                f"head row differs: {f32_bits(got_logit.value):08x}!={f32_bits(want_logit):08x}")

    # Mutation 1: treating hc_head_fn as column-major must visibly change the
    # scalar coefficients/output for this independent fixture.
    bad_fn = []
    for row in range(HC):
        for c in range(FLAT):
            bad_fn.append(fn[c * HC + row])
    bad_h, _, _ = py_hc_head(x, bad_fn, scale, base)
    if first_diff(bad_h, want_h) is None:
        raise AssertionError("transposed hc_head_fn mutation was invisible")

    # Mutation 2: applying stream weights to the next stream is a plausible
    # collapse-axis bug that must not survive the fixture.
    bad_collapse = []
    for d in range(DIM):
        acc = f32(0.0)
        for stream in range(HC):
            acc = f32(acc + f32(want_p[(stream + 1) % HC] * x[stream * DIM + d]))
        bad_collapse.append(acc)
    if first_diff(bad_collapse, want_h) is None:
        raise AssertionError("rotated HC stream mutation was invisible")

    # Mutation 3: RMSNorm computes in F32 and casts only at the end. Rounding
    # the normalized activation to BF16 before multiplying the weight must move
    # at least one result.
    total = f32(0.0)
    for v in hc_bf16:
        total = f32(total + f32(v * v))
    mean = f32(total / f32(float(DIM)))
    rsqrt = f32(f32(1.0) / sqrtf(f32(mean + f32(EPS))))
    bad_norm = []
    for d in range(DIM):
        rounded = bf16_float(bf16_bits(f32(hc_bf16[d] * rsqrt)))
        bad_norm.append(f32(norm_weight[d] * rounded))
    if first_diff(bad_norm, want_norm) is None:
        raise AssertionError("premature BF16 RMSNorm mutation was invisible")

    # Mutation 4: each checkpoint row is one vocabulary output. Reversing its
    # hidden dimension simulates a silent orientation error.
    if f32_bits(py_dot(norm_bf16, list(reversed(head_row)))) == f32_bits(want_logit):
        raise AssertionError("reversed vocabulary-row mutation was invisible")

    print("PASS Gate I/V8 scalar head primitive: independent Python == C F32 bits")
    print("mutations killed: hc-fn transpose, stream rotation, BF16 RMS inner, head-row reversal")
    print("NO CHECKPOINT PAYLOAD / NO FINAL-MODEL LOGIT CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
