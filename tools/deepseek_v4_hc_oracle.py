#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Independent float32 HyperConnection equations for DeepSeek V4.

This module deliberately does not import or link WASTE source. It mirrors the
model's scalar tensor boundary using Python control flow plus the platform's
``expf``/``sqrtf`` from libm. Every multiply, add, divide, normalization and
Sinkhorn step is rounded to IEEE float32 in the same operation order as the
scalar model path.

Ordinary Python double-precision reductions are *not* a stronger oracle here:
they change model dtype semantics and can move BF16 boundary values. This file
exists so acquisition and replay fixtures can prove the actual F32 contract
without sharing WASTE implementation code.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct

HC = 4
MIX = 24


def f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(v)))[0]


_libm_name = ctypes.util.find_library("m")
_libm = ctypes.CDLL(_libm_name) if _libm_name else ctypes.CDLL(None)
_expf = _libm.expf
_expf.argtypes = [ctypes.c_float]
_expf.restype = ctypes.c_float
_sqrtf = _libm.sqrtf
_sqrtf.argtypes = [ctypes.c_float]
_sqrtf.restype = ctypes.c_float


def expf(v: float) -> float:
    return f32(_expf(ctypes.c_float(f32(v))))


def sqrtf(v: float) -> float:
    return f32(_sqrtf(ctypes.c_float(f32(v))))


def sigmoidf(v: float) -> float:
    x = f32(v)
    if x >= 0.0:
        z = expf(f32(-x))
        return f32(f32(1.0) / f32(f32(1.0) + z))
    z = expf(x)
    return f32(z / f32(f32(1.0) + z))


def split(mixes, scale, base, *, iters: int = 20, eps: float = 1e-6):
    if len(mixes) != MIX or len(scale) != 3 or len(base) != MIX:
        raise ValueError("HyperConnection split expects mixes/base=24 and scale=3")
    if iters <= 0 or not eps > 0.0:
        raise ValueError("invalid Sinkhorn contract")
    e = f32(eps)
    sc = [f32(v) for v in scale]
    bs = [f32(v) for v in base]
    mx = [f32(v) for v in mixes]

    pre = []
    post = []
    for j in range(HC):
        v = f32(f32(mx[j] * sc[0]) + bs[j])
        pre.append(f32(sigmoidf(v) + e))
        v = f32(f32(mx[j + HC] * sc[1]) + bs[j + HC])
        post.append(f32(f32(2.0) * sigmoidf(v)))

    comb = [[f32(0.0)] * HC for _ in range(HC)]
    for j in range(HC):
        logits = []
        for k in range(HC):
            p = 2 * HC + j * HC + k
            logits.append(f32(f32(mx[p] * sc[2]) + bs[p]))
        row_max = max(logits)
        ex = [expf(f32(v - row_max)) for v in logits]
        total = f32(0.0)
        for v in ex:
            total = f32(total + v)
        for k in range(HC):
            comb[j][k] = f32(f32(ex[k] / total) + e)

    for k in range(HC):
        total = f32(0.0)
        for j in range(HC):
            total = f32(total + comb[j][k])
        denom = f32(total + e)
        for j in range(HC):
            comb[j][k] = f32(comb[j][k] / denom)

    for _ in range(1, iters):
        for j in range(HC):
            total = f32(0.0)
            for k in range(HC):
                total = f32(total + comb[j][k])
            denom = f32(total + e)
            for k in range(HC):
                comb[j][k] = f32(comb[j][k] / denom)
        for k in range(HC):
            total = f32(0.0)
            for j in range(HC):
                total = f32(total + comb[j][k])
            denom = f32(total + e)
            for j in range(HC):
                comb[j][k] = f32(comb[j][k] / denom)
    return pre, post, comb


def params(x, fn, scale, base, *, dim: int | None = None,
           norm_eps: float = 1e-6, hc_eps: float = 1e-6, iters: int = 20):
    if dim is None:
        if len(x) % HC:
            raise ValueError("residual length is not divisible by HC")
        dim = len(x) // HC
    flat = HC * dim
    if len(x) != flat or len(fn) != MIX * flat:
        raise ValueError("HyperConnection fn/input geometry mismatch")

    xf = [f32(v) for v in x]
    ff = [f32(v) for v in fn]
    total = f32(0.0)
    for v in xf:
        total = f32(total + f32(v * v))
    mean = f32(total / f32(float(flat)))
    shifted = f32(mean + f32(norm_eps))
    root = sqrtf(shifted)
    rsqrt = f32(f32(1.0) / root)

    mixes = []
    for row in range(MIX):
        acc = f32(0.0)
        off = row * flat
        for c in range(flat):
            acc = f32(acc + f32(ff[off + c] * xf[c]))
        mixes.append(f32(acc * rsqrt))
    pre, post, comb = split(mixes, scale, base, iters=iters, eps=hc_eps)
    return pre, post, comb, mixes


def pre_y(x, pre, *, dim: int | None = None):
    if dim is None:
        if len(x) % HC:
            raise ValueError("residual length is not divisible by HC")
        dim = len(x) // HC
    if len(x) != HC * dim or len(pre) != HC:
        raise ValueError("HyperConnection pre geometry mismatch")
    xf = [f32(v) for v in x]
    pf = [f32(v) for v in pre]
    out = []
    for d in range(dim):
        acc = f32(0.0)
        for j in range(HC):
            acc = f32(acc + f32(pf[j] * xf[j * dim + d]))
        out.append(acc)
    return out


def post_y(branch, residual, post, comb, *, dim: int | None = None):
    if dim is None:
        dim = len(branch)
    if len(branch) != dim or len(residual) != HC * dim or len(post) != HC:
        raise ValueError("HyperConnection post geometry mismatch")
    bf = [f32(v) for v in branch]
    rf = [f32(v) for v in residual]
    pf = [f32(v) for v in post]
    cf = [[f32(v) for v in row] for row in comb]
    out = [f32(0.0)] * (HC * dim)
    for k in range(HC):
        for d in range(dim):
            acc = f32(pf[k] * bf[d])
            for j in range(HC):
                acc = f32(acc + f32(cf[j][k] * rf[j * dim + d]))
            out[k * dim + d] = acc
    return out
