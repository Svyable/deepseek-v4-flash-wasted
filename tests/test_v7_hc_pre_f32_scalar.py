#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Pin F32 multiply/add ordering at the HyperConnection pre boundary.

The first real layer-4 chained fixture found a one-BF16 mismatch because an
independent Python oracle rounded the four HC coefficients to F32 but summed
`pre[j] * residual[j,d]` in Python double precision. The C/model path performs
the product/add sequence in F32. This tiny construction intentionally straddles
a BF16 boundary so the distinction cannot disappear again.
"""

import struct

PRE = [
    1.3260564804077148,
    0.370778352022171,
    -0.9432477355003357,
    -0.9716181755065918,
]
X = [
    -1.455226182937622,
    -0.9237891435623169,
    -0.4118974804878235,
    -1.9387576580047607,
]


def f32(v):
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", f32(v)))[0]


def bf16_bits(v):
    u = f32_bits(v)
    exp, frac = u & 0x7F800000, u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def hc_pre_f32(pre, x):
    acc = f32(0.0)
    for a, b in zip(pre, x):
        acc = f32(acc + f32(a * b))
    return acc


def main():
    pre = [f32(v) for v in PRE]
    x = [f32(v) for v in X]
    correct = hc_pre_f32(pre, x)
    double_sum = sum(a * b for a, b in zip(pre, x))
    got = bf16_bits(correct)
    wrong = bf16_bits(double_sum)
    if got != 0x37AD:
        print(f"FAIL: F32-sequential construction drifted: 0x{got:04x} != 0x37ad")
        return 1
    if wrong != 0x37AB:
        print(f"FAIL: double-precision mutation drifted: 0x{wrong:04x} != 0x37ab")
        return 1
    if got == wrong:
        print("FAIL: double accumulation is invisible at the BF16 boundary")
        return 1
    print("PASS V7 HyperConnection hc_pre requires sequential F32 multiply/add before BF16")
    print(f"f32-sequential=0x{got:04x} python-double=0x{wrong:04x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
