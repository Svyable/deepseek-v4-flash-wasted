#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the first real DeepSeek-V4 quantized-linear checkpoint fixture.

The fixture is intentionally bounded: one full 4096-wide input vector, eight
rows of the real layer-0 wq_a FP8 weight, and the matching first E8M0 scale-grid
row. Expected BF16 output is produced independently in Python from release-format
E4M3/E8M0 equations; WASTE runtime code is not imported or executed here.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from fractions import Fraction

import fetch_hf_tensor_slice as slice_fetch

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
K = 4096
ROWS = 8
BLOCK = 128
K_BLOCKS = K // BLOCK


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def bits_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bf16_bits(value):
    """Round binary32 to BF16, round-to-nearest-even."""
    u = f32_bits(value)
    exp = u & 0x7F800000
    frac = u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16_to_f32(bits):
    return bits_f32((bits & 0xFFFF) << 16)


def e4m3_fraction(code):
    sign = -1 if (code & 0x80) else 1
    mag = code & 0x7F
    if mag == 0x7F:
        raise ValueError("E4M3FN NaN is not valid in projection fixture")
    exp = (mag >> 3) & 0xF
    mant = mag & 0x7
    if exp == 0:
        # (mant/8) * 2^(1-7) = mant * 2^-9
        value = Fraction(mant, 512)
    else:
        # (8+mant)/8 * 2^(exp-7)
        power = exp - 10
        value = Fraction(8 + mant, 1)
        value = value * (2 ** power if power >= 0 else Fraction(1, 2 ** (-power)))
    return sign * value


def e4m3_float(code):
    """Decode finite E4M3 to Python float, preserving IEEE signed zero.

    `Fraction` is useful for exact nonzero release-format arithmetic but has no
    signed-zero representation. The E4M3 finite format does: 0x00 is +0 and
    0x80 is -0. Preserve that bit here so QAT/oracle paths match the scalar C
    decoder and the checkpoint runtime at exact-zero boundaries.
    """
    if (code & 0x7F) == 0:
        return -0.0 if (code & 0x80) else 0.0
    return float(e4m3_fraction(code))


def _finite_codes_for_sign(negative):
    sign = 0x80 if negative else 0
    return [sign | mag for mag in range(0x7F)]


def e4m3_encode(value):
    """Independent finite-E4M3 RN-even reference encoder."""
    value = f32(value)
    if math.isnan(value):
        return 0x7F
    negative = math.copysign(1.0, value) < 0.0
    if value == 0.0:
        return 0x80 if negative else 0x00
    value = max(-448.0, min(448.0, value))
    best = None
    best_err = None
    for code in _finite_codes_for_sign(negative):
        decoded = e4m3_float(code)
        err = abs(value - decoded)
        if best is None or err < best_err:
            best, best_err = code, err
        elif err == best_err:
            # RN-even: choose the representation whose retained LSB is zero.
            if (code & 1) == 0 and (best & 1) != 0:
                best = code
    return best


def e8m0_fraction(code):
    if code == 0xFF:
        raise ValueError("E8M0 NaN scale in real checkpoint fixture")
    exponent = code - 127
    return Fraction(2 ** exponent, 1) if exponent >= 0 else Fraction(1, 2 ** (-exponent))


def e8m0_float(code):
    return float(e8m0_fraction(code))


def next_pow2_scale(amax):
    """Pinned official fast_round_scale(max(amax,1e-4), 1/448)."""
    amax = f32(max(f32(amax), f32(1e-4)))
    ratio = f32(amax * f32(1.0 / 448.0))
    bits = f32_bits(ratio)
    exponent = ((bits >> 23) & 0xFF) - 127
    if bits & 0x7FFFFF:
        exponent += 1
    return math.ldexp(1.0, exponent)


def input_value(column):
    """Deterministic dyadic BF16-exact input with varied K128 amplitudes."""
    block = column // BLOCK
    lane = column % BLOCK
    # Permuted -15..15 pattern, all exact dyadic fractions. Block amplitude
    # cycles through 2^-2..2^2 so activation scales exercise several exponents.
    numerator = ((lane * 37 + 11) % 31) - 15
    amplitude_exp = (block % 5) - 2
    value = Fraction(numerator, 16)
    value *= 2 ** amplitude_exp if amplitude_exp >= 0 else Fraction(1, 2 ** (-amplitude_exp))
    return float(value)


def build_input():
    bits = []
    values = []
    for column in range(K):
        b = bf16_bits(input_value(column))
        bits.append(b)
        values.append(bf16_to_f32(b))
    return bits, values


def quantize_input(values):
    q = []
    scales = []
    for block in range(K_BLOCKS):
        start = block * BLOCK
        xb = values[start:start + BLOCK]
        scale = next_pow2_scale(max(abs(x) for x in xb))
        scales.append(scale)
        for x in xb:
            normalized = f32(x / scale)
            normalized = max(-448.0, min(448.0, normalized))
            q.append(e4m3_encode(normalized))
    return q, scales


def source_projection(weight, weight_scales, qx, act_scales):
    if len(weight) != ROWS * K:
        raise ValueError(f"weight slice is {len(weight)} bytes, want {ROWS*K}")
    if len(weight_scales) != K_BLOCKS:
        raise ValueError(f"scale row is {len(weight_scales)} bytes, want {K_BLOCKS}")

    # Three accumulation views. The committed BF16 oracle must be invariant
    # across them, reducing sensitivity to the GPU GEMM's internal reduction
    # order while still testing the official block-scaling algebra.
    sequential = []
    reverse = []
    exact = []

    for row in range(ROWS):
        seq_acc = f32(0.0)
        rev_acc = f32(0.0)
        exact_acc = Fraction(0, 1)
        for block in range(K_BLOCKS):
            start = block * BLOCK
            part = f32(0.0)
            rpart = f32(0.0)
            exact_part = Fraction(0, 1)
            for j in range(BLOCK):
                a = e4m3_float(qx[start + j])
                w = e4m3_float(weight[row*K + start + j])
                part = f32(part + f32(a*w))
                exact_part += e4m3_fraction(qx[start + j]) * e4m3_fraction(weight[row*K + start + j])
            for j in reversed(range(BLOCK)):
                a = e4m3_float(qx[start + j])
                w = e4m3_float(weight[row*K + start + j])
                rpart = f32(rpart + f32(a*w))
            ws = e8m0_float(weight_scales[block])
            seq_acc = f32(seq_acc + f32(part * f32(act_scales[block] * ws)))
            rev_acc = f32(rev_acc + f32(rpart * f32(act_scales[block] * ws)))
            exact_acc += exact_part * Fraction.from_float(act_scales[block]) * e8m0_fraction(weight_scales[block])
        sequential.append(bf16_bits(seq_acc))
        reverse.append(bf16_bits(rev_acc))
        exact.append(bf16_bits(float(exact_acc)))

    if sequential != reverse or sequential != exact:
        raise ValueError(
            "bounded projection is reduction-order-sensitive at BF16; "
            f"sequential={sequential}, reverse={reverse}, exact={exact}")
    return sequential


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2-real-") as td:
        w_path = os.path.join(td, "weight.bin")
        s_path = os.path.join(td, "scale.bin")
        w_info = slice_fetch.fetch_slice(
            args.snapshot, "layers.0.attn.wq_a.weight", f"0:{ROWS}", w_path,
            token=args.token)
        s_info = slice_fetch.fetch_slice(
            args.snapshot, "layers.0.attn.wq_a.scale", "0:1", s_path,
            token=args.token)
        if w_info["revision"] != REV or s_info["revision"] != REV:
            raise ValueError("checkpoint revision drifted")
        if w_info["dtype"] != "F8_E4M3" or w_info["shape"] != [1024,4096]:
            raise ValueError(f"wq_a contract drifted: {w_info['dtype']} {w_info['shape']}")
        if s_info["dtype"] != "F8_E8M0" or s_info["shape"] != [8,32]:
            raise ValueError(f"wq_a.scale contract drifted: {s_info['dtype']} {s_info['shape']}")
        weight = open(w_path, "rb").read()
        scales = open(s_path, "rb").read()

    input_bits, input_values = build_input()
    qx, act_scales = quantize_input(input_values)
    expected = source_projection(weight, scales, qx, act_scales)

    input_path = os.path.join(args.out_dir, "input.bf16.bin")
    weight_path = os.path.join(args.out_dir, "weight-rows0-8.e4m3.bin")
    scale_path = os.path.join(args.out_dir, "weight-scale-row0.e8m0.bin")
    expected_path = os.path.join(args.out_dir, "expected.bf16.bin")
    write_u16(input_path, input_bits)
    shutil.copyfile(w_path, weight_path) if os.path.exists(w_path) else None
    # weight/scales were in a TemporaryDirectory, so persist from bytes instead.
    with open(weight_path, "wb") as f:
        f.write(weight)
    with open(scale_path, "wb") as f:
        f.write(scales)
    write_u16(expected_path, expected)

    prov = {
        "schema_version": 1,
        "gate": "Gate C / V2 real quantized projection",
        "evidence_state": "CHECKPOINT-WEIGHTS + INDEPENDENT-SOURCE-ORACLE",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REV,
        "tensor": w_info["tensor"],
        "weight": {
            "file": os.path.basename(weight_path),
            "dtype": w_info["dtype"], "shape": w_info["shape"],
            "rows": [w_info["row_start"], w_info["row_end"]],
            "absolute_range": [w_info["absolute_start"], w_info["absolute_end"]],
            "payload_sha256": w_info["payload_sha256"],
        },
        "scale": {
            "file": os.path.basename(scale_path),
            "dtype": s_info["dtype"], "shape": s_info["shape"],
            "rows": [s_info["row_start"], s_info["row_end"]],
            "absolute_range": [s_info["absolute_start"], s_info["absolute_end"]],
            "payload_sha256": s_info["payload_sha256"],
        },
        "input": {
            "file": os.path.basename(input_path), "dtype": "BF16", "shape": [K],
            "sha256": sha256_file(input_path),
        },
        "expected": {
            "file": os.path.basename(expected_path), "dtype": "BF16", "shape": [ROWS],
            "sha256": sha256_file(expected_path),
            "bf16_hex": [f"0x{x:04x}" for x in expected],
        },
        "source_contract": {
            "activation_quant": "K128 dynamic E4M3 with fast-round power-of-two scale",
            "weight": "checkpoint-native E4M3 + E8M0 128x128 blocks",
            "output": "f32 block accumulation -> BF16",
            "reduction_guard": "sequential, reversed and exact rational views must agree at BF16",
        },
        "oracle_independence": {
            "producer": "tools/make_v2_real_fixture.py using Fraction/int/struct equations",
            "shares_waste_runtime_helpers": False,
        },
        "non_claims": [
            "eight output rows only",
            "does not prove attention composition",
            "does not complete Gate C/V2 alone",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Gate C real projection fixture built")
    print("expected:", " ".join(prov["expected"]["bf16_hex"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
