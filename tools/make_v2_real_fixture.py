#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the first real-checkpoint Gate C/V2 projection fixture.

The fixture uses eight real rows from the pinned resident projection
``layers.0.attn.wq_a`` and its first real E8M0 scale row. Expected values are
computed by an independent Python implementation of the pinned official 0731
kernel equations -- it does not import or execute WASTE quantization code.

This is intentionally a tiny payload exercise (~41 KiB including input), not a
full-shard or full-checkpoint download.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
import shutil
import struct
import sys

import fetch_hf_tensor_slice as slice_fetch


WEIGHT = "layers.0.attn.wq_a.weight"
SCALE = "layers.0.attn.wq_a.scale"
ROWS = 8
K = 4096
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
    """Decode finite E4M3 while preserving the format's signed zero."""
    # Fraction is exact for nonzero values but has no signed-zero state.
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
        row_base = row * K

        # Normal order, matching the scalar C reference's block walk.
        for block in range(K_BLOCKS):
            start = block * BLOCK
            part = f32(0.0)
            exact_part = Fraction(0, 1)
            for lane in range(BLOCK):
                a_code = qx[start + lane]
                w_code = weight[row_base + start + lane]
                product = e4m3_fraction(a_code) * e4m3_fraction(w_code)
                part = f32(part + f32(float(product)))
                exact_part += product
            scale = e8m0_fraction(weight_scales[block]) * Fraction(act_scales[block])
            seq_acc = f32(seq_acc + f32(part * float(scale)))
            exact_acc += exact_part * scale

        # Reverse block/lane order as a cheap reduction-order perturbation.
        for block in reversed(range(K_BLOCKS)):
            start = block * BLOCK
            part = f32(0.0)
            for lane in reversed(range(BLOCK)):
                a = e4m3_float(qx[start + lane])
                w = e4m3_float(weight[row_base + start + lane])
                part = f32(part + f32(a * w))
            scale = f32(act_scales[block] * e8m0_float(weight_scales[block]))
            rev_acc = f32(rev_acc + f32(part * scale))

        sequential.append(seq_acc)
        reverse.append(rev_acc)
        exact.append(float(exact_acc))

    seq_bf16 = [bf16_bits(x) for x in sequential]
    rev_bf16 = [bf16_bits(x) for x in reverse]
    exact_bf16 = [bf16_bits(x) for x in exact]
    if not (seq_bf16 == rev_bf16 == exact_bf16):
        detail = [
            {
                "row": i,
                "sequential": f"0x{seq_bf16[i]:04x}",
                "reverse": f"0x{rev_bf16[i]:04x}",
                "exact": f"0x{exact_bf16[i]:04x}",
            }
            for i in range(ROWS)
            if not (seq_bf16[i] == rev_bf16[i] == exact_bf16[i])
        ]
        raise ValueError(
            "chosen fixture is not BF16-stable across reduction views: " + json.dumps(detail))
    return sequential, seq_bf16


def write_u16(path, values):
    with open(path, "wb") as f:
        for value in values:
            f.write(struct.pack("<H", value))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="header-only pinned 0731 snapshot")
    ap.add_argument("out_dir", help="fixture output directory")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    weight_path = os.path.join(args.out_dir, "weight-rows0-8.e4m3.bin")
    scale_path = os.path.join(args.out_dir, "weight-scale-row0.e8m0.bin")
    input_path = os.path.join(args.out_dir, "input.bf16.bin")
    expected_path = os.path.join(args.out_dir, "expected.bf16.bin")

    weight_info = slice_fetch.fetch_slice(
        args.snapshot, WEIGHT, f"0:{ROWS}", weight_path, token=args.token)
    scale_info = slice_fetch.fetch_slice(
        args.snapshot, SCALE, "0:1", scale_path, token=args.token)

    with open(weight_path, "rb") as f:
        weight = f.read()
    with open(scale_path, "rb") as f:
        weight_scales = f.read()

    input_bits, input_values = build_input()
    qx, act_scales = quantize_input(input_values)
    output_f32, output_bf16 = source_projection(
        weight, weight_scales, qx, act_scales)

    write_u16(input_path, input_bits)
    write_u16(expected_path, output_bf16)

    provenance = {
        "schema_version": 1,
        "fixture_class": "F3 official-source + real-checkpoint projection",
        "gate": "Gate C / V2",
        "evidence_state": "CHECKPOINT-VERIFIED_SOURCE-ORACLE",
        "model": weight_info["model"],
        "revision": weight_info["revision"],
        "operation": "layers.0.attn.wq_a quantized linear, first 8 output rows",
        "official_source_contract": {
            "kernel_path": "inference/kernel.py",
            "activation": "act_quant: K128 E4M3FN, max(amax,1e-4), power-of-two scale",
            "gemm": "fp8_gemm: K128 FP8 dot, activation_scale * weight_scale, FP32 accumulation, BF16 output"
        },
        "weight": weight_info,
        "weight_scale": scale_info,
        "input": {
            "dtype": "BF16",
            "shape": [1, K],
            "file": os.path.basename(input_path),
            "sha256": sha256_file(input_path),
            "formula": "(((lane*37+11)%31)-15)/16 * 2**((block%5)-2), then BF16 RN-even"
        },
        "activation_quantization": {
            "dtype": "E4M3FN",
            "block_k": BLOCK,
            "scale": "next_power_of_two(max(max(abs(x)),1e-4)/448)",
            "distinct_scale_values": sorted(set(act_scales)),
            "quantized_input_sha256": hashlib.sha256(bytes(qx)).hexdigest()
        },
        "expected": {
            "dtype": "BF16",
            "shape": [1, ROWS],
            "file": os.path.basename(expected_path),
            "sha256": sha256_file(expected_path),
            "bf16_hex": [f"0x{x:04x}" for x in output_bf16],
            "source_oracle_f32_before_bf16": output_f32,
            "reduction_stability": "same BF16 under sequential-f32, reverse-f32, and exact dyadic accumulation"
        },
        "non_claim": "Fixture executes pinned official algebra independently on CPU; it does not execute the official TileLang GPU kernel. GPU/backend parity remains a later backend validation concern."
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    # Slice sidecars are useful during generation, but provenance.json embeds
    # all required fields; remove duplicates from the final frozen fixture.
    for path in (weight_path + ".json", scale_path + ".json"):
        if os.path.exists(path):
            os.remove(path)

    print("Gate C real fixture built")
    print("expected BF16:", " ".join(provenance["expected"]["bf16_hex"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
