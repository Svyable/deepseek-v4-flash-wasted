#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build a real layer-3 expert-2 routed-expert primitive fixture.

The fixture reuses the already-frozen learned-router input and its top routing
weight. It fetches the full real w1/w3 packed FP4 matrices for expert 2 plus
only the first eight output rows of w2. That is enough to evaluate all 2,048
real SwiGLU intermediates and eight true final output dimensions while keeping
frozen checkpoint payload under ~9 MiB.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
import shutil
import struct

import fetch_hf_tensor_slice as slice_fetch

LAYER = 3
EXPERT = 2
INPUT = 4096
INTER = 2048
OUT_ROWS = 8
LIMIT = 10.0
ROUTER_FIXTURE = os.path.join(
    "tests", "fixtures", "deepseek_v4", "v3_router_real")

BASE = f"layers.{LAYER}.ffn.experts.{EXPERT}"
W1 = BASE + ".w1.weight"
S1 = BASE + ".w1.scale"
W3 = BASE + ".w3.weight"
S3 = BASE + ".w3.scale"
W2 = BASE + ".w2.weight"
S2 = BASE + ".w2.scale"


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


def bits_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def bf16_bits(x):
    u = f32_bits(x)
    exp = u & 0x7F800000
    frac = u & 0x007FFFFF
    if exp == 0x7F800000:
        if frac:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16_to_f32(b):
    return bits_f32((b & 0xFFFF) << 16)


def e2m1_fraction(code):
    table = (
        Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
        Fraction(2), Fraction(3), Fraction(4), Fraction(6),
        Fraction(0), Fraction(-1, 2), Fraction(-1), Fraction(-3, 2),
        Fraction(-2), Fraction(-3), Fraction(-4), Fraction(-6),
    )
    return table[code & 0xF]


def e2m1_float(code):
    return float(e2m1_fraction(code))


def e4m3_fraction(code):
    sign = -1 if code & 0x80 else 1
    mag = code & 0x7F
    if mag == 0x7F:
        raise ValueError("E4M3FN NaN in fixture")
    exp = (mag >> 3) & 0xF
    mant = mag & 7
    if exp == 0:
        value = Fraction(mant, 512)
    else:
        power = exp - 10
        value = Fraction(8 + mant)
        value *= 2 ** power if power >= 0 else Fraction(1, 2 ** (-power))
    return sign * value


def e4m3_float(code):
    return float(e4m3_fraction(code))


def e4m3_encode(value):
    value = f32(value)
    if math.isnan(value):
        return 0x7F
    neg = math.copysign(1.0, value) < 0
    if value == 0.0:
        return 0x80 if neg else 0
    value = max(-448.0, min(448.0, value))
    sign = 0x80 if neg else 0
    best, best_err = None, None
    for mag in range(0x7F):
        code = sign | mag
        err = abs(value - e4m3_float(code))
        if best is None or err < best_err or (
                err == best_err and not (code & 1) and (best & 1)):
            best, best_err = code, err
    return best


def e8m0_fraction(code):
    if code == 0xFF:
        raise ValueError("E8M0 NaN in checkpoint expert scale")
    e = code - 127
    return Fraction(2 ** e) if e >= 0 else Fraction(1, 2 ** (-e))


def e8m0_float(code):
    return float(e8m0_fraction(code))


def next_pow2_scale(values):
    amax = f32(max(max(abs(x) for x in values), f32(1e-4)))
    ratio = f32(amax * f32(1.0 / 448.0))
    u = f32_bits(ratio)
    e = ((u >> 23) & 0xFF) - 127
    if u & 0x7FFFFF:
        e += 1
    return math.ldexp(1.0, e)


def quantize_activation(values):
    if len(values) % 128:
        raise ValueError("activation K must divide into K128")
    q = []
    scales = []
    for base in range(0, len(values), 128):
        block = values[base:base + 128]
        scale = next_pow2_scale(block)
        scales.append(scale)
        for x in block:
            q.append(e4m3_encode(max(-448.0, min(448.0, f32(x / scale)))))
    return q, scales


def fp4_code(row, logical_k):
    byte = row[logical_k // 2]
    return (byte >> 4) & 0xF if logical_k & 1 else byte & 0xF


def fp4_linear_rows(packed, scales, rows, k, x, require_stable=True):
    """Independent official-shaped FP8-activation x FP4-weight linear."""
    qx, act_scales = quantize_activation(x)
    packed_row = k // 2
    scale_row = k // 32
    if len(packed) != rows * packed_row or len(scales) != rows * scale_row:
        raise ValueError("FP4 matrix payload dimensions do not match")

    outputs = []
    exact_outputs = []
    reverse_outputs = []
    for r in range(rows):
        wr = packed[r * packed_row:(r + 1) * packed_row]
        sr = scales[r * scale_row:(r + 1) * scale_row]
        acc = f32(0.0)
        exact_acc = Fraction(0)
        parts = []
        for wb in range(scale_row):
            base = wb * 32
            ab = base // 128
            part = f32(0.0)
            exact_part = Fraction(0)
            for j in range(32):
                kk = base + j
                a = e4m3_fraction(qx[kk])
                b = e2m1_fraction(fp4_code(wr, kk))
                product = a * b
                part = f32(part + f32(float(product)))
                exact_part += product
            scale = Fraction(act_scales[ab]) * e8m0_fraction(sr[wb])
            scaled = f32(part * float(scale))
            parts.append(scaled)
            acc = f32(acc + scaled)
            exact_acc += exact_part * scale
        rev = f32(0.0)
        for part in reversed(parts):
            rev = f32(rev + part)
        outputs.append(acc)
        reverse_outputs.append(rev)
        exact_outputs.append(float(exact_acc))

    if require_stable:
        for r in range(rows):
            a, b, c = map(bf16_bits, (
                outputs[r], reverse_outputs[r], exact_outputs[r]))
            if not (a == b == c):
                raise ValueError(
                    f"row {r} is not BF16-stable across reduction views: "
                    f"0x{a:04x} 0x{b:04x} 0x{c:04x}")
    return [bf16_to_f32(bf16_bits(x)) for x in outputs]


def silu_f32(x):
    x = f32(x)
    if x >= 0:
        z = f32(math.exp(-x))
        return f32(x / f32(1.0 + z))
    z = f32(math.exp(x))
    return f32(f32(x * z) / f32(1.0 + z))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[f32(v) for v in values]))


def read_router_input_and_weight(repo_root):
    root = os.path.join(repo_root, ROUTER_FIXTURE)
    with open(os.path.join(root, "provenance.json"), encoding="utf-8") as f:
        p = json.load(f)
    if p["learned"]["ids"][0] != EXPERT:
        raise ValueError(
            f"router fixture top learned expert changed: {p['learned']['ids'][0]}")
    with open(os.path.join(root, "input.bf16.bin"), "rb") as f:
        raw = f.read()
    bits = struct.unpack(f"<{INPUT}H", raw)
    x = [bf16_to_f32(b) for b in bits]
    with open(os.path.join(root, "learned-weights.f32.bin"), "rb") as f:
        weights = struct.unpack("<6f", f.read())
    return raw, x, f32(weights[0]), p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_input, x, route_weight, router_prov = read_router_input_and_weight(repo_root)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = {
        "w1": os.path.join(args.out_dir, "w1.fp4.bin"),
        "s1": os.path.join(args.out_dir, "w1-scale.e8m0.bin"),
        "w3": os.path.join(args.out_dir, "w3.fp4.bin"),
        "s3": os.path.join(args.out_dir, "w3-scale.e8m0.bin"),
        "w2": os.path.join(args.out_dir, "w2-rows0-8.fp4.bin"),
        "s2": os.path.join(args.out_dir, "w2-scale-rows0-8.e8m0.bin"),
    }
    infos = {
        "w1": slice_fetch.fetch_slice(args.snapshot, W1, None, paths["w1"], token=args.token),
        "s1": slice_fetch.fetch_slice(args.snapshot, S1, None, paths["s1"], token=args.token),
        "w3": slice_fetch.fetch_slice(args.snapshot, W3, None, paths["w3"], token=args.token),
        "s3": slice_fetch.fetch_slice(args.snapshot, S3, None, paths["s3"], token=args.token),
        "w2": slice_fetch.fetch_slice(args.snapshot, W2, f"0:{OUT_ROWS}", paths["w2"], token=args.token),
        "s2": slice_fetch.fetch_slice(args.snapshot, S2, f"0:{OUT_ROWS}", paths["s2"], token=args.token),
    }
    expected = {
        "w1": ("I8", [INTER, INPUT // 2]),
        "s1": ("F8_E8M0", [INTER, INPUT // 32]),
        "w3": ("I8", [INTER, INPUT // 2]),
        "s3": ("F8_E8M0", [INTER, INPUT // 32]),
        "w2": ("I8", [4096, INTER // 2]),
        "s2": ("F8_E8M0", [4096, INTER // 32]),
    }
    for key, (dtype, shape) in expected.items():
        if infos[key]["dtype"] != dtype or infos[key]["shape"] != shape:
            raise ValueError(
                f"{key}: got {infos[key]['dtype']} {infos[key]['shape']}, "
                f"expected {dtype} {shape}")

    payload = {}
    for key, path in paths.items():
        with open(path, "rb") as f:
            payload[key] = f.read()

    gate = fp4_linear_rows(payload["w1"], payload["s1"], INTER, INPUT, x)
    up = fp4_linear_rows(payload["w3"], payload["s3"], INTER, INPUT, x)

    hidden_bits = []
    gate_clamped = []
    up_clamped = []
    for g, u in zip(gate, up):
        gc = min(g, LIMIT)
        uc = max(-LIMIT, min(LIMIT, u))
        gate_clamped.append(gc)
        up_clamped.append(uc)
        h = f32(silu_f32(gc) * uc)
        h = f32(h * route_weight)
        hidden_bits.append(bf16_bits(h))
    hidden = [bf16_to_f32(b) for b in hidden_bits]

    out = fp4_linear_rows(
        payload["w2"], payload["s2"], OUT_ROWS, INTER, hidden,
        require_stable=True)
    out_bits = [bf16_bits(v) for v in out]

    input_path = os.path.join(args.out_dir, "input.bf16.bin")
    hidden_path = os.path.join(args.out_dir, "hidden-after-route.bf16.bin")
    gate_path = os.path.join(args.out_dir, "gate-clamped.f32.bin")
    up_path = os.path.join(args.out_dir, "up-clamped.f32.bin")
    out_path = os.path.join(args.out_dir, "expected-out8.bf16.bin")
    with open(input_path, "wb") as f:
        f.write(raw_input)
    write_u16(hidden_path, hidden_bits)
    write_f32(gate_path, gate_clamped)
    write_f32(up_path, up_clamped)
    write_u16(out_path, out_bits)

    provenance = {
        "schema_version": 1,
        "gate": "Gate F / V4 routed-expert primitive",
        "fixture_class": "F3 official-source + real-checkpoint selected routed expert",
        "evidence_state": "CHECKPOINT-VERIFIED_SOURCE-ORACLE",
        "model": infos["w1"]["model"],
        "revision": infos["w1"]["revision"],
        "layer": LAYER,
        "expert": EXPERT,
        "route_weight": route_weight,
        "router_fixture": {
            "learned_ids": router_prov["learned"]["ids"],
            "learned_weights": router_prov["learned"]["weights"],
            "input_sha256": router_prov["input"]["sha256"],
        },
        "source_contract": {
            "path": "inference/model.py Expert.forward + inference/kernel.py fp4_gemm",
            "w1_role": "gate",
            "w3_role": "up",
            "up_clamp": [-LIMIT, LIMIT],
            "gate_clamp": [None, LIMIT],
            "activation": "silu(gate) * up",
            "routing_weight_order": "multiply hidden by route_weight before BF16 cast and w2",
            "w2_input_dtype": "BF16",
            "quantized_linear": "E4M3 activation K128 x packed E2M1 weight with E8M0 K32 scales",
        },
        "tensors": infos,
        "input": {
            "file": os.path.basename(input_path),
            "dtype": "BF16",
            "shape": [INPUT],
            "sha256": sha256_file(input_path),
        },
        "intermediate": {
            "gate_file": os.path.basename(gate_path),
            "gate_sha256": sha256_file(gate_path),
            "up_file": os.path.basename(up_path),
            "up_sha256": sha256_file(up_path),
            "hidden_file": os.path.basename(hidden_path),
            "hidden_dtype": "BF16",
            "hidden_shape": [INTER],
            "hidden_sha256": sha256_file(hidden_path),
            "gate_positive_clamped_count": sum(v >= LIMIT for v in gate_clamped),
            "up_clamped_count": sum(abs(v) >= LIMIT for v in up_clamped),
        },
        "expected": {
            "file": os.path.basename(out_path),
            "dtype": "BF16",
            "shape": [OUT_ROWS],
            "sha256": sha256_file(out_path),
            "bf16_hex": [f"0x{b:04x}" for b in out_bits],
            "reduction_stability": "each quantized linear output used in the fixture is BF16-stable against forward/reverse/exact dyadic accumulation",
        },
        "non_claim": "One selected routed expert with eight final output rows. This is not yet the six-routed-plus-shared combined MoE Gate F block."
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    for path in paths.values():
        side = path + ".json"
        if os.path.exists(side):
            os.remove(side)

    print("routed expert fixture built")
    print("layer/expert:", LAYER, EXPERT, "route weight", route_weight)
    print("output BF16:", " ".join(provenance["expected"]["bf16_hex"]))
    print("gate clamped:", provenance["intermediate"]["gate_positive_clamped_count"])
    print("up clamped:", provenance["intermediate"]["up_clamped_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
