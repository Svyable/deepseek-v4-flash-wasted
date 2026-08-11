#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Acquire a bounded real-state Gate-I final-head primitive fixture.

This is NOT a final-model logits gate. It deliberately feeds the frozen real
layer-4 `[4,4096]` state into the pinned 0731 final-head arithmetic only to
prove HC-head orientation/casts, final RMSNorm, and selected vocabulary-row
dots against real checkpoint bytes. Raw parameter/head-row payloads stay in a
temporary acquisition directory and are never committed.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import fetch_hf_tensor_slice as slices  # noqa: E402
import make_v8_head_surface as surface_gate  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIM = 4096
HC = 4
FLAT = HC * DIM
VOCAB = 129280
ROW_RANGES = ((0, 8), (128796, 128804), (129272, 129280))
ROW_IDS = [r for start, end in ROW_RANGES for r in range(start, end)]

V7 = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v7_layer4_full_real")
SURFACE = os.path.join(REPO, "reference", "deepseek-v4-flash-0731.v8-head.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def bf16_float(bits):
    return struct.unpack("<f", struct.pack("<I", (int(bits) & 0xFFFF) << 16))[0]


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


def read_u16(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 2:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*2}")
    return list(struct.unpack(f"<{count}H", data))


def read_f32(path, count):
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {count*4}")
    return list(struct.unpack(f"<{count}f", data))


def write_u16(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_u32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *[f32(v) for v in values]))


def source_defaults(source_path):
    raw = open(source_path, "rb").read()
    expected_sha = json.load(open(SURFACE, encoding="utf-8"))["official_source"]["sha256"]
    got_sha = hashlib.sha256(raw).hexdigest()
    if got_sha != expected_sha:
        raise ValueError(f"pinned source SHA {got_sha} != Gate-I surface {expected_sha}")
    tree = ast.parse(raw.decode("utf-8"), filename=source_path)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelArgs"), None)
    if cls is None:
        raise ValueError("pinned source lacks ModelArgs")
    out = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return out, got_sha


def effective_config(source_path, inference_config_path):
    defaults, source_sha = source_defaults(source_path)
    config = json.load(open(inference_config_path, encoding="utf-8"))
    effective = dict(defaults)
    effective.update(config)
    required = {
        "vocab_size": VOCAB,
        "dim": DIM,
        "n_layers": 43,
        "hc_mult": HC,
        "norm_eps": 1e-6,
        "hc_eps": 1e-6,
    }
    for key, want in required.items():
        if effective.get(key) != want:
            raise ValueError(f"effective ModelArgs.{key}={effective.get(key)!r}, expected {want!r}")
    return effective, defaults, config, source_sha


def py_hc_head(x, fn, scale, base, norm_eps, hc_eps):
    total = f32(0.0)
    for v in x:
        total = f32(total + f32(v * v))
    mean = f32(total / f32(float(FLAT)))
    rsqrt = f32(f32(1.0) / sqrtf(f32(mean + f32(norm_eps))))
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
        pre.append(f32(sigmoidf(affine) + f32(hc_eps)))
    out = []
    for d in range(DIM):
        acc = f32(0.0)
        for stream in range(HC):
            acc = f32(acc + f32(pre[stream] * x[stream * DIM + d]))
        out.append(acc)
    return out, mixes, pre


def py_rms(x, weight, eps):
    total = f32(0.0)
    for v in x:
        total = f32(total + f32(v * v))
    mean = f32(total / f32(float(DIM)))
    rsqrt = f32(f32(1.0) / sqrtf(f32(mean + f32(eps))))
    return [f32(weight[d] * f32(x[d] * rsqrt)) for d in range(DIM)]


def py_dot(x, row):
    acc = f32(0.0)
    for a, b in zip(x, row):
        acc = f32(acc + f32(a * b))
    return acc


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("no C compiler available for V8 head acquisition")


def build_lib(work):
    out = os.path.join(work, "libv8head.so")
    cmd = [compiler(), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
           "-ffp-contract=off", "-shared", "-fPIC", "-I", os.path.join(REPO, "src"),
           os.path.join(REPO, "src", "deepseek_v4_head_ref.c"), "-lm", "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("V8 head ref compile failed:\n" + p.stdout + p.stderr)
    return ctypes.CDLL(out)


def as_cf(values):
    return (ctypes.c_float * len(values))(*[f32(v) for v in values])


def first_f32_diff(got, want):
    for i, (a, b) in enumerate(zip(got, want)):
        if f32_bits(a) != f32_bits(b):
            return i
    return None


def fetch(snapshot, name, rows, path, token):
    info = slices.fetch_slice(snapshot, name, rows, path, token=token)
    if info.get("revision") != REVISION:
        raise ValueError(f"{name}: revision drifted to {info.get('revision')}")
    side = path + ".json"
    if os.path.exists(side):
        os.remove(side)
    return info


def meta(info):
    return {
        "tensor": info["tensor"],
        "shard": info["shard"],
        "dtype": info["dtype"],
        "shape": info["shape"],
        "rows": [info["row_start"], info["row_end"]],
        "payload_bytes": info["payload_bytes"],
        "payload_sha256": info["payload_sha256"],
        "absolute_http_range_inclusive": [info["absolute_start"], info["absolute_end"]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("source")
    ap.add_argument("inference_config")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    surface = json.load(open(SURFACE, encoding="utf-8"))
    surface_gate.validate_surface(surface)
    if surface["revision"] != REVISION:
        raise ValueError("Gate-I surface revision drifted")

    effective, defaults, source_config, source_sha = effective_config(
        args.source, args.inference_config)
    norm_eps = f32(effective["norm_eps"])
    hc_eps = f32(effective["hc_eps"])

    v7_prov_path = os.path.join(V7, "provenance.json")
    v7 = json.load(open(v7_prov_path, encoding="utf-8"))
    final = v7.get("outputs", {}).get("layer4_final", {})
    input_path = os.path.join(V7, final.get("file", ""))
    expected_input_sha = "f32bec5a013bc5a0da98a6ac940dd4946fe0dad877714ffb6811c177837cf749"
    if final.get("dtype") != "BF16" or final.get("shape") != [HC, DIM] or \
       final.get("sha256") != expected_input_sha or sha(input_path) != expected_input_sha:
        raise ValueError("V8 primitive input is not the frozen real layer-4 final state")
    input_bits = read_u16(input_path, FLAT)
    x = [bf16_float(v) for v in input_bits]

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)
    fixture_input = os.path.join(args.out_dir, "input-layer4-final.bf16.bin")
    shutil.copyfile(input_path, fixture_input)

    with tempfile.TemporaryDirectory(prefix="v8-head-primitive-") as work:
        raw = {}
        infos = {}
        for key, name in (("fn", "hc_head_fn"), ("base", "hc_head_base"),
                          ("scale", "hc_head_scale"), ("norm", "norm.weight")):
            dest = os.path.join(work, key + ".bin")
            infos[key] = fetch(args.snapshot, name, None, dest, args.token)
            raw[key] = dest

        expected_shapes = {
            "fn": ("F32", [HC, FLAT]),
            "base": ("F32", [HC]),
            "scale": ("F32", [1]),
            "norm": ("BF16", [DIM]),
        }
        for key, (dtype, shape) in expected_shapes.items():
            if infos[key]["dtype"] != dtype or infos[key]["shape"] != shape:
                raise ValueError(f"{key}: got {infos[key]['dtype']} {infos[key]['shape']}, expected {dtype} {shape}")

        fn = read_f32(raw["fn"], HC * FLAT)
        base = read_f32(raw["base"], HC)
        scale = read_f32(raw["scale"], 1)
        norm_bits = read_u16(raw["norm"], DIM)
        norm_weight = [bf16_float(v) for v in norm_bits]

        head_infos = []
        head_rows = []
        for start, end in ROW_RANGES:
            dest = os.path.join(work, f"head-{start}-{end}.bf16.bin")
            info = fetch(args.snapshot, "head.weight", f"{start}:{end}", dest, args.token)
            if info["dtype"] != "BF16" or info["shape"] != [VOCAB, DIM] or \
               info["row_start"] != start or info["row_end"] != end:
                raise ValueError(f"head rows {start}:{end}: checkpoint geometry drifted")
            bits = read_u16(dest, (end - start) * DIM)
            for local in range(end - start):
                row = bits[local * DIM:(local + 1) * DIM]
                head_rows.append([bf16_float(v) for v in row])
            head_infos.append(info)
        if len(head_rows) != len(ROW_IDS):
            raise ValueError("selected head-row count drifted")

        # Independent Python source-derived expected side.
        hc_raw, mixes, pre = py_hc_head(x, fn, scale, base, norm_eps, hc_eps)
        hc_bits = [bf16_bits(v) for v in hc_raw]
        hc = [bf16_float(v) for v in hc_bits]
        norm_raw = py_rms(hc, norm_weight, norm_eps)
        normalized_bits = [bf16_bits(v) for v in norm_raw]
        normalized = [bf16_float(v) for v in normalized_bits]
        logits = [py_dot(normalized, row) for row in head_rows]

        # WASTE scalar implementation must reproduce every F32 arithmetic value
        # before the fixture is allowed to freeze.
        lib = build_lib(work)
        c_hc = lib.waste_ds_v4_hc_head_ref
        c_hc.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float,
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float)]
        c_hc.restype = ctypes.c_int
        c_rms = lib.waste_ds_v4_final_rmsnorm_ref
        c_rms.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                          ctypes.c_size_t, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
        c_rms.restype = ctypes.c_int
        c_dot = lib.waste_ds_v4_head_row_ref
        c_dot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                          ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
        c_dot.restype = ctypes.c_int
        c_bf = lib.waste_ds_v4_head_f32_to_bf16
        c_bf.argtypes = [ctypes.c_float]
        c_bf.restype = ctypes.c_uint16
        c_fb = lib.waste_ds_v4_head_bf16_to_f32
        c_fb.argtypes = [ctypes.c_uint16]
        c_fb.restype = ctypes.c_float

        ch = (ctypes.c_float * DIM)()
        cm = (ctypes.c_float * HC)()
        cp = (ctypes.c_float * HC)()
        if c_hc(as_cf(x), DIM, as_cf(fn), as_cf(scale), as_cf(base),
                ctypes.c_float(norm_eps), ctypes.c_float(hc_eps), ch, cm, cp):
            raise ValueError("C hc_head rejected real layer-4 state")
        for label, got, want in (("hc_raw", list(ch), hc_raw),
                                 ("hc_mixes", list(cm), mixes),
                                 ("hc_pre", list(cp), pre)):
            diff = first_f32_diff(got, want)
            if diff is not None:
                raise ValueError(f"C/Python {label} F32 mismatch at {diff}: {f32_bits(got[diff]):08x}!={f32_bits(want[diff]):08x}")
        c_hc_bits = [int(c_bf(v)) for v in ch]
        if c_hc_bits != hc_bits:
            raise ValueError("C/Python hc_head BF16 cast mismatch")
        c_hc_values = [f32(c_fb(v)) for v in c_hc_bits]
        if [f32_bits(v) for v in c_hc_values] != [f32_bits(v) for v in hc]:
            raise ValueError("C BF16 expansion disagrees with independent Python")

        cn = (ctypes.c_float * DIM)()
        if c_rms(as_cf(c_hc_values), as_cf(norm_weight), DIM,
                 ctypes.c_float(norm_eps), cn):
            raise ValueError("C final RMSNorm rejected real HC-head output")
        diff = first_f32_diff(list(cn), norm_raw)
        if diff is not None:
            raise ValueError(f"C/Python RMSNorm F32 mismatch at {diff}")
        c_norm_bits = [int(c_bf(v)) for v in cn]
        if c_norm_bits != normalized_bits:
            raise ValueError("C/Python final RMSNorm BF16 cast mismatch")

        c_logits = []
        for row in head_rows:
            value = ctypes.c_float()
            if c_dot(as_cf(normalized), as_cf(row), DIM, ctypes.byref(value)):
                raise ValueError("C vocabulary head row rejected real normalized state")
            c_logits.append(f32(value.value))
        diff = first_f32_diff(c_logits, logits)
        if diff is not None:
            raise ValueError(f"C/Python selected head-row F32 mismatch at row {ROW_IDS[diff]}")

        # Freeze only compact stage outputs; all checkpoint parameter bytes stay
        # in the TemporaryDirectory and disappear after acquisition.
        hc_path = os.path.join(args.out_dir, "hc-head.bf16.bin")
        mix_path = os.path.join(args.out_dir, "hc-mixes.f32.bin")
        pre_path = os.path.join(args.out_dir, "hc-pre.f32.bin")
        norm_path = os.path.join(args.out_dir, "final-norm.bf16.bin")
        ids_path = os.path.join(args.out_dir, "selected-row-ids.u32.bin")
        logits_path = os.path.join(args.out_dir, "selected-logits.f32.bin")
        write_u16(hc_path, hc_bits)
        write_f32(mix_path, mixes)
        write_f32(pre_path, pre)
        write_u16(norm_path, normalized_bits)
        write_u32(ids_path, ROW_IDS)
        write_f32(logits_path, logits)

        payload = sum(info["payload_bytes"] for info in infos.values()) + \
                  sum(info["payload_bytes"] for info in head_infos)
        provenance = {
            "schema_version": 1,
            "gate": "Gate I/V8 real-state final-head primitive",
            "evidence_state": "CHECKPOINT-BOUND-REAL-STATE-HEAD-PRIMITIVE",
            "model": MODEL,
            "revision": REVISION,
            "input": {
                "producer": "V7 complete real layer-4 final state",
                "source_file": "tests/fixtures/deepseek_v4/v7_layer4_full_real/" + final["file"],
                "source_sha256": expected_input_sha,
                "fixture_file": os.path.basename(fixture_input),
                "fixture_sha256": sha(fixture_input),
                "shape": [HC, DIM],
                "dtype": "BF16",
                "is_true_final_model_hidden": False,
            },
            "surface_dependency": {
                "file": "reference/deepseek-v4-flash-0731.v8-head.json",
                "sha256": sha(SURFACE),
            },
            "effective_inference_config": {
                "vocab_size": effective["vocab_size"],
                "dim": effective["dim"],
                "n_layers": effective["n_layers"],
                "hc_mult": effective["hc_mult"],
                "norm_eps": effective["norm_eps"],
                "hc_eps": effective["hc_eps"],
                "norm_eps_source": "ModelArgs default" if "norm_eps" not in source_config else "inference/config.json",
                "hc_eps_source": "ModelArgs default" if "hc_eps" not in source_config else "inference/config.json",
                "pinned_source_sha256": source_sha,
                "inference_config_sha256": sha(args.inference_config),
            },
            "checkpoint_payload": {
                "total_bytes_fetched": payload,
                "raw_payload_committed": False,
                "full_parameters": {key: meta(info) for key, info in infos.items()},
                "head_row_slices": [meta(info) for info in head_infos],
                "selected_row_ids": ROW_IDS,
            },
            "outputs": {
                "hc_head": {"file": os.path.basename(hc_path), "dtype": "BF16", "shape": [DIM], "sha256": sha(hc_path), "first8_hex": [f"0x{x:04x}" for x in hc_bits[:8]]},
                "hc_mixes": {"file": os.path.basename(mix_path), "dtype": "F32", "shape": [HC], "sha256": sha(mix_path), "f32_bits": [f"0x{f32_bits(x):08x}" for x in mixes]},
                "hc_pre": {"file": os.path.basename(pre_path), "dtype": "F32", "shape": [HC], "sha256": sha(pre_path), "f32_bits": [f"0x{f32_bits(x):08x}" for x in pre]},
                "final_norm": {"file": os.path.basename(norm_path), "dtype": "BF16", "shape": [DIM], "sha256": sha(norm_path), "first8_hex": [f"0x{x:04x}" for x in normalized_bits[:8]]},
                "row_ids": {"file": os.path.basename(ids_path), "dtype": "U32", "shape": [len(ROW_IDS)], "sha256": sha(ids_path)},
                "selected_logits": {"file": os.path.basename(logits_path), "dtype": "F32", "shape": [len(ROW_IDS)], "sha256": sha(logits_path), "f32_bits": [f"0x{f32_bits(x):08x}" for x in logits]},
            },
            "acquisition_verification": {
                "independent_python_vs_waste_hc_raw_exact_f32": DIM,
                "independent_python_vs_waste_hc_mixes_exact_f32": HC,
                "independent_python_vs_waste_hc_pre_exact_f32": HC,
                "independent_python_vs_waste_hc_bf16_cast_exact": DIM,
                "independent_python_vs_waste_rms_raw_exact_f32": DIM,
                "independent_python_vs_waste_rms_bf16_cast_exact": DIM,
                "independent_python_vs_waste_selected_logits_exact_f32": len(ROW_IDS),
            },
            "non_claims": [
                "the input is a real layer-4 hidden state used as a primitive test vector, not the true layer-42/final transformer hidden state",
                "selected F32 row outputs are primitive/orientation checks, not final-model logits",
                "this does not execute the remaining transformer layers or pass V8 final-logit parity",
                "this does not fetch or validate the full 1.059 GB vocabulary head",
                "no generation, converted-container/cache identity, RAM or throughput claim",
            ],
        }
        with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, sort_keys=True)
            f.write("\n")

    print("PASS Gate I/V8 real-state final-head primitive")
    print("input sha256:", expected_input_sha)
    print("checkpoint payload bytes fetched:", provenance["checkpoint_payload"]["total_bytes_fetched"])
    print("selected rows:", ROW_IDS)
    print("hc first8:", " ".join(f"0x{x:04x}" for x in hc_bits[:8]))
    print("norm first8:", " ".join(f"0x{x:04x}" for x in normalized_bits[:8]))
    print("logit bits:", " ".join(f"{row}:0x{f32_bits(v):08x}" for row, v in zip(ROW_IDS, logits)))
    print("PRIMITIVE ONLY — NOT FINAL-MODEL LOGITS / V8 NOT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
