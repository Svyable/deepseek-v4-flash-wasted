#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the Gate H MoE fixture using the proven scalar router's exact F32 weights.

`make_v6_moe_branch_fixture.py` derives the route independently in Python so
selection does not depend on the implementation under test. That is the right
oracle for IDs and stability, but its double-precision reductions differ by a
few 1e-8 from the scalar C router's float32 outputs. Route weights are consumed
inside each expert before the hidden BF16 cast, so the complete-layer fixture
must use the exact scalar-runtime weights, not the nearby oracle weights.

This wrapper keeps both contracts: Python must select the same six IDs and stay
within the Gate-F tolerance; the actual expert acquisition is then driven by
the C router values. Provenance records both vectors and their maximum delta.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import make_v6_moe_branch_fixture as base  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402

CAPTURE = {}
ORIGINAL_DERIVE = base.derive_real_ffn_input


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError("no C compiler available for exact Gate H routing")


def exact_runtime_route(state):
    gate_path = os.path.join(base.RTFIX, "layer3-gate-weight.bf16.bin")
    bias_path = os.path.join(base.RTFIX, "layer3-gate-bias.f32.bin")
    gate_bits = chain.read_u16(gate_path, chain.EXPERTS * base.INPUT)
    bias = chain.read_f32(bias_path, chain.EXPERTS)

    with tempfile.TemporaryDirectory(prefix="gate-h-route-runtime-") as td:
        lib = os.path.join(td, "librouter.so")
        cc = compiler()
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", "-shared", "-fPIC",
               "-I", os.path.join(REPO, "src"),
               os.path.join(REPO, "src", "deepseek_v4_router_ref.c"),
               "-lm", "-o", lib]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError("router compile failed:\n" + p.stdout + p.stderr)
        so = ctypes.CDLL(lib)
        score = so.waste_ds_v4_router_score_ref
        score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_uint16),
                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        score.restype = ctypes.c_int
        learned = so.waste_ds_v4_router_learned_ref
        learned.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        learned.restype = ctypes.c_int

        x = (ctypes.c_float * base.INPUT)(*state["ffn_pre"])
        wb = (ctypes.c_uint16 * len(gate_bits))(*gate_bits)
        raw = (ctypes.c_float * chain.EXPERTS)()
        scores = (ctypes.c_float * chain.EXPERTS)()
        if score(x, base.INPUT, wb, raw, scores):
            raise RuntimeError("scalar router score rejected Gate H ffn_pre")
        ids = (ctypes.c_uint32 * base.TOPK)()
        weights = (ctypes.c_float * base.TOPK)()
        selection = (ctypes.c_float * chain.EXPERTS)()
        bias_c = (ctypes.c_float * chain.EXPERTS)(*bias)
        if learned(scores, bias_c, ctypes.c_float(chain.ROUTE_SCALE),
                   ids, weights, selection):
            raise RuntimeError("scalar learned router rejected Gate H scores")
        c_ids = list(ids)
        c_weights = list(weights)

    py_ids, py_weights, _, _, margin = chain.oracle_router(state["ffn_pre"], gate_bits, bias)
    if c_ids != py_ids or c_ids != base.EXPECTED_IDS:
        raise ValueError(f"runtime/Python route ID disagreement: C={c_ids} Python={py_ids}")
    max_delta = max(abs(a - b) for a, b in zip(c_weights, py_weights))
    if max_delta > 1e-5:
        raise ValueError(f"runtime/Python route weights differ by {max_delta}, above Gate-F contract")
    if not margin > 1e-4:
        raise ValueError(f"Python route margin {margin} is not a stable acquisition boundary")

    CAPTURE.update({
        "ids": c_ids,
        "runtime_weights": c_weights,
        "python_weights": py_weights,
        "max_abs_delta": max_delta,
        "runtime_source_sha256": sha(os.path.join(REPO, "src", "deepseek_v4_router_ref.c")),
    })
    state["ids"] = c_ids
    state["weights"] = c_weights
    return state


def patched_derive():
    return exact_runtime_route(ORIGINAL_DERIVE())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("out_dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args(argv)

    base.derive_real_ffn_input = patched_derive
    forwarded = [args.snapshot, args.out_dir]
    if args.token:
        forwarded += ["--token", args.token]
    rc = base.main(forwarded)
    if rc:
        return rc
    if not CAPTURE:
        raise RuntimeError("Gate H runtime routing evidence was not captured")

    prov_path = os.path.join(args.out_dir, "provenance.json")
    p = json.load(open(prov_path, encoding="utf-8"))
    router = p.setdefault("router", {})
    router["weight_source"] = "src/deepseek_v4_router_ref.c scalar learned-router F32 outputs"
    router["runtime_weights_f32"] = CAPTURE["runtime_weights"]
    router["independent_python_weights"] = CAPTURE["python_weights"]
    router["runtime_vs_python_max_abs"] = CAPTURE["max_abs_delta"]
    router["runtime_source_sha256"] = CAPTURE["runtime_source_sha256"]
    router["ids_runtime_vs_python_exact"] = True
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, sort_keys=True)
        f.write("\n")

    print("Gate H expert acquisition route is bound to scalar runtime F32 weights")
    print("runtime weights:", " ".join(f"{w:.9f}" for w in CAPTURE["runtime_weights"]))
    print("Python oracle max abs delta:", f"{CAPTURE['max_abs_delta']:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
