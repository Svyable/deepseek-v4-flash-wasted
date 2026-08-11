#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Build the first real consecutive two-layer V7 localization trace offline.

No checkpoint access occurs here. All branch outputs and router decisions are
already frozen by earlier gates. This tool binds those real boundaries across
layers 3 and 4 while recomputing HyperConnection composition two ways:

* ``expected`` uses the independent F32 Python HC oracle; and
* ``runtime`` uses WASTE's C scalar HC reference.

Both paths must reproduce every frozen HC boundary exactly before a trace is
written. Attention/MoE/router arithmetic is intentionally reused from its
separate checkpoint-backed fixtures rather than re-proved here. V7's purpose is
consecutive-layer composition and first-divergence localization.

The layer walk itself now lives in :mod:`tools.deepseek_v4_continuation`, which
is parameterized over a prior ``[4, 4096]`` state and an evidence source. This
tool is the two-layer instance of it: the frozen fixtures are one source, and a
checkpoint-backed source for layers 5-42 drops into the same driver. The output
is byte-identical to what the hand-written two-layer version produced.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import deepseek_v4_continuation as cont  # noqa: E402

MODEL = cont.MODEL
REV = cont.REV
DIM = cont.DIM
HC = cont.HC
FLAT = cont.FLAT
MIX = cont.MIX
TOPK = cont.TOPK
BOUNDARIES = cont.BOUNDARIES

sha256_file = cont.sha256_file


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="output fixture directory")
    args = ap.parse_args(argv)

    source = cont.FrozenFixtureSource()
    layers = cont.resolve_layers(source)
    # The walk starts from the exact prior state layer 3 consumes; every later
    # layer's input is the previous layer's computed output, never a copy.
    initial = source.initial_state()

    with cont.OracleBackend() as oracle:
        expected = cont.run_continuation(initial, source, oracle, layers)
    with cont.RuntimeBackend() as runtime_backend:
        runtime = cont.run_continuation(initial, source, runtime_backend, layers)

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # The chained endpoint is the V7 invariant, not an incidental copy.
    for label, walk in (("expected", expected), ("runtime", runtime)):
        for (_, _, a), (number, _, b) in zip(walk, walk[1:]):
            cont.exact_bits(f"{label} layer{number - 1}-output/layer{number}-input chain",
                            a["output"], b["input"])

    expected_manifest = cont.write_trace(
        os.path.join(args.out_dir, "expected"), "real-layer3-to-layer4-expected",
        expected)
    runtime_manifest = cont.write_trace(
        os.path.join(args.out_dir, "runtime"), "real-layer3-to-layer4-runtime",
        runtime)

    described = source.describe()
    provenance = {
        "schema_version": 1,
        "gate": "V7 first real consecutive two-layer localization trace",
        "evidence_state": "END-TO-END-VERIFIED-TWO-LAYER-SCALAR-BOUNDARIES",
        "model": MODEL,
        "revision": REV,
        "layers": list(layers),
        "structural_classes": [cls for _, cls, _ in expected],
        "boundary_order": [b[0] for b in BOUNDARIES],
        "expected_manifest": os.path.relpath(expected_manifest, args.out_dir).replace(os.sep, "/"),
        "runtime_manifest": os.path.relpath(runtime_manifest, args.out_dir).replace(os.sep, "/"),
        "source_provenance_sha256": described["provenance_sha256"],
        "composition_paths": {
            "expected_hyperconnection": cont.OracleBackend.description,
            "runtime_hyperconnection": cont.RuntimeBackend.description,
            "attention_moe_router": "reuse already-frozen real checkpoint-backed boundaries; primitive arithmetic remains owned by Gate E/F/H and the complete layer-4 fixture",
        },
        "chain": {
            "layer3_output_sha256": sha256_file(os.path.join(args.out_dir, "expected", "layer3", "output.bf16.bin")),
            "layer4_input_sha256": sha256_file(os.path.join(args.out_dir, "expected", "layer4", "input.bf16.bin")),
            "byte_identical": True,
        },
        "non_claims": [
            "two consecutive layers only, not all 43 transformer layers",
            "position 0 does not re-prove non-empty ratio-4 compressor/indexer history; Gate E owns that primitive evidence",
            "no final-model logits, generation, converted-container/cache identity, RAM or throughput claim",
        ],
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    print("PASS built real V7 layer3->layer4 expected/runtime trace")
    print("chain sha256:", provenance["chain"]["layer3_output_sha256"])
    print("layer4 final sha256:", sha256_file(os.path.join(args.out_dir, "expected", "layer4", "output.bf16.bin")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
