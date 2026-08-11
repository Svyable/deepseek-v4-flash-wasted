#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Prove the parameterized per-layer continuation engine offline.

Three things are checked, in this order:

1. **Behavior preservation.** Walking layers 3 and 4 through the engine
   reproduces the committed two-layer fixture byte for byte, on both the
   independent F32 oracle and the C scalar HC reference. The generalization did
   not move a single bit of V7's evidence.

2. **Generalization is real.** The driver walks a three-layer synthetic source
   with no layer numbers, layer counts or fixture paths of its own, and
   ``v7_localize`` accepts the resulting trace. This is what layers 5-42 will
   use; the only thing still missing there is the evidence, not the engine.

3. **Fail-closed.** A layer with no evidence, a non-consecutive walk, a wrong
   prior state, a broken chain, a drifted HyperConnection parameter and a
   backend used outside its ``with`` block are each refused by name. The
   declared-boundary assertions are shown to be live by mutating a parameter
   and watching the exact boundary that must catch it.

No checkpoint access occurs here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import deepseek_v4_continuation as cont  # noqa: E402
import make_v7_two_layer_trace as two_layer  # noqa: E402
import v7_localize as loc  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v7_two_layer_real")


class CheckError(AssertionError):
    pass


def refuses(label, fn, *, contains=None):
    """Assert ``fn`` raises ContinuationError, optionally naming a substring."""
    try:
        fn()
    except cont.ContinuationError as exc:
        if contains is not None and contains not in str(exc):
            raise CheckError(
                f"{label}: refused, but message {str(exc)!r} does not mention "
                f"{contains!r}") from None
        return str(exc)
    raise CheckError(f"{label}: was accepted, expected a fail-closed refusal")


def relpaths(root):
    out = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            full = os.path.join(dirpath, name)
            out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


# ---------------------------------------------------------------------------
# 1. behavior preservation
# ---------------------------------------------------------------------------


def check_reproduces_frozen_fixture(work):
    if not os.path.isfile(os.path.join(FIX, "provenance.json")):
        return None
    out = os.path.join(work, "regen")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = two_layer.main([out])
    if rc != 0:
        raise CheckError("two-layer trace generation failed")

    want = relpaths(FIX)
    got = relpaths(out)
    if want != got:
        raise CheckError(
            f"regenerated file set drifted: missing={sorted(set(want) - set(got))} "
            f"extra={sorted(set(got) - set(want))}")
    for rel in want:
        a = open(os.path.join(FIX, rel), "rb").read()
        b = open(os.path.join(out, rel), "rb").read()
        if a != b:
            raise CheckError(f"regenerated {rel} is not byte-identical to the committed fixture")

    # The localizer must accept what the engine produced, on both traces.
    result = loc.compare(
        loc.load_trace(os.path.join(out, "expected", "trace.json")),
        loc.load_trace(os.path.join(out, "runtime", "trace.json")))
    if result.get("status") != "match":
        raise CheckError(f"engine expected/runtime traces diverge: {result}")
    if result.get("boundaries_compared") != 2 * len(cont.BOUNDARIES):
        raise CheckError(
            f"expected {2 * len(cont.BOUNDARIES)} compared boundaries, "
            f"got {result.get('boundaries_compared')}")
    return len(want)


# ---------------------------------------------------------------------------
# 2. generalization
# ---------------------------------------------------------------------------


class SyntheticSource(cont.EvidenceSource):
    """A free-running N-layer source built from one real layer's parameters.

    Only the first layer declares an input; every later layer is free-running,
    so its input is whatever the driver carried forward. That is precisely the
    shape a checkpoint-backed source for layers 5-42 has before those layers'
    outputs are frozen, and it is what proves the driver imposes no layer
    count, numbering or fixture layout of its own.
    """

    def __init__(self, template: cont.LayerEvidence, first: int, count: int):
        self._layers = tuple(range(first, first + count))
        self._template = template

    def available_layers(self):
        return self._layers

    def evidence(self, layer):
        if layer not in self._layers:
            raise self.refuse(layer)
        return dataclasses.replace(
            self._template,
            layer=layer,
            structural_class=f"synthetic-layer{layer}",
            declared_input=self._template.declared_input if layer == self._layers[0] else None,
            declared_attn_pre=None,
            declared_after_attn=None,
            declared_ffn_pre=None,
            declared_output=None,
        )


def check_walks_more_than_two_layers(work, source):
    template = source.evidence(4)
    synthetic = SyntheticSource(template, first=7, count=3)
    with cont.OracleBackend() as oracle:
        walk = cont.run_continuation(
            template.declared_input, synthetic, oracle)
    if [n for n, _, _ in walk] != [7, 8, 9]:
        raise CheckError(f"synthetic walk visited {[n for n, _, _ in walk]}, expected [7, 8, 9]")

    # Chained by construction: each layer's input is the previous output.
    for (_, _, a), (number, _, b) in zip(walk, walk[1:]):
        if a["output"] != b["input"]:
            raise CheckError(f"synthetic chain broke entering layer {number}")
    # A free-running layer must actually move the state, or the walk proves
    # nothing about continuation.
    if walk[0][2]["input"] == walk[-1][2]["output"]:
        raise CheckError("synthetic walk returned the state unchanged")

    root = os.path.join(work, "synthetic")
    manifest = cont.write_trace(root, "synthetic-three-layer", walk)
    trace = loc.load_trace(manifest)
    if [layer.number for layer in trace.layers] != [7, 8, 9]:
        raise CheckError("localizer disagreed about the synthetic layer numbers")
    result = loc.compare(trace, loc.load_trace(manifest))
    if result.get("status") != "match" or \
       result.get("boundaries_compared") != 3 * len(cont.BOUNDARIES):
        raise CheckError(f"localizer rejected the synthetic three-layer trace: {result}")
    return walk


# ---------------------------------------------------------------------------
# 3. fail-closed
# ---------------------------------------------------------------------------


def check_fail_closed(source):
    ev3 = source.evidence(3)
    ev4 = source.evidence(4)
    initial = source.initial_state()

    message = refuses("layer 5 evidence", lambda: source.evidence(5),
                      contains="no checkpoint-bound evidence available")
    for token in ("3", "4"):
        if token not in message:
            raise CheckError(f"refusal does not name available layer {token}: {message!r}")

    refuses("empty walk", lambda: cont.resolve_layers(source, []),
            contains="at least one layer")
    refuses("non-consecutive walk", lambda: cont.resolve_layers(source, [3, 5]),
            contains="must be consecutive")
    refuses("walk past the evidence", lambda: cont.resolve_layers(source, [3, 4, 5]),
            contains="no checkpoint-bound evidence available")

    with cont.OracleBackend() as oracle:
        # A prior state of the wrong width is not a numerical question.
        refuses("short prior state",
                lambda: cont.step_layer(list(initial)[:-1], ev3, oracle),
                contains="expected 16384")

        # One flipped BF16 in the carried state must be caught at the input
        # boundary, before any arithmetic can average it away.
        bad = list(initial)
        bad[0] ^= 1
        refuses("perturbed prior state",
                lambda: cont.step_layer(bad, ev3, oracle),
                contains="carried input")

        # Starting at layer 4 from layer 3's input is a chain break, and the
        # declared input is what makes it visible.
        refuses("chain break entering layer 4",
                lambda: cont.run_continuation(initial, source, oracle, [4]),
                contains="layer 4 carried input")

        # The declared-boundary assertions must be live. Perturb one F32 of the
        # attention HyperConnection base and the very next declared boundary
        # has to refuse: if this passes, every "exact" claim above is theatre.
        base = list(ev3.params.attn_base)
        base[0] = base[0] + 1.0
        drifted = dataclasses.replace(
            ev3, params=dataclasses.replace(ev3.params, attn_base=tuple(base)))
        refuses("drifted attn_base",
                lambda: cont.step_layer(initial, drifted, oracle),
                contains="attn_pre")

        # And the same for the FFN side, which is only reached after the
        # attention boundaries have already passed.
        fbase = list(ev4.params.ffn_base)
        fbase[0] = fbase[0] + 1.0
        fdrift = dataclasses.replace(
            ev4, params=dataclasses.replace(ev4.params, ffn_base=tuple(fbase)))
        refuses("drifted ffn_base",
                lambda: cont.step_layer(ev4.declared_input, fdrift, oracle),
                contains="ffn_pre")

        # A source that answers with the wrong layer is refused rather than
        # silently relabelled.
        class Liar(cont.EvidenceSource):
            def available_layers(self):
                return (3,)

            def evidence(self, layer):
                return ev4

        refuses("source returning the wrong layer",
                lambda: cont.run_continuation(initial, Liar(), oracle, [3]),
                contains="returned layer 4")

    # Geometry is validated before it can size anything.
    refuses("truncated HC parameter",
            lambda: dataclasses.replace(
                ev3, params=dataclasses.replace(
                    ev3.params, ffn_scale=(1.0, 2.0))).validate(),
            contains="ffn_scale")
    refuses("wrong router width",
            lambda: dataclasses.replace(ev3, router_ids=(1, 2, 3)).validate(),
            contains="router_ids")

    # The C backend is a context manager because its library is temporary.
    refuses("runtime backend outside its with block",
            lambda: cont.RuntimeBackend().pre(
                [0.0] * cont.FLAT, ev3.params.attn_fn,
                ev3.params.attn_scale, ev3.params.attn_base),
            contains="outside its `with` block")


def main():
    source = cont.FrozenFixtureSource()
    try:
        source.evidence(3)
    except cont.ContinuationError as exc:
        print(f"SKIP: frozen V6/V7 layer evidence not present ({exc})")
        return 77

    work = tempfile.mkdtemp(prefix="v7-continuation-engine-")
    try:
        files = check_reproduces_frozen_fixture(work)
        if files is None:
            print("SKIP: committed two-layer fixture not present")
            return 77
        walk = check_walks_more_than_two_layers(work, source)
        check_fail_closed(source)
    except (CheckError, cont.ContinuationError) as exc:
        print("FAIL:", exc)
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("PASS V7 parameterized per-layer continuation engine")
    print(f"reproduced the committed two-layer fixture byte for byte: {files} files, "
          f"both composition paths")
    print(f"walked {len(walk)} consecutive synthetic layers "
          f"({len(walk) * len(cont.BOUNDARIES)} boundaries) through the same driver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
