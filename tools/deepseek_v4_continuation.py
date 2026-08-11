#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Parameterized per-layer DeepSeek V4 continuation engine.

V7 proved one thing twice: layer 3 composes into layer 4 exactly, through both
an independent F32 HyperConnection oracle and WASTE's C scalar HC reference.
That proof was written as a two-layer script with layer 3 and layer 4 spelled
out by hand, which is why it stopped at layer 4.

This module is the same arithmetic with the layer identity lifted out of it. It
accepts an exact prior ``[HC, DIM]`` BF16 state and a source of per-layer
evidence, and walks as many consecutive layers as that source can supply,
emitting the same nine-boundary trace contract ``tools/v7_localize.py`` already
consumes.

Three seams make it reusable without weakening anything V7 established:

``EvidenceSource``
    supplies, for one layer, the HyperConnection parameters plus the frozen
    attention and MoE branch outputs. :class:`FrozenFixtureSource` reads the
    committed layer-3/layer-4 fixtures; a checkpoint-backed source for layers
    5-42 implements the same three methods and changes nothing below.

``HcBackend``
    is the composition path. :class:`OracleBackend` is the independent Python
    F32 oracle; :class:`RuntimeBackend` is ``src/deepseek_v4_mhc_ref.c``. Both
    must reproduce every declared boundary exactly.

:func:`run_continuation`
    chains them. The carried state is the *computed* output of the previous
    layer, never a copy of the next layer's frozen input, and each layer's
    declared boundaries are checked against what was computed rather than
    trusted.

Fail-closed is the point. A layer with no evidence is refused by name; a
declared boundary that does not reproduce raises at the first differing
element. Nothing here acquires checkpoint bytes, and nothing here invents a
layer it cannot prove.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import make_v7_layer4_route_fixture as l4route  # noqa: E402
import deepseek_v4_hc_oracle as hc_oracle  # noqa: E402
import test_v6_ffn_route_real as chain  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
DIM = 4096
HC = 4
FLAT = HC * DIM
MIX = 24
TOPK = 6

BASE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4")
L3HC = os.path.join(BASE, "v6_hc_composition_real")
L3ATT = os.path.join(BASE, "v6_attention_branch_real")
L3MOE = os.path.join(BASE, "v6_moe_branch_real")
L4ROUTE = os.path.join(BASE, "v7_layer4_route_real")
L4FULL = os.path.join(BASE, "v7_layer4_full_real")

#: The trace contract. Order is load bearing: ``v7_localize`` requires the
#: first boundary to be ``input`` and the last to be ``output``, and compares
#: boundaries positionally between the expected and actual traces.
BOUNDARIES = (
    ("input", "BF16", [HC, DIM]),
    ("attn_pre", "BF16", [DIM]),
    ("attention_branch", "BF16", [DIM]),
    ("after_attn", "BF16", [HC, DIM]),
    ("ffn_pre", "BF16", [DIM]),
    ("router_ids", "U32", [TOPK]),
    ("router_weights", "F32", [TOPK]),
    ("moe_branch", "BF16", [DIM]),
    ("output", "BF16", [HC, DIM]),
)

#: Boundaries the engine computes rather than reads. Evidence may declare them,
#: in which case they are checked; it may not supply them in their place.
COMPUTED_BOUNDARIES = ("attn_pre", "after_attn", "ffn_pre", "output")


class ContinuationError(ValueError):
    """A layer cannot be continued, or a declared boundary did not reproduce."""


# ---------------------------------------------------------------------------
# byte helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_u16(path: str, count: int) -> list[int]:
    data = open(path, "rb").read()
    if len(data) != count * 2:
        raise ContinuationError(f"{path}: {len(data)} bytes, expected {count * 2}")
    return list(struct.unpack(f"<{count}H", data))


def read_u32(path: str, count: int) -> list[int]:
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ContinuationError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}I", data))


def read_f32(path: str, count: int) -> list[float]:
    data = open(path, "rb").read()
    if len(data) != count * 4:
        raise ContinuationError(f"{path}: {len(data)} bytes, expected {count * 4}")
    return list(struct.unpack(f"<{count}f", data))


def write_u16(path: str, values: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}H", *values))


def write_u32(path: str, values: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def write_f32(path: str, values: list[float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def bf16_floats(bits: list[int]) -> list[float]:
    return [chain.bf16_to_f32(v) for v in bits]


def exact_bits(label: str, got: list[int], want: list[int]) -> None:
    if got == want:
        return
    if len(got) != len(want):
        raise ContinuationError(
            f"{label}: {len(got)} values, expected {len(want)}")
    i = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
    raise ContinuationError(f"{label} mismatch at {i}: {got[i]:04x}!={want[i]:04x}")


def exact_f32(label: str, got: list[float], want: list[float]) -> None:
    gb = [chain.f32_bits(v) for v in got]
    wb = [chain.f32_bits(v) for v in want]
    if gb == wb:
        return
    if len(gb) != len(wb):
        raise ContinuationError(f"{label}: {len(gb)} values, expected {len(wb)}")
    i = next(i for i, (a, b) in enumerate(zip(gb, wb)) if a != b)
    raise ContinuationError(f"{label} F32 mismatch at {i}: {gb[i]:08x}!={wb[i]:08x}")


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HcParams:
    """One layer's HyperConnection parameters, exactly as stored F32."""

    attn_fn: tuple[float, ...]
    attn_scale: tuple[float, ...]
    attn_base: tuple[float, ...]
    ffn_fn: tuple[float, ...]
    ffn_scale: tuple[float, ...]
    ffn_base: tuple[float, ...]

    def validate(self, layer: int) -> None:
        for key, want in (("attn_fn", MIX * FLAT), ("attn_scale", 3),
                          ("attn_base", MIX), ("ffn_fn", MIX * FLAT),
                          ("ffn_scale", 3), ("ffn_base", MIX)):
            got = len(getattr(self, key))
            if got != want:
                raise ContinuationError(
                    f"layer {layer}: {key} has {got} values, expected {want}")


@dataclass(frozen=True)
class LayerEvidence:
    """Everything needed to continue one layer, plus what may be checked.

    ``attention_branch`` and ``moe_branch`` are the branch outputs owned by
    Gates E/F/H; this engine composes them, it does not re-prove them. The
    optional boundaries are assertions: when a source declares one, the engine
    must reproduce it bit for bit or refuse.
    """

    layer: int
    structural_class: str
    params: HcParams
    attention_branch: tuple[int, ...]
    moe_branch: tuple[int, ...]
    router_ids: tuple[int, ...]
    router_weights: tuple[float, ...]
    declared_input: tuple[int, ...] | None = None
    declared_attn_pre: tuple[int, ...] | None = None
    declared_after_attn: tuple[int, ...] | None = None
    declared_ffn_pre: tuple[int, ...] | None = None
    declared_output: tuple[int, ...] | None = None

    def validate(self) -> None:
        if not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0:
            raise ContinuationError(f"invalid layer number {self.layer!r}")
        if not self.structural_class:
            raise ContinuationError(f"layer {self.layer}: structural_class must be non-empty")
        self.params.validate(self.layer)
        for key, want in (("attention_branch", DIM), ("moe_branch", DIM),
                          ("router_ids", TOPK), ("router_weights", TOPK)):
            got = len(getattr(self, key))
            if got != want:
                raise ContinuationError(
                    f"layer {self.layer}: {key} has {got} values, expected {want}")
        for key, want in (("declared_input", FLAT), ("declared_attn_pre", DIM),
                          ("declared_after_attn", FLAT), ("declared_ffn_pre", DIM),
                          ("declared_output", FLAT)):
            value = getattr(self, key)
            if value is not None and len(value) != want:
                raise ContinuationError(
                    f"layer {self.layer}: {key} has {len(value)} values, expected {want}")


class EvidenceSource(abc.ABC):
    """Per-layer evidence provider.

    A checkpoint-backed source for layers 5-42 implements exactly these three
    methods. Nothing downstream needs to change when one appears.
    """

    @abc.abstractmethod
    def available_layers(self) -> tuple[int, ...]:
        """Consecutive-capable layer numbers this source can supply."""

    @abc.abstractmethod
    def evidence(self, layer: int) -> LayerEvidence:
        """Return evidence for ``layer``, or raise :class:`ContinuationError`."""

    def describe(self) -> dict:
        """Provenance for the trace. Sources with frozen inputs list them."""
        return {"source": type(self).__name__}

    def refuse(self, layer: int) -> "ContinuationError":
        have = ", ".join(str(n) for n in self.available_layers()) or "none"
        return ContinuationError(
            f"layer {layer}: no checkpoint-bound evidence available "
            f"(this source has: {have}). Acquire the layer's HyperConnection "
            f"parameters and its attention/MoE branch outputs before "
            f"continuing past it.")


class FrozenFixtureSource(EvidenceSource):
    """Layers 3 and 4, read from the committed V6/V7 fixtures.

    This is the same loading V7's two-layer tool did inline, moved behind the
    source interface so the continuation driver has no layer numbers in it. The
    SHA checks are kept: a fixture whose payload drifted from its provenance is
    refused rather than replayed.
    """

    LAYERS = (3, 4)

    def __init__(self) -> None:
        self._prov_paths = {
            "l3_hc": os.path.join(L3HC, "provenance.json"),
            "l3_att": os.path.join(L3ATT, "provenance.json"),
            "l3_moe": os.path.join(L3MOE, "provenance.json"),
            "l4_route": os.path.join(L4ROUTE, "provenance.json"),
            "l4_full": os.path.join(L4FULL, "provenance.json"),
        }
        self._cache: dict[int, LayerEvidence] = {}
        self._prov: dict[str, dict] | None = None

    # -- fixture plumbing --------------------------------------------------

    def _provenance(self) -> dict[str, dict]:
        if self._prov is not None:
            return self._prov
        for label, path in self._prov_paths.items():
            if not os.path.isfile(path):
                raise ContinuationError(f"{label}: frozen provenance missing")
        prov = {k: load_json(v) for k, v in self._prov_paths.items()}
        for label, p in prov.items():
            if p.get("model") != MODEL or p.get("revision") != REV:
                raise ContinuationError(f"{label}: model/revision drifted")
        self._prov = prov
        return prov

    @staticmethod
    def _checked(root: str, meta: dict, label: str, count: int, reader=read_u16):
        path = os.path.join(root, meta.get("file", ""))
        if not os.path.isfile(path) or sha256_file(path) != meta.get("sha256"):
            raise ContinuationError(f"{label}: file missing or SHA mismatch")
        return reader(path, count)

    @staticmethod
    def _f32_param(root: str, params: dict, key: str, count: int) -> list[float]:
        meta = params.get(key) or {}
        path = os.path.join(root, meta.get("file", ""))
        want = meta.get("payload_sha256", meta.get("sha256"))
        if not os.path.isfile(path) or (want and sha256_file(path) != want):
            raise ContinuationError(f"{key}: parameter missing or SHA mismatch")
        return read_f32(path, count)

    # -- interface ---------------------------------------------------------

    def available_layers(self) -> tuple[int, ...]:
        return self.LAYERS

    def describe(self) -> dict:
        return {
            "source": type(self).__name__,
            "layers": list(self.LAYERS),
            "provenance_sha256": {
                label: sha256_file(path)
                for label, path in self._prov_paths.items()
            },
        }

    def evidence(self, layer: int) -> LayerEvidence:
        if layer in self._cache:
            return self._cache[layer]
        if layer not in self.LAYERS:
            raise self.refuse(layer)
        ev = self._layer3() if layer == 3 else self._layer4()
        ev.validate()
        self._cache[layer] = ev
        return ev

    def _layer3(self) -> LayerEvidence:
        prov = self._provenance()
        hcp = prov["l3_hc"]["parameters"]
        hcs = prov["l3_hc"]["states"]
        params = HcParams(
            attn_fn=tuple(self._f32_param(L3HC, hcp, "attn_fn", MIX * FLAT)),
            attn_scale=tuple(self._f32_param(L3HC, hcp, "attn_scale", 3)),
            attn_base=tuple(self._f32_param(L3HC, hcp, "attn_base", MIX)),
            ffn_fn=tuple(self._f32_param(L3HC, hcp, "ffn_fn", MIX * FLAT)),
            ffn_scale=tuple(self._f32_param(L3HC, hcp, "ffn_scale", 3)),
            ffn_base=tuple(self._f32_param(L3HC, hcp, "ffn_base", MIX)),
        )
        moe_out = prov["l3_moe"]["outputs"]
        return LayerEvidence(
            layer=3,
            structural_class="ratio128-learned",
            params=params,
            attention_branch=tuple(self._checked(
                L3ATT, prov["l3_att"]["outputs"]["branch"],
                "layer3 real attention branch", DIM)),
            moe_branch=tuple(self._checked(
                L3MOE, moe_out["moe_branch"], "layer3 MoE branch", DIM)),
            router_ids=tuple(self._checked(
                L3MOE, moe_out["ids"], "layer3 router ids", TOPK, reader=read_u32)),
            router_weights=tuple(self._checked(
                L3MOE, moe_out["weights"], "layer3 router weights", TOPK,
                reader=read_f32)),
            declared_input=tuple(self._checked(
                L3HC, {"file": hcs["residual"]["file"],
                       "sha256": hcs["residual"]["sha256"]},
                "layer3 input", FLAT)),
            declared_attn_pre=tuple(self._checked(
                L3HC, {"file": hcs["attn_pre"]["file"],
                       "sha256": hcs["attn_pre"]["sha256"]},
                "layer3 attn_pre", DIM)),
            # Gate H froze layer 3's ffn_pre as the MoE fixture's input
            # dependency rather than as one of its outputs.
            declared_ffn_pre=tuple(self._checked(
                L3MOE, prov["l3_moe"]["input_dependency"], "layer3 ffn_pre", DIM)),
            declared_output=tuple(self._checked(
                L3MOE, moe_out["layer3_final"], "layer3 output", FLAT)),
        )

    def _layer4(self) -> LayerEvidence:
        prov = self._provenance()
        route = prov["l4_route"]
        trunk = route["checkpoint_tensors"]["hyperconnection_and_router"]

        def compact(key: str, count: int) -> tuple[float, ...]:
            meta = trunk[key]
            path = os.path.join(L4ROUTE, key.replace("_", "-") + ".f32.bin")
            if not os.path.isfile(path) or sha256_file(path) != meta["payload_sha256"]:
                raise ContinuationError(f"layer4 {key}: compact parameter SHA drifted")
            return tuple(read_f32(path, count))

        params = HcParams(
            attn_fn=compact("attn_fn", MIX * FLAT),
            attn_scale=compact("attn_scale", 3),
            attn_base=compact("attn_base", MIX),
            ffn_fn=compact("ffn_fn", MIX * FLAT),
            ffn_scale=compact("ffn_scale", 3),
            ffn_base=compact("ffn_base", MIX),
        )

        out = route["outputs"]
        router = route["router"]
        ids_path = os.path.join(L4ROUTE, router["ids_file"])
        if not os.path.isfile(ids_path) or sha256_file(ids_path) != router["ids_sha256"]:
            raise ContinuationError("layer4 router ids SHA drifted")
        weights_path = os.path.join(L4ROUTE, router["weights_file"])
        if not os.path.isfile(weights_path) or \
           sha256_file(weights_path) != router["weights_sha256"]:
            raise ContinuationError("layer4 router weights SHA drifted")
        full_out = prov["l4_full"]["outputs"]

        return LayerEvidence(
            layer=4,
            structural_class="ratio4-learned",
            params=params,
            attention_branch=tuple(self._checked(
                L4ROUTE, out["attention_branch"],
                "layer4 real attention branch", DIM)),
            moe_branch=tuple(self._checked(
                L4FULL, full_out["moe_branch"], "layer4 MoE branch", DIM)),
            router_ids=tuple(read_u32(ids_path, TOPK)),
            router_weights=tuple(read_f32(weights_path, TOPK)),
            declared_input=tuple(self._checked(
                L4ROUTE, out["input"], "layer4 input", FLAT)),
            declared_attn_pre=tuple(self._checked(
                L4ROUTE, out["attn_pre"], "layer4 attn_pre", DIM)),
            declared_after_attn=tuple(self._checked(
                L4ROUTE, out["after_attn"], "layer4 after_attn", FLAT)),
            declared_ffn_pre=tuple(self._checked(
                L4ROUTE, out["ffn_pre"], "layer4 ffn_pre", DIM)),
            declared_output=tuple(self._checked(
                L4FULL, full_out["layer4_final"], "layer4 output", FLAT)),
        )

    def initial_state(self) -> tuple[int, ...]:
        """The exact prior state the first available layer consumes."""
        ev = self.evidence(self.LAYERS[0])
        if ev.declared_input is None:
            raise ContinuationError(
                f"layer {ev.layer}: source declares no input state to start from")
        return ev.declared_input


# ---------------------------------------------------------------------------
# composition backends
# ---------------------------------------------------------------------------


class HcBackend(abc.ABC):
    """One HyperConnection composition path."""

    name: str = "abstract"
    description: str = ""

    @abc.abstractmethod
    def pre(self, x: list[float], fn, scale, base):
        """Return ``(y, post, comb)`` for the pre-branch projection."""

    @abc.abstractmethod
    def post(self, branch: list[float], residual: list[float], post, comb) -> list[float]:
        """Return the flat ``[HC, DIM]`` state after the branch is folded in."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class OracleBackend(HcBackend):
    """Independent Python F32 oracle; system libm, no WASTE source linkage."""

    name = "expected"
    description = ("independent Python F32 oracle using system libm; "
                   "no WASTE source linkage")

    def pre(self, x, fn, scale, base):
        pre, post, comb, _ = hc_oracle.params(
            list(x), list(fn), list(scale), list(base), dim=DIM)
        return hc_oracle.pre_y(list(x), pre, dim=DIM), post, comb

    def post(self, branch, residual, post, comb):
        return hc_oracle.post_y(list(branch), list(residual), post, comb, dim=DIM)


class RuntimeBackend(HcBackend):
    """WASTE's C scalar HC reference, ``src/deepseek_v4_mhc_ref.c``.

    The shared object is compiled into a temporary directory, so this backend
    is a context manager: use it in a ``with`` block or the library is left
    behind and the ctypes handle outlives its file.
    """

    name = "runtime"
    description = "src/deepseek_v4_mhc_ref.c through the scalar C reference"

    def __init__(self) -> None:
        self._tmp = None
        self._pre_fn = None
        self._post_fn = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="ds-v4-continuation-")
        self._pre_fn, self._post_fn, _, _ = l4route.build_state_lib(self._tmp.name)
        return self

    def __exit__(self, *exc):
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        self._pre_fn = None
        self._post_fn = None
        return False

    def _ready(self):
        if self._pre_fn is None or self._post_fn is None:
            raise ContinuationError(
                "RuntimeBackend used outside its `with` block; the C reference "
                "library has not been built")

    def pre(self, x, fn, scale, base):
        self._ready()
        return l4route.c_pre(self._pre_fn, list(x), list(fn), list(scale), list(base))

    def post(self, branch, residual, post, comb):
        self._ready()
        return l4route.c_post(self._post_fn, list(branch), list(residual), post, comb)


# ---------------------------------------------------------------------------
# the continuation driver
# ---------------------------------------------------------------------------


def step_layer(state_bits, evidence: LayerEvidence, backend: HcBackend) -> dict:
    """Continue exactly one layer from an exact prior ``[HC, DIM]`` BF16 state.

    ``state_bits`` is the carried state, not the layer's frozen input. When the
    evidence declares an input, the two must be byte-identical: that check is
    what makes a chained walk a chain rather than a sequence of replays.
    """
    state_bits = list(state_bits)
    if len(state_bits) != FLAT:
        raise ContinuationError(
            f"layer {evidence.layer}: prior state has {len(state_bits)} values, "
            f"expected {FLAT}")
    if evidence.declared_input is not None:
        exact_bits(f"layer {evidence.layer} carried input",
                   state_bits, list(evidence.declared_input))

    p = evidence.params
    residual = bf16_floats(state_bits)
    attn = bf16_floats(list(evidence.attention_branch))
    moe = bf16_floats(list(evidence.moe_branch))

    attn_pre_f, apost, acomb = backend.pre(residual, p.attn_fn, p.attn_scale, p.attn_base)
    attn_pre, _ = chain.cast_bf16(attn_pre_f)
    if evidence.declared_attn_pre is not None:
        exact_bits(f"{backend.name} layer {evidence.layer} attn_pre",
                   attn_pre, list(evidence.declared_attn_pre))

    after_bits, after = chain.cast_bf16(backend.post(attn, residual, apost, acomb))
    if evidence.declared_after_attn is not None:
        exact_bits(f"{backend.name} layer {evidence.layer} after_attn",
                   after_bits, list(evidence.declared_after_attn))

    ffn_pre_f, fpost, fcomb = backend.pre(after, p.ffn_fn, p.ffn_scale, p.ffn_base)
    ffn_pre, _ = chain.cast_bf16(ffn_pre_f)
    if evidence.declared_ffn_pre is not None:
        exact_bits(f"{backend.name} layer {evidence.layer} ffn_pre",
                   ffn_pre, list(evidence.declared_ffn_pre))

    output, _ = chain.cast_bf16(backend.post(moe, after, fpost, fcomb))
    if evidence.declared_output is not None:
        exact_bits(f"{backend.name} layer {evidence.layer} output",
                   output, list(evidence.declared_output))

    return {
        "input": state_bits,
        "attn_pre": attn_pre,
        "attention_branch": list(evidence.attention_branch),
        "after_attn": after_bits,
        "ffn_pre": ffn_pre,
        "router_ids": list(evidence.router_ids),
        "router_weights": list(evidence.router_weights),
        "moe_branch": list(evidence.moe_branch),
        "output": output,
    }


def resolve_layers(source: EvidenceSource, layers=None) -> tuple[int, ...]:
    """Validate a layer walk: non-empty, consecutive, and available."""
    if layers is None:
        layers = source.available_layers()
    layers = tuple(layers)
    if not layers:
        raise ContinuationError("a continuation needs at least one layer")
    for a, b in zip(layers, layers[1:]):
        if b != a + 1:
            raise ContinuationError(
                f"layers must be consecutive: got {a} then {b}")
    available = set(source.available_layers())
    for n in layers:
        if n not in available:
            raise source.refuse(n)
    return layers


def run_continuation(initial_state_bits, source: EvidenceSource,
                     backend: HcBackend, layers=None) -> list[tuple[int, str, dict]]:
    """Walk consecutive layers from an exact prior state.

    Returns ``[(layer_number, structural_class, boundaries)]``. The carried
    state after layer *n* is layer *n+1*'s input by construction; no evidence
    is allowed to substitute its own copy for it.
    """
    walk = resolve_layers(source, layers)
    state = list(initial_state_bits)
    out: list[tuple[int, str, dict]] = []
    for number in walk:
        ev = source.evidence(number)
        ev.validate()
        if ev.layer != number:
            raise ContinuationError(
                f"source returned layer {ev.layer} when asked for {number}")
        values = step_layer(state, ev, backend)
        out.append((number, ev.structural_class, values))
        state = values["output"]
    return out


def write_trace(root: str, name: str, layers: list[tuple[int, str, dict]]) -> str:
    """Emit the trace ``tools/v7_localize.py`` consumes."""
    os.makedirs(root, exist_ok=True)
    raw_layers = []
    for number, structural_class, values in layers:
        os.makedirs(os.path.join(root, f"layer{number}"), exist_ok=True)
        raw_boundaries = []
        for boundary, dtype, shape in BOUNDARIES:
            ext = {"U32": "u32.bin", "F32": "f32.bin"}.get(dtype, "bf16.bin")
            rel = os.path.join(f"layer{number}", f"{boundary}.{ext}")
            path = os.path.join(root, rel)
            if dtype == "U32":
                write_u32(path, values[boundary])
            elif dtype == "F32":
                write_f32(path, values[boundary])
            else:
                write_u16(path, values[boundary])
            raw_boundaries.append({
                "name": boundary,
                "file": rel.replace(os.sep, "/"),
                "dtype": dtype,
                "shape": shape,
                "sha256": sha256_file(path),
            })
        raw_layers.append({
            "layer": number,
            "structural_class": structural_class,
            "boundaries": raw_boundaries,
        })

    manifest = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REV,
        "trace_name": name,
        "layers": raw_layers,
    }
    path = os.path.join(root, "trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
