#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""
inventory.py — what is actually inside a DeepSeek-V4-Flash checkpoint?

Every RAM, disk-size and bytes-per-token claim this port makes has to come
from the checkpoint, not from a config-derived estimate. README §1 carries
first-order numbers (≈12.75 MiB per routed expert, 11,008 main expert
records, ≈3.21 GiB of routed bytes for an all-miss decode token) and every
one of them is explicitly marked an estimate. This tool replaces them.

It reads **safetensors headers only** — the 8-byte little-endian header
length and the JSON header that follows it. It never reads, materializes or
dequantizes a tensor, so it runs against a checkpoint directory in seconds
and needs neither torch nor the `safetensors` package.

  tools/inventory.py reference/deepseek-v4-flash-0731
  tools/inventory.py <dir> --json docs/inventory-0731.json
  tools/inventory.py <dir> --by-layer --strict

Two levels of completeness, and the tool always says which one it got:

  index-only   Only config.json and model.safetensors.index.json are
               present. Names can be classified and counted; per-tensor
               byte totals cannot be known, because the index maps names to
               shards without recording dtype or shape. README §22
               deliberately downloads metadata before weights, so this is
               the expected first run.

  headers      The shard files are present. Byte totals are exact, read
               from each tensor's data_offsets.

A missing shard is reported as a SKIP against the affected checks. It is
never silently treated as zero bytes, and no byte total is ever inferred
from config.json — an estimate that looks like a measurement is exactly
what this tool exists to prevent.

Gate 0 (README §5) is evaluated at the end. `--strict` turns any FAIL into
a non-zero exit so the tool can gate CI.

**Unknown tensor names are a FAIL, by design.** The classification table
below was written from the architecture description in README §1, not from
the checkpoint, because huggingface.co was unreachable when this was
written. The first engineer to run this against real weights should expect
to extend `RULES` — and README §5 is explicit about which side wins:

    If the inventory disagrees with this README, update this README.
    The checkpoint wins.
"""

import argparse
import json
import os
import re
import struct
import sys

MiB = 1 << 20
GiB = 1 << 30

# ------------------------------------------------------------- safetensors --

# Byte width of every dtype safetensors can name. Packed 4-bit weights are
# not a safetensors dtype: they arrive as U8 with an already-halved last
# dimension, which is why stored bytes come from data_offsets below and the
# packing is *detected* rather than assumed.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class HeaderOnlyIndex:
    """Tensor metadata for a sharded checkpoint, from headers alone.

    Mirrors the lazy-header trick in the upstream converter's ShardReader
    (tools/convert.py), minus the tensor reads: this class opens each shard
    exactly once and stops after the JSON header.
    """

    def __init__(self, model_dir):
        self.dir = model_dir
        path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            idx = json.load(f)
        self.weight_map = idx["weight_map"]
        self.index_metadata = idx.get("metadata", {})
        self.headers = {}         # shard filename -> parsed JSON header
        self.missing_shards = []  # shards named by the index but not present

        for shard in sorted(set(self.weight_map.values())):
            full = os.path.join(model_dir, shard)
            if not os.path.exists(full):
                self.missing_shards.append(shard)
                continue
            self.headers[shard] = self._read_header(full)

    @staticmethod
    def _read_header(path):
        """Read one safetensors header. Reads 8 + hlen bytes, never more."""
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path}: too short for a safetensors header")
            (hlen,) = struct.unpack("<Q", raw)
            # A corrupt or non-safetensors file can claim a gigantic header.
            # Refuse it rather than attempting the allocation.
            if hlen > (100 << 20):
                raise ValueError(f"{path}: implausible header length {hlen}")
            blob = f.read(hlen)
            if len(blob) != hlen:
                raise ValueError(f"{path}: truncated header")
            hdr = json.loads(blob)
        hdr.pop("__metadata__", None)   # not a tensor
        return hdr

    @property
    def have_headers(self):
        return not self.missing_shards

    def meta(self, name):
        """(dtype, shape, stored_bytes) or None when the shard is absent."""
        shard = self.weight_map[name]
        hdr = self.headers.get(shard)
        if hdr is None or name not in hdr:
            return None
        m = hdr[name]
        beg, end = m["data_offsets"]
        return m["dtype"], list(m["shape"]), end - beg


# ---------------------------------------------------------- classification --

# Scale tensors are recognised by suffix and attributed to the weight they
# scale, so "routed expert bytes" means weights *and* their scales — that is
# what a streamed expert record has to carry (README §6).
SCALE_SUFFIXES = (
    "weight_scale_inv",
    "weight_scale",
    "scale_inv",
    "scales",
    "scale",
)

# DeepSeek's reference names expert matrices w1/w2/w3; the HF export of the
# same architecture names them gate/up/down. Both are accepted and
# normalised, because Gate 0 has to count one triplet per expert either way.
MATRIX_ALIASES = {
    "w1": "w1", "gate_proj": "w1",
    "w3": "w3", "up_proj": "w3",
    "w2": "w2", "down_proj": "w2",
}

# Ordered. First match wins, so the expert rules must precede the router
# rule: `mlp.experts.7.gate_proj` and `mlp.gate` both contain "gate".
RULES = (
    (re.compile(r"\.experts\.(?P<expert>\d+)\."),        "routed_expert"),
    (re.compile(r"\.shared_experts?\."),                 "shared_expert"),
    (re.compile(r"hash|token_to_expert|expert_map"),     "router_hash"),
    (re.compile(r"\.mlp\.gate\.|e_score_correction_bias"
                r"|\brouter\b"),                         "router"),
    (re.compile(r"indexer|\bidx_"),                      "csa_indexer"),
    (re.compile(r"compress|\bkv_c\b|\bc_kv\b"),          "compressor"),
    (re.compile(r"hyper|\bmhc\b|\bhc_|manifold"),        "mhc"),
    # Before the attention rule on purpose: `post_attention_layernorm`
    # contains "attention", and a norm is a norm wherever it sits (§5 gives
    # norms their own bucket, including the ones inside the attention block).
    (re.compile(r"norm"),                                "norm"),
    # `_proj` without a trailing dot: DeepSeek's KV projection is spelled
    # `kv_a_proj_with_mqa`, so anchoring on `_proj.` silently misses it.
    (re.compile(r"_proj|\.attn\.|attention|q_a_|q_b_|kv_a_|kv_b_"),
                                                         "attention"),
    (re.compile(r"embed_tokens|\bembed\b|word_embeddings"), "embedding"),
    (re.compile(r"lm_head|output\.weight$"),             "lm_head"),
)

# Which §5 report bucket each subsystem rolls up into.
BUCKETS = {
    "routed_expert":  "main routed expert",
    "shared_expert":  "shared expert",
    "attention":      "attention/compressor/indexer",
    "compressor":     "attention/compressor/indexer",
    "csa_indexer":    "attention/compressor/indexer",
    "mhc":            "mHC",
    "embedding":      "embed/head",
    "lm_head":        "embed/head",
    "router":         "other",
    "router_hash":    "other",
    "norm":           "other",
    "dspark_only":    "DSpark",
    "unclassified":   "unclassified",
}

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
DSPARK_RE = re.compile(r"dspark|\bspark\b|\bmtp\b|speculat|draft")


def split_scale(name):
    """(base weight name, is_scale). A scale is attributed to its weight."""
    for suf in SCALE_SUFFIXES:
        if name.endswith("." + suf):
            return name[: -(len(suf) + 1)] + ".weight", True
    return name, False


def matrix_kind(name):
    """w1 / w3 / w2 for an expert matrix, else None."""
    for token, canon in MATRIX_ALIASES.items():
        if re.search(r"(^|\.)" + re.escape(token) + r"\.", name):
            return canon
    return None


def classify(name, n_main_layers):
    """Describe one tensor by name alone.

    Returns a dict with subsystem, module (main/dspark), layer, expert and
    the scale relationship. Byte counts are attached separately, because
    they need shard headers and this does not.
    """
    base, is_scale = split_scale(name)

    lm = LAYER_RE.search(name)
    layer = int(lm.group(1)) if lm else None

    # DSpark is an attached module. Two independent signals: an explicit
    # name, or a layer index past the end of the 43-layer main stack — the
    # shape a DeepSeek multi-token module has historically taken. Either is
    # enough; Gate 0 checks that the split is unambiguous.
    by_name = bool(DSPARK_RE.search(name.lower()))
    by_layer = (
        layer is not None
        and n_main_layers is not None
        and layer >= n_main_layers
    )
    module = "dspark" if (by_name or by_layer) else "main"

    subsystem = "unclassified"
    for rx, sub in RULES:
        if rx.search(name):
            subsystem = sub
            break

    # A tensor that exists only inside the attached module needs no
    # main-stack subsystem — DSpark's internals are phase 2 (README §15) and
    # its Markov/draft parameters have no analogue in the 43-layer path.
    # This is a real classification, not a catch-all: it applies only to
    # tensors already proven to be outside the main stack, whose bytes land
    # in the DSpark bucket, and the report counts them separately so the
    # category cannot quietly absorb a surprise. An unrecognised *main*
    # tensor still fails Gate 0.
    if subsystem == "unclassified" and module == "dspark":
        subsystem = "dspark_only"

    em = EXPERT_RE.search(name)
    return {
        "name": name,
        "subsystem": subsystem,
        "module": module,
        "dspark_signal": ("name" if by_name else "layer_index" if by_layer
                          else None),
        "layer": layer,
        "expert": int(em.group(1)) if em else None,
        "matrix": matrix_kind(name),
        "is_scale": is_scale,
        "scale_of": base if is_scale else None,
        # Routed experts are the streamed population; everything else has to
        # be resident for the trunk to run at all (README §16).
        "placement": "streamed" if subsystem == "routed_expert" else "resident",
    }


# ----------------------------------------------------------------- config --

def _cfg(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def load_config(model_dir):
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    # Some releases nest the real config one level down.
    if "num_hidden_layers" not in cfg:
        for v in cfg.values():
            if isinstance(v, dict) and "num_hidden_layers" in v:
                cfg = v
                break
    return {
        "layers":      _cfg(cfg, "num_hidden_layers"),
        "hidden":      _cfg(cfg, "hidden_size"),
        "n_routed":    _cfg(cfg, "n_routed_experts", "num_experts",
                            "num_local_experts"),
        "n_shared":    _cfg(cfg, "n_shared_experts", "num_shared_experts",
                            default=0) or 0,
        "top_k":       _cfg(cfg, "num_experts_per_tok", "moe_top_k"),
        "moe_inter":   _cfg(cfg, "moe_intermediate_size"),
        "vocab":       _cfg(cfg, "vocab_size"),
        "n_hash":      _cfg(cfg, "n_hash_layers", "num_hash_layers",
                            "first_k_hash_replace"),
        "_raw":        cfg,
    }


# ------------------------------------------------------------- inventory --

def build(model_dir):
    idx = HeaderOnlyIndex(model_dir)
    cfg = load_config(model_dir)
    n_layers = cfg["layers"] if cfg else None

    rows = []
    for name in sorted(idx.weight_map):
        row = classify(name, n_layers)
        meta = idx.meta(name)
        if meta is None:
            row.update(dtype=None, shape=None, stored_bytes=None,
                       shard=idx.weight_map[name], resolved=False)
        else:
            dtype, shape, nbytes = meta
            row.update(dtype=dtype, shape=shape, stored_bytes=nbytes,
                       shard=idx.weight_map[name], resolved=True)
            row["packing"] = packing_of(row, cfg)
        rows.append(row)
    return idx, cfg, rows


def packing_of(row, cfg):
    """values-per-byte, when it can be established without reading data.

    Compares the stored element count against dtype width. A routed-expert
    weight whose last dimension is half the config's expectation is packed
    two values per byte — the FP4 E2M1 layout README §7 describes. This is
    reported, never assumed: if the checkpoint is not packed the ratio comes
    back 1 and the converter plan needs revisiting.
    """
    if not row.get("resolved") or row["is_scale"]:
        return None
    width = DTYPE_BYTES.get(row["dtype"])
    if width is None or not row["shape"]:
        return None
    elems = 1
    for d in row["shape"]:
        elems *= d
    if elems == 0:
        return None
    dense = elems * width
    if dense != row["stored_bytes"]:
        return None    # ragged/unknown; do not guess

    if row["subsystem"] != "routed_expert" or not cfg or row["matrix"] is None:
        return 1.0
    expected_in = cfg["hidden"] if row["matrix"] in ("w1", "w3") else cfg["moe_inter"]
    if expected_in and len(row["shape"]) == 2 and row["shape"][1]:
        ratio = expected_in / row["shape"][1]
        # Only 1 (unpacked) and 2 (two 4-bit values per byte) are meaningful
        # here; anything else is a shape surprise worth surfacing as-is.
        if abs(ratio - round(ratio)) < 1e-9:
            return float(round(ratio))
    return 1.0


# --------------------------------------------------------------- gate 0 --

class Gate:
    def __init__(self):
        self.results = []

    def add(self, status, name, detail):
        self.results.append((status, name, detail))

    def ok(self, name, detail):
        self.add("PASS", name, detail)

    def fail(self, name, detail):
        self.add("FAIL", name, detail)

    def skip(self, name, detail):
        self.add("SKIP", name, detail)

    @property
    def failed(self):
        return any(s == "FAIL" for s, _, _ in self.results)


def gate0(idx, cfg, rows):
    """README §5 — inventory sanity."""
    g = Gate()

    # G0.1 — every name classified.
    unknown = [r["name"] for r in rows if r["subsystem"] == "unclassified"]
    if unknown:
        g.fail("all tensor names classified",
               f"{len(unknown)} unclassified, first: {unknown[:3]}")
    else:
        g.ok("all tensor names classified", f"{len(rows)} tensors")

    # G0.2 — no bytes in an unexplained bucket.
    if not idx.have_headers:
        g.skip("no bytes in an unexplained bucket",
               f"{len(idx.missing_shards)} shard(s) absent — byte totals "
               "unavailable in index-only mode")
    else:
        stray = sum(r["stored_bytes"] for r in rows
                    if r["subsystem"] == "unclassified")
        if stray:
            g.fail("no bytes in an unexplained bucket", f"{human(stray)} stray")
        else:
            g.ok("no bytes in an unexplained bucket", "0 stray bytes")

    # G0.3 — dimensions agree with config.json.
    if cfg is None:
        g.skip("dimensions agree with config.json", "no config.json")
    elif not idx.have_headers:
        g.skip("dimensions agree with config.json", "no shard headers")
    else:
        g.add(*_check_dims(cfg, rows))

    # G0.4 — one w1/w2/w3 per routed expert, plus scales.
    g.add(*_check_experts(cfg, rows))

    # G0.5 — the first three layers expose their routing tables.
    g.add(*_check_hash_layers(cfg, rows))

    # G0.6 — DSpark separable from the main stack.
    g.add(*_check_dspark(cfg, rows))

    return g


def _check_dims(cfg, rows):
    name = "dimensions agree with config.json"
    problems = []

    for r in rows:
        if r["subsystem"] != "routed_expert" or r["is_scale"]:
            continue
        if r["matrix"] is None or not r["shape"] or len(r["shape"]) != 2:
            continue
        out, inn = r["shape"]
        pack = r.get("packing") or 1.0
        want_out = cfg["moe_inter"] if r["matrix"] in ("w1", "w3") else cfg["hidden"]
        want_in = cfg["hidden"] if r["matrix"] in ("w1", "w3") else cfg["moe_inter"]
        if want_out and out != want_out:
            problems.append(f"{r['name']}: out {out} != {want_out}")
        if want_in and inn * pack != want_in:
            problems.append(
                f"{r['name']}: in {inn} x packing {pack:g} != {want_in}")
        if len(problems) >= 5:
            break

    for r in rows:
        if r["subsystem"] == "embedding" and r["shape"] and len(r["shape"]) == 2:
            if cfg["vocab"] and r["shape"][0] != cfg["vocab"]:
                problems.append(
                    f"{r['name']}: vocab {r['shape'][0]} != {cfg['vocab']}")
            if cfg["hidden"] and r["shape"][1] != cfg["hidden"]:
                problems.append(
                    f"{r['name']}: hidden {r['shape'][1]} != {cfg['hidden']}")

    if problems:
        return "FAIL", name, "; ".join(problems[:5])
    return "PASS", name, "expert and embedding shapes match config"


def _check_experts(cfg, rows):
    name = "one w1/w2/w3 per routed expert, with scales"
    seen = {}      # (module, layer, expert) -> {matrix: {"weight", "scale"}}
    for r in rows:
        if r["subsystem"] != "routed_expert":
            continue
        if r["layer"] is None or r["expert"] is None or r["matrix"] is None:
            continue
        slot = seen.setdefault((r["module"], r["layer"], r["expert"]), {})
        entry = slot.setdefault(r["matrix"], {"weight": 0, "scale": 0})
        entry["scale" if r["is_scale"] else "weight"] += 1

    if not seen:
        return "SKIP", name, "no routed-expert tensors found"

    bad, no_scale = [], []
    for (module, layer, exp), mats in sorted(seen.items()):
        tag = f"L{layer}/E{exp}" + ("" if module == "main" else " (dspark)")
        for kind in ("w1", "w3", "w2"):
            e = mats.get(kind)
            if e is None or e["weight"] != 1:
                bad.append(f"{tag} {kind}="
                           f"{0 if e is None else e['weight']}")
            elif e["scale"] == 0:
                no_scale.append(f"{tag} {kind}")
        if len(bad) >= 5:
            break

    if bad:
        return "FAIL", name, f"{len(bad)}+ malformed: {bad[:5]}"
    n_main = sum(1 for k in seen if k[0] == "main")
    detail = f"{n_main} main expert records ({len(seen)} incl. DSpark), each with w1+w3+w2"
    if no_scale:
        # Not fatal on its own: a checkpoint could ship unquantized experts.
        # It does contradict README §1, so it must be visible.
        return "FAIL", name, (
            f"{len(no_scale)} matrices carry no scale tensor "
            f"(first: {no_scale[:3]}) — README §1 expects UE8M0 K32 scales")
    return "PASS", name, detail + ", each with a scale tensor"


def _check_hash_layers(cfg, rows):
    name = "first three layers expose token-id -> expert mapping"
    n_hash = (cfg or {}).get("n_hash") or 3
    hash_layers = {r["layer"] for r in rows
                   if r["subsystem"] == "router_hash" and r["layer"] is not None}
    want = set(range(n_hash))
    if not hash_layers:
        return "FAIL", name, (
            f"no hash-routing tensors found; README §11 expects deterministic "
            f"routing in layers 0..{n_hash - 1}. Either the tables are named "
            "unexpectedly (extend RULES) or routing is computed, not stored")
    missing = want - hash_layers
    if missing:
        return "FAIL", name, f"layers {sorted(missing)} have no mapping tensor"
    return "PASS", name, f"layers {sorted(want)} carry mapping tensors"


def _check_dspark(cfg, rows):
    name = "DSpark tensors separable from the main stack"
    ds = [r for r in rows if r["module"] == "dspark"]
    if not ds:
        return "SKIP", name, "no DSpark tensors in this checkpoint"
    if cfg is None or cfg["layers"] is None:
        return "SKIP", name, "no config.json to bound the main stack"

    # Ambiguity is the failure mode that matters, and it is about namespace,
    # not layer number. A DSpark tensor under its own root (`dspark.layers.2`)
    # is unambiguous however it is numbered. A DSpark tensor sharing the main
    # stack's root at a layer inside the main range (`model.layers.2.dspark_*`)
    # is not: a loader walking model.layers.0..42 sweeps it into the 43-layer
    # path. Checking the layer index alone cannot catch that — an out-of-range
    # index is what marks a tensor DSpark in the first place.
    main_roots = {r["name"].split(".", 1)[0] for r in rows
                  if r["module"] == "main" and r["layer"] is not None}
    overlap = [r["name"] for r in ds
               if r["layer"] is not None and r["layer"] < cfg["layers"]
               and r["name"].split(".", 1)[0] in main_roots]
    if overlap:
        return "FAIL", name, (
            f"{len(overlap)} DSpark tensor(s) share the main stack's "
            f"namespace inside layers 0..{cfg['layers'] - 1}: {overlap[:3]}")
    main_layers = {r["layer"] for r in rows
                   if r["module"] == "main" and r["layer"] is not None}
    if main_layers and max(main_layers) >= cfg["layers"]:
        return "FAIL", name, (
            f"main-module tensor at layer {max(main_layers)} >= "
            f"num_hidden_layers {cfg['layers']}")
    return "PASS", name, (
        f"{len(ds)} DSpark tensors, main stack is layers "
        f"0..{cfg['layers'] - 1}")


# ---------------------------------------------------------------- report --

def human(n):
    if n is None:
        return "?"
    if n >= GiB:
        return f"{n / GiB:.2f} GiB"
    if n >= MiB:
        return f"{n / MiB:.2f} MiB"
    return f"{n} B"


def totals(rows):
    """Bucket -> (bytes, tensor count). Bytes are None when unresolved."""
    out = {}
    for r in rows:
        bucket = ("DSpark" if r["module"] == "dspark"
                  else BUCKETS.get(r["subsystem"], "other"))
        b, c, unres = out.get(bucket, (0, 0, 0))
        if r["stored_bytes"] is None:
            unres += 1
        else:
            b += r["stored_bytes"]
        out[bucket] = (b, c + 1, unres)
    return out


BUCKET_ORDER = ("main routed expert", "shared expert",
                "attention/compressor/indexer", "mHC", "embed/head",
                "DSpark", "other", "unclassified")


def report(idx, cfg, rows, gate, by_layer=False):
    mode = "headers" if idx.have_headers else "index-only"
    print(f"\n  DeepSeek-V4-Flash checkpoint inventory   [mode: {mode}]")
    print(f"  {len(rows)} tensors across {len(set(idx.weight_map.values()))} shard(s)")
    if idx.missing_shards:
        print(f"  {len(idx.missing_shards)} shard(s) absent — byte totals are "
              "partial. Not counted as zero.")
    if cfg:
        print(f"  config: {cfg['layers']} layers, hidden {cfg['hidden']}, "
              f"{cfg['n_routed']} routed + {cfg['n_shared']} shared experts, "
              f"top-{cfg['top_k']}, moe_inter {cfg['moe_inter']}")

    tot = totals(rows)
    print("\n  bytes by bucket")
    print(f"  {'bucket':<32} {'stored':>12} {'tensors':>9}  {'unresolved':>10}")
    print(f"  {'-' * 32} {'-' * 12:>12} {'-' * 9:>9}  {'-' * 10:>10}")
    grand = 0
    for name in BUCKET_ORDER:
        if name not in tot:
            continue
        b, c, unres = tot[name]
        grand += b
        print(f"  {name:<32} {human(b):>12} {c:>9}  {unres or '':>10}")
    print(f"  {'-' * 32} {'-' * 12:>12} {'-' * 9:>9}  {'-' * 10:>10}")
    print(f"  {'total':<32} {human(grand):>12} {len(rows):>9}")

    # Named so it cannot hide: these are DSpark tensors with no main-stack
    # analogue, classified by module rather than by a matched rule.
    ds_only = [r for r in rows if r["subsystem"] == "dspark_only"]
    if ds_only:
        names = sorted({r["name"].split(".")[-1] for r in ds_only})
        print(f"\n  {len(ds_only)} DSpark-only tensors matched no main-stack "
              f"rule (suffixes: {', '.join(names[:6])})")

    # Per-token routed traffic: the number README §1 estimates.
    _report_expert_math(cfg, rows)

    if by_layer:
        _report_by_layer(rows)

    print("\n  Gate 0 — inventory sanity")
    for status, name, detail in gate.results:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        print(f"    {mark}  {name}")
        print(f"          {detail}")

    n_fail = sum(1 for s, _, _ in gate.results if s == "FAIL")
    n_skip = sum(1 for s, _, _ in gate.results if s == "SKIP")
    n_pass = sum(1 for s, _, _ in gate.results if s == "PASS")
    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n")


def _report_expert_math(cfg, rows):
    """Exact routed-expert traffic, or nothing at all."""
    recs = {}
    for r in rows:
        if r["subsystem"] != "routed_expert" or r["module"] != "main":
            continue
        if r["layer"] is None or r["expert"] is None:
            continue
        if r["stored_bytes"] is None:
            return       # partial data would produce a misleading figure
        key = (r["layer"], r["expert"])
        recs[key] = recs.get(key, 0) + r["stored_bytes"]
    if not recs:
        return

    sizes = sorted(set(recs.values()))
    n = len(recs)
    layers = len({k[0] for k in recs})
    print("\n  routed-expert traffic (exact, from data_offsets)")
    print(f"    expert records                {n}")
    print(f"    MoE layers                    {layers}")
    if len(sizes) == 1:
        print(f"    bytes per expert record       {human(sizes[0])} "
              f"({sizes[0]} B, payload only, before header/4 KiB padding)")
    else:
        print(f"    bytes per expert record       {human(sizes[0])} .. "
              f"{human(sizes[-1])} ({len(sizes)} distinct sizes)")
    if cfg and cfg["top_k"]:
        per_layer = sum(recs.values()) / n
        cold = per_layer * cfg["top_k"] * layers
        print(f"    all-miss decode token         {human(cold)} "
              f"({cfg['top_k']} experts x {layers} layers)")
        print("    ^ payload only; excludes record headers and 4 KiB "
              "alignment, and assumes zero cache hits")


def _report_by_layer(rows):
    per = {}
    for r in rows:
        if r["layer"] is None or r["stored_bytes"] is None:
            continue
        key = (r["module"], r["layer"])
        streamed, resident = per.get(key, (0, 0))
        if r["placement"] == "streamed":
            streamed += r["stored_bytes"]
        else:
            resident += r["stored_bytes"]
        per[key] = (streamed, resident)
    if not per:
        return
    print("\n  by layer")
    print(f"  {'layer':<12} {'streamed':>12} {'resident':>12}")
    for (module, layer) in sorted(per):
        s, res = per[(module, layer)]
        label = f"{layer}" if module == "main" else f"{layer} (dspark)"
        print(f"  {label:<12} {human(s):>12} {human(res):>12}")


def emit_json(path, idx, cfg, rows, gate):
    tot = totals(rows)
    doc = {
        "mode": "headers" if idx.have_headers else "index-only",
        "missing_shards": idx.missing_shards,
        "index_metadata": idx.index_metadata,
        "config": {k: v for k, v in (cfg or {}).items() if k != "_raw"},
        "totals": {k: {"stored_bytes": b, "tensors": c, "unresolved": u}
                   for k, (b, c, u) in tot.items()},
        "gate0": [{"status": s, "check": n, "detail": d}
                  for s, n, d in gate.results],
        "tensors": rows,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"  wrote {path}")


# ------------------------------------------------------------------ main --

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Inventory a DeepSeek-V4-Flash checkpoint from "
                    "safetensors headers only.")
    ap.add_argument("model_dir",
                    help="directory holding config.json and "
                         "model.safetensors.index.json")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the full per-tensor inventory as JSON")
    ap.add_argument("--by-layer", action="store_true",
                    help="print streamed/resident bytes per layer")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any Gate 0 check fails")
    args = ap.parse_args(argv)

    try:
        idx, cfg, rows = build(args.model_dir)
    except FileNotFoundError as e:
        print(f"inventory: no such file: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"inventory: malformed checkpoint metadata: {e}", file=sys.stderr)
        return 2

    gate = gate0(idx, cfg, rows)
    report(idx, cfg, rows, gate, by_layer=args.by_layer)
    if args.json:
        emit_json(args.json, idx, cfg, rows, gate)

    if args.strict and gate.failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
