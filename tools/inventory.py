#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Inventory a DeepSeek-V4-Flash checkpoint from safetensors headers only.

Every RAM, disk-size and bytes-per-token claim in this port must descend from
checkpoint metadata rather than a config-derived estimate. This tool reads the
8-byte safetensors header length and JSON header only; it never materializes or
dequantizes tensor payloads and requires neither torch nor safetensors.

Examples:

  tools/inventory.py reference/deepseek-v4-flash-0731
  tools/inventory.py reference/deepseek-v4-flash-0731 --strict --by-layer
  tools/inventory.py reference/deepseek-v4-flash-0731 --json docs/inventory-0731.json

Completeness is explicit:

  index-only   config.json + model.safetensors.index.json are present. Names
               can be classified, but dtype/shape/byte totals are unresolved.

  headers      every indexed shard has a readable safetensors header (a real
               shard or a header-only stub from fetch_hf_headers.py). Byte
               totals are exact from data_offsets.

Missing headers produce SKIP/unresolved values, never inferred zeros. Unknown
main-model names are a FAIL by design. Gate A/V0 is the checkpoint-truth gate;
`--strict` makes any Gate A failure non-zero.

The classifier began from synthetic handoff names, but is intentionally amended
only with narrow names established by pinned official source/artifacts. In
particular the 0731 runtime names the bootstrap token->expert table `tid2eid`,
while its converter contains the source spelling `tie2eid`; both are recognized
as router-hash tensors. Unknown main tensors remain fatal.
"""

import argparse
import json
import os
import re
import struct
import sys

MiB = 1 << 20
GiB = 1 << 30

# Stored-width awareness is useful for sanity/packing detection. Exact stored
# bytes always come from safetensors data_offsets, so a new dtype is never
# silently converted into an estimated byte count.
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class HeaderOnlyIndex:
    """Tensor metadata for a sharded checkpoint, from headers alone."""

    def __init__(self, model_dir):
        self.dir = model_dir
        path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            idx = json.load(f)
        self.weight_map = idx["weight_map"]
        self.index_metadata = idx.get("metadata", {})
        self.headers = {}
        self.missing_shards = []

        for shard in sorted(set(self.weight_map.values())):
            full = os.path.join(model_dir, shard)
            if not os.path.exists(full):
                self.missing_shards.append(shard)
                continue
            self.headers[shard] = self._read_header(full)

    @staticmethod
    def _read_header(path):
        """Read exactly 8 + hlen bytes from one safetensors file/stub."""
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path}: too short for a safetensors header")
            (hlen,) = struct.unpack("<Q", raw)
            if hlen == 0 or hlen > (100 << 20):
                raise ValueError(f"{path}: implausible header length {hlen}")
            blob = f.read(hlen)
            if len(blob) != hlen:
                raise ValueError(f"{path}: truncated header")
            hdr = json.loads(blob)
        if not isinstance(hdr, dict):
            raise ValueError(f"{path}: safetensors header is not an object")
        hdr.pop("__metadata__", None)
        return hdr

    @property
    def have_headers(self):
        return not self.missing_shards

    def meta(self, name):
        """Return (dtype, shape, stored_bytes), or None if its shard is absent."""
        shard = self.weight_map[name]
        hdr = self.headers.get(shard)
        if hdr is None or name not in hdr:
            return None
        m = hdr[name]
        beg, end = m["data_offsets"]
        return m["dtype"], list(m["shape"]), end - beg


# ---------------------------------------------------------- classification --

SCALE_SUFFIXES = (
    "weight_scale_inv",
    "weight_scale",
    "scale_inv",
    "scales",
    "scale",
)

MATRIX_ALIASES = {
    "w1": "w1",
    "gate_proj": "w1",
    "w3": "w3",
    "up_proj": "w3",
    "w2": "w2",
    "down_proj": "w2",
}

# Ordered: expert rules precede generic router rules. Official-source additions
# remain narrow. `tid2eid` is the pinned 0731 runtime field; `tie2eid` is the
# spelling handled by the pinned converter before normalized naming.
RULES = (
    (re.compile(r"\.experts\.(?P<expert>\d+)\."), "routed_expert"),
    (re.compile(r"\.shared_experts?\."), "shared_expert"),
    (re.compile(r"tid2eid|tie2eid|hash|token_to_expert|expert_map"),
     "router_hash"),
    (re.compile(r"\.mlp\.gate\.|e_score_correction_bias|\brouter\b"),
     "router"),
    (re.compile(r"indexer|\bidx_"), "csa_indexer"),
    (re.compile(r"compress|\bkv_c\b|\bc_kv\b"), "compressor"),
    (re.compile(r"hyper|\bmhc\b|\bhc_|manifold"), "mhc"),
    (re.compile(r"norm"), "norm"),
    (re.compile(r"_proj|\.attn\.|attention|q_a_|q_b_|kv_a_|kv_b_"),
     "attention"),
    (re.compile(r"embed_tokens|\bembed\b|word_embeddings"), "embedding"),
    (re.compile(r"lm_head|output\.weight$"), "lm_head"),
)

BUCKETS = {
    "routed_expert": "main routed expert",
    "shared_expert": "shared expert",
    "attention": "attention/compressor/indexer",
    "compressor": "attention/compressor/indexer",
    "csa_indexer": "attention/compressor/indexer",
    "mhc": "mHC",
    "embedding": "embed/head",
    "lm_head": "embed/head",
    "router": "other",
    "router_hash": "other",
    "norm": "other",
    "dspark_only": "DSpark",
    "unclassified": "unclassified",
}

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
DSPARK_RE = re.compile(r"dspark|\bspark\b|\bmtp\b|speculat|draft")


def split_scale(name):
    """Return (owning weight name, is_scale)."""
    for suffix in SCALE_SUFFIXES:
        if name.endswith("." + suffix):
            return name[: -(len(suffix) + 1)] + ".weight", True
    return name, False


def matrix_kind(name):
    for token, canon in MATRIX_ALIASES.items():
        if re.search(r"(^|\.)" + re.escape(token) + r"\.", name):
            return canon
    return None


def classify(name, n_main_layers):
    """Classify one tensor by name; byte metadata is attached separately."""
    base, is_scale = split_scale(name)
    lm = LAYER_RE.search(name)
    layer = int(lm.group(1)) if lm else None

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

    # An unknown tensor is only allowed into the DSpark bucket after its module
    # boundary is already established independently. Unknown main tensors fail.
    if subsystem == "unclassified" and module == "dspark":
        subsystem = "dspark_only"

    em = EXPERT_RE.search(name)
    return {
        "name": name,
        "subsystem": subsystem,
        "module": module,
        "dspark_signal": (
            "name" if by_name else "layer_index" if by_layer else None
        ),
        "layer": layer,
        "expert": int(em.group(1)) if em else None,
        "matrix": matrix_kind(name),
        "is_scale": is_scale,
        "scale_of": base if is_scale else None,
        "placement": "streamed" if subsystem == "routed_expert" else "resident",
    }


# ----------------------------------------------------------------- config --

def _cfg(d, *keys, default=None):
    for key in keys:
        if key in d:
            return d[key]
    return default


def load_config(model_dir):
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "num_hidden_layers" not in cfg:
        for value in cfg.values():
            if isinstance(value, dict) and "num_hidden_layers" in value:
                cfg = value
                break
    return {
        "layers": _cfg(cfg, "num_hidden_layers"),
        "hidden": _cfg(cfg, "hidden_size"),
        "n_routed": _cfg(cfg, "n_routed_experts", "num_experts",
                         "num_local_experts"),
        "n_shared": _cfg(cfg, "n_shared_experts", "num_shared_experts",
                         default=0) or 0,
        "top_k": _cfg(cfg, "num_experts_per_tok", "moe_top_k"),
        "moe_inter": _cfg(cfg, "moe_intermediate_size"),
        "vocab": _cfg(cfg, "vocab_size"),
        "n_hash": _cfg(cfg, "n_hash_layers", "num_hash_layers",
                       "first_k_hash_replace"),
        "_raw": cfg,
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
    """Return logical values per stored element when header/config prove it."""
    if not row.get("resolved") or row["is_scale"]:
        return None
    width = DTYPE_BYTES.get(row["dtype"])
    if width is None or not row["shape"]:
        return None
    elems = 1
    for dim in row["shape"]:
        elems *= dim
    if elems == 0 or elems * width != row["stored_bytes"]:
        return None

    if row["subsystem"] != "routed_expert" or not cfg or row["matrix"] is None:
        return 1.0
    expected_in = (
        cfg["hidden"] if row["matrix"] in ("w1", "w3") else cfg["moe_inter"]
    )
    if expected_in and len(row["shape"]) == 2 and row["shape"][1]:
        ratio = expected_in / row["shape"][1]
        if abs(ratio - round(ratio)) < 1e-9:
            return float(round(ratio))
    return 1.0


# ------------------------------------------------------------ Gate A / V0 --
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
        return any(status == "FAIL" for status, _, _ in self.results)


def gate0(idx, cfg, rows):
    """Gate A/V0 — checkpoint inventory sanity.

    Function name remains `gate0` for compatibility with the existing JSON
    schema/tests; documentation and CLI output use the canonical gate name.
    """
    g = Gate()

    unknown = [r["name"] for r in rows if r["subsystem"] == "unclassified"]
    if unknown:
        g.fail("all tensor names classified",
               f"{len(unknown)} unclassified, first: {unknown[:3]}")
    else:
        g.ok("all tensor names classified", f"{len(rows)} tensors")

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

    if cfg is None:
        g.skip("dimensions agree with config.json", "no config.json")
    elif not idx.have_headers:
        g.skip("dimensions agree with config.json", "no shard headers")
    else:
        g.add(*_check_dims(cfg, rows))

    g.add(*_check_experts(cfg, rows))
    g.add(*_check_hash_layers(cfg, rows))
    g.add(*_check_dspark(cfg, rows))
    return g


def _check_dims(cfg, rows):
    name = "dimensions agree with config.json"
    problems = []

    for row in rows:
        if row["subsystem"] != "routed_expert" or row["is_scale"]:
            continue
        if row["matrix"] is None or not row["shape"] or len(row["shape"]) != 2:
            continue
        out_dim, in_dim = row["shape"]
        packing = row.get("packing") or 1.0
        want_out = (
            cfg["moe_inter"] if row["matrix"] in ("w1", "w3") else cfg["hidden"]
        )
        want_in = (
            cfg["hidden"] if row["matrix"] in ("w1", "w3") else cfg["moe_inter"]
        )
        if want_out and out_dim != want_out:
            problems.append(f"{row['name']}: out {out_dim} != {want_out}")
        if want_in and in_dim * packing != want_in:
            problems.append(
                f"{row['name']}: in {in_dim} x packing {packing:g} != {want_in}")
        if len(problems) >= 5:
            break

    for row in rows:
        if row["subsystem"] == "embedding" and row["shape"] and len(row["shape"]) == 2:
            if cfg["vocab"] and row["shape"][0] != cfg["vocab"]:
                problems.append(
                    f"{row['name']}: vocab {row['shape'][0]} != {cfg['vocab']}")
            if cfg["hidden"] and row["shape"][1] != cfg["hidden"]:
                problems.append(
                    f"{row['name']}: hidden {row['shape'][1]} != {cfg['hidden']}")

    if problems:
        return "FAIL", name, "; ".join(problems[:5])
    return "PASS", name, "expert and embedding shapes match config"


def _check_experts(cfg, rows):
    del cfg
    name = "one w1/w2/w3 per routed expert, with scales"
    seen = {}
    for row in rows:
        if row["subsystem"] != "routed_expert":
            continue
        if row["layer"] is None or row["expert"] is None or row["matrix"] is None:
            continue
        slot = seen.setdefault(
            (row["module"], row["layer"], row["expert"]), {})
        entry = slot.setdefault(row["matrix"], {"weight": 0, "scale": 0})
        entry["scale" if row["is_scale"] else "weight"] += 1

    if not seen:
        return "SKIP", name, "no routed-expert tensors found"

    bad = []
    no_scale = []
    for (module, layer, expert), matrices in sorted(seen.items()):
        tag = f"L{layer}/E{expert}" + ("" if module == "main" else " (dspark)")
        for kind in ("w1", "w3", "w2"):
            entry = matrices.get(kind)
            if entry is None or entry["weight"] != 1:
                bad.append(f"{tag} {kind}={0 if entry is None else entry['weight']}")
            elif entry["scale"] == 0:
                no_scale.append(f"{tag} {kind}")
        if len(bad) >= 5:
            break

    if bad:
        return "FAIL", name, f"{len(bad)}+ malformed: {bad[:5]}"
    if no_scale:
        return "FAIL", name, (
            f"{len(no_scale)} matrices carry no scale tensor "
            f"(first: {no_scale[:3]}) — the pinned 0731 source expects "
            "scaled routed experts")

    n_main = sum(1 for key in seen if key[0] == "main")
    return "PASS", name, (
        f"{n_main} main expert records ({len(seen)} incl. DSpark), "
        "each with w1+w3+w2 and a scale tensor")


def _check_hash_layers(cfg, rows):
    name = "first three layers expose token-id -> expert mapping"
    n_hash = (cfg or {}).get("n_hash") or 3
    hash_layers = {
        row["layer"] for row in rows
        if row["subsystem"] == "router_hash" and row["layer"] is not None
    }
    want = set(range(n_hash))
    if not hash_layers:
        return "FAIL", name, (
            f"no bootstrap-routing tensors found; official source expects "
            f"tid2eid semantics in layers 0..{n_hash - 1}. Either the export "
            "uses another proven spelling (extend RULES narrowly) or the "
            "checkpoint/source contract needs reconciliation")
    missing = want - hash_layers
    if missing:
        return "FAIL", name, f"layers {sorted(missing)} have no mapping tensor"
    return "PASS", name, f"layers {sorted(want)} carry mapping tensors"


def _check_dspark(cfg, rows):
    name = "DSpark tensors separable from the main stack"
    dspark = [row for row in rows if row["module"] == "dspark"]
    if not dspark:
        return "SKIP", name, "no DSpark tensors in this checkpoint"
    if cfg is None or cfg["layers"] is None:
        return "SKIP", name, "no config.json to bound the main stack"

    main_roots = {
        row["name"].split(".", 1)[0] for row in rows
        if row["module"] == "main" and row["layer"] is not None
    }
    overlap = [
        row["name"] for row in dspark
        if row["layer"] is not None
        and row["layer"] < cfg["layers"]
        and row["name"].split(".", 1)[0] in main_roots
    ]
    if overlap:
        return "FAIL", name, (
            f"{len(overlap)} DSpark tensor(s) share the main stack namespace "
            f"inside layers 0..{cfg['layers'] - 1}: {overlap[:3]}")

    main_layers = {
        row["layer"] for row in rows
        if row["module"] == "main" and row["layer"] is not None
    }
    if main_layers and max(main_layers) >= cfg["layers"]:
        return "FAIL", name, (
            f"main-module tensor at layer {max(main_layers)} >= "
            f"num_hidden_layers {cfg['layers']}")
    return "PASS", name, (
        f"{len(dspark)} DSpark tensors, main stack is layers "
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
    """Return bucket -> (resolved bytes, tensor count, unresolved count)."""
    out = {}
    for row in rows:
        bucket = (
            "DSpark" if row["module"] == "dspark"
            else BUCKETS.get(row["subsystem"], "other")
        )
        stored, count, unresolved = out.get(bucket, (0, 0, 0))
        if row["stored_bytes"] is None:
            unresolved += 1
        else:
            stored += row["stored_bytes"]
        out[bucket] = (stored, count + 1, unresolved)
    return out


BUCKET_ORDER = (
    "main routed expert",
    "shared expert",
    "attention/compressor/indexer",
    "mHC",
    "embed/head",
    "DSpark",
    "other",
    "unclassified",
)


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

    total_map = totals(rows)
    print("\n  bytes by bucket")
    print(f"  {'bucket':<32} {'stored':>12} {'tensors':>9}  {'unresolved':>10}")
    print(f"  {'-' * 32} {'-' * 12:>12} {'-' * 9:>9}  {'-' * 10:>10}")
    grand = 0
    for bucket in BUCKET_ORDER:
        if bucket not in total_map:
            continue
        stored, count, unresolved = total_map[bucket]
        grand += stored
        print(f"  {bucket:<32} {human(stored):>12} {count:>9}  "
              f"{unresolved or '':>10}")
    print(f"  {'-' * 32} {'-' * 12:>12} {'-' * 9:>9}  {'-' * 10:>10}")
    print(f"  {'total':<32} {human(grand):>12} {len(rows):>9}")

    ds_only = [row for row in rows if row["subsystem"] == "dspark_only"]
    if ds_only:
        suffixes = sorted({row["name"].split(".")[-1] for row in ds_only})
        print(f"\n  {len(ds_only)} DSpark-only tensors matched no main-stack "
              f"rule (suffixes: {', '.join(suffixes[:6])})")

    _report_expert_math(cfg, rows)
    if by_layer:
        _report_by_layer(rows)

    print("\n  Gate A / V0 — checkpoint inventory")
    for status, name, detail in gate.results:
        print(f"    {status:<4}  {name}")
        print(f"          {detail}")

    n_fail = sum(1 for status, _, _ in gate.results if status == "FAIL")
    n_skip = sum(1 for status, _, _ in gate.results if status == "SKIP")
    n_pass = sum(1 for status, _, _ in gate.results if status == "PASS")
    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped\n")


def _report_expert_math(cfg, rows):
    """Report exact routed-expert traffic, or nothing for partial metadata."""
    records = {}
    for row in rows:
        if row["subsystem"] != "routed_expert" or row["module"] != "main":
            continue
        if row["layer"] is None or row["expert"] is None:
            continue
        if row["stored_bytes"] is None:
            return
        key = (row["layer"], row["expert"])
        records[key] = records.get(key, 0) + row["stored_bytes"]
    if not records:
        return

    sizes = sorted(set(records.values()))
    n_records = len(records)
    layers = len({key[0] for key in records})
    print("\n  routed-expert traffic (exact, from data_offsets)")
    print(f"    expert records                {n_records}")
    print(f"    MoE layers                    {layers}")
    if len(sizes) == 1:
        print(f"    bytes per expert record       {human(sizes[0])} "
              f"({sizes[0]} B, payload only, before header/4 KiB padding)")
    else:
        print(f"    bytes per expert record       {human(sizes[0])} .. "
              f"{human(sizes[-1])} ({len(sizes)} distinct sizes)")
    if cfg and cfg["top_k"]:
        average_record = sum(records.values()) / n_records
        cold = average_record * cfg["top_k"] * layers
        print(f"    all-miss decode token         {human(cold)} "
              f"({cfg['top_k']} experts x {layers} layers)")
        print("    ^ payload only; excludes record headers and 4 KiB alignment, "
              "and assumes zero cache hits")


def _report_by_layer(rows):
    per_layer = {}
    for row in rows:
        if row["layer"] is None or row["stored_bytes"] is None:
            continue
        key = (row["module"], row["layer"])
        streamed, resident = per_layer.get(key, (0, 0))
        if row["placement"] == "streamed":
            streamed += row["stored_bytes"]
        else:
            resident += row["stored_bytes"]
        per_layer[key] = (streamed, resident)
    if not per_layer:
        return

    print("\n  by layer")
    print(f"  {'layer':<12} {'streamed':>12} {'resident':>12}")
    for (module, layer) in sorted(per_layer):
        streamed, resident = per_layer[(module, layer)]
        label = f"{layer}" if module == "main" else f"{layer} (dspark)"
        print(f"  {label:<12} {human(streamed):>12} {human(resident):>12}")


def emit_json(path, idx, cfg, rows, gate):
    total_map = totals(rows)
    doc = {
        "mode": "headers" if idx.have_headers else "index-only",
        "missing_shards": idx.missing_shards,
        "index_metadata": idx.index_metadata,
        "config": {key: value for key, value in (cfg or {}).items()
                   if key != "_raw"},
        "totals": {
            key: {
                "stored_bytes": stored,
                "tensors": count,
                "unresolved": unresolved,
            }
            for key, (stored, count, unresolved) in total_map.items()
        },
        # Keep the established machine-readable key while human-facing docs use
        # the canonical Gate A/V0 name.
        "gate0": [
            {"status": status, "check": name, "detail": detail}
            for status, name, detail in gate.results
        ],
        "tensors": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"  wrote {path}")


# ------------------------------------------------------------------ main --
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Inventory a DeepSeek-V4-Flash checkpoint from "
                    "safetensors headers only.")
    ap.add_argument(
        "model_dir",
        help="directory holding config.json and model.safetensors.index.json")
    ap.add_argument(
        "--json", metavar="PATH",
        help="also write the full per-tensor inventory as JSON")
    ap.add_argument(
        "--by-layer", action="store_true",
        help="print streamed/resident bytes per layer")
    ap.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any Gate A/V0 check fails")
    args = ap.parse_args(argv)

    try:
        idx, cfg, rows = build(args.model_dir)
    except FileNotFoundError as exc:
        print(f"inventory: no such file: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"inventory: malformed checkpoint metadata: {exc}", file=sys.stderr)
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
