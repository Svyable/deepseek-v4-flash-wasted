#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Gate A/V0 inventory + acquisition checks.

Every RAM and bytes-per-token claim in this port is supposed to descend from
the checkpoint. That only holds if the inventory tool reports what it actually
read and refuses to fill gaps with plausible numbers. This driver also runs the
header-only Hugging Face acquisition test and the pinned 0731 FP4 convention
fixture so `make check` covers the cheap Gate A -> Gate B handoff.

The synthetic inventory fixture proves tooling mechanics, not checkpoint truth.
The release-convention fixture is separately sourced from the immutable
DeepSeek 0731 release and is replayed through the actual scalar C FP4 path.

  python3 tests/test_inventory.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import inventory                      # noqa: E402
import make_inventory_fixture as fx   # noqa: E402
import test_fetch_hf_headers as fetch_headers_test  # noqa: E402
import test_release_quant_fixture as release_quant_test  # noqa: E402

# Small enough to be fast, big enough that every rule fires: hash-routed
# layers 0-2, a ratio-0 layer without compressor/indexer, and one attached
# DSpark layer past the end of the main stack.
SMALL = dict(layers=5, experts=4, vocab=512, dspark_layers=1)

fails = 0


def ck(cond, msg):
    global fails
    if not cond:
        fails += 1
        print(f"  FAIL {msg}")
    return cond


def make(tmp, name, **over):
    cfg = dict(fx.DEFAULTS)
    cfg.update(SMALL)
    cfg.update(over)
    out = os.path.join(tmp, name)
    fx.build(cfg, out, shard_bytes=4 << 30, with_data=False)
    return out


def run(model_dir):
    """(gate results by check name, rows, idx) for a checkpoint dir."""
    idx, cfg, rows = inventory.build(model_dir)
    gate = inventory.gate0(idx, cfg, rows)
    by_name = {n: (s, d) for s, n, d in gate.results}
    return by_name, rows, idx


def status(gate, substr):
    for name, (st, _) in gate.items():
        if substr in name:
            return st
    return None


def rewrite_index(model_dir, fn):
    """Apply fn to the {name: shard} weight map."""
    ip = os.path.join(model_dir, "model.safetensors.index.json")
    with open(ip) as f:
        idx = json.load(f)
    fn(idx["weight_map"])
    with open(ip, "w") as f:
        json.dump(idx, f)


def edit_header(model_dir, shard, fn):
    """Mutate one shard's JSON header in place, keeping it valid."""
    import struct
    p = os.path.join(model_dir, shard)
    with open(p, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
    fn(hdr)
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def only_shard(model_dir):
    return sorted(x for x in os.listdir(model_dir) if x.endswith(".safetensors"))[0]


def main():
    tmp = tempfile.mkdtemp(prefix="inv-test-")
    try:
        # ---------------------------------------------------- happy path --
        good = make(tmp, "good")
        gate, rows, idx = run(good)

        ck(all(st == "PASS" for st, _ in gate.values()),
           f"a well-formed checkpoint passes every Gate A/V0 check: "
           f"{[(n, s) for n, (s, _) in gate.items() if s != 'PASS']}")
        ck(idx.have_headers, "headers mode is reached when shards are present")
        ck(not any(r["subsystem"] == "unclassified" for r in rows),
           "no tensor lands in the unclassified bucket")

        # ------------------------------------------- header-only promise --
        shard = os.path.join(good, only_shard(good))
        described = max(
            m["data_offsets"][1]
            for m in inventory.HeaderOnlyIndex._read_header(shard).values())
        ck(os.path.getsize(shard) < described,
           f"the fixture really is header-only: {os.path.getsize(shard)} B on "
           f"disk describes {described} B of tensors")

        # ---------------------------------------------- exact arithmetic --
        w1 = [r for r in rows if r["subsystem"] == "routed_expert"
              and r["matrix"] == "w1" and not r["is_scale"]][0]
        ck(w1["packing"] == 2.0,
           f"FP4 packing is detected from shape, not assumed: {w1['packing']}")
        ck(w1["placement"] == "streamed",
           "routed experts are the streamed population")

        per_expert = sum(
            r["stored_bytes"] for r in rows
            if r["subsystem"] == "routed_expert" and r["module"] == "main"
            and r["layer"] == 0 and r["expert"] == 0)
        h, mi = fx.DEFAULTS["hidden"], fx.DEFAULTS["moe_inter"]
        want = 3 * (h * mi // 2) + 3 * (h * mi // 32)
        ck(per_expert == want,
           f"one expert record is w1+w3+w2 with scales: {per_expert} != {want}")

        scales = [r for r in rows if r["is_scale"]]
        ck(scales and all(r["scale_of"] and r["scale_of"].endswith(".weight")
                          for r in scales),
           "every scale tensor is attributed to the weight it scales")

        ck(all(r["placement"] == "resident" for r in rows
               if r["subsystem"] in ("attention", "mhc", "norm", "embedding",
                                     "lm_head", "shared_expert")),
           "trunk subsystems are resident")

        # ------------------------------------------------- index-only mode --
        lonely = make(tmp, "lonely")
        for f in os.listdir(lonely):
            if f.endswith(".safetensors"):
                os.remove(os.path.join(lonely, f))
        gate2, rows2, idx2 = run(lonely)
        ck(not idx2.have_headers, "absent shards are noticed")
        ck(status(gate2, "unexplained bucket") == "SKIP",
           "byte checks SKIP rather than pass on an index-only checkpoint")
        ck(status(gate2, "dimensions agree") == "SKIP",
           "shape checks SKIP without headers")
        ck(all(r["stored_bytes"] is None for r in rows2),
           "unreadable tensors report None bytes, never 0")
        tot = inventory.totals(rows2)
        ck(all(v[2] > 0 for v in tot.values()),
           "every bucket reports its unresolved count")
        ck(status(gate2, "all tensor names classified") == "PASS",
           "classification still runs without weights")

        partial = make(tmp, "partial", layers=9)
        shards = sorted(x for x in os.listdir(partial)
                        if x.endswith(".safetensors"))
        if len(shards) > 1:
            os.remove(os.path.join(partial, shards[0]))
            _, rows3, idx3 = run(partial)
            ck(idx3.missing_shards, "a partly-downloaded checkpoint is flagged")
            ck(any(r["stored_bytes"] is None for r in rows3)
               and any(r["stored_bytes"] is not None for r in rows3),
               "partial data stays partial")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                inventory._report_expert_math(inventory.load_config(partial), rows3)
            ck(buf.getvalue() == "",
               "per-token traffic is withheld when any expert byte is unknown")

        # ------------------------------------------------- injected faults --
        unk = make(tmp, "unknown")
        edit_header(unk, only_shard(unk),
                    lambda h: h.__setitem__(
                        "model.layers.0.mystery_module.weight",
                        {"dtype": "BF16", "shape": [8, 8],
                         "data_offsets": [0, 128]}))
        rewrite_index(unk, lambda wm: wm.__setitem__(
            "model.layers.0.mystery_module.weight", only_shard(unk)))
        g, _, _ = run(unk)
        ck(status(g, "all tensor names classified") == "FAIL",
           "an unrecognised main-stack tensor fails Gate A/V0")
        ck(status(g, "unexplained bucket") == "FAIL",
           "its bytes are reported as stray")

        miss = make(tmp, "missing-w3")
        victim = "model.layers.0.mlp.experts.0.w3.weight"
        edit_header(miss, only_shard(miss), lambda h: h.pop(victim, None))
        rewrite_index(miss, lambda wm: wm.pop(victim, None))
        g, _, _ = run(miss)
        ck(status(g, "one w1/w2/w3") == "FAIL",
           "an expert missing w3 fails the record check")

        bad = make(tmp, "bad-shape")
        edit_header(bad, only_shard(bad), lambda h: h.__setitem__(
            "model.layers.1.mlp.experts.0.w1.weight",
            {"dtype": "U8", "shape": [999, 64], "data_offsets": [0, 999 * 64]}))
        g, _, _ = run(bad)
        ck(status(g, "dimensions agree") == "FAIL",
           "a shape that contradicts config.json fails Gate A/V0")

        nohash = make(tmp, "no-hash")
        gone = "model.layers.1.mlp.gate.hash_table"
        edit_header(nohash, only_shard(nohash), lambda h: h.pop(gone, None))
        rewrite_index(nohash, lambda wm: wm.pop(gone, None))
        g, _, _ = run(nohash)
        ck(status(g, "token-id") == "FAIL",
           "a bootstrap layer without a mapping tensor fails Gate A/V0")

        amb = make(tmp, "ambiguous")
        collide = "model.layers.2.dspark_head.weight"
        edit_header(amb, only_shard(amb), lambda h: h.__setitem__(
            collide, {"dtype": "BF16", "shape": [4, 4],
                      "data_offsets": [0, 32]}))
        rewrite_index(amb, lambda wm: wm.__setitem__(collide, only_shard(amb)))
        g, _, _ = run(amb)
        ck(status(g, "DSpark") == "FAIL",
           "a DSpark tensor in the main namespace fails separability")
        ck(status(gate, "DSpark") == "PASS",
           "an attached module under its own root is not flagged")

        noscale = make(tmp, "no-scale")
        def drop_scales(h):
            for k in [k for k in h if k.endswith(".weight_scale")]:
                h.pop(k)
        edit_header(noscale, only_shard(noscale), drop_scales)
        rewrite_index(noscale, lambda wm: [
            wm.pop(k) for k in list(wm) if k.endswith(".weight_scale")])
        g, _, _ = run(noscale)
        ck(status(g, "one w1/w2/w3") == "FAIL",
           "routed experts without scale tensors fail Gate A/V0")

        # ------------------------------------------------------ interface --
        rc_good = inventory.main([good, "--strict"])
        rc_bad = inventory.main([unk, "--strict"])
        ck(rc_good == 0, f"--strict exits 0 on a clean checkpoint ({rc_good})")
        ck(rc_bad == 1, f"--strict exits 1 when Gate A/V0 fails ({rc_bad})")

        jp = os.path.join(tmp, "inv.json")
        inventory.main([good, "--json", jp])
        with open(jp) as f:
            doc = json.load(f)
        ck(doc["mode"] == "headers" and doc["tensors"]
           and "gate0" in doc and doc["totals"],
           "--json emits mode, totals, gate results and per-tensor rows")

        rc_missing = inventory.main([os.path.join(tmp, "nope")])
        ck(rc_missing == 2, f"a missing checkpoint exits 2 ({rc_missing})")

        junk = make(tmp, "junk")
        with open(os.path.join(junk, only_shard(junk)), "wb") as f:
            f.write(b"\xff" * 64)
        rc_junk = inventory.main([junk])
        ck(rc_junk == 2, f"a corrupt shard is refused, not guessed ({rc_junk})")

        # ------------------------------------------- acquisition + V1b --
        # These are separate evidence classes from the synthetic inventory:
        # fake HTTP validates transport/safety, while the F3 release fixture
        # validates the actual scalar C path against pinned official source.
        ck(fetch_headers_test.main() == 0,
           "header-only Hugging Face fetcher passes its offline Range suite")
        release_rc = release_quant_test.main()
        ck(release_rc == 0,
           "scalar FP4 path matches pinned 0731 nibble/scale convention fixture")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("INVENTORY/REFERENCE FAILED" if fails else "INVENTORY/REFERENCE OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
