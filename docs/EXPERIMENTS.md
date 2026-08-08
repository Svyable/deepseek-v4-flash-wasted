# Experiments and learned results — DeepSeek V4 port

This is the DeepSeek-specific append-only experiment log. Imported `docs/LEARNED.md` remains WASTE's Kimi history and should not be edited to make it look like DeepSeek evidence.

## Rules

1. **Append; do not rewrite history.** If a result is later disproved, append a correction and link the old entry to it.
2. State the evidence level: synthetic, checkpoint, real-model, or end-to-end.
3. Record enough configuration to reproduce the experiment.
4. Negative results are first-class output.
5. Do not promote an upstream Kimi result to a DeepSeek result without re-running it.
6. Distinguish a failed hypothesis from a broken/invalid experiment.

Entry template:

```text
## N — YYYY-MM-DD — short title

Question:
Protects / why run it:
Evidence state:
Port commit:
Model revision:
Container/converter:
Hardware/environment:
Command/method:
Result:
Verdict:
Consequence:
Follow-up:
```

---

## 1 — 2026-08-08 — A header-only inventory can gate the model before tensor reads

**Question:** Can the port classify the checkpoint and establish exact stored byte totals from safetensors metadata without loading/dequantizing model tensors?

**Protects:** downloading/converting a very large model while basic tensor-family assumptions are wrong.

**Evidence state:** SYNTHETIC-VERIFIED tooling; real 0731 checkpoint BLOCKED in the bootstrap environment.

**Implementation:** PR #1 added `tools/inventory.py`, a safetensors index/header parser that reads the 8-byte header length and JSON header only. It does not import torch or `safetensors` and does not read tensor payloads.

**Test method:** `tools/make_inventory_fixture.py` creates shards that end immediately after the header while their declared tensor offsets extend beyond EOF. `tests/test_inventory.py` exercises classification and fault cases. Any accidental tensor read would therefore fail instead of quietly succeeding.

**Result:** the inventory test passes as part of the project suite. PR #1 reported **32 passed, 0 failed, 12 skipped** in total.

**Faults explicitly tested:**

- unknown main tensor;
- routed expert missing one of the expected triplet matrices;
- shape contradicting config;
- bootstrap layer missing expected hash-routing data;
- DSpark tensor sharing the main namespace;
- expert weights with no associated scale tensors.

**Verdict:** Keep inventory/header parsing as Gate 0. Full tensor materialization is unnecessary for initial name/shape/byte accounting.

**Consequence:** future converter work must consume the generated inventory rather than repeat tensor-name heuristics independently.

---

## 2 — 2026-08-08 — The synthetic 3.21 GiB all-miss figure is not checkpoint evidence

**Question:** Does reproducing the README's proposed FP4 expert math from the synthetic fixture validate the released model's storage shape?

**Evidence state:** SYNTHETIC-VERIFIED arithmetic only.

**Method:** the fixture is generated from the same architecture assumptions written in README §1; `tools/inventory.py` rolls those headers into record and per-token totals.

**Result:** the tool reproduces the expected fixture counts/bytes.

**Verdict:** **No independent corroboration.** The fixture cannot validate the assumptions from which it was generated.

**Consequence:** until real 0731 headers are read, do not use the synthetic expert-size/count/all-miss figures for RAM requirements, final container size, storage recommendations, or tok/s projections.

This is recorded as an experiment because it is exactly the kind of self-confirming result an automated agent might otherwise cite later as measured evidence.

---

## 3 — 2026-08-08 — Unknown tensor names must fail Gate 0

**Question:** Should unfamiliar checkpoint names be assigned to a generic `other` bucket to let inventory complete?

**Evidence state:** DESIGN decision enforced by SYNTHETIC tests.

**Result:** PR #1 makes unrecognized **main-stack** tensor names a failure. DSpark-only unknown internals may be classified under the separately proven DSpark module boundary, but their bytes remain separately visible.

**Verdict:** Preserve strictness.

**Consequence:** the first real 0731 inventory is expected to fail if the current regex rules were inferred incorrectly. Fix the rules and tests against real names; never suppress the failure by creating a silent catch-all.

---

## 4 — 2026-08-08 — Do not fabricate missing upstream legal/source material

**Question:** When the official DeepSeek repository is unreachable, should the port create a standard MIT license text/reference source from memory so work can proceed?

**Evidence state:** provenance/process decision from PR #1.

**Result:** No. PR #1 records the missing DeepSeek license under `LICENSES/` rather than committing a fabricated legal document. No DeepSeek-derived source was added.

**Verdict:** Keep this rule.

**Consequence:** resolve the exact official license/notice before copying/adapting official inference or encoding source. Third-party code is not a substitute for official source provenance.

---

## 5 — pending — Real Gate 0 inventory

**Question:** What did DeepSeek actually ship in `DeepSeek-V4-Flash-0731 @ 9e165c3`?

**Evidence state:** BLOCKED in the original bootstrap environment by Hugging Face CONNECT 403.

**Run when access is available:** follow `INVENTORY-0731.md` and `CONVERSION.md` through the index-only then strict-header inventory.

**Must record:**

- exact tensor-name families;
- exact layer/expert counts;
- exact dtypes/stored shapes/bytes;
- scale ownership/granularity;
- bootstrap/hash-routing source;
- DSpark separation;
- exact resident/routed/DSpark checkpoint bytes.

**Do not pre-fill the result.**

---

## Candidate experiments after base correctness

These are hypotheses, not planned conclusions. Run only after the gate that makes the result interpretable.

- Native FP4 versus WASTE VQ3R/VQ4P quality/footprint/performance.
- LFRU versus LRU versus other bounded-cache policies on real V4 routing traces.
- Deterministic early-layer prefetch if official routing confirms it.
- Predictive lookahead for learned routers.
- Chunked prefill distinct-expert reduction.
- Expert-parallel versus row-parallel CPU execution.
- FP4 SIMD variants and activation quantization.
- Resident embedding/head versus row-on-demand strategies if the RAM floor requires it.
- Direct I/O versus page-cache behavior by OS/filesystem.
- CPU placement/thread counts.
- Metal/CUDA offload only after profiling identifies a movable bottleneck.
- DSpark acceptance/speed/memory tradeoff after base V9 correctness.

Each candidate gets its own numbered entry whether it succeeds or fails.