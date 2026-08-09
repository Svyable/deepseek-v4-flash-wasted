# Experiments and learned results — DeepSeek V4 port

This is the DeepSeek-specific append-only experiment log. Imported `docs/LEARNED.md` remains WASTE's Kimi history and should not be edited to make it look like DeepSeek evidence.

> **Gate vocabulary note:** this is an append-only historical record, so older entries below deliberately retain the wording they had when written, including “Gate 0” and standalone `V1`. Read those through the current canonical concordance in `docs/VALIDATION.md` §4a: historical inventory “Gate 0” means **README Gate A / V0**; the PR #3 quantization entry protects **README Gate B / V1**. New entries must use the current A–N/V-level terminology and must not introduce new local gate numbering.

## Rules

1. **Append; do not rewrite history.** If a result is later disproved, append a correction and link the old entry to it.
2. State the evidence level: synthetic, checkpoint, real-model, or end-to-end.
3. Record enough configuration to reproduce the experiment.
4. Negative results are first-class output.
5. Do not promote an upstream Kimi result to a DeepSeek result without re-running it.
6. Distinguish a failed hypothesis from a broken/invalid experiment.
7. Name the current README gate letter(s) and V/system gate separately when adding a new entry.

Entry template:

```text
## N — YYYY-MM-DD — short title

Question:
Protects / why run it:
README gate letter(s):
Operational V-level / systems gate:
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

## 3 — 2026-08-08 — A test that shares a constant with the code cannot detect that constant being wrong

**Question:** Do the new `tests/test_quant.c` checks actually detect faults in `src/quant/`, or do they only confirm that the code agrees with itself?

**Protects:** every later gate. V1 is the seam that all model arithmetic is debugged through — a decode bug that survives its own test surfaces as "the model is wrong" at V6 or V8, with 43 layers of candidates in between.

**Evidence state:** SYNTHETIC-VERIFIED (format conformance). Not checkpoint evidence.

**Port commit:** this entry's commit. **Environment:** Ubuntu 24.04, gcc 13.3.0, x86-64.

**Method:** ten single-line mutations were applied to the decoders one at a time — wrong E2M1 table entry, dropped sign on negative zero, UE8M0 bias off by one, wrong NaN code, E4M3 treated as IEEE (top exponent reserved) instead of the finite variant, subnormal given an implicit leading one, E4M3 exponent bias off by one, FP8 scale grid floored instead of ceiled, FP4 scale-plane row stride dropped, and FP4 nibble order flipped. Each was rebuilt and the suite re-run.

**Result:** **nine of ten were caught. The tenth was the nibble order** — the single assumption the header itself flags as the highest-risk unverified choice.

**Cause:** the test packed its fixtures through a helper compiled from the same `WASTE_FP4_LOW_NIBBLE_IS_EVEN` macro as the decoder. Flipping the macro flipped the packer and the unpacker together, so the round trip stayed consistent and every assertion still passed.

**Verdict:** the suite was strong everywhere it derived expected values independently, and blind exactly where it shared a definition with the code under test. Exhaustive enumeration did not help: all 16 E2M1 codes were checked, and all 16 were checked through the same wrong lens.

**Consequence:** the byte layout is now pinned with a literal — byte `0x21` must decode to `0.5` at column 0 and `1.0` at column 1 — which is independent of the macro. Re-running the mutation set gives ten of ten. Generalizing: a fixture generator must not import the convention it is meant to prove. The same trap applies to `tools/make_inventory_fixture.py`, which shares its architecture assumptions with `README.md` §1 (entry 2), and it will apply to any golden-fixture writer built from our own encoder rather than the official one.

**Follow-up:** when Gate 0 makes the official reference readable, the pinned literal is the assertion expected to fail if DeepSeek packs the other way. That failure is the designed outcome, not a regression — changing the convention should cost a deliberate edit to a stated constant.

---

## 6 — 2026-08-09 — Eleven gate replays ran under one name, and the name was wrong

**Question:** `make check` reported 34 passed before this entry and had reported 34 passed since PR #3, across four gate PRs that each froze a real-checkpoint fixture. Were Gates B, C, D and F actually being replayed?

**Protects:** every gate verdict this project publishes. The gates are the whole claim — a frozen fixture nobody replays is a screenshot, and a replay whose failure names the wrong subsystem costs the debugging time the fixture was meant to save.

**README gate letter(s):** A, B, C, D, F — the reporting of all of them, not their arithmetic.

**Operational V-level / systems gate:** V0–V4 replay legibility. No model arithmetic changed.

**Evidence state:** MEASURED on this checkout. No checkpoint access was required or used; Hugging Face remains CONNECT 403 here (entry 5).

**Port commit:** this entry's commit. **Environment:** Ubuntu 24.04, gcc 13.3.0, x86-64, no container, `tests/run.sh /nonexistent`.

**Method:** traced the call graph of every `tests/test_*.py`, then ran three mutations through the full suite: (A) flip bit 0 of the frozen Gate F expected output `v4_moe_real/combined-out8.bf16.bin`; (B) corrupt Gate D's `expected-post.bf16.bin` and Gate F's output together; (C) remove the `v3_mhc_real/` and `v4_moe_real/` fixture directories from the checkout.

**Result:** they were running. `tests/run.sh` called `test_inventory.py`, which imported and called `test_fetch_hf_headers.main()` and `test_release_quant_fixture.main()`, which imported and called the other nine. The whole ladder arrived as one line: *"inventory measures what it read and refuses to guess the rest."* That is why the total never moved.

Three defects followed from the shape, and the mutations pinned each:

- **Misattribution.** Mutation A reported `FAIL inventory.py` with the sub-lines "all tensor names classified" and "no bytes in an unexplained bucket" — a fault in the MoE combination pointing a reader at the tensor classifier.
- **Masking.** The driver returned on first failure, so under mutation B the earliest broken gate hid every later one; Gate F never ran.
- **Invisibility.** Nothing in a green run said Gate C, D or F existed, so the count staying at 34 across four gate PRs read as normal.

**Verdict:** not a coverage gap — a reporting defect, and the more dangerous kind, because the suite was green and the evidence was real. The count that should have raised it was visible for four PRs and was read as "unchanged" rather than "unchanged despite four gates landing".

**Consequence:** `tests/run.sh` gained a "DeepSeek gate replays" section that invokes all eleven by name, mapping exit 77 to SKIP; the chaining was removed from `test_inventory.py` and `test_release_quant_fixture.py`, which now each test one thing. Re-running the mutations: A names Gate F/V4 and leaves inventory passing; B names Gate D/V3 and Gate F/V4 independently (43 passed, 2 failed); C reports both as SKIP with their reasons (43 passed, 0 failed, 14 skipped) rather than passing. Suite is now 45 passed, 0 failed, 12 skipped.

**Follow-up:** the replays are enumerated rather than globbed, so a Gate E/V5 attention fixture is not covered until its line is added. That is deliberate — a glob hides an unreplayed fixture exactly as well as the import chain did — and it is the one maintenance cost this fix creates.

---

## Candidate experiments after base correctness

These are hypotheses, not planned conclusions. Run only after the canonical gate that makes the result interpretable.

- Native FP4 versus WASTE VQ3R/VQ4P quality/footprint/performance — after Gate I/V8 + Gate K/V9 native baseline.
- LFRU versus LRU versus other bounded-cache policies on real V4 routing traces — Gate M, while Gate G remains true.
- Deterministic early-layer prefetch if Gate A/V0 confirms official routing representation — Gate G correctness, Gate M performance.
- Predictive lookahead for learned routers — Gate G correctness, then Gate M/performance.
- Chunked prefill distinct-expert reduction — preserve Gate G and Gate I/V8/K/V9 numerics.
- Expert-parallel versus row-parallel CPU execution — after the relevant model V-level is stable.
- FP4 SIMD variants and activation quantization — after Gate B/V1 + Gate C/V2.
- Resident embedding/head versus row-on-demand strategies if the RAM floor requires it — planner + Gate G/L measurement.
- Direct I/O versus page-cache behavior by OS/filesystem — Gate G correctness, Gate L performance.
- CPU placement/thread counts — benchmark only after the measured model path is sufficiently correct.
- Metal/CUDA offload only after profiling identifies a movable bottleneck and the scalar/official gate passes.
- DSpark acceptance/speed/memory tradeoff — Gate N after Gate I/V8 + Gate K/V9.

Each candidate gets its own numbered entry whether it succeeds or fails.