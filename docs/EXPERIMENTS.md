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

**Addendum, same day, before this entry merged.** PR #9 landed ratio-0 attention and the grouped output projection on `main` while this change was open, and added its three replays *to the import chain* — `test_v5_attention_scalar`, `test_v5_attention_ratio0_real`, `test_v5_attention_output_real`, imported into `test_release_quant_fixture.py`. So the chain reached fourteen replays under the one "inventory" line before anyone noticed eleven. The merge resolution keeps the decoupling and gives the three their own lines; suite is 48 passed, 0 failed, 12 skipped. Two further mutations confirm the new lines bite: corrupting `v5_attn_ratio0_real/q-after-rope.bf16.bin` names *Gate E/V5 — real ratio-0 attention core* and leaves inventory passing (47 passed, 1 failed), and removing `v5_attn_output_group0_real/` reports SKIP with its reason rather than passing.

This is the maintenance cost being paid within hours of being written down, and it is also the evidence that the cost is worth it: the chain absorbed three new gates silently, which is exactly how it absorbed the first eleven. **Ratio-128 and ratio-4/CSA each need their `run.sh` line when their fixtures are frozen.**

---

## 7 — 2026-08-09 — A mutation check placed where the two schemes provably coincide

**Question:** `tests/test_v5_compressor_scalar.py` claims to pin that the ratio-128 compressor's compressed YaRN RoPE differs from ordinary base-10000 RoPE. Does that check discriminate?

**Protects:** Gate E/V5 ratio-128, and every attention mode built on the compressor. A wrong RoPE frequency is not a crash — it is a model that reads slightly the wrong positions, which surfaces at V8 logits with 43 layers of candidates in between.

**README gate letter(s):** E. **Operational V-level:** V5 ratio-128 model-free half.

**Evidence state:** MEASURED on this checkout. No checkpoint access used; Hugging Face remains CONNECT 403 here (entry 5).

**Port commit:** this entry's commit. **Environment:** Ubuntu 24.04, gcc 13.3.0, x86-64.

**Method:** the check was failing outright, which is what drew attention to it. It asserted that rope pair 0 at position 128 differs between compressed YaRN (base 160000, factor 16) and plain base-10000 RoPE. Then six single-line mutations were applied to `yarn_freq` and the rotation in `src/deepseek_v4_compressor_ref.c`, each rebuilt and re-run.

**Result:** the assertion is unsatisfiable, and the surrounding checks were blind.

`base**(-0/dim) == 1` for every base, so pair 0 has inv_freq 1.0 whether the base is 10000 or 160000. With `beta_fast=32, beta_slow=1` the YaRN ramp spans low=15 to high=25, so pair 0 is fully extrapolated (`smooth=1`) and `factor` never reaches the angle either. Both schemes compute exactly `angle = 128.0` and produce bit-identical BF16. No implementation can satisfy the assertion.

Worse, pair 0 was also the *only* pair the test verified against an independent YaRN equation — the one pair where base, factor and ramp are all irrelevant. Mutating `yarn_freq` to `return freq`, dropping the interpolation entirely, left every check up to that point passing.

| mutation | before | after |
|---|---|---|
| drop the YaRN blend, return raw freq | **survived** | killed |
| always divide by factor, no ramp blend | killed | killed |
| `smooth = ramp` (roles swapped) | killed | killed |
| base 10000 instead of 160000 | killed | killed |
| RoPE applied to the head instead of the tail | killed | killed |
| sin sign flipped (rotation direction) | killed | killed |

**Verdict:** a discrimination check is only as good as the point it is evaluated at. This is entry 3's lesson in a new shape: there the fixture shared a constant with the code, here it samples the one coordinate where the two candidate semantics are provably equal. Both produce a green check that cannot fail for the right reason.

**Consequence:** the check moved to rope pair 20, which sits at ramp 0.5 — the blended interior, where the result is neither pure interpolation nor pure extrapolation. Its position-128 angles are 0.038 against 0.072 and separate cleanly in BF16. Pair 31 was considered and rejected: it clamps to ramp=1, but its angles (7.3e-05 against 1.2e-03) both round to the same BF16 near 0.5, so it cannot see a dropped blend either. Pair 0's coincidence is now asserted as the invariant it actually is — the two schemes *must* agree there — which turns a check that could only fail into one that can only fail if the geometry changes. Six of six mutations now die.

**Follow-up:** when the real ratio-128 fixture is frozen, it will exercise all 32 pairs at once and subsume this. Until then the model-free half carries the whole gate, which is why it had to discriminate. The general rule for the remaining CSA/indexer work: **when pinning that A differs from B, first check that A and B are not equal by construction at the point sampled.**

---

## 8 — 2026-08-09 — A refusal named in the gate contract but not implemented

**Question:** Gate H/V6's contract names four wirings the full-layer test must reject. Does the model-free composition test reject all four?

**Protects:** Gate H, and every layer built on it. A layer that composes its HyperConnection transitions wrongly still produces finite, plausible hidden states; the error surfaces at V8 logits with nothing pointing back to the wiring.

**README gate letter(s):** H. **Operational V-level:** V6 model-free half.

**Evidence state:** MEASURED on this checkout.

**Port commit:** this entry's commit. **Environment:** Ubuntu 24.04, gcc 13.3.0, x86-64.

**Method:** read the four refusals the PR contract names — reusing the original residual for the FFN `hc_pre`, swapping the attention and FFN HC parameter sets, dropping either `hc_post` transition, and feeding an independently correct branch output produced from the wrong branch input — against what `tests/test_v6_layer_composition_scalar.py` actually asserts.

**Result:** three of four were implemented. **Dropping either `hc_post` transition was not.**

The gap is not obvious from reading the test, because "drop a step" sounds like it would fail loudly. It does not. `hc_post` maps a `[DIM]` branch and an `[HC*DIM]` residual to a new `[HC*DIM]` state as `out[k*DIM+d] = post[k]*branch[d] + Σ_j comb[j][k]*residual[j*DIM+d]`. Omitting it entirely breaks the shapes and would be caught immediately. The realistic wrong implementation is to write the ordinary transformer residual add — `out[k*DIM+d] = residual[k*DIM+d] + branch[d]` — which has the right shape, compiles, runs, and is precisely the thing mHC replaces. Nothing in the suite refused it.

Both variants are now asserted, and all five refusals are measured against the independent oracle before the C library is involved:

| refused wiring | max_abs vs oracle |
|---|---:|
| reuse the original residual for the FFN `hc_pre` | 0.760 |
| replace the attention `hc_post` with a plain residual add | 0.654 |
| swap the attention and FFN HC parameter sets | 0.403 |
| feed a correct branch output from the wrong branch input | 0.203 |
| replace the FFN `hc_post` with a plain residual add | 0.193 |

All are three orders of magnitude above the 1e-3 visibility threshold, so none is marginal.

**Verdict:** the contract was right and the implementation of it was incomplete. Distinct from entries 3 and 7 — there the check existed and could not discriminate; here the check was simply absent while the prose said otherwise, which is harder to notice because the document reads as evidence.

**Consequence:** both `hc_post` refusals added. `VALIDATION.md` §V6 now lists the five refusals with their margins, so the claim and the assertions can be compared without reading the test.

**Follow-up:** when a gate's prose names the mutations it rejects, treat that list as a checklist to diff against the test, not as a description of it. The remaining Gate H work — real attention and real MoE composed in place of the current stubs — should have its own refusals named and then verified the same way.

---

## 9 — 2026-08-10 — A stub branch routes to six entirely different experts

**Question:** Gate H's remaining step needs the real routed experts evaluated at the layer-3 `ffn_pre` state. Which six experts are those, and could the existing stub-branch composition stand in while checkpoint access is unavailable?

**Protects:** a routed expert record is 13.4 MB, and fetching the wrong six produces a fixture that looks authoritative and tests the wrong thing. This is the working rule applied literally — run the cheap real test that could kill the expensive step before running the expensive step.

**README gate letter(s):** H, with F/V4's router. **Operational V-level:** V6 step 4 and the selection half of step 5.

**Evidence state:** CHECKPOINT-VERIFIED for the routing decision — the complete layer-3 gate `[256,4096]` and its 256 correction biases are already frozen in `v3_router_real/`, so no new acquisition was needed. No expert is executed.

**Port commit:** this entry's commit. **Environment:** Ubuntu 24.04, gcc 13.3.0, x86-64, 2 s wall.

**Method:** compose forward to the real FFN branch input — `residual -> hc_pre(hc_attn_*) -> real 64-head attention -> hc_post -> hc_pre(hc_ffn_*)` — then run the real layer-3 learned router on it. C against an independent Python router built from the pinned source contract. Repeat with the stub attention branch in place of the real one.

**Result:**

```text
real attention branch  -> experts [255, 30, 99, 40, 44, 238]
stub attention branch  -> experts [217,  0, 172,  9, 241,  74]
Gate F's earlier input -> experts [  2, 29, 225, 220, 108,  69]
```

**Zero overlap between the real and stub selections.** All six differ. The stub composition is not a weaker version of the real one for routing purposes — it is a different question with a different answer, and a MoE fixture frozen against it would have exercised six experts the real layer never touches.

Supporting numbers:

```text
top-k boundary margin              0.011921765   (Gate F's was 0.000405312)
selection ignoring correction bias [255,30,40,99,44,238]  — same set, 99/40 reordered
route weights                      0.2606 0.2582 0.2489 0.2525 0.2422 0.2376
```

The weights are deliberately non-monotonic in selection order: selection ranks by `score + bias` while the weights are gathered from the unbiased `score`, which is the Gate F contract holding at a new input. The margin is 29x wider than Gate F's, comfortably outside BF16 resolution, so this selection is a stable acquisition list rather than a coin flip at the boundary.

**Verdict:** the stub cannot stand in, and the acquisition is now bounded. Gate H step 5 needs exactly layer-3 routed experts **30, 40, 44, 99, 238, 255** plus the shared expert — six records, not a search.

**Consequence:** `tests/test_v6_ffn_route_real.py` replays this in the ordinary suite and refuses a top-k margin below 1e-4, so a future state whose sixth expert is decided by rounding noise cannot silently become an acquisition list. Three router mutations die against it: `sqrt` dropped from `sqrt(softplus)`, the correction bias ignored for selection, and the routed scaling factor dropped.

**Follow-up:** when those six records are fetched, the resulting fixture should declare `ffn-pre.bf16.bin` as its `input_dependency` with the digest pinned, exactly as `v6_attention_branch_real` declares `attn-pre.bf16.bin`. That is what made the attention half composable offline afterwards, and it would make the final Gate H composition an offline step rather than a third acquisition round.

---

## 10 — 2026-08-10 — Gate F's eight-row shared fixture hid a 128-row scale-grid boundary

**Question:** can the already-proven shared FP8 expert helper be used unchanged for Gate H's full 4,096 output rows?

**Evidence:** model-free scalar regression plus the existing real Gate-F fixture. Gate F only supplied eight `w2` rows and one E8M0 scale-grid row; the helper therefore carried a fixture-era `out_rows <= 128` rejection.

**Result:** the limit was not a model semantic. A 129-row construction makes row 128 consume scale-grid row 1: row 127 is `0.75`, row 128 is `1.5` when the second scale is 2, and returns to `0.75` when that scale is changed to 1. Gate-F real routed/shared fixtures remain bit-exact after removing the cap.

**Verdict:** passed only after making scale-grid coverage explicit. A bounded fixture must not silently become a runtime dimension limit.

---

## 11 — 2026-08-10 — Full real Gate H MoE arithmetic agrees on all 28,672 expert outputs

**Question:** do the six experts selected by the real layer-3 FFN input and the shared expert reproduce independently at full model width?

**Method:** derive the SHA-pinned real `ffn_pre`, fetch exactly experts `[255,30,99,40,44,238]` plus the shared expert from immutable 0731, evaluate all 4,096 outputs with a 4,096-row specialization of standalone `tools/v4_moe_oracle.c`, then replay the same bytes through WASTE's scalar expert refs. Raw checkpoint expert payloads are deleted after acquisition.

**Result:** `6 x 4096 = 24,576` routed BF16 values plus `4,096` shared BF16 values agree exactly — **28,672 / 28,672**. The combined branch SHA-256 is `809f1468f034d21909da7127d08d2c0b6249013630ffd32912a148473044a659`. Dropping any routed branch changes 4,070–4,075 BF16 values; dropping shared changes 4,087.

**Verdict:** PASSED. Gate H's MoE half is a real checkpoint-backed branch, not a stub or a first-eight-row extrapolation.

---

## 12 — 2026-08-10 — One final BF16 exposed an unpinned F32 HyperConnection boundary

**Question:** why did the first full-layer replay disagree at exactly one of 16,384 final BF16 values after every expert output had already matched?

**Result:** final index 13,648 was `0x378d` in the independent Python calculation and `0x378e` in C. The Python Sinkhorn oracle retained double precision for `post`/`comb`; the model/runtime boundary consumes F32 tensors. Rounding independent `post`/`comb` to F32 before `hc_post` changes exactly that one BF16 value and restores exact agreement.

**Verdict:** this was an oracle-boundary bug, not a checkpoint/expert mismatch. No tolerance was introduced. Provenance now records the F32 boundary and the one changed index explicitly.

---

## 13 — 2026-08-10 — Independent router weights were close enough numerically but still the wrong branch inputs

**Question:** should Gate H feed experts the independently computed Python route weights or the exact scalar-runtime router weights?

**Result:** IDs agree exactly at `[255,30,99,40,44,238]`, but the two F32 vectors differ by up to `1.49662139e-07`. Route weight is multiplied before the expert hidden BF16 cast, so the complete-layer fixture was reacquired using the scalar runtime's exact F32 weights while retaining the Python router as an independent ID/tolerance oracle.

Reacquisition changed only the six-F32 weight file and provenance; the 24,576 routed BF16 outputs, 4,096 shared outputs, combined MoE branch, and final layer state were unchanged for this input.

**Verdict:** semantic inputs should come from the implementation boundary being composed even when an independent oracle is numerically very close. Independence verifies the boundary; it does not replace it.

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