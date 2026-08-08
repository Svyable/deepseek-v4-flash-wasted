# Roadmap — DeepSeek V4 Flash on WASTE

This roadmap converts the implementation plan in `README.md` into gated PR-sized work. It is deliberately ordered so a cheap failure prevents an expensive download, conversion, or optimization.

Status values: **DONE**, **IN PROGRESS**, **BLOCKED**, **NEXT**, **LATER**.

## Gate vocabulary

The phases below are a **schedule**, not a fourth gate vocabulary.

- `README.md` §18 defines **14 stable design gates A–N**.
- `docs/VALIDATION.md` defines operational `V0–V11` levels.
- `docs/VALIDATION.md` §4a maps **every README gate A–N** to its V-level or systems/performance owner, keeps the handoff concept column, and maps both to the phases below.

When both identifiers exist, cite both (`Gate B / V1`, `Gate H / V6`, `Gate K / V9`). README Gates `G/L/M/N` intentionally have no V-number. Operational `V7/V11` intentionally have no README letter.

Use the phase only to say **when** the work happens.

## Current snapshot

| Work | Canonical gate | Status | Evidence |
|---|---|---|---|
| PR #1 — WASTE bootstrap + attribution + inventory tool | supports **Gate A / V0** | **DONE** | merged as `edd2a41b66332e5a54ed54bcbb196fec19664079` |
| PR #2 — documentation foundation | process | **DONE** | merged as `7c5c8d95fa7e1a9588b744aba4a6389bf77e98f7` |
| PR #3 — scalar native quantization | **Gate B / V1 half** | **PARTIALLY DONE** | merged as `91c36b8f4168349e6893a9911a3f60075d62d973`; exhaustive public-format conformance, official convention agreement blocked |
| Real 0731 metadata/header inventory | **Gate A / V0** | **BLOCKED** in original bootstrap environment | Hugging Face CONNECT 403; see `docs/REFERENCE_ACCESS.md` and `docs/INVENTORY-0731.md` |
| Official one-projection oracle | **Gate C / V2** | **BLOCKED** on reference/checkpoint access | V2 needs an official quantized-linear oracle |
| DeepSeek transformer implementation | **Gates D/E/F/H/I/K; V3–V9** | **NOT STARTED** | no ported forward path exists |
| Encoding/server | **Gate J / V10 + V11** | **NOT STARTED** | official encoding path not ported |
| Storage/cache feasibility | **Gates G/L/M** | **LATER** | mechanisms imported; DeepSeek measurements absent |
| DSpark | **Gate N** | **LATER** | base model must pass Gate I/V8 and Gate K/V9 first |

Current verified model-free suite after PR #3:

```text
make check -> 34 passed, 0 failed, 12 skipped
make asan  -> 33 passed, 0 failed
```

PR #3 also mutation-tested ten one-line decoder faults; the final suite kills all ten after the nibble-order fixture was made independent of the implementation macro.

## Immediate priority — acquire Tier-0/Tier-1 official reference material

The highest-leverage next action is **not** a full checkpoint download. It is authorized access to the pinned official release's small legal/metadata/reference assets:

```text
license/notice
config.json
generation_config.json
model.safetensors.index.json
inference/
encoding/
tokenizer assets
```

That unlocks or sharply advances:

- exact official licensing/provenance;
- **Gate A / V0** real tensor-name classification;
- **Gate B / V1** FP4 nibble-order and FP8/scale-direction reconciliation;
- **Gate C / V2** oracle harness work;
- **Gate J / V10** encoder/parser study;
- the decision about which shard/header/tensor material is actually needed next.

Follow `docs/REFERENCE_ACCESS.md`. Do not treat “full weights are unavailable” as “all model work is blocked”; PR #3 already disproved that equivalence.

---

## Phase 0 — provenance and baseline

**Status: DONE in PR #1.**

Deliverables:

- import WASTE `d9b919a791148b571e643d0af666bf19b4d733ab` without replacing this project's README;
- preserve Apache-2.0 `LICENSE` and `NOTICE`;
- record provenance in `UPSTREAM.md`;
- run upstream model-free tests before model modifications;
- document parked CI rather than weakening the upstream matrix.

Exit condition:

- upstream baseline builds and model-free tests pass;
- provenance is reproducible.

This phase is prerequisite work, not a separate README gate.

---

## Phase 1 — checkpoint inventory — README Gate A / V0

**Status: tool DONE; real input BLOCKED in the original environment.**

Existing deliverables from PR #1:

- `tools/inventory.py`;
- `tools/make_inventory_fixture.py`;
- `tests/test_inventory.py`;
- `docs/INVENTORY-0731.md`.

Next actions when Hugging Face is reachable:

1. fetch Tier-0/Tier-1 pinned assets using `docs/REFERENCE_ACCESS.md` and resolve the full immutable model SHA;
2. retrieve the exact official DeepSeek license before importing/adapting official source;
3. run index-only inventory immediately;
4. extend `RULES` until all real main-stack names are classified — do not hide unknown names in a catch-all;
5. reconcile routed-expert/scale/hash/DSpark naming assumptions;
6. obtain safetensors headers with the smallest provenance-preserving mechanism available and run strict header inventory;
7. commit `docs/inventory-0731.json` or equivalent machine-readable result;
8. update `README.md`, `docs/TENSOR_MAP.md`, `docs/NUMERICS.md`, and `docs/MEMORY_AND_IO.md` with checkpoint-derived facts.

A useful tooling follow-up, if the official host supports reliable Range requests, is remote safetensors-header fetching: first 8 bytes for header length, then only the JSON header. That could close storage truth without downloading whole shards merely to inspect their headers.

Exit gate — **Gate A / V0**:

- zero unexplained main-stack tensor names/bytes;
- config and tensor shapes agree or the docs are corrected;
- routed expert triplets/scales are proven from the checkpoint;
- hash-routing data for bootstrap layers is either proven or the architecture plan is corrected;
- DSpark tensors are cleanly separable from the base path;
- exact stored-byte totals exist;
- DeepSeek-specific quantization conventions needed by Gate B/V1 are identified from official source/artifacts.

No performance forecast is promoted to a project result before Gate A/V0.

---

## Phase 2 — official reference/oracle harness

**Status: BLOCKED in the original environment; NEXT immediately after Tier-1 reference access.**

This phase supplies fixtures for several gates rather than being a gate itself.

Deliverables:

- pinned copy/reference mechanism for official `inference/` and `encoding/` sources consistent with their license;
- `tools/deepseek_ref.py` or equivalent wrapper that can dump named intermediate activations;
- tiny deterministic fixture inputs/tokens;
- golden metadata that records model revision, source hashes, dtype/device, seed and operation;
- no WASTE implementation code required to generate the expected answer;
- frozen fixtures consumable offline by the C/Python test suite.

Required oracle seams, in increasing cost:

1. FP4 nibble order and scale application — **Gate B / V1**;
2. FP8 scale direction/layout and E4M3 target convention — **Gate B / V1**;
3. one quantized linear projection — **Gate C / V2**;
4. mHC transform/residual operation — **Gate D / V3**;
5. RoPE and model primitives — **V3**;
6. routing/shared/routed MoE — **Gate F / V4**;
7. attention compression/indexer selection — **Gate E / V5**;
8. one full transformer block — **Gate H / V6**;
9. multi-layer localization — **V7**;
10. final norm/head/logits — **Gate I / V8**;
11. greedy generation — **Gate K / V9**;
12. encoder token sequence and output parser behavior — **Gate J / V10**.

All fixtures must follow `docs/FIXTURES.md`: the expected side may not import the convention/helper being tested.

Exit condition: fixtures are deterministic, independently generated from official code, provenance-pinned, and replayable without network access.

---

## Phase 3 — scalar quantization kernels — README Gates B/C, V1/V2

**Status: PARTIALLY DONE in PR #3.** The number formats are public specifications, so the decoders were implementable and exhaustively verifiable without the checkpoint. Only the DeepSeek-specific conventions/reference agreement remain blocked.

Deliverables:

- [x] scalar E2M1 FP4 unpack/reference matvec path — `src/quant/fp4_e2m1.*`;
- [x] UE8M0 K32 scale decode/indexing path — **official DeepSeek use remains unverified**;
- [x] scalar FP8 E4M3/block-scale path — `src/quant/fp8_e4m3.*`, finite (`e4m3fn`) variant, ragged scale grids;
- [x] exhaustive public-format tests — every E2M1/E4M3 code and UE8M0 exponent state;
- [x] literal independent nibble-layout assertion so fixture producer/consumer cannot silently flip together;
- [x] double-accumulating scalar matvec references retained for later SIMD comparison;
- [ ] official-oracle fixtures for nibble/scale conventions — **Gate B / V1, BLOCKED**;
- [ ] official one-projection comparison — **Gate C / V2, BLOCKED**.

Exit gates:

- [x] scalar path lands before any SIMD, and SIMD is still absent;
- [x] clean under ASan/UBSan and warning-free build in PR #3;
- [x] all ten decoder mutations used in PR #3 are caught by the final suite;
- [ ] **Gate B / V1** official DeepSeek storage/convention agreement;
- [ ] **Gate C / V2** one quantized projection passes its documented tolerance.

The unresolved conventions are intentionally explicit:

```text
FP4 nibble order: current choice even column -> low nibble
FP8 scale direction: current choice stored scale multiplies decoded value
```

Neither is considered verified merely because the local round trip works. See `docs/NUMERICS.md`, `docs/FIXTURES.md`, and `docs/VALIDATION.md`.

---

## Phase 4 — DeepSeek-specific container and converter

**Status: DESIGN; exact schema waits on Gate A/V0 tensor truth.**

Deliverables:

- format/model-family discriminator that cannot be confused with WASTE Kimi v0;
- manifest schema/provenance fields;
- resident trunk layout preserving native quantization;
- one independently readable aligned record per routed expert containing the exact matrices/scales required at inference;
- resumable converter;
- header/record verification;
- synthetic DeepSeek-shaped container generator;
- conversion dry-run that calculates required disk space before writing model data.

Exit condition:

- converter round-trips selected real tensors to the scalar/official oracle;
- synthetic container is fully exercised in model-free tests;
- corruption/bounds tests fail safely;
- converter records source and code revisions.

See `docs/CONTAINER_V4.md` and `docs/CONVERSION.md`.

---

## Phase 5 — base model arithmetic — README Gates D/E/F/H/I/K, V3–V9

**Status: NOT STARTED; tensor binding and most primitives need real names/oracles.**

Recommended operational order:

1. config loader + tensor binding;
2. mHC and primitive seams — **Gate D / V3**;
3. router primitives — still V3-level localization;
4. routing/shared/routed MoE — **Gate F / V4**;
5. RoPE and attention projections;
6. sliding-window attention;
7. compression + compressed attention;
8. CSA indexer/selection — together completing **Gate E / V5**;
9. one full block — **Gate H / V6**;
10. multi-block forward/localization — **V7**;
11. final norm/head/logits — **Gate I / V8**;
12. greedy raw/known-token generation — **Gate K / V9**.

The V-order deliberately maps README F→V4 before E→V5. The README letters stay stable; operational test order follows `VALIDATION.md`.

Each PR gets an independent official-oracle differential test before moving to the next seam.

Exit condition:

- selected intermediate tensors pass;
- a full base-model forward pass produces matching logits within the agreed tolerance;
- greedy token sequence matches on a small deterministic known-token prompt/input;
- cache-on/cache-off produces equivalent numerics (Gate G).

---

## Phase 6 — memory planner and streaming behavior — README Gates G/L/M

**Status: PARTLY REUSED, DEEPSEEK MEASUREMENT NOT STARTED.**

Reuse WASTE's bounded-cache/direct-I/O design, then re-derive every size for DeepSeek.

Deliverables:

- exact resident trunk accounting;
- exact attention/state/context accounting;
- scratch/thread accounting;
- minimum expert-buffer requirement;
- direct-I/O record-size/alignment validation;
- **Gate G** cache/disk/prefetch identity tests;
- **Gate L** real target-volume storage measurement;
- real routing trace and **Gate M** cache sweep/recommendation;
- learned/preload/prefetch experiments only after baseline correctness.

Exit gates:

- `plan` accounts for every persistent allocation;
- budgets below the floor fail before loading;
- no benchmark relies on uncontrolled swap/page cache;
- **Gate G:** cache placement/prefetch never changes output;
- **Gate L:** real aligned record reads are measured on target storage;
- **Gate M:** cache recommendations come from a real 0731 routing curve, not Kimi or synthetic traffic.

---

## Phase 7 — encoding, CLI and OpenAI-compatible server — Gate J / V10 + V11

**Status: NOT STARTED.**

Deliverables:

- official DeepSeek encoding/parser port with differential tests — **Gate J / V10**;
- public C API remains model-agnostic;
- CLI features required by the model are exposed through the public API first;
- existing Python `serve/` path is extended, not replaced by an unrelated runtime;
- streaming/tool/structured-output behavior is enabled only where official encoding semantics support it;
- OpenAI-compatible request/response parity — extra operational rung **V11**.

Exit gates:

- token-for-token encoder parity on official test cases — Gate J/V10;
- parser parity on official cases plus malformed/truncated output tests — Gate J/V10;
- `/v1/chat/completions` works against the local engine with deterministic smoke tests — V11;
- user-facing generated content still rests on Gate K/V9 base-generation correctness.

---

## Phase 8 — performance work — reinforce Gates G/L/M

**Status: LATER.**

Only begin after base correctness.

Candidate work, gated by measurement:

- SIMD FP4/FP8 kernels — compare to scalar **and** official oracle;
- expert-parallel execution;
- chunked prefill adapted to real DeepSeek routing;
- asynchronous read-ahead;
- deterministic prefetch for hash-routed layers if Gate A/V0 confirms the mapping;
- cache-policy/routing trace study;
- CPU placement and thread tuning;
- Metal/CUDA experiments only when a measured bottleneck justifies them.

Every result goes to `docs/BENCHMARKS.md`; rejected ideas and test blind spots go to `docs/EXPERIMENTS.md`.

Exit condition: performance changes pass the same numerical tests as the scalar baseline, remain within the memory planner, preserve Gate G, and use Gate L/M methodology where relevant.

---

## Phase 9 — DSpark — README Gate N

**Status: LATER / explicitly blocked on base-model correctness.**

Deliverables and gates are in `docs/DSPARK.md`.

Minimum entry condition:

- base path passes **Gate I / V8** final-logit parity and **Gate K / V9** greedy-generation parity with DSpark disabled.

Exit gate — **Gate N**:

- disabling DSpark reproduces base behavior;
- acceptance/rejection semantics match the official reference;
- speedup is reported together with acceptance rate and memory cost;
- wrong speculative tokens can never alter final output.

---

## Definition of done for a project PR

A PR is complete when:

- implementation and relevant tests are in the same change;
- README §18 gate letter(s) are named where applicable;
- operational V-level/system gate and ROADMAP phase are named separately;
- claims have an evidence state from `docs/README.md`;
- expected fixture values are independent of the implementation/convention under test;
- the matching design/result doc is updated;
- manual gates run while CI is parked;
- licensing/provenance for copied/adapted material is explicit;
- an expensive next step is not taken when a cheaper gate is still unresolved.