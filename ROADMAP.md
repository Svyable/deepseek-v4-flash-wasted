# Roadmap — DeepSeek V4 Flash on WASTE

This roadmap converts the implementation plan in `README.md` into gated PR-sized work. It is deliberately ordered so a cheap failure prevents an expensive download, conversion, or optimization.

Status values: **DONE**, **IN PROGRESS**, **BLOCKED**, **NEXT**, **LATER**.

## Gate vocabulary

The phases below are a **schedule**, not a fourth gate vocabulary.

- `README.md` §18 defines **14 stable design gates A–N**.
- `docs/VALIDATION.md` defines operational `V0–V11` levels.
- `docs/VALIDATION.md` §4a maps every README gate to its V-level or systems/performance owner.

When both identifiers exist, cite both (`Gate B / V1`, `Gate H / V6`, `Gate K / V9`). README Gates `G/L/M/N` intentionally have no V-number. Operational `V7/V11` intentionally have no README letter.

## Current snapshot

| Work | Canonical gate | Status | Evidence |
|---|---|---|---|
| PR #1 — WASTE bootstrap + inventory tool | supports **Gate A / V0** | **DONE** | merged as `edd2a41b66332e5a54ed54bcbb196fec19664079` |
| PR #2 — documentation foundation | process | **DONE** | merged as `7c5c8d95fa7e1a9588b744aba4a6389bf77e98f7` |
| PR #3 — scalar public quantization | **Gate B / V1a** | **DONE** | merged as `91c36b8f4168349e6893a9911a3f60075d62d973`; exhaustive public-format conformance + mutation testing |
| PR #4 — canonical gate/docs contract | process | **DONE** | merged as `3ce11ef4d391208d0c455796bf21803620910948` |
| PR #5 — official header acquisition + quantization source gates | **A/V0, B/V1, C/V2 preflight** | **IN PROGRESS / DRAFT** | exact release/license pinned; Range-safe header fetcher; official FP4/FP8 conventions; scalar official-linear preflight |
| Real 0731 48-shard header inventory | **Gate A / V0** | **NEXT / environment-dependent** | tooling exists; exact official headers have not yet been consumed in this repo |
| Native DeepSeek convention replay | **Gate B / V1** | **SOURCE-VERIFIED; branch test pending** | pinned F3 fixture + C replay wired into `make check`; fresh PR #5 checkout run still required |
| Official one-projection oracle | **Gate C / V2** | **NEXT** | source-derived scalar linear exists; real projection fixture still required |
| DeepSeek transformer implementation | **Gates D/E/F/H/I/K; V3–V9** | **NOT STARTED** | no ported transformer forward path yet |
| Encoding/server | **Gate J / V10 + V11** | **NOT STARTED** | official encoding path not ported |
| Storage/cache feasibility | **Gates G/L/M** | **LATER** | mechanisms imported; DeepSeek measurements absent |
| DSpark | **Gate N** | **LATER** | base model must pass Gate I/V8 and Gate K/V9 first |

Last merged executable baseline remains PR #3:

```text
make check -> 34 passed, 0 failed, 12 skipped
make asan  -> 33 passed, 0 failed
```

PR #5 deliberately remains draft until a checkout-capable environment runs the integrated suite. Narrow source-derived checks were executed during development, but they are not promoted to a full branch test.

## Immediate priority

The highest-leverage sequence has moved forward. Tier-0 licensing and the immutable release SHA are resolved. The next actions are now:

1. run `tools/fetch_hf_headers.py` at the pinned release SHA from a Hub-capable checkout;
2. run strict Gate A/V0 inventory over all 48 header stubs and reconcile every real tensor name/shape/byte;
3. freeze one real resident quantized projection and pass Gate C/V2 against `deepseek_v4_linear_ref`;
4. only then begin Gate D/V3 / Gate F/V4 transformer primitives.

Pinned release baseline:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

Do not silently replace that baseline with a later model-card-only repository tip.

---

## Phase 0 — provenance and baseline

**Status: DONE.**

Delivered across PR #1 and PR #5:

- WASTE imported at `d9b919a791148b571e643d0af666bf19b4d733ab`;
- Apache-2.0 root license/notice preserved;
- DeepSeek release MIT license now vendored exactly from the pinned release;
- upstream/model-free baseline recorded;
- parked CI documented rather than weakened.

Exit condition is satisfied for provenance. This phase is prerequisite work, not a separate README gate.

---

## Phase 1 — checkpoint inventory — README Gate A / V0

**Status: IN PROGRESS in PR #5. Acquisition/inventory tooling exists; real 48-shard header truth is NEXT.**

Delivered:

- `tools/inventory.py` — header-only inventory and strict Gate A checks;
- `tools/make_inventory_fixture.py` — synthetic structural fixture;
- `tests/test_inventory.py` — model-free inventory/failure tests;
- `tools/fetch_hf_headers.py` — immutable, Range-safe Hugging Face header acquisition;
- `tests/test_fetch_hf_headers.py` — offline fake-Hub transport/safety tests;
- classifier support for pinned official `tid2eid` and converter `tie2eid` bootstrap-route spellings;
- `F8_E8M0` dtype awareness;
- pinned full release SHA and exact DeepSeek license.

The header fetcher intentionally writes native safetensors header-only stubs:

```text
8-byte little-endian header length
JSON safetensors header
(no tensor payload)
```

`inventory.py` therefore remains the single implementation of dtype/shape/byte accounting. The fetcher requires HTTP `206 Partial Content`; a server/proxy that ignores Range is refused rather than allowed to download a giant shard unexpectedly.

Next Gate A run:

```bash
python3 tools/fetch_hf_headers.py \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --out reference/deepseek-v4-flash-0731

python3 tools/inventory.py reference/deepseek-v4-flash-0731 \
  --strict --by-layer --json docs/inventory-0731.json
```

Expected engineering behavior: the first real run may still fail on tensor names the current classifier has not seen. Fix only against official names; never add a generic main-stack catch-all.

Exit gate — **Gate A / V0**:

- zero unexplained main-stack names/bytes;
- all 48 shard headers accounted for at the pinned SHA;
- config and tensor shapes agree or docs/code are corrected;
- routed expert triplets/scales are complete;
- bootstrap-routing tensors are mapped exactly;
- DSpark tensors are cleanly separable;
- exact stored-byte totals exist;
- generated inventory provenance is committed.

No performance forecast is promoted to a project result before Gate A/V0.

---

## Phase 2 — official reference/oracle harness

**Status: STARTED in PR #5. Source-level contracts are pinned; real numerical fixtures continue incrementally.**

Delivered so far:

- immutable official release baseline;
- exact license/provenance;
- `docs/OFFICIAL-0731-SOURCE.md`;
- independent F3 FP4 convention fixture;
- source-level routing, quantization and expert-clamp findings;
- scalar official-linear-shaped Gate C preflight.

Still needed:

- `tools/deepseek_ref.py` or equivalent wrapper for named official intermediates;
- real projection fixture for Gate C/V2;
- subsequent mHC/router/attention/expert fixtures;
- encoder/parser fixtures for Gate J/V10.

Oracle seams, in increasing cost:

1. FP4 nibble/scale convention — **Gate B / V1** — source established, F3 fixture added;
2. activation quantization + one real trunk projection — **Gate C / V2** — source preflight exists, real fixture next;
3. mHC/residual operation — **Gate D / V3**;
4. RoPE/model primitives — **V3**;
5. routing/shared/routed MoE — **Gate F / V4**;
6. attention compression/indexer — **Gate E / V5**;
7. one full transformer block — **Gate H / V6**;
8. multi-layer localization — **V7**;
9. final logits — **Gate I / V8**;
10. greedy generation — **Gate K / V9**;
11. encoder/parser — **Gate J / V10**.

All fixtures follow `docs/FIXTURES.md`: expected values may not import the implementation/convention being tested.

---

## Phase 3 — scalar quantization kernels — README Gates B/C, V1/V2

**Status: Gate B semantics substantially advanced in PR #5; Gate C real projection remains NEXT.**

Delivered:

- [x] scalar E2M1 FP4 decode/matvec — `src/quant/fp4_e2m1.*`;
- [x] UE8M0 K32 scale decode/indexing;
- [x] scalar finite E4M3FN + 128x128 block-scale decode/matvec — `src/quant/fp8_e4m3.*`;
- [x] exhaustive public-format tests and PR #3 mutation coverage;
- [x] official source confirms low-nibble-first FP4 packing;
- [x] official source confirms routed `[out, in/32]` E8M0 scale geometry and multiplication semantics;
- [x] official source confirms finite E4M3FN resident path and 128x128 weight-scale geometry;
- [x] independent pinned F3 source-convention fixture;
- [x] C replay test wired into the existing `make check` path;
- [x] source-derived scalar activation quantization + FP8 linear reference in `deepseek_v4_linear_ref.*`;
- [x] closed-form Gate C preflight test;
- [ ] fresh full PR #5 checkout test run;
- [ ] real Gate C/V2 official one-projection fixture + fixed tolerance.

Important distinction:

**Gate B source semantics are no longer blocked on nibble order or scale direction.** Exact checkpoint tensor storage remains Gate A, and Gate C still needs actual numerical projection evidence.

Exit gates:

- **Gate B / V1:** integrated pinned convention replay passes on the branch and remains consistent with Gate A checkpoint truth;
- **Gate C / V2:** one real official quantized projection matches the scalar official-linear-shaped path under a justified fixed tolerance and catches known-wrong scale/activation/rounding mutations.

No SIMD before Gate C/V2.

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

**Status: NOT STARTED; do not begin full transformer integration until Gate C/V2 is real.**

Recommended operational order:

1. config loader + exact tensor binding from Gate A;
2. mHC and primitive seams — **Gate D / V3**;
3. router primitives — V3-level localization;
4. routing/shared/routed MoE — **Gate F / V4**;
5. RoPE and attention projections;
6. sliding-window attention;
7. compression + compressed attention;
8. CSA indexer/selection — **Gate E / V5**;
9. one full block — **Gate H / V6**;
10. multi-block localization — **V7**;
11. final norm/head/logits — **Gate I / V8**;
12. greedy raw/known-token generation — **Gate K / V9**.

Each implementation PR gets an independent official-oracle differential test before moving downstream.

Exit condition:

- selected intermediate tensors pass;
- full base-model forward matches logits within the agreed budget;
- greedy token sequence matches a deterministic official fixture;
- cache-on/cache-off remains numerically equivalent (Gate G).

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
- OpenAI-compatible request/response parity — **V11**.

Exit gates:

- token-for-token encoder parity — Gate J/V10;
- parser parity including malformed/truncated output — Gate J/V10;
- `/v1/chat/completions` deterministic smoke tests — V11;
- user-facing generation still rests on Gate K/V9 model correctness.

---

## Phase 8 — performance work — reinforce Gates G/L/M

**Status: LATER.**

Only begin after base correctness.

Candidate work:

- SIMD FP4/FP8 kernels — compare to scalar and official oracle;
- expert-parallel execution;
- chunked prefill adapted to real routing;
- asynchronous read-ahead;
- deterministic bootstrap prefetch after Gate A confirms exact mapping;
- cache-policy/routing trace study;
- CPU placement/thread tuning;
- Metal/CUDA only when profiling justifies it.

Every result goes to `docs/BENCHMARKS.md`; rejected ideas/test blind spots go to `docs/EXPERIMENTS.md`.

Exit condition: performance changes pass the same numerical tests, remain within the planner, preserve Gate G, and use Gate L/M methodology.

---

## Phase 9 — DSpark — README Gate N

**Status: LATER / explicitly blocked on base-model correctness.**

Minimum entry condition:

- base path passes **Gate I / V8** and **Gate K / V9** with DSpark disabled.

Exit gate — **Gate N**:

- disabling DSpark reproduces base behavior;
- acceptance/rejection semantics match the official reference;
- speedup is reported with acceptance rate and memory cost;
- wrong speculative tokens can never alter committed output.

---

## Definition of done for a project PR

A PR is complete when:

- implementation and relevant tests are in the same change;
- README §18 gate letter(s) are named where applicable;
- operational V-level/system gate and roadmap phase are named separately;
- claims have an evidence state from `docs/README.md`;
- expected fixture values are independent of the implementation/convention under test;
- matching docs are updated in the same PR;
- manual gates run while CI is parked;
- licensing/provenance for copied/adapted material is explicit;
- a cheaper unresolved gate is not skipped for a more expensive next step.
