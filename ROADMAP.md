# Roadmap — DeepSeek V4 Flash on WASTE

This roadmap converts the implementation plan in `README.md` into gated PR-sized work. It is deliberately ordered so a cheap failure prevents an expensive download, conversion, or optimization.

Status values: **DONE**, **IN PROGRESS**, **BLOCKED**, **NEXT**, **LATER**.

The phases below are a *schedule*. They are not a fourth gate vocabulary
alongside `README.md` §18's Gates A–N, `README.md` §20's PR 1–13, and
`docs/VALIDATION.md`'s `V`-levels. [`docs/VALIDATION.md`
§4a](docs/VALIDATION.md) maps all of them to each other — cite the `V`-level
in a PR, and use the phase only to say *when* the work happens.

## Current snapshot

| Phase | Status | Evidence |
|---|---|---|
| PR #1 — WASTE bootstrap + attribution + inventory tool | **DONE** | merged as `edd2a41b66332e5a54ed54bcbb196fec19664079` |
| Documentation foundation | **DONE** | merged as `7c5c8d95fa7e1a9588b744aba4a6389bf77e98f7` |
| Phase 3 — scalar quantization decoders | **PARTIALLY DONE** | `src/quant/`, exhaustive spec conformance; reference agreement blocked |
| Real 0731 metadata/header inventory | **BLOCKED** in bootstrap environment | Hugging Face CONNECT 403; see `docs/INVENTORY-0731.md` |
| DeepSeek inference implementation | **NOT STARTED** | no ported forward code exists |
| DSpark | **LATER** | base model must pass first |

The model-free suite is **34 passed, 0 failed, 12 skipped** as of the
quantization work; it was 32/0/12 after PR #1.

The last verified model-free suite from PR #1 is **32 passed, 0 failed, 12 skipped**.

---

## Phase 0 — provenance and baseline

**Status: DONE in PR #1.**

Deliverables:

- import WASTE `d9b919a791148b571e643d0af666bf19b4d733ab` without replacing this project's README;
- preserve Apache-2.0 `LICENSE` and `NOTICE`;
- record provenance in `UPSTREAM.md`;
- run upstream model-free tests before model modifications;
- document parked CI rather than weakening the upstream matrix.

Exit gate:

- upstream baseline builds and model-free tests pass;
- provenance is reproducible.

---

## Phase 1 — checkpoint inventory / Gate 0

**Status: tool DONE; real input BLOCKED in the original environment.**

Existing deliverables from PR #1:

- `tools/inventory.py`;
- `tools/make_inventory_fixture.py`;
- `tests/test_inventory.py`;
- `docs/INVENTORY-0731.md`.

Next actions when Hugging Face is reachable:

1. fetch pinned metadata/reference assets for `DeepSeek-V4-Flash-0731 @ 9e165c3`;
2. run index-only inventory;
3. extend `RULES` until all real main-stack names are classified;
4. fetch shard headers/weights as needed and run strict header inventory;
5. commit `docs/inventory-0731.json` or equivalent machine-readable result;
6. update `README.md`, `docs/TENSOR_MAP.md`, and `docs/MEMORY_AND_IO.md` with checkpoint-derived facts;
7. retrieve the exact official DeepSeek license before importing/adapting official source.

Exit gate:

- zero unexplained main-stack tensor names/bytes;
- config and tensor shapes agree or the docs are corrected;
- routed expert triplets/scales are proven from the checkpoint;
- hash-routing data for bootstrap layers is either proven or the architecture plan is corrected;
- DSpark tensors are cleanly separable from the base 43-layer path;
- exact stored-byte totals exist.

No performance forecast is promoted to a project result before this gate.

---

## Phase 2 — official reference/oracle harness

**Status: NEXT after official reference access.**

Deliverables:

- pinned copy/reference mechanism for official `inference/` and `encoding/` sources consistent with their license;
- `tools/deepseek_ref.py` or equivalent wrapper that can dump named intermediate activations;
- tiny deterministic fixture inputs/tokens;
- golden metadata that records model revision, dtype/device, seed and operation;
- no WASTE code required to generate the oracle.

Required oracle seams:

- E2M1 FP4 decode + scale application;
- FP8 decode/block-scale behavior;
- one linear projection;
- mHC transform/residual operation;
- RoPE;
- attention compression/indexer selection;
- router scores, selected expert IDs, normalized weights;
- shared expert;
- routed expert SwiGLU;
- one full transformer block;
- final norm/head/logits;
- encoder token sequence and output parser behavior.

Exit gate: fixtures are deterministic and can be regenerated from official code at the pinned revision.

---

## Phase 3 — scalar quantization kernels

**Status: PARTIALLY DONE.** The number formats are public specifications, so
the decoders were implementable and exhaustively verifiable without the
checkpoint. Only the DeepSeek-specific conventions are still blocked.

Deliverables:

- [x] scalar E2M1 FP4 unpack/reference matvec path — `src/quant/fp4_e2m1.*`;
- [x] UE8M0 K32 scale handling — decode and block indexing done; **"exactly
      matching the official reference" is unverified**, see below;
- [x] scalar FP8 E4M3/block-scale path for non-expert quantized tensors —
      `src/quant/fp8_e4m3.*`, finite (`e4m3fn`) variant, ragged grids;
- [ ] C tests driven by oracle-generated fixtures — **BLOCKED**. The
      standing tests are spec-derived and exhaustive, not oracle-derived;
- [x] explicit NaN/Inf/subnormal/rounding behavior — every code of both
      formats enumerated, including both E4M3 NaN encodings, subnormals at
      each end, and UE8M0's 2^-127 which is subnormal in binary32.

Exit gate:

- [x] scalar path lands before any SIMD, and SIMD is still absent;
- [x] clean under ASan/UBSan; ten mutations of the decoders each caught
      (`docs/EXPERIMENTS.md` entry 3);
- [ ] decoded values compared against official reference decode semantics —
      **BLOCKED** on checkpoint/reference access.

Two conventions stay open because no specification settles them: FP4 nibble
order and FP8 scale direction. Both are pinned by literal-byte assertions
that will fail loudly if the official reference disagrees. See
`docs/VALIDATION.md` V1 "Status: half satisfied".

---

## Phase 4 — DeepSeek-specific container and converter

**Status: DESIGN.**

Deliverables:

- format/model-family discriminator that cannot be confused with WASTE Kimi v0;
- manifest schema/provenance fields;
- resident trunk layout preserving native quantization;
- one independently readable aligned record per routed expert containing the exact matrices/scales required at inference;
- resumable converter;
- header/record verification;
- synthetic DeepSeek-shaped container generator;
- conversion dry-run that calculates required disk space before writing model data.

Exit gate:

- converter round-trips selected real tensors to the scalar oracle;
- synthetic container is fully exercised in model-free tests;
- corruption/bounds tests fail safely;
- converter records source and code revisions.

See `docs/CONTAINER_V4.md` and `docs/CONVERSION.md`.

---

## Phase 5 — base model arithmetic

**Status: NOT STARTED.**

Recommended PR order:

1. config loader + tensor binding;
2. mHC;
3. RoPE and attention projections;
4. sliding-window attention;
5. compression + compressed attention;
6. CSA indexer/selection;
7. deterministic/bootstrap routing;
8. learned router/top-6 semantics;
9. shared expert;
10. streamed routed experts;
11. one full block;
12. multi-block forward;
13. final norm/head/logits.

Each PR gets an oracle differential test before moving to the next seam.

Exit gate:

- selected intermediate tensors pass;
- a full base-model forward pass produces matching logits within the agreed tolerance;
- greedy token sequence matches on a small deterministic prompt;
- cache-on/cache-off produces equivalent numerics.

---

## Phase 6 — memory planner and streaming behavior

**Status: PARTLY REUSED, DEEPSEEK MEASUREMENT NOT STARTED.**

Reuse WASTE's bounded-cache/direct-I/O design, then re-derive every size for DeepSeek.

Deliverables:

- exact resident trunk accounting;
- exact attention/state/context accounting;
- scratch/thread accounting;
- minimum expert-buffer requirement;
- recommended bounded-cache policy based on measured DeepSeek routing;
- direct-I/O record-size/alignment validation;
- learned/preload/prefetch experiments only after baseline correctness.

Exit gate:

- `plan` accounts for every persistent allocation;
- budgets below the floor fail before loading;
- no benchmark relies on uncontrolled swap/page cache;
- cache placement never changes output.

---

## Phase 7 — encoding, CLI and OpenAI-compatible server

**Status: NOT STARTED.**

Deliverables:

- official DeepSeek encoding/parser port with differential tests;
- public C API remains model-agnostic;
- CLI features required by the model are exposed through the public API first;
- existing Python `serve/` path is extended, not replaced by an unrelated runtime;
- streaming/tool/structured-output behavior is enabled only where official encoding semantics support it.

Exit gate:

- token-for-token encoder parity on official test cases;
- parser parity on official cases plus malformed/truncated output tests;
- `/v1/chat/completions` works against the local engine with deterministic smoke tests.

---

## Phase 8 — performance work

**Status: LATER.**

Only begin after base correctness.

Candidate work, gated by measurement:

- SIMD FP4/FP8 kernels;
- expert-parallel execution;
- chunked prefill adapted to real DeepSeek routing;
- asynchronous read-ahead;
- deterministic prefetch for hash-routed layers if Gate 0 confirms the mapping;
- cache-policy/routing trace study;
- CPU placement and thread tuning;
- Metal/CUDA experiments only when a measured bottleneck justifies them.

Every result goes to `docs/BENCHMARKS.md`; rejected ideas go to `docs/EXPERIMENTS.md`.

Exit gate: performance changes pass the same numerical tests as the scalar baseline.

---

## Phase 9 — DSpark

**Status: LATER / explicitly blocked on base-model correctness.**

Deliverables and gates are in `docs/DSPARK.md`.

Minimum entry condition:

- base 43-layer path passes final-logit and greedy-generation parity with DSpark disabled.

Exit gate:

- disabling DSpark reproduces base behavior;
- acceptance/rejection semantics match the official reference;
- speedup is reported together with acceptance rate and memory cost;
- wrong speculative tokens can never alter final output.

---

## Definition of done for a project PR

A PR is complete when:

- implementation and relevant tests are in the same change;
- claims have an evidence state from `docs/README.md`;
- the matching design/result doc is updated;
- manual gates run while CI is parked;
- licensing/provenance for copied/adapted material is explicit;
- an expensive next step is not taken when a cheaper gate is still unresolved.