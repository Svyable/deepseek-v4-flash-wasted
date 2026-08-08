# AGENTS.md — DeepSeek V4 Flash WASTE port operating guide

This is the first file an implementation agent should read after `README.md`.

The repository contains a verbatim WASTE import plus DeepSeek-specific additions. **Do not treat the imported Kimi implementation as the target architecture.** Read [`UPSTREAM.md`](UPSTREAM.md) and [`docs/README.md`](docs/README.md) to understand ownership and evidence state.

## Current state

Merged baseline through PR #3:

- WASTE imported from `sqliteai/waste @ d9b919a791148b571e643d0af666bf19b4d733ab`.
- Upstream model-free baseline before local model work: `31 passed, 0 failed, 12 skipped`.
- After PR #1 inventory tooling: `32 passed, 0 failed, 12 skipped`.
- Documentation foundation merged in PR #2 as `7c5c8d95fa7e1a9588b744aba4a6389bf77e98f7`.
- PR #3 merged as `91c36b8f4168349e6893a9911a3f60075d62d973` and added the first DeepSeek-specific arithmetic: scalar E2M1/UE8M0 and finite-E4M3 decoders/matvecs under `src/quant/`.
- Current model-free suite: **34 passed, 0 failed, 12 skipped**.
- PR #3 sanitizer run: **33 passed, 0 failed** under ASan/UBSan; clean rebuild had no `-Wall -Wextra` warnings.
- README **Gate B / V1** is **half satisfied**: public number-format conformance and local indexing/matvec consistency pass, but official DeepSeek nibble order, scale direction, scale layout and oracle projection agreement remain unverified.
- No DeepSeek transformer forward path has been ported.
- The official 0731 checkpoint/reference files were unreachable from the original bootstrap environment because the proxy returned CONNECT 403 for Hugging Face.
- `docs/INVENTORY-0731.md` therefore still contains no checkpoint-derived totals.
- CI is parked under `.github/workflows-disabled/`; run the gates manually.

Do not write code or documentation that implies a later phase has happened.

## Gate terminology — do not invent a fourth vocabulary

Before naming a gate, read `docs/VALIDATION.md` §4a.

- `README.md` §18 defines **14 stable design gates A–N**.
- `docs/VALIDATION.md` defines operational `V0–V11` levels.
- `ROADMAP.md` phases are schedule only.

When a README gate has a V-level, cite both: `Gate B / V1`, `Gate H / V6`, `Gate K / V9`.

README systems/performance gates `G`, `L`, `M`, and `N` intentionally have no V-number. Cite the letter directly. Operational `V7` and `V11` intentionally have no README letter.

Do **not** create local “Gate 0”, “Gate 1”, “Gate 2” numbering in new docs or PRs. Historical shorthand in the README maps to the canonical table:

```text
inventory sanity     -> Gate A / V0
mHC parity           -> Gate D / V3
router/MoE parity    -> Gate F / V4 (router primitive work begins in V3)
```

The README letter identifies the original design rationale; the V-level identifies the maintained operational test rung.

## Source-of-truth order

When two sources disagree, use this order:

1. pinned official `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint tensors and metadata;
2. pinned official 0731 `inference/` implementation;
3. pinned official 0731 `encoding/` implementation and tests;
4. official/public numeric-format specifications for semantics they actually define;
5. official config/model card/report;
6. this repository's measured tests and DeepSeek-specific docs;
7. WASTE's generic systems implementation and measured Kimi results;
8. third-party implementations such as vLLM, SGLang, llama.cpp or GGUF releases.

A public numeric spec can prove E2M1/E4M3 code semantics; it cannot prove DeepSeek-specific nibble order, tensor layout or scale direction. A third-party runtime is never a reason to change official DeepSeek semantics.

## Required reading by task

Before any model work:

1. `README.md`
2. `UPSTREAM.md`
3. `docs/README.md`
4. `docs/VALIDATION.md` §4a
5. `links.md`
6. `docs/INVENTORY-0731.md`
7. `docs/REFERENCE_ACCESS.md`
8. `ROADMAP.md`

For storage/runtime work also read upstream `docs/ENGINE.md`, `docs/FORMAT.md`, `docs/GATES.md`, `docs/LEARNED.md`, `src/ecache.*`, `src/memory.*`, `src/platform.h`, and `src/waste_format.h`.

For model arithmetic, read local:

- `docs/ARCHITECTURE.md`;
- `docs/TENSOR_MAP.md`;
- `docs/NUMERICS.md`;
- `docs/FIXTURES.md`;
- `docs/VALIDATION.md`;
- the pinned official DeepSeek reference once available.

Kimi `src/kda.*`, `docs/K3.md`, and `docs/KDA.md` are historical/reference material only.

## Non-negotiable invariants

### 1. Correctness before speed

The first implementation of every primitive is a scalar/reference path. Optimized SIMD/GPU paths come only after differential tests pass. PR #3 deliberately leaves the scalar quantization path available as the future SIMD reference.

### 2. The checkpoint wins

Do not infer tensor names, shapes, byte counts, or quantization storage conventions when they can be read from the pinned checkpoint/reference. `tools/inventory.py` must fail on unexplained main-stack tensors rather than absorb them into `other`.

### 3. Synthetic evidence is labeled synthetic

The fixture reproducing 11,008 expert records and approximately 3.21 GiB all-miss traffic proves only the fixture arithmetic. Until real 0731 headers are read, those values remain estimates.

Likewise, PR #3's exhaustive number-format tests prove public-format conformance, not that DeepSeek uses the current nibble/scale convention.

### 4. A fixture must not import the convention it is meant to prove

This is a hard rule after PR #3 mutation testing.

A round trip through our own packer and unpacker can stay green when both are wrong in the same way. The FP4 nibble-order mutation survived exhaustive 16-code coverage because fixture generation and decode shared `WASTE_FP4_LOW_NIBBLE_IS_EVEN`.

For silent conventions, require literal bytes/values or official-oracle artifacts independent of the implementation. Read `docs/FIXTURES.md`.

### 5. Native quantization first

Preserve the official FP4 expert and FP8 non-expert semantics for the first correct container/runtime. Do not introduce WASTE VQ3R, pruning, substitute experts, or other lossy transformations into the correctness baseline.

The public-format half is implemented in `src/quant/`; DeepSeek-specific convention reconciliation remains part of README Gate B / V1.

### 6. Placement must not change numerics — README Gate G

An expert loaded from disk and the same bytes served from the bounded cache must produce identical arithmetic. Prefetch may change timing only; it must never decide routing.

### 7. Hard RAM accounting

The engine must retain WASTE's configured maximum-RAM behavior. Every persistent allocation belongs in the memory plan. Do not depend on swap or an unbounded OS page cache to make a configuration work. Real target-storage and cache recommendations are README Gates L and M.

### 8. Model-family separation

Do not mutate the Kimi v0 format so a DeepSeek container can be mistaken for it. A DeepSeek model family/version must be explicit in the manifest and loader.

### 9. DSpark is phase 2 / README Gate N

Base final logits and greedy generation (Gate I/V8 and Gate K/V9) must pass before DSpark is enabled. DSpark may be optional at runtime and must have an easy off switch for parity tests.

### 10. Encoder semantics are part of model correctness — README Gate J / V10

Port the official code-based encoder/parser and differential-test it. Do not invent a Jinja template because it is more convenient for the server.

### 11. Record negative results and test blind spots

If an optimization is measured and rejected, append the result to `docs/EXPERIMENTS.md` rather than deleting the experiment from history.

The same applies to a test that falsely passed a known-wrong mutation. Test blind spots are engineering evidence and should survive in the record.

## Working sequence

Follow `ROADMAP.md`. Current position:

1. inventory tooling exists; real **Gate A / V0** input is blocked in the original environment;
2. official reference/oracle harness is the next dependency once reference access exists;
3. scalar E2M1/UE8M0/E4M3 paths are implemented and public-format-tested; finish **Gate B / V1** against official conventions before SIMD;
4. pass **Gate C / V2** with one official quantized linear projection;
5. define/write the DeepSeek-specific container only after real tensor names/layouts are known;
6. implement mHC and pass **Gate D / V3**;
7. implement routing/MoE and attention through **Gate F / V4** and **Gate E / V5**;
8. prove **Gate H / V6**, extra localization rung `V7`, then **Gate I / V8** final logits;
9. prove **Gate K / V9** greedy generation;
10. port encoding/parser and pass **Gate J / V10**;
11. integrate CLI/server and pass extra operational rung `V11`;
12. retain **Gate G** while profiling, then measure **Gate L** storage and **Gate M** cache behavior;
13. add DSpark only as **Gate N** after the base path is stable.

Do not skip a cheap gate to begin an expensive full conversion.

## Manual gates while CI is parked

At minimum before pushing source changes:

```bash
make
make check
```

For changes touching parsers, low-level memory, container I/O, or arithmetic, also run when practical:

```bash
make asan
make fuzz
```

PR #3 moved the authoritative SPDX check into `tests/run.sh` and made it recursive so `src/quant/` is actually covered. `make check` therefore exercises the live check.

If running it independently, use a recursive file selection rather than the old `src/*.c`/`src/*.h` glob:

```bash
git ls-files \
  | grep -E '^(src|cli|tests)/.*\.(c|h|m)$|^tools/.*\.(py|sh)$' \
  | xargs grep -L 'SPDX-License-Identifier'
```

Empty output is a pass. The parked imported workflow still has the old non-recursive glob and must be widened before restoration; see `.github/workflows-disabled/README.md`.

Python-only additions must have deterministic model-free tests whenever possible. Tests that require unavailable real weights should skip clearly rather than silently become no-ops.

## Mutation-testing guidance

Use mutation tests selectively at silent semantic seams. High-value targets include:

- bit masks/shifts;
- nibble/byte order;
- multiply-vs-divide scale application;
- block-index boundaries;
- matrix transpose/stride assumptions;
- router top-k ordering/ties;
- container offset/identity checks;
- tokenizer structure-vs-content mode.

Ask: **what one-line wrong implementation would still produce plausible numbers/text, and does a test kill it?**

A surviving mutation should cause the fixture to be strengthened before proceeding downstream.

## Licensing and provenance

WASTE is Apache-2.0 and its root `LICENSE`/`NOTICE` must remain with the redistributed source. `LICENSES/` records the multi-source attribution plan.

PR #1 deliberately did **not** fabricate a DeepSeek MIT license text when the official repository was unreachable. Until the exact official file is retrieved, do not copy DeepSeek source into this repository. Once reference code is copied/adapted, resolve the license file first and preserve the appropriate notice/SPDX information.

Whenever external source is adapted, state the source path and pinned revision in the commit/PR or nearby source comment.

## Documentation discipline

Use the evidence vocabulary in `docs/README.md`. Every important number should answer:

- what artifact/spec was measured or consulted?
- at which revision?
- on which hardware/configuration where relevant?
- by which command/tool?
- is it public-spec, checkpoint-derived, synthetic, or end-to-end?

Every implementation PR should also answer:

- which README gate letter(s) A–N does this affect?
- which operational V-level, if any, does it pass or depend on?
- which ROADMAP phase schedules the work?

Update the matching document in the same PR as an implementation change. A passing code path with stale docs is not finished.

Specific mappings now include:

- quantization arithmetic/conventions -> `docs/NUMERICS.md`;
- fixture/oracle methodology -> `docs/FIXTURES.md`;
- official reference acquisition/blocker -> `docs/REFERENCE_ACCESS.md`;
- gate/concordance changes -> README §18 + `docs/VALIDATION.md` §4a + `ROADMAP.md` + `docs/BENCHMARKS.md` + contributor guidance.

## What to do when official files are still blocked

Do not work around an organization egress policy.

Productive work that remains valid without the checkpoint includes:

- improving model-free/synthetic tests;
- hardening inventory/parser behavior;
- deriving/testing public numeric formats from their public specifications;
- mutation-testing silent local conventions;
- documenting interfaces and gates;
- extracting generic WASTE runtime boundaries.

Do **not** reconstruct DeepSeek reference source from memory or from third-party ports and label it official.

The highest-leverage next action is authorized access to the pinned official Tier-0/Tier-1 files described in `docs/REFERENCE_ACCESS.md`; a full weight download is not required to answer most remaining Gate A/V0 and Gate B/V1 convention questions.