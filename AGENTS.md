# AGENTS.md — DeepSeek V4 Flash WASTE port operating guide

This is the first file an implementation agent should read after `README.md`.

The repository contains a verbatim WASTE import plus a small set of DeepSeek-specific additions. **Do not treat the imported Kimi implementation as the target architecture.** Read [`UPSTREAM.md`](UPSTREAM.md) and [`docs/README.md`](docs/README.md) to understand ownership and evidence state.

## Current state

PR #1 established the baseline:

- WASTE imported from `sqliteai/waste @ d9b919a791148b571e643d0af666bf19b4d733ab`.
- Upstream model-free baseline before local model work: `31 passed, 0 failed, 12 skipped`.
- After `tools/inventory.py` and its tests: `32 passed, 0 failed, 12 skipped`.
- No DeepSeek inference code has been ported.
- The official 0731 checkpoint/reference files were unreachable from the bootstrap environment because the proxy returned CONNECT 403 for Hugging Face.
- `docs/INVENTORY-0731.md` therefore contains no checkpoint-derived totals.
- CI is parked under `.github/workflows-disabled/`; run the gates manually.

Do not write code or documentation that implies a later phase has happened.

## Source-of-truth order

When two sources disagree, use this order:

1. pinned official `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint tensors and metadata;
2. pinned official 0731 `inference/` implementation;
3. pinned official 0731 `encoding/` implementation and tests;
4. official config/model card/report;
5. this repository's measured tests and DeepSeek-specific docs;
6. WASTE's generic systems implementation and measured Kimi results;
7. third-party implementations such as vLLM, SGLang, llama.cpp or GGUF releases.

A third-party runtime is never a reason to change official DeepSeek semantics.

## Required reading by task

Before any model work:

1. `README.md`
2. `UPSTREAM.md`
3. `docs/README.md`
4. `links.md`
5. `docs/INVENTORY-0731.md`
6. `ROADMAP.md`

For storage/runtime work also read upstream `docs/ENGINE.md`, `docs/FORMAT.md`, `docs/GATES.md`, `docs/LEARNED.md`, `src/ecache.*`, `src/memory.*`, `src/platform.h`, and `src/waste_format.h`.

For model arithmetic, use local `docs/ARCHITECTURE.md`, `docs/TENSOR_MAP.md`, and `docs/VALIDATION.md` plus the official DeepSeek reference. Kimi `src/kda.*`, `docs/K3.md`, and `docs/KDA.md` are historical/reference material only.

## Non-negotiable invariants

### 1. Correctness before speed

The first implementation of every primitive is a scalar/reference path. Optimized SIMD/GPU paths come only after differential tests pass.

### 2. The checkpoint wins

Do not infer tensor names, shapes, byte counts, or quantization packing when they can be read from the pinned checkpoint. `tools/inventory.py` must fail on unexplained main-stack tensors rather than absorb them into `other`.

### 3. Synthetic evidence is labeled synthetic

The fixture reproducing 11,008 expert records and approximately 3.21 GiB all-miss traffic proves only the fixture arithmetic. Until real 0731 headers are read, those values remain estimates.

### 4. Native quantization first

Preserve the official FP4 expert and FP8 non-expert semantics for the first correct container/runtime. Do not introduce WASTE VQ3R, pruning, substitute experts, or other lossy transformations into the correctness baseline.

### 5. Placement must not change numerics

An expert loaded from disk and the same bytes served from the bounded cache must produce identical arithmetic. Prefetch may change timing only; it must never decide routing.

### 6. Hard RAM accounting

The engine must retain WASTE's configured maximum-RAM behavior. Every persistent allocation belongs in the memory plan. Do not depend on swap or an unbounded OS page cache to make a configuration work.

### 7. Model-family separation

Do not mutate the Kimi v0 format so a DeepSeek container can be mistaken for it. A DeepSeek model family/version must be explicit in the manifest and loader.

### 8. DSpark is phase 2

Base 43-layer logits and generation must pass before DSpark is enabled. DSpark may be optional at runtime and must have an easy off switch for parity tests.

### 9. Encoder semantics are part of model correctness

Port the official code-based encoder/parser and differential-test it. Do not invent a Jinja template because it is more convenient for the server.

### 10. Record negative results

If an optimization is measured and rejected, append the result to `docs/EXPERIMENTS.md` rather than deleting the experiment from history.

## Working sequence

Follow `ROADMAP.md`. At a high level:

1. establish/refresh inventory;
2. build official Python oracle fixtures;
3. implement scalar FP4/FP8 decode and matvec tests;
4. define and write the DeepSeek-specific container;
5. implement mHC;
6. implement attention/compression/indexer;
7. implement routing/shared/routed MoE;
8. prove one layer, then multi-layer, then final logits;
9. port encoding/parser;
10. integrate CLI/server;
11. profile and optimize;
12. add DSpark only after the base path is stable.

Do not skip a cheap gate to begin an expensive full conversion.

## Manual gates while CI is parked

At minimum before pushing source changes:

```bash
make
make check
```

For changes touching parsers, low-level memory, container I/O, or arithmetic, also run the relevant upstream targets when practical:

```bash
make asan
make fuzz
```

Run the SPDX check used by the imported CI:

```bash
git ls-files 'src/*.c' 'src/*.h' 'src/*.m' 'cli/*.c' 'tests/*.c' \
             'tools/*.py' 'tools/*.sh' \
  | xargs grep -L 'SPDX-License-Identifier'
```

Empty output is a pass.

Python-only additions must have deterministic model-free tests whenever possible. Tests that require unavailable real weights should skip clearly rather than silently become no-ops.

## Licensing and provenance

WASTE is Apache-2.0 and its root `LICENSE`/`NOTICE` must remain with the redistributed source. `LICENSES/` records the multi-source attribution plan.

PR #1 deliberately did **not** fabricate a DeepSeek MIT license text when the official repository was unreachable. Until the exact official file is retrieved, do not copy DeepSeek source into this repository. Once reference code is copied/adapted, resolve the license file first and preserve the appropriate notice/SPDX information.

Whenever external source is adapted, state the source path and pinned revision in the commit/PR or nearby source comment.

## Documentation discipline

Use the evidence vocabulary in `docs/README.md`. Every important number should answer:

- what artifact was measured?
- at which revision?
- on which hardware/configuration?
- by which command/tool?
- is it checkpoint-derived, synthetic, or end-to-end?

Update the matching document in the same PR as an implementation change. A passing code path with stale docs is not finished.

## What to do when official files are still blocked

Do not work around an organization egress policy. Productive work that remains valid without the checkpoint includes:

- improving model-free/synthetic tests;
- hardening inventory/parser behavior;
- documenting interfaces and gates;
- extracting generic WASTE runtime boundaries;
- adding scalar quantization tests only when their bit-level semantics are grounded in accessible official material already recorded in the repo.

Do **not** reconstruct DeepSeek reference source from memory or from third-party ports and label it official.