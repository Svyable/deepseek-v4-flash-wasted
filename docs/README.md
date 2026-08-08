# Documentation map

This directory contains two different kinds of material and they must not be confused:

1. **DeepSeek V4 Flash port documents** written in this repository.
2. **Imported WASTE documents** copied verbatim from `sqliteai/waste @ d9b919a791148b571e643d0af666bf19b4d733ab` during PR #1.

Read [`../UPSTREAM.md`](../UPSTREAM.md) before assuming that a document in this directory describes DeepSeek. The bootstrap intentionally preserved upstream Kimi/K3 documentation so the systems design, measurements, and negative results remain available as evidence.

## Current evidence state

As of merged PR #3 (`91c36b8f4168349e6893a9911a3f60075d62d973`):

- WASTE is imported from `sqliteai/waste @ d9b919a791148b571e643d0af666bf19b4d733ab` and the documentation foundation from PR #2 is merged;
- `make check` reports **34 passed, 0 failed, 12 skipped** after the native quantization tests were added;
- PR #3 added the repository's first DeepSeek-specific arithmetic: scalar E2M1/UE8M0 and finite-E4M3 (`e4m3fn`) decoders/matvecs under `src/quant/`;
- those public number-format semantics and model-free scale-indexing paths are **SYNTHETIC-VERIFIED** by exhaustive tests;
- README **Gate B / V1** is **half satisfied**: public-format conformance passes, but DeepSeek-specific nibble order, scale direction/application, and official projection parity remain unverified;
- mutation testing exposed and repaired a shared-assumption fixture bug in the FP4 nibble-order test; see [`FIXTURES.md`](FIXTURES.md) and `EXPERIMENTS.md` entry 3;
- `tools/inventory.py` remains tested against a synthetic safetensors fixture, but the official `DeepSeek-V4-Flash-0731` checkpoint/reference has **not** been read in the original bootstrap environment because the proxy denied `huggingface.co:443` with CONNECT 403;
- therefore no checkpoint-derived byte totals, exact tensor-name/shape map, RAM floor, disk footprint, routing trace, quality result, full projection oracle, or throughput result exists yet;
- no DeepSeek transformer forward path has been ported yet;
- CI remains parked under `.github/workflows-disabled/`; manual `make check`/sanitizer validation remains authoritative until Actions can actually start jobs.

The checkpoint/reference-access limitation and the smallest useful acquisition sequence are documented in [`REFERENCE_ACCESS.md`](REFERENCE_ACCESS.md). Do not convert synthetic arithmetic or public-format conformance into claims about the official checkpoint.

## Gate vocabulary — one map, three purposes

The project has three related labels and they are not interchangeable:

1. **README §18 gates A–N** — 14 stable design gates. These letters identify *why* a gate exists and must not disappear from cross-references.
2. **`VALIDATION.md` V0–V11** — operational numerical/semantic validation levels. Not every README gate has a V-number, and two V-levels have no README letter.
3. **`ROADMAP.md` phases 0–9** — schedule only. A phase says *when* work is expected, not what correctness proof passed.

The canonical mapping is [`VALIDATION.md` §4a](VALIDATION.md), which maps **all 14 README gates A–N** to their V-level or systems/performance owner while preserving the concept column and roadmap phase.

Important asymmetries are intentional:

- README **G/L/M/N** are real systems/performance gates with no invented V-number;
- **V7** (multi-layer localization) and **V11** (API parity) are maintained operational levels with no README letter;
- README **E** maps to `V5` while **F** maps to `V4` because operational bring-up order differs from the original handoff ordering;
- README **J** maps to `V10` while **K** maps to `V9` because raw deterministic generation can be validated before the full chat encoder/parser surface.

In implementation PRs, cite both identifiers when both exist, for example `Gate B / V1`, `Gate H / V6`, or `Gate K / V9`. Cite `Gate G`, `Gate L`, `Gate M`, or `Gate N` directly for the systems/performance gates.

Do not introduce new local labels such as “Gate 0/1/2.” Existing historical shorthand in the handoff should be read through the canonical mapping.

## Evidence labels used by local docs

Local documents should identify important claims with one of these states:

| State | Meaning |
|---|---|
| **UPSTREAM-MEASURED** | Measured in WASTE on its Kimi targets. Useful systems evidence, not a DeepSeek measurement. |
| **OFFICIAL-SPEC** | Taken from a pinned public/official specification or DeepSeek release/config/reference material, but not independently measured here. |
| **SYNTHETIC-VERIFIED** | Exercised by a repository-generated fixture, exhaustive public-format test, or other model-free test. |
| **CHECKPOINT-VERIFIED** | Derived from the pinned official 0731 checkpoint headers/tensors/reference outputs. |
| **END-TO-END-VERIFIED** | Observed in this port running the real model. |
| **DESIGN** | Proposed behavior or format that is not implemented yet. |
| **BLOCKED** | Cannot currently be established because a required external artifact/environment is unavailable. |

A later, stronger state supersedes an earlier one. Never silently promote a claim.

## DeepSeek-specific documents

These documents are authoritative for this port unless the official checkpoint/reference code proves them wrong:

- [`INVENTORY-0731.md`](INVENTORY-0731.md) — exact checkpoint inventory status and README **Gate A / V0** procedure.
- [`REFERENCE_ACCESS.md`](REFERENCE_ACCESS.md) — smallest official-reference acquisition sequence; primarily unblocks **Gate A / V0**, **Gate B / V1**, and **Gate C / V2** before full-weight download.
- [`DEEPSEEK_V4.md`](DEEPSEEK_V4.md) — compact DeepSeek-specific architecture/porting reference and the assumptions Gate A/V0 must verify.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system decomposition and what is reused versus rewritten; owns major model-seam rationale for Gates D/E/F/H/I.
- [`TENSOR_MAP.md`](TENSOR_MAP.md) — tensor-family mapping contract and the open nibble/scale/tensor-layout questions.
- [`NUMERICS.md`](NUMERICS.md) — scalar E2M1/UE8M0/E4M3FN arithmetic contract, exactness rules, unresolved DeepSeek conventions, and **Gate B/V1 → Gate C/V2** handoff.
- [`FIXTURES.md`](FIXTURES.md) — independent-fixture and mutation-testing rules; prevents producer/consumer tests from sharing the convention they are supposed to prove.
- [`VALIDATION.md`](VALIDATION.md) — canonical operational ladder. Its §4a maps every README Gate A–N to a V-level/system owner, concept, roadmap phase, and owning docs.
- [`CONTAINER_V4.md`](CONTAINER_V4.md) — proposed DeepSeek-specific WASTE container contract.
- [`CONVERSION.md`](CONVERSION.md) — download, inventory, conversion, verification, and recovery runbook.
- [`MEMORY_AND_IO.md`](MEMORY_AND_IO.md) — RAM-floor and storage/I/O accounting methodology; owns README **Gate G**, **Gate L**, and **Gate M** alongside validation/benchmarks.
- [`BENCHMARKS.md`](BENCHMARKS.md) — benchmark schema and results ledger; records README gate letters and V-levels separately.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — append-only experiment and negative-result log for this port; historical gate wording is preserved with a translation note.
- [`DSPARK.md`](DSPARK.md) — README **Gate N** speculative-decoding boundary and validation plan.
- [`API.md`](API.md) — library/CLI/OpenAI-compatible server integration contract; owns **Gate J/V10**, contributes to **Gate K/V9**, and owns extra operational level `V11`.
- [`PLATFORM.md`](PLATFORM.md) — macOS/Linux/Windows I/O and backend requirements; supports Gates G/L/M.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — failure diagnosis ordered by the project's canonical gates.

Root/local companion documents:

- [`../README.md`](../README.md) — implementation handoff and the stable 14-gate A–N design rationale.
- [`../links.md`](../links.md) — curated source pack and source-of-truth hierarchy.
- [`../AGENTS.md`](../AGENTS.md) — instructions for coding agents and contributors.
- [`../ROADMAP.md`](../ROADMAP.md) — phase schedule and current project status; not a separate gate vocabulary.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — contribution scope, testing, documentation and evidence requirements.
- [`../UPSTREAM.md`](../UPSTREAM.md) — import provenance and local-vs-upstream ownership.
- [`../reference/README.md`](../reference/README.md) — local official-artifact staging/provenance policy; raw model assets are ignored.

## Imported WASTE documents

The following arrived from upstream and should retain their original historical meaning until intentionally forked:

- `ENGINE.md` — WASTE library/CLI/memory design, mostly reusable systems reasoning.
- `FORMAT.md` — WASTE v0 Kimi container format. **Not** the DeepSeek format.
- `GATES.md` — upstream feasibility-gate history; it is not this port's A–N/V-level concordance.
- `EFFICIENCY.md` — upstream Kimi performance measurements.
- `LEARNED.md` — upstream append-only negative-result/measurement log.
- `BACKENDS.md` — upstream CPU/Metal/backend work.
- `RESEARCH.md` and `TECHNICAL.md` — upstream research notes and detailed measurements.
- `SERVE.md` — upstream OpenAI-compatible server and Kimi prompt-rendering design.
- `K3.md`, `KDA.md` — Kimi-specific architecture; reference only, never a DeepSeek implementation contract.

When adapting a generic lesson from these documents, cite the upstream result and then re-measure it on DeepSeek before treating it as a DeepSeek conclusion.

## Documentation update rule

Every implementation PR should update the smallest relevant local document in the same PR. In particular:

- gate definition/concordance changes -> `README.md` §18, `VALIDATION.md` §4a, `ROADMAP.md`, `BENCHMARKS.md`, and contributor/PR guidance as applicable;
- checkpoint inventory/access changes -> `INVENTORY-0731.md`, `REFERENCE_ACCESS.md`, and `TENSOR_MAP.md`;
- native quantization/convention changes -> `NUMERICS.md`, `VALIDATION.md`, and `TENSOR_MAP.md`;
- fixture/oracle methodology changes -> `FIXTURES.md` and `VALIDATION.md`;
- model-semantics changes -> `DEEPSEEK_V4.md` and `ARCHITECTURE.md` when applicable;
- container/converter changes -> `CONTAINER_V4.md` and `CONVERSION.md`;
- numerical tolerances/results -> `VALIDATION.md`;
- memory or disk observations -> `MEMORY_AND_IO.md`;
- performance measurements -> `BENCHMARKS.md`;
- rejected optimizations or surprising findings -> append to `EXPERIMENTS.md`;
- project phase completion -> `ROADMAP.md`.

Documentation is part of the gate, not cleanup after the implementation.