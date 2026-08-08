# Documentation map

This directory contains two different kinds of material and they must not be confused:

1. **DeepSeek V4 Flash port documents** written in this repository.
2. **Imported WASTE documents** copied verbatim from `sqliteai/waste @ d9b919a791148b571e643d0af666bf19b4d733ab` during PR #1.

Read [`../UPSTREAM.md`](../UPSTREAM.md) before assuming that a document in this directory describes DeepSeek. The bootstrap intentionally preserved upstream Kimi/K3 documentation so the systems design, measurements, and negative results remain available as evidence.

## Current evidence state

As of the merge of PR #1 (`edd2a41b66332e5a54ed54bcbb196fec19664079`):

- the WASTE source tree is imported and its model-free baseline built successfully;
- `make check` reports **32 passed, 0 failed, 12 skipped** after the new inventory test was added;
- `tools/inventory.py` exists and is tested against a synthetic safetensors fixture;
- the official `DeepSeek-V4-Flash-0731` checkpoint has **not** been read in the bootstrap environment because the proxy denied `huggingface.co:443` with HTTP CONNECT 403;
- therefore no checkpoint-derived byte totals, exact tensor-name map, RAM floor, disk footprint, routing trace, quality result, or throughput result exists yet;
- CI is parked under `.github/workflows-disabled/` because the imported Actions matrix could not start jobs in this private repository. Manual `make check` remains the gate.

The checkpoint-access limitation is recorded in [`INVENTORY-0731.md`](INVENTORY-0731.md). Do not convert the synthetic fixture's arithmetic into claims about the real release.

## Evidence labels used by local docs

Local documents should identify important claims with one of these states:

| State | Meaning |
|---|---|
| **UPSTREAM-MEASURED** | Measured in WASTE on its Kimi targets. Useful systems evidence, not a DeepSeek measurement. |
| **OFFICIAL-SPEC** | Taken from the pinned official DeepSeek release/config/reference material, but not independently measured here. |
| **SYNTHETIC-VERIFIED** | Exercised by a repository-generated fixture or model-free test. |
| **CHECKPOINT-VERIFIED** | Derived from the pinned official 0731 checkpoint headers or tensors. |
| **END-TO-END-VERIFIED** | Observed in this port running the real model. |
| **DESIGN** | Proposed behavior or format that is not implemented yet. |
| **BLOCKED** | Cannot currently be established because a required external artifact/environment is unavailable. |

A later, stronger state supersedes an earlier one. Never silently promote a claim.

## DeepSeek-specific documents

These documents are authoritative for this port unless the official checkpoint/reference code proves them wrong:

- [`INVENTORY-0731.md`](INVENTORY-0731.md) — exact checkpoint inventory status and Gate 0 procedure.
- [`DEEPSEEK_V4.md`](DEEPSEEK_V4.md) — compact DeepSeek-specific architecture/porting reference and the assumptions Gate 0 must verify.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system decomposition and what is reused versus rewritten.
- [`TENSOR_MAP.md`](TENSOR_MAP.md) — tensor-family mapping contract and checklist populated by `tools/inventory.py`.
- [`VALIDATION.md`](VALIDATION.md) — numerical correctness ladder and merge gates. Its §4a is the **gate concordance**: one table mapping this project's `V`-levels to `README.md` §18's Gates A–N and to `ROADMAP.md` phases. Start there whenever a document, a PR template or an issue names a gate, so the same rung is not tracked under three names.
- [`CONTAINER_V4.md`](CONTAINER_V4.md) — proposed DeepSeek-specific WASTE container contract.
- [`CONVERSION.md`](CONVERSION.md) — download, inventory, conversion, verification, and recovery runbook.
- [`MEMORY_AND_IO.md`](MEMORY_AND_IO.md) — RAM-floor and storage/I/O accounting methodology.
- [`BENCHMARKS.md`](BENCHMARKS.md) — benchmark schema and results ledger; no marketing numbers without provenance.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — append-only experiment and negative-result log for this port.
- [`DSPARK.md`](DSPARK.md) — phase-2 speculative-decoding boundary and validation plan.
- [`API.md`](API.md) — library/CLI/OpenAI-compatible server integration contract.
- [`PLATFORM.md`](PLATFORM.md) — macOS/Linux/Windows I/O and backend requirements.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — failure diagnosis ordered by the project's gates.

Root-level companion documents:

- [`../README.md`](../README.md) — implementation handoff and design plan.
- [`../links.md`](../links.md) — curated source pack and source-of-truth hierarchy.
- [`../AGENTS.md`](../AGENTS.md) — instructions for coding agents and contributors.
- [`../ROADMAP.md`](../ROADMAP.md) — PR/gate sequence and current project status.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — contribution scope, testing, documentation and evidence requirements.
- [`../UPSTREAM.md`](../UPSTREAM.md) — import provenance and local-vs-upstream ownership.

## Imported WASTE documents

The following arrived from upstream and should retain their original historical meaning until intentionally forked:

- `ENGINE.md` — WASTE library/CLI/memory design, mostly reusable systems reasoning.
- `FORMAT.md` — WASTE v0 Kimi container format. **Not** the DeepSeek format.
- `GATES.md` — upstream feasibility-gate history.
- `EFFICIENCY.md` — upstream Kimi performance measurements.
- `LEARNED.md` — upstream append-only negative-result/measurement log.
- `BACKENDS.md` — upstream CPU/Metal/backend work.
- `RESEARCH.md` and `TECHNICAL.md` — upstream research notes and detailed measurements.
- `SERVE.md` — upstream OpenAI-compatible server and Kimi prompt-rendering design.
- `K3.md`, `KDA.md` — Kimi-specific architecture; reference only, never a DeepSeek implementation contract.

When adapting a generic lesson from these documents, cite the upstream result and then re-measure it on DeepSeek before treating it as a DeepSeek conclusion.

## Documentation update rule

Every implementation PR should update the smallest relevant local document in the same PR. In particular:

- checkpoint inventory changes -> `INVENTORY-0731.md` and `TENSOR_MAP.md`;
- model-semantics changes -> `DEEPSEEK_V4.md` and `ARCHITECTURE.md` when applicable;
- container/converter changes -> `CONTAINER_V4.md` and `CONVERSION.md`;
- numerical tolerances/results -> `VALIDATION.md`;
- memory or disk observations -> `MEMORY_AND_IO.md`;
- performance measurements -> `BENCHMARKS.md`;
- rejected optimizations or surprising findings -> append to `EXPERIMENTS.md`;
- project phase completion -> `ROADMAP.md`.

Documentation is part of the gate, not cleanup after the implementation.