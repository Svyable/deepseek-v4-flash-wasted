# Documentation map

This directory contains two different kinds of material and they must not be confused:

1. **DeepSeek V4 Flash port documents** written in this repository.
2. **Imported WASTE documents** copied verbatim from `sqliteai/waste @ d9b919a791148b571e643d0af666bf19b4d733ab` during PR #1.

Read [`../UPSTREAM.md`](../UPSTREAM.md) before assuming that a document in this directory describes DeepSeek. The bootstrap intentionally preserved upstream Kimi/K3 documentation so the systems design, measurements, and negative results remain available as evidence.

## Current evidence state

PR #3 (`91c36b8f4168349e6893a9911a3f60075d62d973`) reported:

```text
make check -> 34 passed, 0 failed, 12 skipped
make asan  -> 33 passed, 0 failed
```

PRs #5–#8 have since merged Gates A/V0, B/V1, C/V2, D/V3 and F/V4 against
real 0731 checkpoint fixtures. Current:

```text
make check -> 45 passed, 0 failed, 12 skipped
```

Those eleven gate replays were running under PR #3's count too — through an
import chain that reported all of them as one "inventory" line. They are now
invoked by name; `EXPERIMENTS.md` entry 6 has the mutation evidence.

PR #4 merged the canonical gate/documentation contract as `3ce11ef4d391208d0c455796bf21803620910948`.

Draft PR #5 advances the next implementation seams but is **not yet a merged/tested executable baseline**:

- exact official release baseline pinned to `deepseek-ai/DeepSeek-V4-Flash-0731 @ 9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- exact publisher MIT license vendored; the old missing-license marker is removed;
- `tools/fetch_hf_headers.py` implements immutable, HTTP-Range-only safetensors header acquisition without tensor payloads;
- `tools/inventory.py` recognizes pinned-source bootstrap routing spellings `tid2eid`/`tie2eid` while keeping unknown main tensors fatal;
- README **Gate B / V1** high-risk FP4/FP8 packing/scale semantics are now **OFFICIAL-SPEC / SOURCE-VERIFIED** for the pinned release rather than blocked assumptions;
- an independent F3 fixture pins the official FP4 nibble/scale convention and a C replay test is wired into the existing inventory test path;
- a deliberately slow source-derived Gate C scalar linear reference now covers official activation E4M3 quantization, power-of-two K128 scaling, scaled FP8 block accumulation and BF16 output rounding;
- Gate C/V2 is **not passed** because no real released quantized projection fixture has been compared yet;
- Gate A/V0 is **not passed** because the real 48 shard headers have not yet been consumed by this repository;
- no DeepSeek transformer forward path, final logits, generation, real RAM floor, routing trace, storage benchmark, cache curve or model throughput result exists yet;
- CI remains parked under `.github/workflows-disabled/`; a fresh full PR #5 checkout run is still required before that draft is eligible to merge.

Narrow development probes during PR #5 confirmed the source-derived literal arithmetic (`0x21` + E8M0 `0x80` -> `[1,2]`) and the closed-form two-K-block linear preflight (`320`), but those are not a replacement for the integrated repository test run or real Gate A/C artifacts.

See [`OFFICIAL-0731-SOURCE.md`](OFFICIAL-0731-SOURCE.md), [`NUMERICS.md`](NUMERICS.md), [`VALIDATION.md`](VALIDATION.md), and [`REFERENCE_ACCESS.md`](REFERENCE_ACCESS.md) for the exact evidence boundaries.

## Gate vocabulary — one map, three purposes

The project has three related labels and they are not interchangeable:

1. **README §18 gates A–N** — 14 stable design gates identifying *why* a gate exists.
2. **`VALIDATION.md` V0–V11** — operational numerical/semantic validation levels.
3. **`ROADMAP.md` phases 0–9** — schedule only.

The canonical mapping is [`VALIDATION.md` §4a](VALIDATION.md), which maps all 14 README gates A–N to their V-level or systems/performance owner while preserving the concept and roadmap phase.

Intentional asymmetries:

- README **G/L/M/N** are real systems/performance gates with no invented V-number;
- **V7** and **V11** are operational levels with no README letter;
- README **E** maps to `V5` while **F** maps to `V4`;
- README **J** maps to `V10` while **K** maps to `V9`.

In implementation PRs, cite both identifiers when both exist, for example `Gate B / V1`, `Gate H / V6`, or `Gate K / V9`. Cite `Gate G`, `Gate L`, `Gate M`, or `Gate N` directly for systems/performance gates.

Do not introduce new local labels such as “Gate 0/1/2.” Historical shorthand is read through the canonical map.

## Evidence labels used by local docs

| State | Meaning |
|---|---|
| **UPSTREAM-MEASURED** | Measured in WASTE on Kimi targets; useful systems evidence, not a DeepSeek measurement. |
| **OFFICIAL-SPEC** | Taken from pinned official specification/release/config/reference source, but not necessarily exercised against real checkpoint payloads here. |
| **SYNTHETIC-VERIFIED** | Exercised by repository-generated/model-free tests. |
| **CHECKPOINT-VERIFIED** | Derived from pinned official 0731 checkpoint headers/tensors/reference outputs. |
| **END-TO-END-VERIFIED** | Observed in this port running the real model. |
| **DESIGN** | Proposed behavior/format not yet implemented or validated. |
| **BLOCKED** | Cannot currently be established because a required artifact/environment is unavailable. |

A later, stronger state supersedes an earlier one. Never silently promote source-level evidence into checkpoint or end-to-end evidence.

## DeepSeek-specific documents

These documents are authoritative for this port unless the official checkpoint/reference proves them wrong:

- [`INVENTORY-0731.md`](INVENTORY-0731.md) — exact checkpoint inventory status and **Gate A / V0** procedure.
- [`REFERENCE_ACCESS.md`](REFERENCE_ACCESS.md) — executable artifact-acquisition sequence for Gates A–C; Tier 0 and the header Range tooling are now resolved/implemented.
- [`OFFICIAL-0731-SOURCE.md`](OFFICIAL-0731-SOURCE.md) — pinned release-source findings: quantization, routing, expert activation, release provenance and evidence boundaries.
- [`DEEPSEEK_V4.md`](DEEPSEEK_V4.md) — compact architecture/porting reference and assumptions Gate A/V0 must verify.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system decomposition and WASTE reuse boundary.
- [`TENSOR_MAP.md`](TENSOR_MAP.md) — tensor-family mapping contract; exact exported names/shapes remain Gate A work.
- [`NUMERICS.md`](NUMERICS.md) — scalar E2M1/UE8M0/E4M3FN contract, official 0731 packing/scale semantics, activation quantization and Gate B→C handoff.
- [`FIXTURES.md`](FIXTURES.md) — independent-fixture and mutation-testing policy.
- [`VALIDATION.md`](VALIDATION.md) — canonical operational ladder and A–N/V-level concordance.
- [`CONTAINER_V4.md`](CONTAINER_V4.md) — proposed DeepSeek-specific WASTE container contract.
- [`CONVERSION.md`](CONVERSION.md) — checkpoint conversion runbook.
- [`MEMORY_AND_IO.md`](MEMORY_AND_IO.md) — RAM/storage/I/O accounting methodology; owns Gates G/L/M alongside validation/benchmarks.
- [`BENCHMARKS.md`](BENCHMARKS.md) — reproducible benchmark ledger; records README gate letters and V-levels separately.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — append-only experiment/negative-result log; historical gate wording is preserved with translation notes.
- [`DSPARK.md`](DSPARK.md) — Gate N speculative-decoding boundary.
- [`API.md`](API.md) — library/CLI/OpenAI-compatible server contract; Gate J/V10 + V11.
- [`PLATFORM.md`](PLATFORM.md) — macOS/Linux/Windows storage/backend requirements.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — diagnostic playbook ordered by canonical gates.

Root/local companions:

- [`../README.md`](../README.md) — implementation handoff and stable 14-gate A–N rationale.
- [`../links.md`](../links.md) — curated source pack.
- [`../AGENTS.md`](../AGENTS.md) — coding-agent/contributor operating rules.
- [`../ROADMAP.md`](../ROADMAP.md) — phase schedule/current status.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — contribution/testing/evidence requirements.
- [`../UPSTREAM.md`](../UPSTREAM.md) — import provenance and local ownership.
- [`../reference/README.md`](../reference/README.md) — local official-artifact staging policy.

## Imported WASTE documents

These retain their original historical meaning until intentionally forked:

- `ENGINE.md` — WASTE library/CLI/memory design.
- `FORMAT.md` — WASTE v0 Kimi container format, **not** DeepSeek format.
- `GATES.md` — upstream feasibility-gate history, not this port's concordance.
- `EFFICIENCY.md` — upstream Kimi performance measurements.
- `LEARNED.md` — upstream negative-result history.
- `BACKENDS.md` — upstream CPU/Metal/backend work.
- `RESEARCH.md`, `TECHNICAL.md` — upstream research/measurements.
- `SERVE.md` — upstream server + Kimi prompt design.
- `K3.md`, `KDA.md` — Kimi-specific architecture reference only.

When adapting a generic lesson, cite the upstream result and re-measure it on DeepSeek before treating it as a DeepSeek conclusion.

## Documentation update rule

Every implementation PR updates the smallest relevant maintained document in the same PR. In particular:

- gate/concordance changes -> `README.md` §18, `VALIDATION.md` §4a, `ROADMAP.md`, benchmark/contributor guidance as applicable;
- official source/provenance changes -> `OFFICIAL-0731-SOURCE.md`, `REFERENCE_ACCESS.md`, `LICENSES/`;
- checkpoint inventory/access -> `INVENTORY-0731.md`, `REFERENCE_ACCESS.md`, `TENSOR_MAP.md`;
- quantization/activation/projection semantics -> `NUMERICS.md`, `VALIDATION.md`;
- fixture/oracle methodology -> `FIXTURES.md`, `VALIDATION.md`;
- model semantics -> `DEEPSEEK_V4.md`, `ARCHITECTURE.md`;
- container/converter -> `CONTAINER_V4.md`, `CONVERSION.md`;
- memory/disk -> `MEMORY_AND_IO.md`;
- performance -> `BENCHMARKS.md`;
- failed/surprising findings -> append `EXPERIMENTS.md`;
- phase completion -> `ROADMAP.md`.

Documentation is part of the gate, not cleanup after implementation.
