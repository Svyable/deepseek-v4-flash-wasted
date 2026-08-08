# Upstream provenance

This repository is bootstrapped from WASTE. Everything outside `README.md`, this file, and the paths listed under "Local additions" below arrived verbatim from the import described here.

```text
WASTE upstream: sqliteai/waste
Imported commit: d9b919a791148b571e643d0af666bf19b4d733ab
Imported date:   2026-08-08
Initial target:  deepseek-ai/DeepSeek-V4-Flash-0731 @ 9e165c3
```

The import was performed with `git archive` at the pinned commit, excluding only `README.md` so that this repository's DeepSeek handoff document survives:

```bash
git -C <waste-clone> archive d9b919a791148b571e643d0af666bf19b4d733ab \
  | tar -x --exclude=README.md -C .
```

At the time of import `d9b919a` was also the tip of `sqliteai/waste@main`.

## Baseline verification

The upstream model-free suite was run **before** any DeepSeek modification, per `README.md` §4. On the bootstrap machine (Ubuntu 24.04, gcc 13.3.0, x86-64, 4 cores):

```text
make -j4     -> exit 0
make check   -> 31 passed, 0 failed, 12 skipped
```

All 12 skips require assets this environment does not have: a real `.waste` container, source model weights, a tokenizer, or a K3 release directory. None of them indicate a broken bootstrap.

After PR #1 added the model-free DeepSeek inventory test:

```text
make check   -> 32 passed, 0 failed, 12 skipped
```

After PR #3 added the native quantization decoders/tests:

```text
make check   -> 34 passed, 0 failed, 12 skipped
make asan    -> 33 passed, 0 failed
```

PR #3 also reported a clean rebuild with no `-Wall -Wextra` warnings and mutation-tested ten one-line decoder faults; the final suite catches all ten.

Re-run the appropriate baseline before blaming the DeepSeek port for a failure in imported code.

## Canonical gate terminology

This provenance file uses the same gate vocabulary as the maintained project docs:

- README §18 defines 14 stable design gates **A–N**;
- `docs/VALIDATION.md` defines operational `V0–V11` levels;
- `ROADMAP.md` phases are schedule only.

The PR #1 inventory work supports **Gate A / V0**. The PR #3 native quantization work partially satisfies **Gate B / V1**. Do not call either “Gate 0” or “Gate V0” as a standalone project gate; use the paired canonical identifier where one exists.

## CI is parked, not deleted

`.github/workflows/ci.yml` was moved to `.github/workflows-disabled/ci.yml`. The file is unmodified — GitHub simply only reads `.github/workflows/`. Its first run here failed all eight jobs in seconds with no log output, an account-level Actions condition rather than a code failure. See `.github/workflows-disabled/README.md` for the evidence and the one-line restore.

While it is parked, `make check` is the manual validation entrypoint and nothing runs it automatically. PR #3 made the live suite's SPDX check recursive so local subdirectories such as `src/quant/` are covered. The parked imported workflow still contains the old non-recursive glob and must be widened before restoration; this is documented beside the parked workflow.

## Licensing

WASTE is Apache-2.0. The upstream `LICENSE` and `NOTICE` are preserved at the repository root exactly as imported, and are also copied into `LICENSES/` for clarity now that this tree will hold code under more than one license. See `LICENSES/README.md`.

PR #1 did not vendor or fabricate DeepSeek's license text while the official repository was unreachable. `LICENSES/DEEPSEEK-MIT.txt.MISSING` records that blocker. Resolve the exact official file before copying/adapting DeepSeek source into this tree.

`docs/REFERENCE_ACCESS.md` and `reference/README.md` define how official material should be staged and provenance-pinned once access is available.

## Imported files that still describe Kimi, not DeepSeek

The import is deliberately verbatim, so several files still describe the upstream Kimi/K3 target and must not be read as guidance for this port:

- `CLAUDE.md` — upstream agent/build instructions, Kimi-specific throughout.
- `docs/K3.md`, `docs/KDA.md` — Kimi architecture notes.
- `src/kda.*`, `src/vision.*`, `src/image.c` — Kimi Delta Attention and the Kimi vision path.
- `tools/kimi_*.py`, `tools/k3parts_ref.py`, `tools/kda_ref.py` — Kimi oracles.
- `CHANGELOG.md` — upstream release history.

Other imported docs such as `docs/ENGINE.md`, `docs/FORMAT.md`, `docs/GATES.md`, `docs/LEARNED.md`, `docs/EFFICIENCY.md`, `docs/BACKENDS.md` and `docs/SERVE.md` remain valuable upstream evidence, but their measurements, format details and Kimi prompt/model semantics are not automatically DeepSeek facts. `docs/README.md` is the project-specific navigation layer.

`README.md` §2 lists what to reuse and what to rewrite. Nothing in the Kimi list above is a generic primitive merely because it exists in the imported source tree.

## Local additions

Files/paths in this tree that are **not** verbatim upstream WASTE include:

### Project handoff, collaboration, and provenance

- `README.md` — DeepSeek V4 Flash port plan, stable Gates A–N, and handoff.
- `links.md` — curated source pack and source-of-truth hierarchy.
- `UPSTREAM.md` — this file.
- `AGENTS.md` — agent/contributor operating rules for this port.
- `ROADMAP.md` — implementation schedule mapped to canonical gates.
- `CONTRIBUTING.md` — contribution scope, tests, evidence and documentation rules.
- `.github/pull_request_template.md` — project-specific A–N/V-level/evidence PR checklist.
- `LICENSES/` — collected license/attribution material and the unresolved DeepSeek-license marker.
- `.github/workflows-disabled/README.md` — local explanation for parking the otherwise-unmodified imported CI workflow.
- `reference/README.md`, `reference/.gitignore` — local official-artifact staging/provenance policy and raw-model ignore rules.

### README Gate A / V0 inventory implementation from PR #1

- `tools/inventory.py` — DeepSeek checkpoint inventory.
- `tools/make_inventory_fixture.py` — synthetic checkpoint generator for tests.
- `tests/test_inventory.py` — tests for the inventory tool.
- `docs/INVENTORY-0731.md` — inventory status/results (checkpoint access pending).

### README Gate B / V1 native quantization work from PR #3

- `src/quant/fp4_e2m1.{c,h}` — E2M1 + UE8M0 K32 scalar decode and matvec.
- `src/quant/fp8_e4m3.{c,h}` — E4M3 finite variant + 128×128 block scales.
- `tests/test_quant.c` — exhaustive public-format conformance, scale/indexing, matvec consistency, literal nibble-order pin.

PR #3 satisfies only the public-format/model-free half of Gate B/V1. Official DeepSeek convention agreement remains pending.

### Imported files modified locally

Apache-2.0 §4(b) requires a modified file to say so, and each of these carries that notice in its own header/comment:

- `Makefile` — `src/quant/` added to `SRC`; `test_quant` target added.
- `tests/run.sh` — quantization decode and checkpoint inventory sections, plus a recursive SPDX check.
- `.github/workflows/ci.yml` — **not modified**, only moved to `.github/workflows-disabled/`; byte-identical to the import.

### DeepSeek-specific documentation

- `docs/README.md` — documentation/evidence map and canonical gate-vocabulary guide.
- `docs/REFERENCE_ACCESS.md` — minimal official-reference acquisition/unblock runbook for Gates A–C / V0–V2.
- `docs/DEEPSEEK_V4.md` — compact architecture/porting reference mapped to canonical gates.
- `docs/ARCHITECTURE.md` — WASTE reuse boundary and DeepSeek system design.
- `docs/TENSOR_MAP.md` — Gate A/V0 checkpoint tensor-family mapping contract and Gate B/V1 convention handoff.
- `docs/NUMERICS.md` — Gates B/C native quantization scalar arithmetic/convention contract.
- `docs/FIXTURES.md` — independent fixture/oracle and mutation-testing policy using F1–F4 fixture classes, not gate letters.
- `docs/VALIDATION.md` — operational V-ladder and §4a full A–N concordance.
- `docs/CONTAINER_V4.md` — proposed DeepSeek-specific container contract.
- `docs/CONVERSION.md` — official-checkpoint conversion runbook.
- `docs/MEMORY_AND_IO.md` — RAM/storage/I/O accounting and Gates G/L/M methodology.
- `docs/BENCHMARKS.md` — reproducible benchmark ledger with separate README-gate and V-level fields.
- `docs/EXPERIMENTS.md` — append-only DeepSeek experiment/negative-result log.
- `docs/DSPARK.md` — Gate N speculative-decoding boundary.
- `docs/API.md` — Gate J/V10 + V11 library/CLI/server integration contract.
- `docs/PLATFORM.md` — cross-platform runtime/storage requirements.
- `docs/TROUBLESHOOTING.md` — diagnostics ordered by canonical gate.

Keep this list current. It is what tells a reader which code/document carries upstream history and which is maintained by this port.