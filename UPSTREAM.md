# Upstream provenance

This repository is bootstrapped from WASTE. Everything outside `README.md`,
this file, and the paths listed under "Local additions" below arrived verbatim
from the import described here.

```text
WASTE upstream: sqliteai/waste
Imported commit: d9b919a791148b571e643d0af666bf19b4d733ab
Imported date:   2026-08-08
Initial target:  deepseek-ai/DeepSeek-V4-Flash-0731 @ 9e165c3
```

The import was performed with `git archive` at the pinned commit, excluding
only `README.md` so that this repository's DeepSeek handoff document survives:

```bash
git -C <waste-clone> archive d9b919a791148b571e643d0af666bf19b4d733ab \
  | tar -x --exclude=README.md -C .
```

At the time of import `d9b919a` was also the tip of `sqliteai/waste@main`.

## Baseline verification

The upstream model-free suite was run **before** any DeepSeek modification, per
`README.md` §4. On the bootstrap machine (Ubuntu 24.04, gcc 13.3.0, x86-64,
4 cores):

```text
make -j4     -> exit 0
make check   -> 31 passed, 0 failed, 12 skipped
```

All 12 skips require assets this environment does not have: a real `.waste`
container, source model weights, a tokenizer, or a K3 release directory. None
of them indicate a broken bootstrap.

After PR #1 added the model-free DeepSeek inventory test, the full suite was:

```text
make check   -> 32 passed, 0 failed, 12 skipped
```

Re-run that baseline before blaming the DeepSeek port for a failure in
imported code.

## CI is parked, not deleted

`.github/workflows/ci.yml` was moved to `.github/workflows-disabled/ci.yml`.
The file is unmodified — GitHub simply only reads `.github/workflows/`. Its
first run here failed all eight jobs in seconds with no log output, an
account-level Actions condition rather than a code failure. See
`.github/workflows-disabled/README.md` for the evidence and the one-line
restore.

While it is parked, `make check` is the gate and nothing runs it
automatically. Run it, and the SPDX check, before pushing.

## Licensing

WASTE is Apache-2.0. The upstream `LICENSE` and `NOTICE` are preserved at the
repository root exactly as imported, and are also copied into `LICENSES/` for
clarity now that this tree will hold code under more than one license. See
`LICENSES/README.md`.

PR #1 did not vendor or fabricate DeepSeek's license text while the official
repository was unreachable. `LICENSES/DEEPSEEK-MIT.txt.MISSING` records that
blocker. Resolve the exact official file before copying/adapting DeepSeek
source into this tree.

## Imported files that still describe Kimi, not DeepSeek

The import is deliberately verbatim, so several files still describe the
upstream Kimi/K3 target and must not be read as guidance for this port:

- `CLAUDE.md` — upstream agent/build instructions, Kimi-specific throughout.
- `docs/K3.md`, `docs/KDA.md` — Kimi architecture notes.
- `src/kda.*`, `src/vision.*`, `src/image.c` — Kimi Delta Attention and the
  Kimi vision path.
- `tools/kimi_*.py`, `tools/k3parts_ref.py`, `tools/kda_ref.py` — Kimi oracles.
- `CHANGELOG.md` — upstream release history.

Other imported docs such as `docs/ENGINE.md`, `docs/FORMAT.md`,
`docs/GATES.md`, `docs/LEARNED.md`, `docs/EFFICIENCY.md`, `docs/BACKENDS.md`
and `docs/SERVE.md` remain valuable upstream evidence, but their measurements,
format details and Kimi prompt/model semantics are not automatically DeepSeek
facts. `docs/README.md` is the project-specific navigation layer.

`README.md` §2 lists what to reuse and what to rewrite. Nothing in the Kimi
list above is a generic primitive merely because it exists in the imported
source tree.

## Local additions

Files/paths in this tree that are **not** verbatim upstream WASTE include:

### Project handoff, collaboration, and provenance

- `README.md` — DeepSeek V4 Flash port plan and handoff.
- `links.md` — curated source pack and source-of-truth hierarchy.
- `UPSTREAM.md` — this file.
- `AGENTS.md` — agent/contributor operating rules for this port.
- `ROADMAP.md` — gated implementation/PR sequence.
- `CONTRIBUTING.md` — contribution scope, tests, evidence and documentation rules.
- `.github/pull_request_template.md` — project-specific gate/evidence PR checklist.
- `LICENSES/` — collected license/attribution material and the unresolved
  DeepSeek-license marker.
- `.github/workflows-disabled/README.md` — local explanation for parking the
  otherwise-unmodified imported CI workflow.

### Gate 0 implementation from PR #1

- `tools/inventory.py` — DeepSeek checkpoint inventory (README §5).
- `tools/make_inventory_fixture.py` — synthetic checkpoint generator for tests.
- `tests/test_inventory.py` — tests for the inventory tool.
- `docs/INVENTORY-0731.md` — inventory status/results (checkpoint access pending).

### DeepSeek-specific documentation

- `docs/README.md` — documentation/evidence map separating local docs from the
  imported WASTE/Kimi history.
- `docs/DEEPSEEK_V4.md` — compact architecture/porting reference.
- `docs/ARCHITECTURE.md` — WASTE reuse boundary and DeepSeek system design.
- `docs/TENSOR_MAP.md` — checkpoint tensor-family mapping contract.
- `docs/VALIDATION.md` — numerical/oracle validation ladder.
- `docs/CONTAINER_V4.md` — proposed DeepSeek-specific container contract.
- `docs/CONVERSION.md` — official-checkpoint conversion runbook.
- `docs/MEMORY_AND_IO.md` — RAM/storage/I/O accounting methodology.
- `docs/BENCHMARKS.md` — reproducible benchmark ledger.
- `docs/EXPERIMENTS.md` — append-only DeepSeek experiment/negative-result log.
- `docs/DSPARK.md` — speculative-decoding phase boundary.
- `docs/API.md` — library/CLI/server integration contract.
- `docs/PLATFORM.md` — cross-platform runtime/storage requirements.
- `docs/TROUBLESHOOTING.md` — project-specific diagnostic playbook.

Keep this list current. It is what tells a reader which code/document carries
upstream history and which is maintained by this port.