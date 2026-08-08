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

## Imported files that still describe Kimi, not DeepSeek

The import is deliberately verbatim, so several files still describe the
upstream Kimi/K3 target and must not be read as guidance for this port:

- `CLAUDE.md` — upstream agent/build instructions, Kimi-specific throughout.
- `docs/K3.md`, `docs/KDA.md` — Kimi architecture notes.
- `src/kda.*`, `src/vision.*`, `src/image.c` — Kimi Delta Attention and the
  Kimi vision path.
- `tools/kimi_*.py`, `tools/k3parts_ref.py`, `tools/kda_ref.py` — Kimi oracles.
- `CHANGELOG.md` — upstream release history.

`README.md` §2 lists what to reuse and what to rewrite. Nothing in the list
above is a generic primitive.

## Local additions

Files in this tree that are **not** from upstream WASTE:

- `README.md` — DeepSeek V4 Flash port plan and handoff.
- `UPSTREAM.md` — this file.
- `LICENSES/` — collected license/attribution material.
- `tools/inventory.py` — DeepSeek checkpoint inventory (README §5).
- `tools/make_inventory_fixture.py` — synthetic checkpoint generator for tests.
- `tests/test_inventory.py` — tests for the inventory tool.
- `docs/INVENTORY-0731.md` — inventory results (pending checkpoint access).

Keep this list current. It is what tells a reader which code carries upstream's
copyright and which is ours.
