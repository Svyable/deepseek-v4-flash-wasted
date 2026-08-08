# Disabled workflows

GitHub only runs workflows under `.github/workflows/`. Files here are
parked: present, unmodified, and inert.

## ci.yml

Upstream WASTE's CI, imported verbatim with the bootstrap (`e6f5b01`). It is
**unmodified** — this directory is a rename, not an edit, so restoring it is
a `git mv` and nothing else.

### Why it is parked

Its first and only run in this repository ([run
31271243157](https://github.com/Svyable/deepseek-v4-flash-wasted/actions/runs/31271243157))
failed all eight jobs 2–5 seconds after starting, with no log output on any
of them (HTTP 404 on every log download, empty `output` on every check run).
Jobs that fail that way never executed a step. Plain `ubuntu-latest` failed
identically to `macos-latest`, `ubuntu-24.04-arm` and `windows-latest`,
which rules out runner availability — that would be selective.

The cause is an account-level Actions condition on a private repository,
most likely exhausted minutes or a spending limit (this matrix asks for
macOS, which bills at 10×, and Windows at 2×), or an Actions policy
blocking `actions/checkout@v4` before the first step. Neither is fixable
from a commit.

Parking it was a deliberate call: the alternative was trimming the matrix to
dodge the block, which would have quietly deleted coverage upstream added on
purpose. `ci.yml`'s own header explains that the cross-platform matrix
exists because Windows and AVX-512 paths were invisible from the
development machine and broke unnoticed. That reasoning does not stop being
true because the runner is unavailable this week.

### Restoring it

```bash
git mv .github/workflows-disabled/ci.yml .github/workflows/ci.yml
```

Do this once Actions can run — check **Settings → Actions** and the
account's billing page first, otherwise the next PR goes red again for the
same reason.

**Widen the SPDX glob when you restore it.** The workflow's header check
globs `src/*.c` and `src/*.h`, which does not recurse — it would not see
`src/quant/` at all, and would report "ok: all source files carry the
header" while checking none of them. The port now keeps sources in
subdirectories, so that step needs `src/**/*.c` and `src/**/*.h` added.

Until then the recursive version runs in `tests/run.sh` under "quantization
decode", so the check is live; it is only the parked copy that is stale.
That is the safer failure of the two, but it is still a check that would
pass by looking in the wrong place — fix the glob rather than assume the
suite covers it forever.

### Meanwhile

The suite still runs, and is still the gate; it just runs by hand:

```bash
make            # libwaste.a, waste CLI, libwaste.$(SOEXT), libwastevq
make check      # tests/run.sh — the whole C-side suite
make asan       # ASan/UBSan rebuild + suite
make fuzz       # container parser fuzzer
```

Plus the SPDX check CI enforces, which is cheap to run directly:

```bash
git ls-files 'src/*.c' 'src/*.h' 'src/*.m' 'cli/*.c' 'tests/*.c' \
             'tools/*.py' 'tools/*.sh' \
  | xargs grep -L 'SPDX-License-Identifier'
```

Empty output is a pass. Run all of these before pushing while CI is parked —
nothing else is watching.
