# Contributing to DeepSeek V4 Flash WASTED

This project is an architecture port, not a conventional feature backlog. Small errors in tensor mapping, quantization, state updates, or storage accounting can produce plausible but wrong text. Contributions therefore move through explicit gates.

Read first:

1. `README.md`
2. `AGENTS.md`
3. `UPSTREAM.md`
4. `docs/README.md`
5. `ROADMAP.md`
6. the task-specific document under `docs/`

## Branch and PR scope

Prefer one testable boundary per PR:

```text
inventory mapping
oracle fixture support
one quantization primitive
one model primitive
one attention mode
router/MoE seam
container/converter change
API/encoder integration
one measured optimization
```

Avoid combining a new arithmetic primitive, a new container format, a SIMD rewrite and a server feature in one change. If final logits fail, the review needs to know which boundary moved.

## Before implementation

State in the PR/issue:

- which `ROADMAP.md` phase this belongs to;
- source of truth used (checkpoint/reference/upstream WASTE);
- highest existing validation gate;
- new gate the change intends to pass;
- whether official 0731 assets are required;
- whether the work copies/adapts external source and its license/provenance.

If a cheaper gate can invalidate an expensive operation, run the cheap gate first.

## Correctness requirements

Follow `docs/VALIDATION.md`.

For arithmetic changes:

- add a deterministic scalar/oracle fixture before or with the implementation;
- compare the smallest primitive before a full layer;
- keep selected expert IDs/token IDs exact where semantics require it;
- do not loosen numerical tolerances without diagnosing the mismatch;
- optimized paths must pass the same fixture suite as the scalar baseline.

For storage/cache changes:

- cache-on and cache-off must preserve model output;
- prefetch can change timing only;
- parser/record bounds must remain hardened;
- RAM allocations must be included in the planner.

## Checkpoint claims

Use the evidence states in `docs/README.md`.

Do not describe a value as measured from DeepSeek when it came from:

- the README design estimate;
- `tools/make_inventory_fixture.py`;
- imported Kimi documentation;
- a GGUF/third-party runtime;
- config arithmetic without reading the corresponding checkpoint metadata.

Checkpoint-derived facts should be accompanied by the pinned model revision and the command/tool that produced them.

## Testing while CI is parked

PR #1 preserved the imported Actions workflow under `.github/workflows-disabled/` because the repository/account environment prevented jobs from starting.

At minimum for source changes:

```bash
make
make check
```

For parser, memory, container and low-level arithmetic work when practical:

```bash
make asan
make fuzz
```

SPDX check:

```bash
git ls-files 'src/*.c' 'src/*.h' 'src/*.m' 'cli/*.c' 'tests/*.c' \
             'tools/*.py' 'tools/*.sh' \
  | xargs grep -L 'SPDX-License-Identifier'
```

Empty output passes.

If a test requires real weights unavailable to normal CI, keep a model-free/synthetic version where possible and make the real-model gate explicit/manual. Do not convert a required test into a silent no-op.

## Documentation requirement

Update the relevant local document in the same PR:

| Change | Update |
|---|---|
| checkpoint inventory/name mapping | `docs/INVENTORY-0731.md`, `docs/TENSOR_MAP.md` |
| model semantics | `docs/DEEPSEEK_V4.md`, possibly `docs/ARCHITECTURE.md` |
| validation/tolerance | `docs/VALIDATION.md` |
| format/converter | `docs/CONTAINER_V4.md`, `docs/CONVERSION.md` |
| RAM/storage/I/O | `docs/MEMORY_AND_IO.md` |
| performance result | `docs/BENCHMARKS.md` |
| failed/surprising experiment | append `docs/EXPERIMENTS.md` |
| DSpark | `docs/DSPARK.md` |
| API/server | `docs/API.md` |
| platform behavior | `docs/PLATFORM.md` |
| phase completion | `ROADMAP.md` |

Imported WASTE documents should remain historical upstream evidence unless a PR intentionally forks one and updates `UPSTREAM.md` ownership.

## Performance changes

A speedup is mergeable only when:

- the relevant correctness gate still passes;
- memory use remains inside the planner;
- measurement includes commit/model/container/hardware/cache/I/O configuration;
- the result is entered in `docs/BENCHMARKS.md`;
- rejected or hardware-specific conclusions are recorded honestly in `docs/EXPERIMENTS.md`.

Microbenchmark improvements are not automatically end-to-end improvements.

## Licensing/provenance

WASTE content is Apache-2.0 and its root `LICENSE`/`NOTICE` are preserved.

Before copying/adapting official DeepSeek code, resolve the exact official license/notice currently marked missing under `LICENSES/`. Do not generate legal/source text from memory.

For adapted external source, preserve required headers and identify the pinned source path/revision.

## PR description checklist

Every implementation PR should answer:

```text
What changed?
Why this boundary now?
Roadmap phase / validation gate?
Source of truth?
Tests/commands run?
Evidence state of new claims?
Memory/container/API compatibility impact?
Documentation updated?
Licensing/provenance impact?
What remains blocked or deliberately deferred?
```

A narrow PR with a strong oracle and explicit negative result is more valuable than a large port that merely generates plausible text.