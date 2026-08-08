## What changed?

<!-- Keep the scope narrow enough that a failing gate can localize the problem. -->

## Roadmap / gate

<!-- Cite the V-level from docs/VALIDATION.md. Use ROADMAP phase only for schedule. -->

- Roadmap phase:
- Highest gate passing before this PR:
- Gate this PR adds/passes:

## Source of truth

<!-- Pinned official checkpoint/reference path, public numeric spec, repository artifact, or upstream WASTE evidence. -->

- Source:
- Revision:
- Evidence state: `UPSTREAM-MEASURED` / `OFFICIAL-SPEC` / `SYNTHETIC-VERIFIED` / `CHECKPOINT-VERIFIED` / `END-TO-END-VERIFIED` / `DESIGN` / `BLOCKED`

## Validation

Commands/tests run:

```text

```

Numerical/semantic result, when applicable:

```text
exact IDs/tokens:
max_abs:
max_rel:
RMS:
argmax/greedy parity:
```

### Fixture independence / mutations

<!-- Required for arithmetic, packing/layout, routing, parser, tokenizer or other silent semantic seams. A round trip through two helpers sharing the same convention is not sufficient evidence; see docs/FIXTURES.md. -->

- What produces the expected answer?
- Why is it independent of the implementation/convention being tested?
- Literal/raw convention fixture included where applicable?
- Known-wrong mutation(s) tested? Which ones survived/failed?
- Official fixture provenance recorded, if applicable?

## Memory / storage / API impact

- RAM planner changed? If yes, which allocation is newly accounted for?
- Container/format changed? If yes, compatibility/version impact?
- Expert cache/I/O behavior changed?
- Public C API / CLI / server behavior changed?
- DSpark behavior changed?

## Documentation

Updated as applicable:

- [ ] `docs/INVENTORY-0731.md` / `docs/TENSOR_MAP.md`
- [ ] `docs/REFERENCE_ACCESS.md` / `reference/README.md`
- [ ] `docs/NUMERICS.md`
- [ ] `docs/FIXTURES.md`
- [ ] `docs/DEEPSEEK_V4.md` / `docs/ARCHITECTURE.md`
- [ ] `docs/VALIDATION.md`
- [ ] `docs/CONTAINER_V4.md` / `docs/CONVERSION.md`
- [ ] `docs/MEMORY_AND_IO.md`
- [ ] `docs/BENCHMARKS.md`
- [ ] `docs/EXPERIMENTS.md`
- [ ] `docs/DSPARK.md`
- [ ] `docs/API.md`
- [ ] `docs/PLATFORM.md`
- [ ] `ROADMAP.md`
- [ ] none needed — explain why below

## Licensing / provenance

- External source copied/adapted?
- License/notice verified from the pinned source?
- Required SPDX/attribution preserved?
- Official reference artifacts pinned/hashes recorded?

## Deferred / blocked

<!-- Explicitly list what this PR does NOT prove. Public-spec/synthetic fixtures are not checkpoint/reference evidence unless they actually consume the pinned official artifact. -->
