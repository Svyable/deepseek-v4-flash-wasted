## What changed?

<!-- Keep the scope narrow enough that a failing gate can localize the problem. -->

## Roadmap / gate

<!-- Cite the V-level from docs/VALIDATION.md §4a. That table maps V-levels to
     README §18's Gates A–N and to ROADMAP phases, so one answer covers all
     three. Gates G/L/M/N are systems/performance gates with no V-level —
     name them directly if that is what this PR moves. -->

- Roadmap phase:
- Highest gate passing before this PR:
- Gate this PR adds/passes:

## Source of truth

<!-- Pinned official checkpoint/reference path, repository artifact, or upstream WASTE evidence. -->

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

## Memory / storage / API impact

- RAM planner changed? If yes, which allocation is newly accounted for?
- Container/format changed? If yes, compatibility/version impact?
- Expert cache/I/O behavior changed?
- Public C API / CLI / server behavior changed?
- DSpark behavior changed?

## Documentation

Updated as applicable:

- [ ] `docs/INVENTORY-0731.md` / `docs/TENSOR_MAP.md`
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

## Deferred / blocked

<!-- Explicitly list what this PR does NOT prove. Synthetic fixtures are not real-checkpoint evidence. -->
