# Benchmarks — DeepSeek V4 WASTE port

**Status: NO DEEPSEEK PERFORMANCE RESULTS YET.**

This file is the authoritative ledger for performance measurements produced by this port. It starts empty on purpose. Upstream WASTE measurements are useful design evidence but are not DeepSeek benchmark rows.

Gate terminology follows the canonical concordance in `docs/VALIDATION.md` §4a:

- README §18 defines the stable design gates **A–N**;
- numerical/semantic validation uses **V0–V11** where a V-level exists;
- README systems/performance gates **G, L, M, N** keep their letter and intentionally do not receive invented V-numbers.

A benchmark entry should record both labels when applicable. For example, a real generation benchmark may be `Gate K / V9`; a storage benchmark is `Gate L`; a cache sweep is `Gate M`; a DSpark result is `Gate N` plus the base-model V-levels it depends on.

## 1. Rules

A performance number can be added only if the entry records enough information to reproduce the storage, memory, model and hardware conditions.

Every entry must state:

- date;
- port commit;
- source model repository + resolved revision;
- container format/converter commit and conversion policy;
- base model versus DSpark enabled;
- hardware model, CPU, RAM;
- OS/version;
- storage device/interface/filesystem and model container location;
- direct-I/O/page-cache bypass status;
- context/prompt size;
- generation length;
- thread count/CPU placement;
- RAM budget;
- expert-cache bytes/policy;
- prefetch/lookahead settings;
- backend/kernel selection;
- README §18 gate letter(s) the measurement is intended to satisfy/support;
- highest operational `VALIDATION.md` V-level passed by the tested binary/container, when applicable;
- command/workload;
- raw measurement or a link/path to it.

A row missing model/container commit or RAM/cache configuration is not comparable enough to publish here.

## 2. Do not mix these metrics

Keep separate:

### Prefill

```text
prompt tokens
wall seconds
tokens/s
expert reads
bytes read
distinct experts/chunk
```

### Decode

```text
generated tokens
wall seconds
tokens/s or seconds/token
expert record/byte hit rate
bytes read/token
prefetch useful/wasted bytes
direct-I/O state
```

### Time to first token

Includes encoding + prefill + first decode. Report separately from steady decode.

### Disk microbenchmark — README Gate L

Storage throughput/latency only. Never label the disk upper bound as model tok/s.

### Cache sweep — README Gate M

A cache curve is a system/performance gate, not a numerical V-level. It still requires a sufficiently correct model path for the measured routing/workload to mean anything.

### Kernel microbenchmark

One operation with shapes/dtypes/threads. Never present isolated kernel speedup as end-to-end speedup.

## 3. Gate prerequisite and labels

Before a result is eligible for the primary table, record the stable README gate and the highest operational V-level passed.

Common pairs:

```text
Gate B / V1   native quantization decode
Gate C / V2   one quantized trunk linear
Gate H / V6   one complete transformer block
Gate I / V8   real final logits
Gate K / V9   real greedy generation
Gate J / V10  encoding/parser parity
V11           API parity; no README letter
Gate L         real target-storage feasibility; no V-level
Gate M         real cache curve; no V-level
Gate N         DSpark correctness/performance; no V-level of its own
```

`Gate G` (disk/cache identity) is a correctness prerequisite for any benchmark that changes placement, caching, direct-I/O mode, or prefetch. A benchmark is invalid as an optimization result if enabling the measured mechanism changes model meaning.

A benchmark from a path that has not passed Gate I/V8 and Gate K/V9 may be recorded under **development measurements**, but it must not be summarized as full-model performance.

## 4. Primary end-to-end table

_No results yet._

| Date | Port commit | Model rev | Hardware | RAM budget / cache | Context | Configuration | Prefill tok/s | Decode tok/s | Read GiB/token | Cache hit | README gate | Validation |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|

For normal base-model generation, expect `README gate = K` and `Validation >= V9`. Storage/cache studies should additionally state `L` and/or `M` in the configuration/notes or use their dedicated tables below.

## 5. Development measurements

Use this section for useful engineering data before end-to-end parity. Clearly identify the limited scope.

_No DeepSeek development performance measurements recorded yet._

| Date | Scope | Commit | Fixture/model | Hardware | Configuration | Result | README gate | Correctness state | Notes |
|---|---|---|---|---|---|---|---|---|---|

## 6. Storage benchmarks — README Gate L

Record the actual model/container volume, not a generic temporary filesystem if the purpose is to predict streaming performance.

A Gate L claim is not satisfied by marketing SSD specifications or a cached sequential read. Use the expert-record size/access pattern, relevant concurrency, and page-cache-bypass mode the runtime will actually use.

| Date | Device / interface | FS | bypass | record size | threads | workload size | seq read | random record read | model/container rev | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|

_No DeepSeek-target storage measurement recorded in this repository yet._

## 7. Cache sweep template — README Gate M

A cache result should normally be a sweep performed with one binary/container/workload so comparisons share everything except cache size/policy.

Gate M requires **real 0731 routing** or an explicitly labeled routing trace derived from the real model. Synthetic expert IDs are useful for testing cache mechanics but do not establish the DeepSeek cache curve.

```text
Experiment id:
Date:
Commit:
Model/container:
Highest V-level:
Prompt/workload:
Hardware/storage:
Threads/backend:
Prefetch:
Gate G cache/disk identity checked?:
```

| RAM budget | Expert cache | Record capacity | Policy | Hit records | Hit bytes | Read GiB/token | Decode tok/s | RSS | swap/page faults |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|

Do not choose the automatic cache heuristic from a single cache point.

## 8. Prefetch sweep template

| Mode / depth | Reads issued/token | Useful reads | Wasted reads | Useful lead time | Read GiB/token | Cache hit | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|

Separate deterministic bootstrap prefetch from predictive lookahead if the official architecture confirms both.

Prefetch changes timing only. If prefetch on/off changes selected experts or output, README Gate G has failed and the timing result is not an optimization benchmark.

## 9. Context sweep template

| Context | Attention/state RAM | Total floor | Prefill tok/s | Decode tok/s | Notes |
|---:|---:|---:|---:|---:|---|

This table is needed because DeepSeek's advertised long context can change state/memory behavior even if expert traffic per decode token is similar.

## 10. Thread/backend sweep template

| Threads | CPU placement | backend | trunk time/token | expert compute | I/O wait | decode tok/s | power/thermal notes |
|---:|---|---|---:|---:|---:|---:|---|

Run long enough to reveal thermal throttling on laptops.

Optimized backend measurements must identify the scalar/reference gate they inherit. A faster SIMD path that only agrees with an unverified scalar convention is not an end-to-end DeepSeek result.

## 11. Conversion benchmark template

Conversion performance is separate from inference:

| Source storage | Output storage | jobs | source bytes | output bytes | wall time | peak RSS | avg read/write | verified? |
|---|---|---:|---:|---:|---:|---:|---|---|

Record whether the run is native-preserving or an experimental lossy conversion.

## 12. DSpark benchmark template — README Gate N

Do not report DSpark speedup until the base path has passed Gate I/V8 and Gate K/V9 and the speculative implementation has passed the correctness levels defined in `DSPARK.md`.

Record together:

```text
base committed tok/s
DSpark committed tok/s
proposal length
acceptance rate by position
committed tokens / verification step
expert read GiB / committed token
RAM floor delta
expert-cache delta
TTFT delta
```

A higher evaluated-token rate is not Gate N. The requirement is target-model-valid committed output plus positive wall-clock value under the same resource accounting.

## 13. Reporting ranges

For noisy end-to-end measurements:

- warm up explicitly and say how;
- report multiple runs or a stable range;
- retain surprising slow runs if they reveal paging/thermal behavior;
- do not cherry-pick the best token interval;
- state whether prompt/token sampling changed between runs.

If a benchmark is invalidated later, do not erase it. Mark it **INVALIDATED** with the reason and point to the corrected result, following the spirit of upstream `LEARNED.md`.

## 14. JSON result format target

The CLI should eventually support a machine-readable benchmark/stats output containing fields like:

```json
{
  "port_commit": "...",
  "model_revision": "...",
  "container_revision": "...",
  "readme_gates": ["K"],
  "validation_gate": "V9",
  "hardware": {},
  "context_tokens": 0,
  "prompt_tokens": 0,
  "generated_tokens": 0,
  "prefill_seconds": 0.0,
  "decode_seconds": 0.0,
  "decode_tokens_per_second": 0.0,
  "expert_cache_bytes": 0,
  "expert_record_hits": 0,
  "expert_record_misses": 0,
  "expert_bytes_read": 0,
  "prefetch_bytes_issued": 0,
  "prefetch_bytes_useful": 0,
  "direct_io": false,
  "ram_budget_bytes": 0,
  "backend": "..."
}
```

For a storage-only Gate L run, `validation_gate` may be null while `readme_gates` contains `L`; the entry still needs enough container/record provenance to be meaningful. For a cache Gate M run, include both `G` and `M` when cache/disk identity was revalidated as part of the experiment.

The exact schema can evolve; stable machine-readable provenance is the goal.

## 15. Current project baseline that is *not* a benchmark

PR #3 reported:

```text
make check -> 34 passed, 0 failed, 12 skipped
make asan  -> 33 passed, 0 failed
```

Current, after the Gate A–F replays were given their own suite lines:

```text
make check -> 48 passed, 0 failed, 12 skipped
```

That is a test-suite state, not a performance result. It belongs in project status/PR history and is repeated here only to prevent someone from mistaking the absence of DeepSeek benchmarks for missing documentation.

## 16. First benchmark sequence after Gate K / V9

When real generation works, run in this order:

1. `waste plan` and measured process memory sanity;
2. **Gate L** target-volume disk benchmark using real expert-record size;
3. decode with cache disabled or minimum practical cache;
4. **Gate M** cache sweep, while retaining **Gate G** cache/disk identity;
5. routing trace + cache simulation comparison;
6. prefetch off/on, retaining Gate G;
7. prefill sequential/chunked;
8. thread sweep;
9. context sweep;
10. **Gate N** DSpark only after base results are stable.

That order tells us whether time is going to disk, trunk compute, expert compute, synchronization, or state handling before we optimize the wrong component.