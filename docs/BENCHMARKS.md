# Benchmarks — DeepSeek V4 WASTE port

**Status: NO DEEPSEEK PERFORMANCE RESULTS YET.**

This file is the authoritative ledger for performance measurements produced by this port. It starts empty on purpose. Upstream WASTE measurements are useful design evidence but are not DeepSeek benchmark rows.

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
- correctness gate passed by the tested binary/container;
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

### Disk microbenchmark

Storage throughput/latency only. Never label the disk upper bound as model tok/s.

### Kernel microbenchmark

One operation with shapes/dtypes/threads. Never present isolated kernel speedup as end-to-end speedup.

## 3. Correctness prerequisite

Before a result is eligible for the primary table, record the highest `VALIDATION.md` gate passed.

Suggested labels:

```text
V2  quantized linear only
V6  one complete layer
V8  real final logits
V9  real greedy generation
V11 API-level parity
```

A benchmark from a path that has not passed V8/V9 may be recorded under **development measurements**, but it must not be summarized as model performance.

## 4. Primary end-to-end table

_No results yet._

| Date | Port commit | Model rev | Hardware | RAM budget / cache | Context | Configuration | Prefill tok/s | Decode tok/s | Read GiB/token | Cache hit | Validation |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|

## 5. Development measurements

Use this section for useful engineering data before end-to-end parity. Clearly identify the limited scope.

_No DeepSeek development performance measurements recorded yet._

| Date | Scope | Commit | Fixture/model | Hardware | Configuration | Result | Correctness state | Notes |
|---|---|---|---|---|---|---|---|---|

## 6. Storage benchmarks

Record the actual model/container volume, not a generic temporary filesystem if the purpose is to predict streaming performance.

| Date | Device / interface | FS | bypass | record size | threads | workload size | seq read | random record read | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|

_No DeepSeek-target storage measurement recorded in this repository yet._

## 7. Cache sweep template

A cache result should normally be a sweep performed with one binary/container/workload so comparisons share everything except cache size/policy.

```text
Experiment id:
Date:
Commit:
Model/container:
Prompt/workload:
Hardware/storage:
Threads/backend:
Prefetch:
```

| RAM budget | Expert cache | Record capacity | Policy | Hit records | Hit bytes | Read GiB/token | Decode tok/s | RSS | swap/page faults |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|

Do not choose the automatic cache heuristic from a single cache point.

## 8. Prefetch sweep template

| Mode / depth | Reads issued/token | Useful reads | Wasted reads | Useful lead time | Read GiB/token | Cache hit | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|

Separate deterministic bootstrap prefetch from predictive lookahead if the official architecture confirms both.

## 9. Context sweep template

| Context | Attention/state RAM | Total floor | Prefill tok/s | Decode tok/s | Notes |
|---:|---:|---:|---:|---:|---|

This table is needed because DeepSeek's advertised long context can change state/memory behavior even if expert traffic per decode token is similar.

## 10. Thread/backend sweep template

| Threads | CPU placement | backend | trunk time/token | expert compute | I/O wait | decode tok/s | power/thermal notes |
|---:|---|---|---:|---:|---:|---:|---|

Run long enough to reveal thermal throttling on laptops.

## 11. Conversion benchmark template

Conversion performance is separate from inference:

| Source storage | Output storage | jobs | source bytes | output bytes | wall time | peak RSS | avg read/write | verified? |
|---|---|---:|---:|---:|---:|---:|---|---|

Record whether the run is native-preserving or an experimental lossy conversion.

## 12. Reporting ranges

For noisy end-to-end measurements:

- warm up explicitly and say how;
- report multiple runs or a stable range;
- retain surprising slow runs if they reveal paging/thermal behavior;
- do not cherry-pick the best token interval;
- state whether prompt/token sampling changed between runs.

If a benchmark is invalidated later, do not erase it. Mark it **INVALIDATED** with the reason and point to the corrected result, following the spirit of upstream `LEARNED.md`.

## 13. JSON result format target

The CLI should eventually support a machine-readable benchmark/stats output containing fields like:

```json
{
  "port_commit": "...",
  "model_revision": "...",
  "container_revision": "...",
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

The exact schema can evolve; stable machine-readable provenance is the goal.

## 14. Current project baseline that is *not* a benchmark

PR #1 reported:

```text
make check -> 32 passed, 0 failed, 12 skipped
```

That is a test-suite state, not a performance result. It belongs in project status/PR history and is repeated here only to prevent someone from mistaking the absence of DeepSeek benchmarks for missing documentation.

## 15. First benchmark sequence after V9

When real generation works, run in this order:

1. `waste plan` and measured process memory sanity;
2. target-volume disk benchmark using real expert-record size;
3. decode with cache disabled or minimum practical cache;
4. cache sweep;
5. routing trace + cache simulation comparison;
6. prefetch off/on;
7. prefill sequential/chunked;
8. thread sweep;
9. context sweep;
10. DSpark only after base results are stable.

That order tells us whether time is going to disk, trunk compute, expert compute, synchronization, or state handling before we optimize the wrong component.