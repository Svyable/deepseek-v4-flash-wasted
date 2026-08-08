# Memory and I/O accounting

**Status: METHODOLOGY ONLY for DeepSeek. No real 0731 RAM floor, container size, cache hit rate, bytes/token measurement, or throughput result exists yet.**

WASTE's core advantage comes from controlling where model bytes live. That makes memory and storage accounting part of correctness: an engine that “works” only because the OS swaps or page-caches hundreds of gigabytes has violated the design.

## 1. Evidence boundary

Upstream WASTE contains valuable measured Kimi evidence about:

- hard RAM budgeting;
- cache-size versus paging cliffs;
- direct I/O/page-cache bypass;
- expert-cache policies;
- chunked prefill;
- NVMe bandwidth as a first-order bottleneck.

Those are **UPSTREAM-MEASURED** lessons. Reuse the mechanisms and gate structure, but re-measure every DeepSeek quantity.

PR #1 adds only a synthetic DeepSeek-shaped inventory fixture. Its approximate expert-record/all-miss arithmetic remains explicitly unverified against the real checkpoint.

## 2. Required memory equation

For a given context/configuration, define:

```text
RAM floor = resident model bytes
          + persistent attention/context state
          + persistent model state
          + mandatory scratch
          + mandatory I/O buffers
          + minimum cache/read buffers required to make one token progress
          + allocator/alignment overhead that is material at this scale
```

Then:

```text
expert cache budget = configured RAM budget - RAM floor
```

If the configured budget is below the floor, fail before loading/allocating the full model.

Do not “solve” an under-floor budget by letting the kernel swap the resident trunk.

## 3. Resident model accounting

Gate 0 should produce exact stored bytes per subsystem from checkpoint headers. The runtime planner then needs exact **in-memory** bytes, which can differ from stored bytes depending on representation.

Track at least:

| Subsystem | Stored bytes | In-memory representation | Planned RAM | Evidence |
|---|---:|---|---:|---|
| embeddings | TBD | TBD | TBD | Gate 0 |
| LM head | TBD | TBD | TBD | Gate 0 |
| norms | TBD | TBD | TBD | Gate 0 |
| mHC | TBD | TBD | TBD | Gate 0/reference |
| attention projections | TBD | native FP8 or proven representation | TBD | Gate 0 |
| compressors | TBD | TBD | TBD | Gate 0 |
| CSA indexers | TBD | TBD | TBD | Gate 0 |
| routers/hash maps | TBD | TBD | TBD | Gate 0 |
| shared experts | TBD | native/proven representation | TBD | Gate 0 |
| tokenizer/encoding runtime | n/a/small | host structures | TBD | implementation |
| DSpark | excluded from base floor initially | optional | TBD | phase 2 |

Do not assume “stored FP8 byte == one RAM byte” if the first kernel expands weights or builds auxiliary lookup structures. Account for the actual runtime representation.

## 4. Context/state accounting

The current handoff describes a hybrid compressed-attention model with very long maximum context. That does **not** imply a conventional full KV cache, and it does not imply context state is negligible.

The official reference must establish the exact persistent state per layer/mode.

For each attention mode, document a formula of the form:

```text
bytes(layer, context) = fixed_state
                      + context_dependent_state(context)
                      + ring/window_state
                      + index/compression metadata
```

Then sum over the actual layer schedule from config.

Required checkpoints:

- context = 1 token;
- short interactive context;
- sliding-window boundary;
- representative medium context;
- maximum supported context or a mathematically exact extrapolation from proven allocation formulas.

Never allocate for the advertised maximum context by accident when the user requested a much smaller context.

## 5. Scratch accounting

Scratch often gets omitted from model-size estimates and then consumes the cache budget in reality.

Include:

- activation buffers;
- quantized activation shadows/LUTs if optimized kernels require them;
- per-thread matvec/GEMM scratch;
- attention temporary scores/selected-position buffers;
- compressor/indexer temporary data;
- MoE expert-parallel temporary outputs;
- logits/sampling buffers;
- chunked-prefill staging;
- image/multimodal scratch only if the target actually includes such a path (do not inherit Kimi vision scratch by default).

Planner output should show scratch as a separate row, not hide it in “other.”

## 6. I/O buffers and alignment

Direct/unbuffered I/O can require aligned:

- file offsets;
- read sizes;
- destination buffers.

The cache needs enough buffers to overlap useful reads without exceeding the hard RAM budget.

Track:

```text
reader thread count
buffers per reader
buffer bytes
record alignment
record bytes by layer/format
outstanding read ceiling
```

Reader concurrency is a memory knob as well as an I/O knob.

## 7. Routed expert traffic

Exact cold traffic per token is:

```text
cold_routed_bytes(token)
  = sum over layers of bytes(record(layer, each selected expert))
```

If record sizes are uniform within a layer:

```text
cold_routed_bytes
  = sum(layer_record_bytes * selected_expert_count_for_layer)
```

Do not substitute `layers * top_k * estimated_record_size` once the actual manifest exists.

Effective read traffic is:

```text
read_bytes = cold_routed_bytes - cache_hit_bytes
           + wasted_prefetch_bytes
           + retry/error overhead
```

Report **byte hit rate** as well as record hit rate if record sizes differ.

## 8. PR #1 synthetic estimate boundary

The current README/synthetic fixture models:

- 43 main layers;
- 256 routed experts/layer;
- top-6;
- 25,165,824 logical weights/expert;
- packed FP4 plus K32 scales;
- approximately 12.75 MiB/expert before any record-header/alignment changes;
- approximately 3.21 GiB routed bytes for an all-miss decode token.

State: **SYNTHETIC-VERIFIED ARITHMETIC, CHECKPOINT-UNVERIFIED**.

These values may be useful for testing `inventory.py`; they must not be used to publish:

- final disk requirement;
- RAM requirement;
- tok/s prediction;
- NVMe requirement;
- cache-size recommendation.

Gate 0 replaces them.

## 9. Cache sizing method

Do not derive a “recommended cache” only from percent of total expert bytes.

Measure real routing traces and sweep cache sizes in one controlled implementation. Record:

- cache bytes and record capacity;
- policy;
- cold/warm state;
- prompt/workload;
- record and byte hit rates;
- bytes read/token;
- I/O latency;
- decode time;
- process RSS/footprint;
- swap/page-fault evidence where available.

WASTE's upstream history shows why this matters: a larger logical cache can make throughput worse if it pushes the process into paging. The DeepSeek resolver must be based on DeepSeek working sets, not copied Kimi multipliers.

## 10. Routing trace requirements

Once the official model can route tokens, add a trace tool that emits at minimum:

```text
token index
layer
selected expert ids
routing weights
routing mode (bootstrap/hash or learned, if applicable)
```

Derived statistics:

- unique records/token;
- next-token reuse;
- per-layer concentration;
- global hot-set concentration;
- reuse distance;
- bootstrap-layer determinism;
- cache simulation for LRU/LFRU/current policy;
- prefetch usefulness opportunity.

Simulation predicts cache policy; only the real bounded cache validates it.

## 11. Prefetch accounting

Prefetch metrics must distinguish:

```text
issued records
useful records (real router later requested them)
wasted records
bytes issued/useful/wasted
lead time before use
cache entries displaced by prefetch
```

A prefetch optimization can increase nominal hit rate while making total I/O or cache pollution worse. Benchmark end-to-end.

If bootstrap routing is confirmed deterministic, measure it separately from predictive lookahead because it has different risk characteristics.

## 12. Disk benchmark methodology

Before attributing inference speed to model code, benchmark the exact target volume with the engine's expected pattern:

- direct/page-cache-bypassed reads if runtime uses them;
- record-sized random reads;
- realistic concurrency;
- working set larger than RAM/page cache;
- report whether bypass actually engaged.

Measure:

```text
sequential read/write (conversion context)
random expert-record read, 1 thread
random expert-record read, N reader threads
latency distribution if available
filesystem/device/model
```

Do not use a cached `dd` result as NVMe evidence.

## 13. Throughput model

A first-order decomposition after real measurements exist:

```text
token_time ~= trunk_compute
           + attention_compute/state
           + expert_compute
           + non-overlapped expert I/O
           + synchronization/dispatch
           + sampling/API overhead
```

Disk-only upper bound:

```text
io_lower_bound_seconds = read_bytes_per_token / measured_effective_GBps
```

This is a lower bound on token time, **not** a tok/s prediction. Upstream WASTE learned this distinction the hard way: I/O may be only one component of decode.

## 14. Memory-plan CLI target

`waste plan` for a DeepSeek container should eventually report something like:

```text
resident trunk                    <exact>
attention/context state           <exact for requested ctx>
model persistent state            <exact>
scratch (threads=N)               <exact>
I/O + minimum expert buffers      <exact>
-------------------------------------------------
RAM FLOOR                         <exact>
configured/auto budget            <exact>
expert cache                      <exact>
expert record capacity            <exact or range by layer>
DSpark (disabled/enabled)          <separate>
```

No hidden multi-gigabyte allocation may occur after this plan without being added to the planner.

## 15. OOM/paging diagnostics

When throughput collapses, record before changing cache policy:

- resolved budget;
- actual RSS/footprint;
- swap usage;
- page faults if available;
- direct-I/O status;
- cache bytes;
- bytes read/token;
- CPU utilization/thread placement;
- storage throughput.

A high cache hit rate with low speed may mean cache pages themselves are being faulted back from swap/page cache. `TROUBLESHOOTING.md` uses this order.

## 16. Required result tables

Once measurements begin, keep three separate tables:

### Storage inventory

Checkpoint/container bytes by subsystem — independent of hardware.

### RAM plan

Context/thread/config-specific allocated bytes — dependent on runtime choices.

### Runtime traffic/performance

Prompt/cache/hardware-specific reads and timings.

Do not merge them into one “model size” number.

## 17. Completion criteria

This document becomes a measured DeepSeek design only when:

- Gate 0 exact stored bytes are recorded;
- the runtime representation of every resident tensor is known;
- attention/state formulas are validated against actual allocations;
- planner sum agrees with measured process memory within explained allocator/OS overhead;
- a real routing trace exists;
- cache sweeps exist on at least one target system;
- disk benchmarks use the real record size/pattern;
- benchmark results are copied to `BENCHMARKS.md` with full provenance.