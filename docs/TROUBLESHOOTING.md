# Troubleshooting playbook

This document starts with the known PR #1 environment failures, then orders future converter/runtime failures by the cheapest evidence that can disambiguate them.

General rule: **fix the earliest failing canonical gate, not the latest visible symptom.** Bad generated text is usually the end of a much earlier mismatch.

Gate terminology follows `docs/VALIDATION.md` §4a: README §18 gates A–N are stable design identifiers, V0–V11 are operational validation levels, and ROADMAP phases are schedule only.

## 1. Hugging Face access returns CONNECT 403

Known bootstrap failure:

```text
huggingface.co:443 — proxy answered 403 to CONNECT
```

Interpretation recorded by PR #1: organization egress policy denial.

Do:

- report the environment policy block;
- run metadata/inventory commands later from an authorized environment;
- keep checkpoint-derived docs marked BLOCKED.

Do not:

- retry repeatedly as though it were a transient model-host outage;
- route around the policy through mirrors/proxies;
- substitute a GGUF/third-party checkpoint and call it official;
- reconstruct official source from memory.

See `REFERENCE_ACCESS.md`, `INVENTORY-0731.md`, and `CONVERSION.md`.

## 2. GitHub Actions jobs fail in seconds with no step logs

PR #1 preserved the imported workflow under `.github/workflows-disabled/` because all jobs failed before executing steps and logs were unavailable.

Check:

- repository Settings -> Actions permissions;
- account/org Actions policy;
- billing/minutes/spending limits, especially for macOS/Windows runners;
- whether `actions/checkout@v4` is allowed.

Do not “fix” this by deleting macOS/Windows jobs from the upstream matrix. Restore the workflow once the account condition is resolved.

Meanwhile run manual gates in `AGENTS.md`.

## 3. `tools/inventory.py` reports unknown main tensors — Gate A / V0

Expected on the first real 0731 run. The current regex rules were inferred before the checkpoint was reachable.

Do:

1. inspect exact checkpoint name;
2. identify its official-reference role;
3. extend the narrowest correct classifier rule;
4. add a regression test to `tests/test_inventory.py`;
5. re-run index-only inventory.

Do not add a generic main-model catch-all.

Unknown DSpark internals may remain in the explicit DSpark bucket only after their module boundary is independently proven.

## 4. Inventory is `index-only` and byte totals are unresolved — Gate A / V0 incomplete

This is not a bug. `model.safetensors.index.json` maps names to shards but does not provide all dtype/shape/data-offset information needed for exact stored bytes.

Fetch/make shard headers available, then run the strict header mode described in `INVENTORY-0731.md`.

Do not turn unresolved bytes into zero or config-derived estimates.

## 5. Gate A / V0 says an expert is missing w1/w2/w3 or scales

Possible causes:

- classifier alias is wrong;
- matrix lives under a different official name;
- scale naming/ownership heuristic is wrong;
- expert format is not the proposed triplet;
- checkpoint architecture differs from the README assumption.

Resolution order:

1. inspect all names for one `(layer, expert)`;
2. inspect official inference code binding;
3. update `TENSOR_MAP.md` and classifier/tests;
4. if there is an additional required per-expert tensor, redesign `CONTAINER_V4.md` before converter work.

Do not force the checkpoint into a three-matrix record if official arithmetic requires more.

If storage identity is correct but nibble/scale arithmetic is wrong, that is **Gate B / V1**, not Gate A/V0.

## 6. DeepSeek license/source attribution file is still missing

Do not copy/adapt official `inference/` or `encoding/` source until the exact license/notice has been retrieved from the pinned source.

The placeholder under `LICENSES/` records this intentionally. A generic MIT template is not a substitute for the publisher's exact file/provenance.

## 7. Converter dry run cannot account for all bytes — Gate A/V0 or format-design failure

Stop before conversion.

Check:

- inventory version matches source revision;
- all resident tensors have a destination;
- scale tensors are attributed to owning weights;
- DSpark excluded/included consistently;
- alignment/padding counted;
- tokenizer/sidecar assets counted separately from tensors;
- no transform changes stored size without an explicit formula.

A dry run with unexplained output is a failed prerequisite, not a warning.

## 8. Converter fails midway

A correct converter should be resumable by verified units.

Check:

- `.tmp` file left by interrupted unit;
- final bank exists and passes structural verification;
- source/converter/format revisions still match the progress metadata;
- disk is not full;
- peak RSS did not OOM the process;
- worker count is appropriate for available RAM.

Never mark a bank complete based only on its filename existing.

## 9. Runtime rejects model family/format

Good failure if the manifest is a Kimi v0 container or an unknown DeepSeek draft format.

Check:

- `container_family`/model-family discriminator;
- format version;
- converter commit;
- runtime commit supports that version.

Do not weaken version checks to open an old/incompatible container. Add an explicit migration/converter path if needed.

## 10. `waste plan` says budget is below floor

Do not increase the cache or allow swap.

Inspect the floor breakdown:

- resident trunk;
- attention/context state;
- persistent state;
- scratch/thread buffers;
- I/O/minimum expert buffers;
- DSpark if enabled.

Reduce only a real configurable component (context, threads, DSpark) if semantics allow. If a mandatory allocation is missing from the planner, fix the planner before rerunning.

## 11. Process is OOM-killed despite plan fitting

Likely planner/accounting bug or wrong usable-memory detection.

Check:

- cgroup/container memory limits versus host RAM;
- actual RSS and anonymous mappings;
- allocations performed after plan/open;
- thread-local scratch multiplied by threads;
- cache and I/O buffers counted twice;
- resident quantized weights expanded unexpectedly;
- DSpark enabled without planner delta.

Treat this as a correctness defect in the hard-budget contract.

## 12. Direct I/O/page-cache bypass is false — Gate L measurement context

First determine whether correctness still works through the documented fallback.

Check:

- filesystem supports requested bypass;
- record offset/size alignment;
- userspace buffer alignment;
- Windows `FILE_FLAG_NO_BUFFERING` constraints;
- Linux `O_DIRECT` constraints;
- macOS bypass API return status.

Do not benchmark the buffered fallback as though it were the intended Gate L streaming path without labeling it.

## 13. Expert read returns short/bad record

Generation must stop.

Check error detail for:

- layer/expert identity;
- expected/actual bytes;
- header magic/version;
- offset/length bounds;
- checksum if enabled.

Then run the offline verifier on the bank/container. Do not continue with partially read or corrupted expert data.

## 14. Cache-on output differs from cache-off — README Gate G failure

This violates a core invariant.

Likely causes:

- cache key aliases wrong `(layer, expert)`;
- buffer reused before compute finished;
- record read/padding parsed differently on cache hit;
- mutation of cached quantized payload;
- race between reader/eviction and compute;
- format metadata not part of key when needed.

Reproduce single-threaded with a tiny synthetic container and compare payload bytes before debugging model arithmetic.

Do not publish Gate M cache-performance results while Gate G fails.

## 15. Router selects different expert IDs from official reference — V3 primitive / Gate F / V4

Do not debug expert kernels yet.

Check in this order:

1. layer input hidden state matches;
2. router projection weight binding/transposition;
3. quantization decode for router if quantized;
4. score transform order;
5. correction/bias used for selection;
6. top-k tie/order semantics;
7. normalization/scaling;
8. bootstrap/hash mapping source for early layers.

Selected expert IDs/order should be treated as exact semantic checks. Small router primitive mismatches localize in V3; routing + one complete MoE layer is README Gate F / V4.

## 16. One expert output differs — Gates B/C or V3/F-V4

Check:

- correct expert identity/record;
- `w1/w3/w2` order;
- logical versus stored dimensions;
- FP4 packing order — Gate B/V1;
- scale axis/block and scale direction — Gate B/V1;
- matrix transpose/row-major assumptions;
- activation/clamp order — V3;
- accumulation dtype;
- down projection shape.

Return to Gate B/V1, Gate C/V2, then V3/one-expert fixtures in that order. Do not loosen full-layer tolerances.

## 17. Attention matches at short context but fails near a boundary — Gate E / V5

Suspect state/indexing rather than general projection math.

Check:

- sliding-window start/end index;
- position/RoPE index;
- compression update cadence;
- compressor state lifetime;
- CSA indexer selection coordinates;
- context length versus capacity;
- off-by-one around the first compressed/windowed token.

Maintain explicit boundary fixtures at `boundary-1`, `boundary`, `boundary+1`.

## 18. Full layer fails but primitives pass — Gate H / V6

Capture official and C intermediates at the composition boundaries:

```text
layer input
after attention
mHC/residual state
router input
shared expert output
routed expert outputs
combined MoE
layer output
```

The first mismatch determines the next investigation. Do not refactor multiple components at once while diagnosing.

## 19. Final logits differ but every sampled layer looked close — Gate I / V8

Increase checkpoint density to find the first divergence. Use extra operational rung V7 for multi-layer localization.

Also check:

- exact prompt/token IDs;
- embedding row;
- hidden-state/state reset between runs;
- final norm;
- LM head binding/quantization;
- context position;
- accidentally enabled DSpark/prefetch path changing state;
- stale golden fixtures from another model revision.

Logit max-relative error alone can be misleading near zero; use `VALIDATION.md`'s metric set.

## 20. Greedy output diverges after several matching tokens — Gate K / V9

Capture logits at the first divergent position for both sides.

If argmax differs because two logits are extremely close, that is still a parity issue to understand; do not dismiss deterministic divergence as sampling noise.

Check state updates after the last matching token, especially compressed attention and any session persistence.

## 21. Server output differs from direct CLI/C generation — Gate J/V10 or V11

If underlying Gate K/V9 passes, suspect the API/encoding layer:

- exact encoded prompt token sequence — Gate J/V10;
- generation options/defaults;
- stop/EOS configuration;
- parser dropping/rewriting tokens — Gate J/V10;
- streaming parser state — Gate J/V10/V11 seam;
- model alias resolving a different container — V11;
- DSpark enabled in one path only — Gate N configuration.

Compare token IDs, not rendered strings first.

## 22. Streaming emits tokens later rejected by DSpark — Gate N critical failure

Only committed tokens may leave the server.

Disable DSpark and reproduce. Then inspect speculative acceptance/commit boundary and parser feed location. See `DSPARK.md`.

## 23. High cache hit rate but decode becomes dramatically slower — Gate M analysis

Suspect memory pressure/paging before assuming cache policy is good.

Record:

- RAM budget and planned floor;
- actual RSS;
- swap/page faults;
- expert-cache bytes;
- bytes read/token;
- direct-I/O state;
- storage throughput;
- CPU utilization.

Upstream WASTE measured this class of failure on Kimi: more cache can increase logical hit rate while paging destroys throughput. DeepSeek must be re-measured, but the diagnostic lesson transfers.

A high logical hit rate alone does not satisfy Gate M.

## 24. Disk benchmark is fast but model is slow — Gate L is not tok/s

Disk throughput is only a lower bound on token time.

Profile:

- trunk compute;
- attention/state;
- expert compute;
- non-overlapped I/O wait;
- thread dispatch/synchronization;
- cache lookup/eviction;
- API/sampling overhead.

Do not infer that storage is unused simply because total token time exceeds `bytes / GBps`.

## 25. Optimized SIMD path fails while scalar passes

Keep scalar as local implementation oracle, but check whether the scalar baseline has itself passed the official gate.

Check:

- lane packing/unpacking;
- tail handling;
- alignment assumptions;
- scale-block boundaries;
- accumulator width/overflow;
- reduction order/tolerance;
- runtime dispatch selecting the intended kernel;
- multi-thread race versus pure kernel error.

For native quantization, an optimized path matching a scalar implementation that has only passed V1a but not Gate B/V1b is still not official DeepSeek parity.

Do not remove the scalar path to make binary size smaller while the port is still evolving.

## 26. What to include in a bug report/PR

```text
port commit
model revision
container/converter revision
OS/hardware/storage
command
RAM budget/context/threads/backend
DSpark/prefetch/cache settings
README §18 gate letter(s) A-N involved
highest operational V-level/system gate that passes
first canonical gate that fails
exact error detail
smallest reproducible independent fixture if available
```

That information is usually more useful than a screenshot of incorrect generated text.