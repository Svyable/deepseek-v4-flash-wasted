# DeepSeek V4 Flash porting status

> Canonical re-entry point for active porting work. Update this file when a gate
> is promoted, invalidated, or superseded. Historical design/validation notes
> remain useful, but this file describes what `main` can claim now.

## Baseline

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Pinned revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Merged foundation: through PR #24
- Corrected canonical Gate-H layer-3 endpoint:
  `c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7`
- Refused historical layer-3 endpoint:
  `0e65c4ecb328d5067f2274e724f9f46e4a13218e7fdeb706f7d1a465c0ee4761`

## Proven on `main`

### Model-semantic arithmetic

- Gate A/V0 checkpoint inventory and storage geometry;
- Gate B/V1 native E2M1/E8M0/E4M3FN semantics;
- Gate C/V2 real resident quantized projection;
- Gate D/V3 real mHC/Sinkhorn semantics with corrected F32 operation order;
- Gate F/V4 learned/hash routing, routed FP4 experts, shared FP8 expert, and
  exact six-branch combination;
- Gate E/V5 attention structural classes including ratio-0, ratio-128, and
  ratio-4/indexer paths at the stated scalar/model-semantic boundary;
- Gate H/V6 complete real layer-3 transition;
- V7 consecutive layer 3 ratio128 -> layer 4 ratio4 with an 18-boundary exact
  trace and first-divergence localization;
- V7 layer walk generalized into a parameterized per-layer continuation engine
  (`tools/deepseek_v4_continuation.py`) that accepts an exact prior `[4,4096]`
  state and an evidence source, reproduces the committed two-layer fixture byte
  for byte on both composition backends, and walks an arbitrary consecutive run
  through the same driver;
- Gate-I/V8 final-head surface frozen from immutable headers/source;
- bounded final-head primitive validated on a real frozen V7 layer-4 state:
  hc_head collapse, final RMSNorm, and 24 selected vocabulary-row logits.

The V8 primitive input is deliberately **not** the true final transformer hidden
state, so final-model logits remain open.

### Evidence freshness

- V7 derives the current parent endpoint from Gate-H provenance and verifies the
  actual frozen BF16 payload;
- the refused historical endpoint cannot be reintroduced as a copied chain SHA;
- bounded payload acquisition is revision/header/SHA bound and permanently
  replayed offline;
- first-divergence localization validates trace integrity and exact inter-layer
  chaining before numeric comparison.

### Fail-closed loader/runtime groundwork

- Gate-A model geometry is encoded as strict C contracts;
- native routed expert payload geometry is validated as packed E2M1 + raw E8M0
  scales without a dequantized shadow representation;
- resident E4M3 + E8M0 tile geometry is validated;
- an opaque expert-cache record can be mapped to six native routed planes via a
  manifest-supplied, non-overlapping byte map;
- a v1 DeepSeek-family manifest parses fail-closed: mandatory exact `family`,
  pinned revision, Gate-A geometry compared rather than range-checked, derived
  plane sizes against declared offsets, pairwise non-overlap for every routed
  and resident span, strict integer number reading, and a refused
  stepping/generation declaration;
- validated resident planes can be applied through the normal WASTE backend
  dispatch, whose universal DeepSeek FP8 slot is the scalar reference until an
  optimized backend earns parity;
- validated routed records can be fetched through the existing `waste_ecache`
  callback/cache path and only then bound to the six native FP4/E8M0 planes;
- a production-facing positional source can deep-copy an explicit
  `[layer][expert] -> byte offset` index, bind it to the exact validated routed
  map, bounds-check every record against its bank and signed-64-bit WASTE I/O
  range, tolerate legal short reads until an exact record lands, and feed that
  source directly into the existing runtime cache seam without inferring expert
  ordering;
- an explicit file-resource owner can load the manifest-declared resident trunk
  directly into its final aligned allocation, open caller-resolved routed bank
  files, derive their byte limits from the OS rather than caller claims, bind
  them through the positional source, and unwind every partially-open resource
  in reverse dependency order;
- an evidence-backed resolver derives the resident path, per-layer bank paths
  and explicit expert offsets from the container's own `files` declaration,
  confines every path under the container root, requires each main layer to be
  declared exactly once, refuses overlapping records within a bank, and
  constructs no filename of its own;
- the runtime, positional-source, file-resource, and resolver checks remain
  model-free/synthetic infrastructure only: no evidence-backed DeepSeek
  directory resolver or Gate-G disk-vs-cache identity result is claimed;
- no on-disk filename/header/order/alignment/direct-I/O convention has been
  prematurely frozen;
- no DeepSeek public stepping/generation is enabled.

## Open gates

1. **True final hidden state.** V7 stops after layer 4. The reusable
   continuation path now exists and is tested; what layers 5-42 still need is
   *evidence* — per-layer HyperConnection parameters and acquired attention/MoE
   branch outputs behind an `EvidenceSource`. Acquisition is blocked wherever
   the pinned checkpoint is unreachable.
2. **Final-model logits.** The final-head primitive is proven only on a bounded
   real test vector, not the true final transformer state.
3. **Deterministic greedy generation / V9.** Blocked on final logits.
4. **Family container/open path.** The v1 family manifest parser, resident
   backend binding, routed cache binding, explicit positional source, explicit
   native-file ownership layer, and the declaration-driven resolver are all
   built and fail-closed. What remains is a **real container** to point them
   at: the resolver reads a `files` declaration it has only ever been given
   synthetically. No filename, record header, ordering, alignment, or
   direct-I/O rule is frozen by the generic runtime substrate.
5. **Encoding/API and serving.** Downstream of raw model arithmetic.
6. **Storage/cache/performance.** Correct native record geometry, a hardened
   exact-read placement seam, and owned native resources exist; real
   container/cache identity and performance remain open.

## Next load-bearing work

### Numerical path

1. ~~Generalize the V7 layer-4 replay into a parameterized per-layer
   continuation engine that accepts an exact prior `[4,4096]` state and
   preserves the same trace/localizer contract.~~ **Done** —
   `tools/deepseek_v4_continuation.py`, checked by
   `tests/test_v7_continuation_engine.py`.
2. Advance through layers 5-42 with evidence-driven acquisition of only the six
   routed experts plus the shared expert selected at each layer. Implement a
   checkpoint-backed `EvidenceSource`; the driver, trace contract and localizer
   need no further change.
3. Apply the already-proven final-head primitive to the true layer-42 output and
   pin selected/full logits.
4. Only then begin deterministic greedy generation (V9).

### Runtime path in parallel

1. ~~Add a DeepSeek-family manifest/parser that validates Gate-A geometry,
   resident FP8 descriptors, and routed record maps.~~ **Done** —
   `src/deepseek_v4_manifest.{c,h}`, checked by
   `tests/test_deepseek_v4_manifest.c`; see `docs/CONTAINER_V4.md` §4a.
2. ~~Bind resident matrices to the normal WASTE backend and routed records to
   the existing expert-cache fetch seam.~~ **Done as model-free substrate** —
   `src/deepseek_v4_runtime.{c,h}` routes resident E4M3/E8M0 matrices through
   `waste_k` and routed records through `waste_ecache`, checked by
   `tests/test_deepseek_v4_runtime.c`. This does not yet open a real DeepSeek
   container or claim Gate-G disk/cache identity.
3. ~~Harden the routed storage boundary into an exact positional source without
   inventing bank order.~~ **Done as model-free substrate** — callers supply an
   explicit offset for every `[layer][expert]`; initialization deep-copies and
   bounds-checks the index, exact reads tolerate short progress and reject EOF,
   zero progress or over-reporting readers, the routed map is frozen with the
   source, and a native-fd adapter reuses WASTE's signed-64-bit `pread` seam.
4. ~~Own explicit native file resources around the positional source.~~ **Done
   as model-free substrate** — `src/deepseek_v4_file_runtime.{c,h}` opens
   caller-resolved paths without naming assumptions; the resident file must
   exactly match the manifest's trunk size and is loaded once into its final
   aligned allocation; routed bank limits come from actual OS file sizes;
   partial opens unwind runtime, source, descriptors, and resident bytes through
   one reverse-order cleanup path. Real local-file coverage lives in
   `tests/test_deepseek_v4_file_runtime.c` and is included in `make check` and
   `make asan`.
5. ~~Build the evidence-backed family resolver: read the real DeepSeek
   container metadata, derive actual resident/bank paths and explicit expert
   offsets, and hand those facts to `waste_ds_v4_file_runtime_open`.~~ **Done
   as declaration-driven substrate** — `src/deepseek_v4_resolver.{c,h}` reads
   the container's `files` section, confines paths under the container root,
   requires exactly one declaration per main layer, accepts an explicit offset
   table or a declared base/stride that materializes byte-identically, refuses
   overlapping records, and constructs no filename (CI greps for it). Checked
   by `tests/test_deepseek_v4_resolver.c`; see `docs/DEEPSEEK_RUNTIME_IO.md`.
   The remaining gap is a real container document to resolve.
6. Use real container bytes to close Gate G: compare miss/hit bytes and expert
   arithmetic, then compare buffered versus page-cache-bypassing I/O only after
   actual alignment requirements are known.
7. Keep model step/generation unconditionally refused until numerical gates are
   closed. `waste_ds_v4_manifest_step_refused` is that refusal, and CI checks
   it stays unconditional.

## Merge discipline

Keep one numerical frontier and one runtime frontier at a time. Merge an earned
base before stacking further evidence. Never repair stale downstream evidence by
editing hashes; reacquire from the corrected parent endpoint.
