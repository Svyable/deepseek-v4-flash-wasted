# DeepSeek V4 Flash porting status

> Canonical re-entry point for active porting work. Update this file when a gate
> is promoted, invalidated, or superseded. Historical design/validation notes
> remain useful, but this file describes what `main` can claim now.

## Baseline

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Pinned revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Merged foundation: through PR #19 plus fail-closed loader contracts from #17
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
- no on-disk header/order/alignment has been prematurely frozen;
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
4. **Family manifest/parser/binding.** The v1 family manifest parser is built
   and fail-closed (`src/deepseek_v4_manifest.{c,h}`); binding validated
   resident planes to the WASTE backend and routed records to the expert-cache
   fetch seam remains to be built. No on-disk record header/order/alignment is
   frozen.
5. **Encoding/API and serving.** Downstream of raw model arithmetic.
6. **Storage/cache/performance.** Correct native record geometry exists; real
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
2. Bind resident matrices to the normal WASTE backend and routed records to the
   existing expert-cache fetch seam. The manifest exposes the two seams
   (`waste_ds_v4_manifest_resident_plane`,
   `waste_ds_v4_manifest_bind_routed_record`); neither is wired to the engine
   yet, and `deepseek_v4_manifest.o` is deliberately not in `libwaste.a`.
3. Keep model step/generation unconditionally refused until numerical gates are
   closed. `waste_ds_v4_manifest_step_refused` is that refusal, and CI checks
   it stays unconditional.

## Merge discipline

Keep one numerical frontier and one runtime frontier at a time. Merge an earned
base before stacking further evidence. Never repair stale downstream evidence by
editing hashes; reacquire from the corrected parent endpoint.
