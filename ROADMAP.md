# Roadmap — DeepSeek V4 Flash on WASTE

This roadmap turns the implementation plan into gated PR-sized work. Cheap exact gates precede expensive conversion, full-layer integration, and performance work.

Gate vocabulary:

- `README.md` §18 — stable design Gates A–N;
- `docs/VALIDATION.md` — operational V0–V11 levels;
- roadmap phases — schedule only.

Pinned model:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

## Current snapshot

| Work | Gate | Status |
|---|---|---|
| PR #1 WASTE bootstrap/inventory tooling | supports A/V0 | **DONE** |
| PR #2 docs foundation | process | **DONE** |
| PR #3 scalar public quantization | B/V1a | **DONE** |
| PR #4 gate/docs concordance | process | **DONE** |
| PR #5 real checkpoint inventory + official quant contracts | **A/V0 + B/V1** | **DONE**, `fad91e7c7c9c9888670953b51d7e35338db9575e` |
| PR #6 real resident quantized projection | **C/V2** | **DONE**, `8456aeef6ccdf417894c3ea97fc0aef9568ab7a1` |
| PR #7 real layer-0 mHC | **D/V3 mHC** | **DONE**, `eac344236d5f3bd5544188ab2a229faa8e3ead6c` |
| PR #8 learned/hash routing + routed/shared MoE | **F/V4** | **DONE**, `632f0b3ba36a033713702fda2daecf44ac0946aa` |
| PR #9 ratio-0 attention + grouped output projection | **E/V5 partial** | **DONE**, `093081efc4a9f583a1962bcb69e05a20d57c1585` |
| ratio-128 compressor | **E/V5 partial** | **CHECKPOINT-PASSED**, 4,096 exact BF16 diagnostics |
| ratio-128 compressed-history composition | **E/V5 partial** | **CHECKPOINT-PASSED**, 130 indices + 512 BF16 exact |
| coherent same-input ratio-128 forward | **E/V5 partial** | **CHECKPOINT-PASSED**, one head / row 255 through inverse compressed RoPE |
| ratio-4 CSA/indexer attention | **E/V5** | **CHECKPOINT-PASSED** — real Indexer + coherent selected attention |
| Gate E attention by type | **E/V5** | **PASSED** at stated scalar/model-semantic evidence level |
| one full transformer layer | **H/V6** | **PASSED scalar/model-semantic** — real layer-3 HC + 64-head attention + exact runtime router + six routed/shared experts + final BF16 state |
| multi-layer localization/logits/generation | V7, I/V8, K/V9 | later base bring-up |
| encoding/parser/API | J/V10 + V11 | after raw model arithmetic |
| storage/cache/performance | G/L/M | after container/base correctness |
| DSpark | N | after base logits/generation |

## Completed correctness milestones

### Gate A / V0

```text
48 / 48 headers
72,317 tensors
155.417748 GiB payload
11,008 routed expert records
6 PASS, 0 FAIL, 0 SKIP
```

### Gate B / V1

- public E2M1/E8M0/E4M3FN semantics;
- official FP4 nibble and scale conventions;
- real checkpoint quantized-storage geometry.

### Gate C / V2

- real `layers.0.attn.wq_a` bytes;
- independent pinned-source oracle;
- scalar C exact BF16 match on eight outputs.

### Gate D / V3 mHC

- real `layers.0.hc_attn_{fn,base,scale}`;
- independent `hc_pre` / Sinkhorn / `hc_post` oracle;
- exact BF16 match on 4,096 pre + 16,384 post outputs;
- diagnostic max_abs `4.76837158e-07`.

### Gate F / V4

- learned layer-3 router exact top-6 IDs `[2,29,225,220,108,69]`;
- hash layer-0 token-4242 IDs `[150,142,245,248,174,119]`;
- real routed expert 3/2 exact hidden/output BF16;
- all six selected routed branches independently evaluated;
- real resident shared FP8 expert exact hidden/output BF16;
- official ascending-expert f32 accumulation + shared-add + final BF16 exact;
- final representative out8 `b848 ba7a 3b1a bb78 bbb7 3ab7 ba25 3982`.

### Gate E / V5 — PASSED at scalar/model-semantic evidence level

Real layer-0 ratio-0 core:

```text
2 tokens x 2 heads
post-RoPE Q                 2,048 BF16 values exact
post-K64-QAT KV             1,024 BF16 values exact
post-inverse-RoPE attention 2,048 BF16 values exact
all 5,120 diagnostics exact
```

Representative real signatures:

```text
Q pos0/head0 first8
bf2b 3f58 bfb6 beba bee2 bfaa be58 3da2

Q pos1/head0 RoPE tail
402f 3fb0 3fac bf83 bf8b 3fc3 bf46 bd40

attention pos1/head0 first8
be21 3e02 be52 be33 3de1 3f07 3e54 3e27
```

Shared grouped-output projection sub-seam with real checkpoint `wo_a/wo_b`:

```text
group latent first8
3a65 3dcb 3d2b 3d09 3cba bc02 bcab 3d9e

wo_b output first8
ba34 bce1 bd35 3bd0 3d87 3d77 bc85 bd46
```

The output fixture uses a sparse structurally valid 8-head group seeded from real ratio-0 heads 0/1. It proves checkpoint `wo_a` FP8→BF16 dequant/orientation, group placement, and quantized `wo_b`; it does not claim heads 2–7 are independently generated real attention heads.

See `docs/ATTENTION.md` for source cast boundaries, fixture provenance, and non-claims.

Ratio-4 CSA/indexer completion:

```text
model-free: normalized Hadamard + K32 FP4 + overlap + causal top-512
real Indexer: layer 2, 64 heads x 128, scores bc8e / bc63, top-k [9,8]
coherent CSA: same positions 3/7 input -> main overlap compressor -> selected sparse attention -> inverse compressed RoPE
```

The shared grouped output projection remains independently proved rather than duplicated inside each mode fixture. Gate H/V6 is the next numerical integration rung.

**Gate H/V6 is now PASSED at the scalar/model-semantic one-layer level.** The checkpoint-backed layer-3 path includes both real HC transitions, all 64 attention heads, exact scalar-runtime routing, six routed FP4 experts, the shared FP8 expert, and an exact final BF16 `[4,4096]` state. Raw expert records remain acquisition-only rather than repository payload.

**Immediate priority: V7 multi-layer localization, then Gate I/V8 logits.** Do not jump straight to generation: establish a reproducible way to carry the frozen layer-state contract across structurally different layers and stop at the first divergent layer. Gate K/V9 follows only after final hidden/norm/logits are pinned.

---

## Phase 0 — provenance / baseline — DONE

- WASTE pinned at `d9b919a791148b571e643d0af666bf19b4d733ab`;
- Apache-2.0 provenance preserved;
- exact DeepSeek MIT release license vendored;
- immutable 0731 model revision pinned.

---

## Phase 1 — checkpoint inventory — Gate A / V0 — DONE

Durable evidence:

- `docs/INVENTORY-0731.md`;
- `reference/deepseek-v4-flash-0731.gate-a.json`.

Exact highlights:

```text
48 shards
72,317 tensors
166,878,536,440 payload bytes
147,169,738,752 B main routed expert payload
13,369,344 B / expert record
3.212402 GiB routed payload / cold all-miss decode token
```

Bootstrap `tid2eid` tables are checkpoint-resident in layers 0–2.

---

## Phase 2 — reference/oracle harness — ACTIVE FOUNDATION

Reusable tooling now supports:

- immutable header acquisition;
- bounded tensor-row/payload Range acquisition;
- exact safetensors offsets and SHA-256 provenance;
- independent source-equation evaluation;
- frozen offline fixtures;
- scalar C replay inside ordinary `make check`.

Completed oracle seams:

1. native quantization convention — B/V1;
2. real resident quantized projection — C/V2;
3. real mHC/Sinkhorn primitive — D/V3;
4. learned + hash routing — F/V4;
5. routed FP4 + shared FP8 expert and six-branch MoE combination — F/V4;
6. ratio-0 attention core + grouped output projection — E/V5;
7. coherent same-input ratio-128 compressor/history/attention — E/V5;
8. ratio-4 CSA compressor + Indexer selection + selected sparse attention — E/V5;
9. complete real layer-3 HC + attention + router + six-routed/shared-MoE transition — H/V6.

Next oracle seams:

1. multi-layer localization — V7;
2. final hidden/norm/logits — I/V8;
3. deterministic greedy generation — K/V9;
4. encoder/parser — J/V10.

Reuse the same independent-fixture pattern rather than inventing a new oracle mechanism per subsystem.

---

## Phase 3 — native quantization — Gates B/C — DONE AT SCALAR MODEL-SEMANTIC LEVEL

Delivered:

- E2M1/E8M0 routed decode;
- finite E4M3FN resident decode;
- official low-nibble-first convention;
- official scale multiplication semantics;
- real checkpoint storage geometry;
- K128 activation quantization reference;
- frozen real `wq_a` projection;
- exact scalar-C BF16 replay;
- fail-closed bounded Range payload fetcher.

The resident-linear reference now quantizes each input row once and reuses the exact activation bytes/scales across output rows. This preserves Gate-C/F results while making attention-sized scalar fixtures practical.

---

## Phase 4 — DeepSeek-specific container / converter

**Status: DESIGN / schema can now use real Gates A, C, and F facts.**

Deliverables:

- DeepSeek model-family/format discriminator;
- resident trunk mapping from real tensor names;
- aligned routed-expert record with real packed `w1/w3/w2` + E8M0 scales;
- resumable conversion/provenance;
- bounds/checksum/header validation;
- dry-run disk accounting;
- selected real tensor round-trip to proven scalar/oracle seams.

Container work may proceed in parallel, but storage must not become the only path for correctness fixtures.

---

## Phase 5 — base model arithmetic — D/F/E/H/I/K, V3–V9

### Gate D / V3 mHC — PASSED

See `docs/MHC.md`.

### Gate F / V4 routing + MoE — PASSED

PR #8 is merged. Ordinary `make check` permanently replays model-free routing, real learned/hash routing, a full routed expert, the shared expert, all six routed output slices, and final combination.

### Gate E / V5 attention — PASSED scalar/model-semantic

Permanently replayed structural modes now include ratio-0, coherent ratio-128 compressed history, and coherent ratio-4 CSA/Indexer attention, with the shared grouped `wo_a/wo_b` output projection separately proven. See `docs/ATTENTION.md` and `docs/VALIDATION.md` for the exact evidence boundaries.

### Gate H / V6 complete transformer layer — PASSED scalar/model-semantic

One real layer-3 transition is frozen end to end at the model-semantic boundary:

```text
residual -> hc_pre(attn) -> real 64-head attention -> hc_post
         -> hc_pre(ffn)  -> real router [255,30,99,40,44,238]
         -> six routed FP4 experts + shared FP8 expert -> hc_post
         -> exact BF16 [4,4096]
```

Acquisition checked 28,672 expert BF16 outputs against an independent standalone oracle. The exact scalar-router F32 weights are recorded separately from the independent Python router (same IDs; max weight delta `1.49662139e-07`). The canonical final state SHA-256 is `c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7`.

A later V7 review exposed that Gate H's original Python Sinkhorn still executed its internal reductions/normalizations in double precision and only rounded `post`/`comb` at the boundary. That is not model-equivalent. `tools/deepseek_v4_hc_oracle.py` now executes every HC operation in F32 and is bit-exact with `src/deepseek_v4_mhc_ref.c`. The correction changes 9 of 16,384 final BF16 values (indices 1186, 10135, 10503, 10877, 11565, 11672, 11675, 13648, 13717) while leaving `attn_pre`, `ffn_pre`, routing, all seven expert outputs, and the MoE branch unchanged. No checkpoint reacquisition was required.

Permanent suite floor after the Gate H precision correction:

```text
make check  65 passed, 0 failed, 13 skipped
make asan   64 passed, 0 failed, 14 skipped
```

Continue in this order:

1. multi-layer localization — **V7**;
2. final hidden/norm/head/logits — **I/V8**;
3. deterministic known-token generation — **K/V9**;
4. only then broaden to encoding/API and optimization gates.

---

## Phase 6 — memory planner + streaming — G/L/M

Reuse WASTE's bounded-cache/direct-I/O mechanisms, but derive all DeepSeek sizes from Gate A and real routing traces.

Deliverables:

- exact resident trunk/state/scratch accounting;
- expert cache floor/recommendation;
- Gate G cache/disk/prefetch identity;
- Gate L real aligned-record I/O measurement;
- Gate M real routing-derived cache curve;
- chunked prefill/read-ahead only after numerical identity is proven.

---

## Phase 7 — encoding / server — J/V10 + V11

Port the official code-based `encoding/` semantics separately from model arithmetic. Extend the existing WASTE OpenAI-compatible server rather than adding an unrelated serving stack.

---

## Phase 8 — optimization

Only optimize a seam with a proven scalar/oracle path. Candidates include FP4/FP8 SIMD, expert parallelism, async read-ahead, deterministic bootstrap prefetch, routing-informed cache policy, CPU placement, and accelerators justified by profiling.

Every optimization must preserve frozen correctness fixtures and Gate G.

---

## Phase 9 — DSpark — Gate N

Begin only after base Gate I/V8 logits and Gate K/V9 greedy generation pass with DSpark disabled. Wrong speculative tokens must never alter committed output.

---

## Definition of done for a project PR

A PR is complete when:

- implementation and relevant tests land together;
- gate/V-level is named;
- expected values are independent of the implementation under test;
- immutable source/checkpoint provenance is recorded;
- maintained docs are updated;
- model-free and relevant real-fixture gates pass;
- evidence boundaries/non-claims are explicit;
- a cheaper unresolved gate is not skipped for a more expensive downstream step.
