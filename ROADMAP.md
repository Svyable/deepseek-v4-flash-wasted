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
| PR #7 real layer-0 mHC | **D/V3 mHC** | **IN PROGRESS; real validation passed** |
| router primitives + one MoE block | V3 → **F/V4** | **NEXT** |
| attention by type | **E/V5** | after primitive/MoE seams |
| one full transformer layer | **H/V6** | later base bring-up |
| multi-layer localization/logits/generation | V7, I/V8, K/V9 | later base bring-up |
| encoding/parser/API | J/V10 + V11 | after raw model arithmetic |
| storage/cache/performance | G/L/M | after container/base correctness |
| DSpark | N | after base logits/generation |

Completed correctness milestones:

```text
Gate A / V0
  48 / 48 headers
  72,317 tensors
  155.417748 GiB payload
  11,008 routed expert records
  6 PASS, 0 FAIL, 0 SKIP

Gate B / V1
  public format semantics
  + official nibble/scale conventions
  + real checkpoint storage geometry

Gate C / V2
  real layers.0.attn.wq_a bytes
  independent pinned-source oracle
  scalar C exact BF16 match on 8 outputs

Gate D / V3 mHC
  real layers.0.hc_attn_{fn,base,scale}
  independent hc_pre/Sinkhorn/hc_post oracle
  scalar C exact BF16 match on 4,096 pre + 16,384 post outputs
  diagnostic max_abs 4.76837158e-07
```

The immediate priority is now **router/MoE primitive bring-up**, not more checkpoint metadata or quantization work.

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

Reusable oracle tooling now supports:

- immutable header acquisition;
- bounded tensor row/payload Range acquisition;
- exact safetensors offsets and SHA-256 provenance;
- independent pinned-source equation evaluation;
- frozen offline fixtures;
- scalar C replay inside ordinary `make check`.

Completed oracle seams:

1. native quantization convention — B/V1;
2. real resident quantized projection — C/V2;
3. real mHC/Sinkhorn primitive — D/V3.

Next oracle seams:

1. learned router score/selection/weighting;
2. hash-router IDs + weights;
3. shared/routed expert SwiGLU and combination — F/V4;
4. attention/compressor/indexer — E/V5;
5. complete layer — H/V6;
6. final logits/generation — I/V8 + K/V9;
7. encoder/parser — J/V10.

Reuse the same fixture/provenance pattern rather than inventing a new oracle mechanism per subsystem.

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

Optimized backends may now use these frozen fixtures as arithmetic oracles, but optimization is not the immediate correctness priority.

---

## Phase 4 — DeepSeek-specific container / converter

**Status: DESIGN / schema can now use real Gate A facts.**

Deliverables:

- DeepSeek-specific model-family/format discriminator;
- resident trunk mapping from real tensor names;
- one independently readable aligned routed-expert record containing real `w1/w3/w2` packed bytes + E8M0 scales;
- resumable conversion/provenance;
- bounds/checksum/header validation;
- dry-run disk-space accounting;
- selected real tensor round-trip to proven scalar/oracle seams.

Container work may proceed in parallel, but storage must not become the only path for correctness fixtures.

---

## Phase 5 — base model arithmetic — D/F/E/H/I/K, V3–V9

### Gate D / V3 mHC — VALIDATED

PR #7 adds:

- scalar `hc_split_sinkhorn`, `hc_pre`, `hc_post`;
- model-free Sinkhorn/orientation tests;
- frozen real layer-0 attention-HC parameters and independent oracle;
- exact BF16 parity across the full pre/post state.

See `docs/MHC.md`.

### Next: router primitives → Gate F / V4

Bring up routing in two cases from the pinned release:

1. **learned router**, representative layer 3;
2. **hash/bootstrap router**, representative layer 0 with real `tid2eid`.

Required exact semantics:

```text
raw = gate_linear(hidden)
score = sqrt(softplus(raw))
selection_score = score + correction_bias
ids = topk(selection_score, 6)
weights = score[ids]            # original transformed score, not biased score
weights /= sum(weights)
weights *= 1.5
```

For the first three layers, `tid2eid[token_id]` provides the selected IDs; routing weights still come from the score path.

After router primitives:

- shared expert SwiGLU/clamp;
- one real routed expert;
- exact selected-expert combination;
- direct bytes versus storage/cache bytes identity;
- **Gate F / V4** one MoE block.

Then continue:

1. sliding-window attention;
2. compressed attention + compressor;
3. CSA indexer/selection — **E/V5**;
4. complete transformer layer — **H/V6**;
5. multi-layer localization — **V7**;
6. final norm/head/logits — **I/V8**;
7. deterministic known-token generation — **K/V9**.

Each seam gets an independent source/checkpoint fixture before becoming part of a full layer.

---

## Phase 6 — memory planner + streaming — G/L/M

Reuse WASTE's bounded-cache/direct-I/O mechanisms, but derive all DeepSeek sizes from Gate A and real routing traces.

Deliverables:

- exact resident trunk/state/scratch accounting;
- expert cache floor/recommendation;
- Gate G cache/disk/prefetch arithmetic identity;
- Gate L real aligned-record I/O measurement;
- Gate M real routing-derived cache curve;
- chunked prefill/read-ahead only after numerical identity is proven.

---

## Phase 7 — encoding / server — J/V10 + V11

Port the official code-based `encoding/` semantics separately from model arithmetic. Extend the existing WASTE OpenAI-compatible server rather than adding an unrelated serving stack.

---

## Phase 8 — optimization

Only optimize a seam with a proven scalar/oracle path.

Candidates:

- FP4/FP8 SIMD;
- expert-parallel execution;
- async read-ahead;
- deterministic prefetch for layers 0–2 using real `tid2eid`;
- routing-informed cache policy;
- CPU/thread placement;
- accelerator backends where profiling justifies them.

Every optimization must preserve frozen correctness fixtures and Gate G.

---

## Phase 9 — DSpark — Gate N

Begin only after base-model Gate I/V8 logits and Gate K/V9 greedy generation pass with DSpark disabled.

Wrong speculative tokens must never alter committed output.

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
