# Roadmap — DeepSeek V4 Flash on WASTE

This roadmap turns the implementation plan into gated PR-sized work. The ordering is intentional: cheap exact gates precede expensive conversion, transformer integration, and performance work.

Gate vocabulary:

- `README.md` §18 — stable design Gates A–N;
- `docs/VALIDATION.md` — operational V0–V11 levels;
- roadmap phases — schedule only.

## Current snapshot

Pinned model:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

| Work | Gate | Status |
|---|---|---|
| PR #1 WASTE bootstrap/inventory tooling | supports A/V0 | **DONE** |
| PR #2 docs foundation | process | **DONE** |
| PR #3 scalar public quantization | B/V1a | **DONE** |
| PR #4 gate/docs concordance | process | **DONE** |
| PR #5 real checkpoint inventory + official quant contracts | **A/V0 + B/V1** | **DONE**, merged as `fad91e7c7c9c9888670953b51d7e35338db9575e` |
| PR #6 real resident quantized projection | **C/V2** | **IN PROGRESS; validation passed, merge pending** |
| mHC/model primitives | **D/V3** | **NEXT** |
| routing + one MoE block | **F/V4** | after D |
| attention by type | **E/V5** | after primitive/MoE seams |
| one full transformer layer | **H/V6** | later base bring-up |
| multi-layer localization/logits/generation | V7, I/V8, K/V9 | later base bring-up |
| encoding/parser/API | J/V10 + V11 | after raw model arithmetic |
| storage/cache/performance | G/L/M | after container/base correctness |
| DSpark | N | after base logits/generation |

Checkpoint gates now completed:

```text
Gate A / V0
  48 / 48 headers
  72,317 tensors
  155.417748 GiB payload
  11,008 routed expert records
  6 PASS, 0 FAIL, 0 SKIP

Gate B / V1
  public format semantics + official nibble/scale conventions
  + real checkpoint storage geometry

Gate C / V2
  real layers.0.attn.wq_a bytes
  independent pinned-source oracle
  scalar C exact BF16 match on 8 outputs
```

The immediate priority is no longer metadata or quantization. It is **Gate D/V3 mHC**.

---

## Phase 0 — provenance/baseline — DONE

- WASTE baseline pinned at `d9b919a791148b571e643d0af666bf19b4d733ab`;
- Apache-2.0 provenance preserved;
- exact DeepSeek MIT release license vendored;
- immutable 0731 model revision pinned.

---

## Phase 1 — checkpoint inventory — Gate A / V0 — DONE

`tools/fetch_hf_headers.py` and `tools/inventory.py` have been run against every official shard header at the pinned revision.

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

Any future model-revision change reruns this phase.

---

## Phase 2 — reference/oracle harness — ACTIVE FOUNDATION

Reusable oracle tooling now includes:

- immutable header acquisition;
- bounded row-range tensor payload acquisition;
- provenance/SHA-256 recording;
- independent source-equation fixture generation;
- offline C replay.

This same pattern should be reused for mHC, routing, MoE and attention rather than inventing a new oracle mechanism for each subsystem.

Completed oracle seams:

- FP4 nibble/scale convention — Gate B/V1;
- real resident quantized projection — Gate C/V2.

Next oracle seams:

1. mHC/Sinkhorn transform — Gate D/V3;
2. router score/selection/weighting primitives — V3 then Gate F/V4;
3. shared/routed expert arithmetic — Gate F/V4;
4. attention/compressor/indexer — Gate E/V5;
5. complete block — Gate H/V6;
6. final logits/generation — I/V8 + K/V9;
7. encoder/parser — J/V10.

---

## Phase 3 — native quantization — Gates B/C — COMPLETE AT SCALAR MODEL-SEMANTIC LEVEL

Delivered:

- scalar E2M1/E8M0 routed decode;
- scalar finite E4M3FN resident decode;
- source-verified nibble order and scale multiplication;
- real checkpoint storage geometry;
- official K128 activation quantization reference;
- frozen real resident `wq_a` projection;
- exact scalar-C BF16 replay;
- fail-closed bounded Range payload fetcher.

PR #6's real Gate C fixture expected bits:

```text
0x3e79 0xbf84 0x3f8d 0x400a 0x3ff3 0xbf9b 0x3f82 0x3ff0
```

Do not reinterpret this as optimized backend parity. SIMD/accelerator paths may now use the frozen Gate C fixture as a real arithmetic oracle.

---

## Phase 4 — DeepSeek-specific container/converter

**Status: DESIGN / can now use real Gate A schema.**

Next converter/container work can stop guessing tensor names and record sizes.

Deliverables:

- DeepSeek-specific model-family/format discriminator;
- resident trunk mapping from real tensor names;
- one independently readable aligned routed-expert record containing the six real payload tensors (`w1/w3/w2` + E8M0 scales);
- resumable conversion and provenance;
- bounds/checksum/header validation;
- conversion dry-run and disk-space accounting;
- selected real tensor round-trip to proven scalar/oracle seams.

Container work may proceed in parallel with arithmetic gates, but it must not become the only way to supply bytes to correctness tests.

---

## Phase 5 — base model arithmetic — D/F/E/H/I/K, V3–V9

**Next milestone: Gate D / V3 — mHC.**

Recommended order:

1. exact mHC/Sinkhorn primitive fixture and scalar C implementation — **D/V3**;
2. normalization/RoPE/SwiGLU/router primitives — remaining V3 seams;
3. routing + shared/routed expert combination — **F/V4**;
4. sliding-window attention;
5. compressed attention + compressor;
6. CSA indexer/selection — completing **E/V5**;
7. complete layer — **H/V6**;
8. multi-layer localization — **V7**;
9. final norm/head/logits — **I/V8**;
10. deterministic known-token greedy generation — **K/V9**.

Each implementation PR gets an independent frozen source/checkpoint fixture before becoming part of a full layer.

No “generated text looks plausible” debugging until the smaller gates pass.

---

## Phase 6 — memory planner + streaming — G/L/M

Reuse WASTE's systems mechanisms, but use DeepSeek checkpoint-derived sizes and real routing traces.

Deliverables:

- exact resident trunk accounting;
- scratch/context/state accounting;
- expert cache minimum and recommended sizes;
- Gate G cache/disk/prefetch identity;
- Gate L real aligned-record I/O measurement;
- Gate M real routing-derived cache curve;
- chunked prefill/read-ahead only after numerical identity is proven.

---

## Phase 7 — encoding/server — J/V10 + V11

Port official `encoding/` semantics separately from raw model arithmetic. Extend WASTE's existing OpenAI-compatible server rather than adding an unrelated serving stack.

Entry condition: enough raw model arithmetic exists to distinguish encoder bugs from model bugs.

---

## Phase 8 — optimization

Only optimize a seam with a proven scalar/oracle path.

Candidates:

- FP4/FP8 SIMD;
- expert-parallel execution;
- async read-ahead;
- deterministic prefetch for the first three `tid2eid` layers;
- routing-informed cache policy;
- CPU/thread placement;
- accelerator backends where profiling justifies them.

Every optimization must preserve the frozen correctness fixtures and Gate G.

---

## Phase 9 — DSpark — Gate N

Begin only after base-model Gate I/V8 logits and Gate K/V9 greedy generation pass with DSpark disabled.

Wrong speculative tokens must never alter final committed output.

---

## Definition of done for a project PR

A PR is complete when:

- implementation and relevant tests land together;
- the README gate/V-level is named;
- expected values are independent of the implementation under test;
- immutable model/source provenance is recorded;
- the maintained docs are updated;
- model-free and relevant real-fixture gates pass;
- a cheaper unresolved gate is not skipped for a more expensive downstream step;
- evidence boundaries/non-claims are explicit.
