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
| PR #8 learned/hash routing + routed/shared MoE | **F/V4** | **DONE**, `632f0b3` |
| attention by type | **E/V5** | **NEXT — ratio-0 layer-0 attention first** |
| one full transformer layer | **H/V6** | after E/V5 attention variants |
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

Gate F / V4
  learned layer-3 router exact top-6 IDs [2,29,225,220,108,69]
  hash layer-0 token 4242 IDs [150,142,245,248,174,119]
  real routed expert 3/2 exact gate/up + hidden/output BF16
  all six selected routed expert output slices independently evaluated
  real resident shared FP8 expert exact hidden/output BF16
  official ascending-expert f32 accumulation + shared-add + final BF16 exact
  final representative out8 = b848 ba7a 3b1a bb78 bbb7 3ab7 ba25 3982
```

The immediate priority is now **Gate E / V5 attention**, beginning with the structurally simplest ratio-0 layer before compressed/CSA attention.

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

Reusable oracle tooling supports:

- immutable header acquisition;
- bounded tensor row/payload Range acquisition;
- exact safetensors offsets and SHA-256 provenance;
- independent pinned-source equation evaluation;
- frozen offline fixtures;
- scalar C replay inside ordinary `make check`.

Completed oracle seams:

1. native quantization convention — B/V1;
2. real resident quantized projection — C/V2;
3. real mHC/Sinkhorn primitive — D/V3;
4. learned + hash routing — F/V4;
5. routed FP4 expert + resident shared FP8 expert + six-branch MoE combination — F/V4.

The Gate F fixture generator also establishes a reusable speed pattern: a standalone C expected-value producer shares no WASTE runtime helpers and is anchored bit-for-bit to the earlier exact Fraction-based routed-expert fixture before producing additional expert outputs.

Next oracle seams:

1. ratio-0 sliding-window attention — first E/V5 tranche;
2. ratio-128 compressed attention;
3. ratio-4 CSA compressor/indexer/selection — completes E/V5;
4. complete layer — H/V6;
5. final logits/generation — I/V8 + K/V9;
6. encoder/parser — J/V10.

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

Optimized backends may use these frozen fixtures as arithmetic oracles, but optimization is not the immediate correctness priority.

---

## Phase 4 — DeepSeek-specific container / converter

**Status: DESIGN / schema can now use real Gates A, C, and F facts.**

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

### Gate D / V3 mHC — PASSED

PR #7 delivered:

- scalar `hc_split_sinkhorn`, `hc_pre`, `hc_post`;
- model-free Sinkhorn/orientation tests;
- frozen real layer-0 attention-HC parameters and independent oracle;
- exact BF16 parity across the full pre/post state.

See `docs/MHC.md`.

### Gate F / V4 routing + MoE — PASSED

PR #8 proves both routing modes and both expert storage classes.

Learned representative, layer 3:

```text
ids     = [2,29,225,220,108,69]
weights = [0.263384104,0.251154065,0.248866215,
           0.247819692,0.244902447,0.243873596]
```

Bootstrap representative, layer 0 token 4242:

```text
ids = [150,142,245,248,174,119]
```

The frozen real MoE representative additionally proves:

- correction bias changes selection but does not contaminate route weights;
- selected routed FP4 experts apply route weight before hidden BF16 cast and `w2`;
- resident shared expert uses the FP8 E4M3/E8M0 path without route weight;
- routed expert results are accumulated in ascending expert ID in f32;
- the shared expert is added once after routed branches;
- final MoE result is cast to BF16;
- the complete first-eight output values match exactly:

```text
b848 ba7a 3b1a bb78 bbb7 3ab7 ba25 3982
```

A model-free non-associativity pin distinguishes official expert-ID accumulation order from router top-k order even though that order difference happens to disappear after BF16 rounding in the selected real fixture.

Permanent ordinary `make check` replay covers model-free routing, real learned/hash routing, one full routed expert, the shared expert, all six branch output slices, and the final combination.

Each of those replays is now a named line in `tests/run.sh` under "DeepSeek gate replays". Until 2026-08-09 they reached `make check` only by being imported from `tests/test_inventory.py`, so the replay was real but every gate reported as one "inventory" line — and a corrupted Gate F fixture blamed the tensor classifier. `docs/EXPERIMENTS.md` entry 6 records the mutations. A new gate is replayed when its line is added; the list is deliberately not a glob.

### Next: Gate E / V5 attention by type

Operational order:

1. **ratio 0 / layer 0** — q/kv projections, normalization, K64 KV QAT, RoPE, 128-token causal window, sink-softmax sparse attention, inverse RoPE, output projection seam;
2. **ratio 128** — compressor + compressed history attention;
3. **ratio 4 / CSA** — compressor + indexer scoring/top-512 selection + sparse attention.

Gate E is complete only when every structurally distinct attention path has its own independent real fixture.

Then continue:

1. complete transformer layer — **H/V6**;
2. multi-layer localization — **V7**;
3. final norm/head/logits — **I/V8**;
4. deterministic known-token generation — **K/V9**.

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
