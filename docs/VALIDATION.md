# Validation — correctness gates for the DeepSeek V4 port

**Status: Gates A/V0, B/V1, C/V2, README Gate D's mHC seam, and Gate F/V4 routing+MoE are passed at their stated scalar/model-semantic evidence levels for the pinned 0731 release. Gate E/V5 attention is next. Storage/cache identity remains the separate Gate G systems proof.**

Pinned model:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

Validation proceeds from the smallest exact seam to final logits. A later gate may rely on an earlier one only at the evidence level actually proved.

Primary maintained evidence docs:

- `INVENTORY-0731.md` — real Gate A checkpoint result;
- `NUMERICS.md` — Gates B/C quantization + real projection;
- `MHC.md` — real Gate D Hyper-Connection result;
- `ROADMAP.md` — current passed/next milestone sequence;
- `FIXTURES.md` — independence/mutation policy;
- `OFFICIAL-0731-SOURCE.md` — pinned source findings;
- `REFERENCE_ACCESS.md` — bounded official-artifact acquisition.

---

## 1. Oracle and evidence policy

The numerical/model authority is the pinned official release plus exact checkpoint bytes.

Different facts come from different evidence:

- public format spec → E2M1/E8M0/E4M3FN code semantics;
- pinned official source → DeepSeek operation semantics;
- real safetensors headers → exported names, dtypes, shapes, offsets and bytes;
- bounded real payload slices → actual checkpoint parameter/weight bytes;
- independent source oracle → expected model-semantic outputs;
- end-to-end WASTE port → final model correctness and performance.

Third-party runtimes are useful smoke/reference aids, not the numerical authority.

Every frozen official fixture records model/revision, source operation/path, exact tensor/shard/range identity, raw-input hashes, output hashes, shapes/dtypes, and generator/port provenance.

Do not describe a source-equation CPU oracle as an official GPU-kernel execution. Conversely, do not require a GPU to prove byte layout and scalar model semantics when independent checkpoint/source fixtures can settle them.

### Fixture independence

Expected values must not be generated through the WASTE implementation under test.

Round trips prove self-consistency. Silent conventions require independent literals, real checkpoint bytes, or independently evaluated official-source equations. Where practical, add a known-wrong mutation/orientation case that the fixture must reject.

Gate F adds one useful pattern for expensive fixtures: a standalone expected-value producer may be optimized for fixture generation only if it shares no WASTE runtime helpers and is first cross-validated against an earlier exact independent fixture. `tools/v4_moe_oracle.c` is anchored bit-for-bit to the Fraction-based expert-2 fixture before producing the other routed/shared expected values.

---

## 2. Tolerance policy

Prefer exact equality where the fixture can support it:

- tensor names/shapes/dtypes;
- expert/token IDs and order;
- nibble/block selection;
- raw payload hashes;
- BF16 output bits when the fixture is reduction-stable.

Introduce floating tolerances only after independent evidence demonstrates unavoidable backend/dtype/reduction differences. Record `max_abs`, `max_rel`, and RMS where useful, and keep a known-wrong mutation outside the accepted threshold.

Never loosen a tolerance merely to make a new implementation pass.

---

## 3. Operational validation ladder

### V0 — checkpoint inventory — **PASSED / Gate A**

Owners:

- `tools/fetch_hf_headers.py`;
- `tools/inventory.py`;
- `INVENTORY-0731.md`;
- `reference/deepseek-v4-flash-0731.gate-a.json`.

Pinned result:

```text
48 / 48 shards
72,317 tensors
166,878,536,440 payload bytes = 155.417748 GiB
0 unclassified tensors / 0 unexplained bytes
11,008 routed expert records
13,369,344 B / routed expert record
Gate A checks: 6 PASS, 0 FAIL, 0 SKIP
```

Checkpoint-resident bootstrap maps exist in layers 0–2 as `ffn.gate.tid2eid`, I64 `[129280,6]`.

Any model-revision move reruns V0.

### V1 — native quantization decode/conventions — **PASSED / Gate B**

Evidence includes:

- exhaustive E2M1/E8M0/E4M3FN public-format tests;
- FP4 K32 and FP8 128x128 scale-index tests;
- pinned official low-nibble-first semantics;
- pinned scale-multiplication semantics;
- real checkpoint routed `I8` packed-FP4 + `F8_E8M0` shapes;
- real checkpoint resident `F8_E4M3` + `F8_E8M0` layout;
- independent F3 literal (`0x21`, E8M0 `0x80` → `[1,2]`).

Optimized native kernels must continue matching this contract.

### V2 — one real quantized resident projection — **PASSED / Gate C**

Real target:

```text
layers.0.attn.wq_a.weight  F8_E4M3 [1024,4096]
layers.0.attn.wq_a.scale   F8_E8M0 [8,32]
```

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v2_wq_a_real/
```

It contains eight real weight rows, the corresponding scale row, deterministic BF16 input, and eight BF16 outputs. The independent source oracle implements K128 activation E4M3 quantization and scaled FP8 block accumulation.

Scalar C matches exact BF16 output bits:

```text
0x3e79 0xbf84 0x3f8d 0x400a 0x3ff3 0xbf9b 0x3f82 0x3ff0
```

Gate C passes the scalar/model-semantic projection seam. Official accelerator execution was not claimed; optimized backends must match the frozen fixture later.

### V3 — model primitives

V3 is a bucket of independently provable primitive seams. It remains broader than any single README gate.

#### README Gate D — mHC — **PASSED**

Frozen real fixture:

```text
tests/fixtures/deepseek_v4/v3_mhc_real/
```

Real layer-0 attention-HC parameters:

```text
layers.0.hc_attn_fn     F32 [24,16384]
layers.0.hc_attn_base   F32 [24]
layers.0.hc_attn_scale  F32 [3]
```

The independent source oracle covers:

```text
flatten [4,4096] -> F32
rsqrt(mean(x^2)+1e-6)
24 learned mixes
pre/post sigmoid transforms
4x4 row-softmax + 20-iteration Sinkhorn order
hc_pre weighted collapse
hc_post comb[source,output] residual expansion
```

Scalar C result:

```text
hc_pre:  exact BF16 equality at all 4,096 outputs
hc_post: exact BF16 equality at all 16,384 outputs
mixes/pre/post/comb diagnostic max_abs: 4.76837158e-07
```

Final real combination matrix columns sum to approximately one; final rows intentionally do not, because the pinned algorithm ends with column normalization. A separate non-symmetric model-free fixture pins `comb[source,output]` orientation.

See `MHC.md` for hashes and exact diagnostics.

#### Other V3 primitive seams

Routing primitive semantics are now exercised as part of passed Gate F/V4. Still isolate as attention bring-up requires them:

- normalization outside the mHC-specific RMS calculation;
- RoPE + inverse RoPE;
- K64 KV QAT;
- attention projection/compression/indexer helpers.

### V4 — routing + MoE — **PASSED / Gate F**

Pinned real learned-router representative, layer 3:

```text
selected IDs = [2,29,225,220,108,69]
weights      = [0.263384104,0.251154065,0.248866215,
                0.247819692,0.244902447,0.243873596]
top-k boundary margin = 0.00040531158447265625
```

Pinned bootstrap/hash representative, layer 0 token 4242:

```text
selected IDs = [150,142,245,248,174,119]
weights      = [0.248431414,0.267507851,0.247496858,
                0.241705239,0.246138424,0.248720214]
```

The learned and hash scalar paths reproduce exact IDs/order. Float routing diagnostics have observed `max_abs = 9.53674316e-07` while retaining exact discrete decisions.

#### Routed FP4 expert

Frozen representative:

```text
layer 3 / expert 2
route weight = 0.2633841037750244
```

Real full `w1`/`w3` packed FP4 bytes + E8M0 scales and real first-eight `w2` rows replay through scalar C with exact gate/up diagnostics, exact hidden BF16, and exact output BF16:

```text
b96c b83c 39bf ba1c b988 3a81 389d 3a46
```

#### All six selected routed branches + shared resident FP8 expert

Fixture:

```text
tests/fixtures/deepseek_v4/v4_moe_real/
```

All six selected routed experts were fetched from the pinned checkpoint and independently evaluated. To avoid freezing six redundant 12.75 MiB expert records, the extra five retain compact BF16 output slices plus source hashes; expert 2 remains the full permanent routed fixture.

The shared expert is a distinct resident arithmetic path, so its full real `w1`/`w3` FP8 E4M3 weights + E8M0 scales and the first eight `w2` rows are frozen permanently. Scalar C reproduces its hidden and output BF16 exactly.

Shared output first eight:

```text
3a2f ba40 b9b8 bad6 bb24 3a2e ba32 ba3e
```

#### Official combination order

Pinned source semantics are tested explicitly:

1. routed expert returns are BF16 values;
2. selected branches accumulate into f32 in **ascending expert ID** because the source loops over expert IDs, not router top-k slots;
3. the BF16 shared expert output is added once after routed accumulation;
4. the result is finally cast to BF16.

For the real representative the ascending-ID order is:

```text
[2,29,69,108,220,225]
```

The real top-k-versus-ID order happens to round to the same final BF16 vector. Therefore a separate model-free non-associativity mutation pins the ordering: IDs `[5,1,3]` with routed values `[1,2^30,-2^30]` yield `1` under official expert-ID order but `0` under top-k slot order in f32.

Final complete real MoE first-eight output:

```text
b848 ba7a 3b1a bb78 bbb7 3ab7 ba25 3982
```

The successful branch gate also passed the inherited A–D suite and ASan/UBSan. The frozen Gate F fixtures are replayed by ordinary `make check`, not only by the temporary acquisition workflow.

Since 2026-08-09 that replay is also *legible*: `tests/run.sh` invokes each Gate A–F replay as its own named line under "DeepSeek gate replays". Previously they were imported transitively from `tests/test_inventory.py`, so all eleven were reported as a single "inventory" line and a corrupted Gate F fixture failed under the tensor classifier's name. `docs/EXPERIMENTS.md` entry 6 records the mutation evidence. When a V5 attention fixture is frozen it needs its own line there — the list is enumerated, not globbed.

**Evidence boundary:** Gate F proves routing and MoE arithmetic from direct real checkpoint bytes. It does not yet prove that a converted WASTE expert record/cache supplies identical bytes. That placement identity is canonical **Gate G**, and must be proven when the DeepSeek container/storage path lands.

### V5 — attention by type — **NEXT / Gate E**

Create distinct fixtures for every proven attention mode. Operational order:

1. ratio-0 sliding-window-only layer — projections, normalization, K64 KV QAT, RoPE, causal 128-window, sink-softmax sparse attention, inverse RoPE, output seam;
2. ratio-128 compressed-history attention;
3. ratio-4 CSA compressor + indexer/top-k sparse attention.

Gate E passes only when all structurally distinct modes are independently checkpoint/source verified.

### V6 — complete transformer layer — **Gate H**

Compare layer input, attention/mHC merge, routing/MoE output and final layer output. Cover structurally distinct layer classes.

### V7 — multi-layer localization

Checkpoint hidden states across multiple layers and stop at the first divergent layer.

### V8 — final hidden/logits — **Gate I**

Compare final hidden state, norm and logits. No model-correctness claim before this level.

### V9 — deterministic greedy generation — **Gate K**

With identical known-token inputs/stopping rules and sampling disabled, compare generated token IDs. Divergence returns to V8 diagnostics.

### V10 — encoder/parser — **Gate J**

Port the official code-based encoding/parser semantics with official differential fixtures. Do not replace it with guessed Jinja behavior.

### V11 — OpenAI-compatible API

Test direct-C versus API generation parity, streaming/non-streaming behavior, cancellation and only the structured/tool semantics supported by the official encoder.

---

## 4. Canonical README gate concordance

| README gate | Operational owner | State / phase |
|---|---|---|
| **A** checkpoint inventory | **V0** | **PASSED** |
| **B** native quantization | **V1** | **PASSED** |
| **C** quantized trunk linear | **V2** | **PASSED scalar/model-semantic** |
| **D** mHC | **V3 mHC seam** | **PASSED scalar/model-semantic** |
| **E** attention by type | **V5** | **NEXT — ratio 0 first** |
| **F** routing + one MoE block | **V4** | **PASSED scalar/model-semantic** |
| **G** disk/cache identity | systems correctness | streaming/container phase |
| **H** complete transformer block | **V6** | after E/V5 |
| **I** 43-layer base forward/logits | **V8** | base-model bring-up |
| **J** tokenizer/encoding | **V10** | encoding/API phase |
| **K** generation | **V9** | after logits |
| **L** real storage | performance feasibility | after container/storage path |
| **M** cache curve | performance feasibility | after real routing trace |
| **N** DSpark | speculative correctness/perf | after base model |
| — | **V7** multi-layer localization | base-model bring-up |
| — | **V11** API parity | API phase |

README letters remain stable design identifiers even where operational V-order differs.

---

## 5. Cache/I/O correctness — Gate G

For identical token/container inputs, these pairs must preserve selected experts and numerical output:

| A | B |
|---|---|
| direct checkpoint/reference bytes | converted WASTE record bytes |
| cache disabled | cache enabled |
| direct I/O | fallback/page-cache path |
| cold cache | preloaded/hot cache |
| prefetch off | prefetch on |
| sequential prefill | chunked prefill |

Storage placement is never allowed to change model meaning. Gate F's direct-byte arithmetic fixture is the oracle Gate G should reuse for the first converted expert-record identity test.

---

## 6. Failure triage order

When downstream outputs differ, diagnose:

1. prompt/token IDs;
2. tensor binding/name/shape/transpose;
3. quantized storage/scale/indexing;
4. activation quantization / primitive arithmetic;
5. mHC/residual order;
6. attention state/position/compression/indexing;
7. router IDs/order/weights;
8. expert identity/data;
9. layer-state persistence;
10. final norm/head.

Do not start by changing tolerances.

---

## 7. Optimization acceptance rule

An optimized path lands only when:

- the proven scalar/reference seam remains available;
- it passes the same frozen independent fixtures;
- known-wrong mutations remain rejected;
- placement/cache changes preserve Gate G;
- benchmark hardware/configuration is recorded;
- memory remains within the planner;
- optional backend failure/fallback is safe.

Gates C, D, and F now give later SIMD/backend work real checkpoint-backed arithmetic oracles. Gate E attention correctness is the next base-model priority before broad optimization.

---

## 8. Result recording template

```text
Date:
Port commit:
Model revision:
Reference source path/hash:
Checkpoint tensor/shard/range/hash:
Hardware / OS:
README gate:
Operational V-level:
Operation:
Fixture provenance:
max_abs / max_rel / RMS (when applicable):
exact semantic checks:
known-wrong mutations tested:
Verdict:
Evidence boundary / non-claims:
Notes:
```

A naked statement like “matches PyTorch” is not sufficient evidence.
