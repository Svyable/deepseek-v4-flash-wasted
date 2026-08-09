# Validation — correctness gates for the DeepSeek V4 port

**Status: Gate A/V0 is CHECKPOINT-VERIFIED on all 48 pinned 0731 shard headers. Gate B/V1 native-format/storage semantics are passed for the pinned release. Gate C/V2 is now passed at the scalar/model-semantic level on a frozen real checkpoint projection: independent source oracle and scalar C agree exactly at BF16 for eight outputs. The next open arithmetic gate is Gate D/V3 (mHC/model primitives).**

Pinned model:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

Validation proceeds from the smallest exact seam to final logits. A later gate may rely on an earlier one only at the evidence level that earlier gate actually proved.

See:

- `INVENTORY-0731.md` — real Gate A checkpoint result;
- `NUMERICS.md` — Gates B/C arithmetic and real projection fixture;
- `FIXTURES.md` — independence/mutation policy;
- `OFFICIAL-0731-SOURCE.md` — pinned source findings;
- `REFERENCE_ACCESS.md` — bounded official-artifact acquisition.

---

## 1. Oracle and evidence policy

The numerical/model authority is the pinned official release plus the exact checkpoint bytes.

Different facts come from different evidence:

- public format spec → E2M1/E8M0/E4M3FN code semantics;
- pinned official source → DeepSeek operation semantics;
- real safetensors headers → exported names, dtypes, shapes, offsets and bytes;
- bounded real payload slices → actual quantized tensor bytes;
- independent source oracle → expected model-semantic outputs;
- end-to-end port → final model correctness/performance.

Third-party runtimes are smoke/reference aids, not numerical authority.

Every frozen official fixture records at least:

```text
model repository + immutable revision
source operation/path
exact tensor/shard identity
raw-input hashes
output hashes
shape/dtype
fixture generator/port commit
reference device/backend when one was actually used
```

Do not describe a source-equation CPU oracle as an official GPU-kernel execution. Conversely, do not require a GPU merely to prove byte layout and model-semantic scalar algebra when an independent exact fixture can settle them.

### Fixture independence

Expected values must not be generated through the WASTE implementation under test.

Round trips prove self-consistency. Silent conventions require independent literals, checkpoint bytes or independently evaluated official-source equations. Where practical, mutate nibble order, scale direction, block indexing, rounding or reduction order and ensure the fixture rejects the wrong implementation.

---

## 2. Tolerance policy

Use exact equality for discrete semantics and exact-format fixtures when possible:

- tensor names/shapes/dtypes;
- expert IDs/order;
- token IDs;
- nibble/block selection;
- raw payload hashes;
- BF16 expected bits when the fixture is reduction-stable.

Floating-point tolerances are introduced only after independent evidence shows unavoidable backend/dtype reduction differences. Record `max_abs`, `max_rel`, RMS where useful, and keep a known-wrong mutation outside the threshold.

Never loosen a tolerance simply to make a new path pass.

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

Checkpoint-resident bootstrap maps are confirmed in layers 0–2 as `ffn.gate.tid2eid`, I64 `[129280,6]`.

Any model-revision move reruns V0.

### V1 — native quantization decode/conventions — **PASSED / Gate B**

Evidence includes:

- exhaustive E2M1/E8M0/E4M3FN public-format tests;
- FP4 K32 and FP8 128x128 scale-index tests;
- pinned official low-nibble-first source semantics;
- pinned weight-scale multiplication semantics;
- real checkpoint routed `I8` packed-FP4 + `F8_E8M0` shapes;
- real checkpoint resident `F8_E4M3` + `F8_E8M0` layout;
- independent F3 literal fixture (`0x21`, scale `0x80` → `[1,2]`).

Optimized native kernels must continue matching this scalar/evidence contract.

### V2 — one quantized resident projection — **PASSED / Gate C**

Real target:

```text
layers.0.attn.wq_a.weight  F8_E4M3 [1024,4096]
layers.0.attn.wq_a.scale   F8_E8M0 [8,32]
model-00002-of-00048.safetensors
```

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v2_wq_a_real/
```

It contains eight real weight rows (32,768 B), the corresponding real 32-byte scale row, deterministic BF16 input, and an eight-element BF16 expected output. Real payload is fetched with exact HTTP Range offsets and SHA-256 provenance.

The independent Python oracle implements pinned source algebra only:

```text
K128 activation block
  -> max(amax,1e-4)
  -> next-power-of-two scale of amax/448
  -> E4M3FN activation quantization
  -> FP8 K128 dot with real checkpoint weight
  -> activation_scale * weight_scale
  -> FP32 accumulation
  -> BF16 output
```

The fixture generator requires identical final BF16 under sequential f32, reverse f32 and exact dyadic accumulation. Scalar C then matches all eight BF16 outputs exactly:

```text
0x3e79 0xbf84 0x3f8d 0x400a 0x3ff3 0xbf9b 0x3f82 0x3ff0
```

Validation run `31286320991`:

```text
make check: 32 passed, 0 failed, 13 skipped
real Gate C replay: PASS exact BF16
make asan: 31 passed, 0 failed, 14 skipped
```

Evidence boundary: this passes the scalar/model-semantic Gate C seam. The official TileLang GPU kernel was not executed on the ordinary GitHub runner. Future optimized CPU/SIMD/GPU paths must match this frozen fixture and may need backend-specific numerical tolerances.

### V3 — model primitives / mHC — **NEXT / Gate D**

Bring up primitives independently before a transformer block. Required seams include:

- mHC Sinkhorn/manifold mixing and residual transformation;
- normalization;
- RoPE;
- SwiGLU/clamp behavior;
- router score transform before full MoE;
- attention projection substeps as independent arithmetic helpers.

Gate D should start with a small independent official-source fixture for the exact mHC operation, preferably including bounded real checkpoint parameters after Gate A has identified their names.

### V4 — one MoE block — **Gate F**

Fixture should contain:

```text
input hidden state
router raw/transformed scores
selected expert IDs/order
routing weights
shared-expert output
selected routed-expert outputs
combined MoE result
```

Test expert bytes both directly and through WASTE's storage/cache abstraction. Placement may change latency, never arithmetic.

### V5 — one attention block — **Gate E**

Create distinct fixtures for every proven attention mode, including sliding-window-only and compressed/indexed variants. Capture enough intermediates to localize projection, compression, index selection, position and accumulation errors.

### V6 — complete transformer layer — **Gate H**

Compare layer input, attention/mHC merge, routing/MoE output and final layer output. Cover each structurally distinct layer class.

### V7 — multi-layer localization

Checkpoint hidden states across multiple layers. Stop at the first divergent layer.

### V8 — final hidden/logits — **Gate I**

Compare final hidden state, norm and logits. Record numerical errors and top-N ordering. No model-correctness claim before this gate.

### V9 — deterministic greedy generation — **Gate K**

With identical token inputs/stopping rules and sampling disabled, compare generated token IDs. Divergence returns to V8 diagnostics.

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
| **D** mHC | **V3** | **NEXT** |
| **E** attention by type | **V5** | later base-model bring-up |
| **F** routing + one MoE block | **V4** | after primitive bring-up |
| **G** disk/cache identity | systems correctness | streaming phase |
| **H** complete transformer block | **V6** | base-model bring-up |
| **I** 43-layer base forward/logits | **V8** | base-model bring-up |
| **J** tokenizer/encoding | **V10** | encoding/API phase |
| **K** generation | **V9** | after logits |
| **L** real storage | performance feasibility | after container/storage path |
| **M** cache curve | performance feasibility | after real routing trace |
| **N** DSpark | speculative correctness/perf | after base model |
| — | **V7** multi-layer localization | base-model bring-up |
| — | **V11** API parity | API phase |

README letters remain stable design identifiers even where V-order differs (F→V4 before E→V5; K→V9 before J→V10 for raw known-token arithmetic validation).

---

## 5. Cache/I/O correctness — Gate G

For identical token/container inputs, these pairs must preserve selected experts and numerical output:

| A | B |
|---|---|
| cache disabled | cache enabled |
| direct I/O | fallback/page-cache path |
| cold cache | preloaded/hot cache |
| prefetch off | prefetch on |
| sequential prefill | chunked prefill |

Storage placement is never allowed to change model meaning.

---

## 6. Failure triage order

When downstream outputs differ, diagnose in this order:

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
- hardware/configuration is recorded for performance claims;
- memory stays within the planner;
- optional backends fail/fallback safely.

Gate C now gives SIMD/backend work a real checkpoint-backed arithmetic oracle, but Gate D/V3 model semantics remain the next correctness priority before broad optimization.

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
