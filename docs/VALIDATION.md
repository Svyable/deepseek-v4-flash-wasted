# Validation — correctness gates for the DeepSeek V4 port

**Status: Gates A/V0, B/V1, C/V2, README Gate D's mHC seam, Gate E/V5 attention-by-type, and Gate F/V4 are passed at their stated scalar/model-semantic evidence levels. Gate E includes ratio-0, shared grouped output, coherent ratio-128, and coherent ratio-4 CSA/indexer checkpoint evidence. Gate H/V6 is PARTIAL: layer-3 block wiring, the real HC composition, and the attention half composed for real (residual → hc_pre → real 64-head attention → hc_post) are passed; the FFN half still runs a stub, so the real router/MoE composed at the ffn_pre state is what remains. Gate G still owns converted-record/cache identity.**

Pinned model:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

Validation moves from the smallest exact seam to final logits. A later gate may rely on an earlier one only at the evidence level actually proved.

Primary evidence owners:

- `INVENTORY-0731.md` — Gate A checkpoint truth;
- `NUMERICS.md` — Gates B/C quantization and projection;
- `MHC.md` — Gate D Hyper-Connection evidence;
- `ATTENTION.md` — Gate E attention-mode bring-up;
- `ROADMAP.md` — passed/next schedule;
- `FIXTURES.md` — independence/mutation policy;
- `OFFICIAL-0731-SOURCE.md` — pinned source findings;
- `REFERENCE_ACCESS.md` — bounded artifact acquisition.

---

## 1. Oracle and evidence policy

The numerical/model authority is the pinned official release plus exact checkpoint bytes.

Different facts come from different evidence:

- public specification → E2M1/E8M0/E4M3FN semantics;
- pinned official source → DeepSeek operation/cast/order semantics;
- real safetensors headers → exported names, dtypes, shapes, offsets and bytes;
- bounded real payload slices → checkpoint parameter bytes;
- independent source oracle → expected model-semantic outputs;
- end-to-end WASTE port → final model correctness and performance.

Do not call a source-equation CPU oracle an official GPU-kernel execution. Conversely, a GPU is not required to establish byte layout or scalar model semantics when an independent checkpoint/source fixture can prove them.

### Fixture independence

Expected values may not be produced by the WASTE implementation under test.

Round trips establish self-consistency, not external correctness. Silent semantics require independent literals, real checkpoint bytes, or independently evaluated source equations. When possible, each high-risk seam includes a known-wrong mutation/orientation case that must fail.

Examples already enforced:

- FP4 nibble order is pinned by a literal byte, not the runtime pack macro;
- Gate F's fast standalone MoE oracle is cross-anchored to the earlier exact Fraction expert fixture;
- Gate E's standalone ratio-0 oracle must reproduce Gate C's existing `wq_a` BF16 output before larger attention expectations are accepted;
- grouped output projection requires sequential/reverse/exact reductions to agree at BF16 before freezing expected values.

---

## 2. Tolerance policy

Prefer exact equality whenever the fixture supports it:

- tensor names/shapes/dtypes;
- token/expert IDs and ordering;
- nibble/block/group selection;
- payload hashes;
- BF16 bits when reduction-order stability is demonstrated.

Introduce a tolerance only after independent evidence demonstrates an unavoidable backend/reduction difference. Record `max_abs`, `max_rel`, and RMS where useful; keep a known-wrong mutation outside the accepted threshold.

Never loosen a tolerance simply to make a new path pass.

---

## 3. Operational validation ladder

### V0 — checkpoint inventory — **PASSED / Gate A**

Pinned result:

```text
48 / 48 shards
72,317 tensors
166,878,536,440 payload bytes = 155.417748 GiB
0 unexplained main tensors / 0 unexplained bytes
11,008 routed expert records
13,369,344 B / routed expert record
6 PASS, 0 FAIL, 0 SKIP
```

Bootstrap maps exist in layers 0–2 as `ffn.gate.tid2eid`, I64 `[129280,6]`.

Any model-revision move reruns V0.

### V1 — native quantization — **PASSED / Gate B**

Evidence includes:

- exhaustive E2M1/E8M0/E4M3FN tests;
- FP4 K32 / FP8 128×128 scale-index tests;
- official low-nibble-first semantics;
- official scale multiplication;
- real routed `I8` packed-FP4 + `F8_E8M0` geometry;
- real resident `F8_E4M3` + `F8_E8M0` geometry;
- independent literal `0x21`, scale `0x80` → `[1,2]`.

### V2 — real resident quantized projection — **PASSED / Gate C**

Real target:

```text
layers.0.attn.wq_a.weight  F8_E4M3 [1024,4096]
layers.0.attn.wq_a.scale   F8_E8M0 [8,32]
```

Frozen fixture `tests/fixtures/deepseek_v4/v2_wq_a_real/` matches exact BF16:

```text
3e79 bf84 3f8d 400a 3ff3 bf9b 3f82 3ff0
```

This passes the scalar/model-semantic projection seam, not official accelerator bit parity.

### V3 — model primitives

#### README Gate D — mHC — **PASSED**

Frozen real layer-0 fixture:

```text
tests/fixtures/deepseek_v4/v3_mhc_real/
```

Real parameters:

```text
layers.0.hc_attn_fn     F32 [24,16384]
layers.0.hc_attn_base   F32 [24]
layers.0.hc_attn_scale  F32 [3]
```

The independent oracle covers learned mixes, pre/post transforms, 4×4 row softmax, 20 Sinkhorn iterations, `hc_pre`, and `hc_post` orientation.

```text
hc_pre  exact BF16 across 4,096 outputs
hc_post exact BF16 across 16,384 outputs
mix/pre/post/comb diagnostic max_abs 4.76837158e-07
```

See `MHC.md`.

Other primitive seams are validated inside the attention/MoE gates where appropriate, but remain individually diagnosable: learned RMSNorm, direct per-head Q normalization, RoPE, K64 KV QAT, compressor/indexer primitives.

### V4 — routing + MoE — **PASSED / Gate F**

Learned layer-3 representative:

```text
IDs     [2,29,225,220,108,69]
weights [0.263384104,0.251154065,0.248866215,
         0.247819692,0.244902447,0.243873596]
top-k boundary margin 0.00040531158447265625
```

Hash layer-0 token 4242:

```text
IDs [150,142,245,248,174,119]
```

Real routed expert 3/2 exact BF16 output:

```text
b96c b83c 39bf ba1c b988 3a81 389d 3a46
```

Real shared FP8 expert first-eight output:

```text
3a2f ba40 b9b8 bad6 bb24 3a2e ba32 ba3e
```

All six selected routed branches are independently evaluated. Official accumulation is ascending expert ID in f32, then shared BF16 add, then final BF16 cast.

Complete real first-eight output:

```text
b848 ba7a 3b1a bb78 bbb7 3ab7 ba25 3982
```

A model-free non-associativity fixture pins expert-ID ordering even though the selected real case hides that ordering after final BF16 rounding.

**Boundary:** direct checkpoint-byte MoE arithmetic is passed. Converted WASTE record/cache identity is Gate G.

Since 2026-08-09 the replay is also *legible*: `tests/run.sh` invokes each Gate A–F replay as its own named line under "DeepSeek gate replays", and Gate E's three joined them on merge. Previously every replay was imported transitively from `tests/test_inventory.py`, so the whole ladder was reported as a single "inventory" line and a corrupted Gate F fixture failed under the tensor classifier's name. `docs/EXPERIMENTS.md` entry 6 records the mutation evidence. The list is enumerated, not globbed: a ratio-128 or CSA fixture is replayed when its line is added.

### V5 — attention by type — **PASSED / Gate E**

Gate E has three structurally distinct attention modes plus a shared output projection. All are now independently checkpoint/source proved at the stated scalar/model-semantic evidence level.

#### Ratio 0 — attention core — **PASSED sub-seam**

Frozen real fixture:

```text
tests/fixtures/deepseek_v4/v5_attn_ratio0_real/
```

Scope: layer 0, two tokens × two heads, through inverse-RoPE sparse-attention output.

The fixture proves:

- `wq_a` / learned `q_norm`;
- selected real `wq_b` heads;
- distinct direct BF16 per-head Q normalization;
- base RoPE on last 64 dimensions;
- `wkv` / learned `kv_norm`;
- K64 E4M3 QAT simulation on non-RoPE 448 KV dimensions;
- causal window-128 indices;
- sink-softmax sparse attention with 64-position online-softmax blocks;
- inverse RoPE on the attention output.

Exact comparisons:

```text
post-RoPE Q                 [2,2,512] 2,048 BF16 values
post-K64-QAT KV             [2,512]   1,024 BF16 values
post-inverse-RoPE attention [2,2,512] 2,048 BF16 values
TOTAL                                    5,120 exact BF16 values
```

Representative signatures:

```text
Q pos0/head0
bf2b 3f58 bfb6 beba bee2 bfaa be58 3da2

Q pos1/head0 RoPE tail
402f 3fb0 3fac bf83 bf8b 3fc3 bf46 bd40

attention pos1/head0
be21 3e02 be52 be33 3de1 3f07 3e54 3e27
```

The expected-value producer is standalone and must reproduce Gate C's prior token-0 `wq_a` output before the attention expectation is accepted.

#### Shared grouped output projection — **PASSED sub-seam**

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v5_attn_output_group0_real/
```

The official converter dequantizes checkpoint `wo_a` E4M3×E8M0 to BF16 before the grouped einsum. `wo_b` remains quantized.

A sparse structural 8-head group seeded from real ratio-0 heads 0/1 proves:

- `[8 heads,512] -> group 4096` orientation;
- real checkpoint `wo_a` group-0 FP8→BF16 block dequantization;
- `[1024,4096]` group einsum;
- `[8 groups,1024] -> 8192` group placement/flattening;
- real checkpoint quantized `wo_b` rows;
- group-index mutation changes output as required.

Independent equations require sequential/reverse/exact reduction agreement at BF16.

```text
group latent first8
3a65 3dcb 3d2b 3d09 3cba bc02 bcab 3d9e

wo_b output first8
ba34 bce1 bd35 3bd0 3d87 3d77 bc85 bd46
```

**Boundary:** heads 2–7 in this projection fixture are deterministic structural transforms seeded from real heads 0/1, not independently generated checkpoint attention heads. This proves the shared projection structure/arithmetic, not a full 64-head layer output.

#### Ratio 128 compressor — **PASSED real-checkpoint sub-seam**

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v5_compressor_ratio128_real/
```

Pinned real checkpoint tensors are BF16 `wkv/wgate`, F32 APE, and **BF16** compressor norm storage. The reference implementation upcasts RMSNorm parameters to f32 after load; the fixture preserves checkpoint BF16 bytes and performs that upcast only in the source-equation oracle/runtime semantics.

Two reduction-safe 128-token structural chunks match exactly at pooled-before-norm, post-norm, post-YaRN-RoPE and post-K64-QAT compressed KV: **4,096 exact BF16 values**. The compressed-KV SHA-256 is `61dba51c0f59a6f0b93fbed1d0416e5be072f86a3f84da3725ebb18a04419fe5`.

The pair-20 YaRN mutation remains required: pair 0 is invariant to the parameters this check is meant to witness.

#### Ratio 128 compressed-history composition — **PASSED checkpoint-derived sub-seam**

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v5_attention_history_ratio128_real/
```

For layer 3 / prefill row 255 it derives a checkpoint-real head-0 Q, all 256 checkpoint-real local KV entries, and a real `attn_sink`, then composes them with the independently frozen real compressor KV. The source namespace is pinned exactly:

```text
local visible      128..255
compressed visible 256,257
combined top-k     130 entries
```

The C replay matches **130/130 indices and 512/512 BF16 attention values exactly**. Mutations that drop compressed history or use the window-size prefill offset instead of full `seqlen` fail.

**Boundary:** the compressed KV and Q/local-KV fixtures use different structural input sequences. This proves checkpoint-derived cache/index/sparse-attention composition, not one coherent same-input `Attention.forward`.

#### Ratio 128 coherent forward — **PASSED real-checkpoint sub-seam**

Frozen fixture: `tests/fixtures/deepseek_v4/v5_attention_ratio128_coherent_real/`. A single 256-token structural input has only positions 0 and 255 non-zero. The same input drives real layer-3 head-0 Q, local KV, both ratio-128 compressor chunks, row-255 local+compressed sparse attention, and inverse compressed YaRN.

The offline C replay regenerates every stage from the frozen raw checkpoint slices and the already-proven compressor payload. The row-255 namespace is 130 exact indices (`128..255,256,257`). Pre-inverse and post-inverse attention are each 512 exact BF16 values. Pair 20 changes from `3abf 3b49` to `3add 3b41`; replacing coherent compressed KV with the older unrelated compressor fixture also changes attention.

**Boundary:** one head / one query row only. The shared grouped output projection is already proved separately; ratio-4 CSA/indexer remains the final Gate-E attention mode.

#### Ratio 4 / CSA — **PASSED real-checkpoint sub-seams**

Frozen official layer-2 header inventory: `reference/deepseek-v4-flash-0731.layer2-csa.json`. It pins the separate 128-wide Indexer compressor, 64x128 Indexer query path, BF16 `weights_proj`, and 512-wide main overlapping compressor before arithmetic.

Model-free fixtures pin normalized Walsh-Hadamard, K32 E2M1 FP4 simulation, previous-FIRST/current-SECOND overlap, BF16 score/ReLU/weight boundaries, causal masking before top-k, and a genuine 520-candidate -> 512 cutoff.

Real Indexer fixture: `tests/fixtures/deepseek_v4/v5_csa_indexer_real/`. The same 8-token sparse input uses positions 3 and 7 so chunk-1 overlap is load-bearing. All 8,192 real Indexer `wq_b` outputs pass forward/reverse reduction stability. Real BF16 scores are `bc8e` and `bc63`, yielding compressed selection `[9,8]` after offset 8.

Coherent main fixture: `tests/fixtures/deepseek_v4/v5_csa_attention_real/`. The frozen real Indexer selection is consumed as a dependency; the main head-0 Q/local-KV path and 512-wide overlapping compressor are independently replayed from the same positions 3/7 input, followed by selected sink sparse attention and inverse compressed YaRN. Mutations that ignore overlap, drop Indexer-selected compressed entries, or omit inverse YaRN fail.

**Boundary:** one main attention head / one query row. Shared grouped output projection is independently proved. Gate H/V6 now owns complete-layer composition.

See `ATTENTION.md` for detailed cast boundaries and fixture provenance.

### V6 — complete transformer layer — **Gate H** — **PARTIAL**

The target: compare layer input, attention/mHC merge, routing/MoE output and final layer output for every structurally distinct layer class.

#### Block wiring — **PASSED, model-free**

The layer-3 composition order is pinned as `hc_pre(attn) -> branch -> hc_post -> hc_pre(ffn) -> branch -> hc_post`, with the attention and FFN HyperConnection parameter sets distinct.

Five wirings are refused, each chosen because it produces plausible numbers rather than an error:

```text
reuse the original residual for the FFN hc_pre        max_abs 0.760
swap the attention and FFN HC parameter sets          max_abs 0.403
feed a correct branch output from the wrong input     max_abs 0.203
replace the FFN hc_post with an ordinary residual add max_abs 0.193
replace the attention hc_post the same way            max_abs 0.654
```

Each is measured against the independent oracle before the C library is involved, so a vacuous construction cannot pass. The last two matter most: the natural wrong implementation of "drop the hc_post transition" is not omitting a step — the shapes would not line up — but writing `out[k] = residual[k] + branch`, the ordinary transformer residual add that mHC exists to replace. That compiles and runs. It was named in the gate's contract but not enforced until `EXPERIMENTS.md` entry 8.

#### Real layer-3 HC composition — **PASSED sub-seam**

`tests/fixtures/deepseek_v4/v6_hc_composition_real/` replays the composition with real layer-3 `hc_attn_*` and `hc_ffn_*` parameters, SHA-pinned per file, exact at BF16 across attention hc_pre, attention hc_post, FFN hc_pre and the final state.

**Evidence boundary — both branches are stubs in this fixture.** It proves the *wiring and the HyperConnection arithmetic* with real parameters; it does not run real attention or real MoE inside the composition. The frozen files say so by name (`attn-stub-branch.bf16.bin`, `ffn-stub-branch.bf16.bin`), and the test separately pins that each stub is bound to the frozen hc_pre state so a stub that ignored its input would fail.

#### Real attention composed through the transition — **PASSED sub-seam**

`tests/test_v6_attention_composition_real.py` replaces the attention stub with the real 64-head branch from `v6_attention_branch_real`, giving the attention half of the layer end to end from checkpoint bytes:

```text
residual -> hc_pre(hc_attn_*) -> real 64-head attention -> hc_post
```

Chaining two separately frozen fixtures is only legitimate if the second was produced from the first's state, so that is **checked, not assumed**: the attention fixture records the producing file and its SHA-256, the test recomputes that digest, and it also re-derives `attn_pre` through `hc_pre` and requires the frozen BF16 state back. Feeding a correct branch output produced from the wrong branch input is a wiring error this gate names explicitly; hash equality plus re-derivation is what rules it out.

`post`/`comb` come from an independent Sinkhorn rather than from the `hc_pre` under test. That distinction is load-bearing — with them read out of `hc_pre`, a fault inside the Sinkhorn normalization cancels on both sides of the comparison and the test cannot see it. Measured against the C:

```text
hc_post f32 max_abs vs independent oracle       2.38418579e-07
hc_pre post/comb max_abs vs independent Sinkhorn 7.13651154e-08
composed after-attn state                        exact at BF16, 16,384 values
```

Refused: the stub branch in place of the real one, the FFN HC `post`/`comb` on the attention transition, a plain residual add instead of `hc_post`, and a composed state equal to the frozen stub composition. Four `src/deepseek_v4_mhc_ref.c` mutations die against it — `comb` transposed inside `hc_post`, `comb` transposed inside the Sinkhorn normalization, the `post[]` branch scaling dropped, and the residual indexed by `k` instead of `j`.

#### What Gate H still needs

- the real router + routed/shared MoE composed in place of the FFN stub, at the `ffn_pre` state — this needs expert weights selected by the router at that state, so it needs checkpoint access;
- both halves in one composition, giving a complete layer-3 state transition replayable offline, under the normal and ASan/UBSan suites.

Gate H does not close, and no full-layer claim is made, until those hold together.

### V7 — multi-layer localization

Checkpoint hidden states across multiple layers and stop at the first divergent layer.

### V8 — final hidden/logits — **Gate I**

Compare final hidden state, final norm and logits. No model-correctness claim before this level.

### V9 — deterministic greedy generation — **Gate K**

With identical known-token inputs/stopping rules and sampling disabled, compare generated token IDs. Divergence returns to V8 diagnostics.

### V10 — encoder/parser — **Gate J**

Port official code-based encoding/parser semantics with official differential fixtures. Do not replace them with guessed Jinja behavior.

### V11 — OpenAI-compatible API

Test direct-C versus API generation parity, streaming/non-streaming behavior, cancellation and only structured/tool behavior supported by the official encoder.

---

## 4. Canonical README gate concordance

| README gate | Operational owner | State / phase |
|---|---|---|
| **A** checkpoint inventory | **V0** | **PASSED** |
| **B** native quantization | **V1** | **PASSED** |
| **C** quantized trunk linear | **V2** | **PASSED scalar/model-semantic** |
| **D** mHC | **V3 mHC seam** | **PASSED scalar/model-semantic** |
| **E** attention by type | **V5** | **PASSED scalar/model-semantic — ratio 0 + output + coherent ratio-128 + coherent ratio-4 CSA checkpoint-passed** |
| **F** routing + one MoE block | **V4** | **PASSED scalar/model-semantic** |
| **G** disk/cache identity | systems correctness | streaming/container phase |
| **H** complete transformer block | **V6** | **PARTIAL** — wiring, real HC composition and real attention half done; MoE half stubbed |
| **I** 43-layer base forward/logits | **V8** | base-model bring-up |
| **J** tokenizer/encoding | **V10** | encoding/API phase |
| **K** generation | **V9** | after logits |
| **L** real storage | performance feasibility | after container/storage path |
| **M** cache curve | performance feasibility | after real routing trace |
| **N** DSpark | speculative correctness/perf | after base model |
| — | **V7** multi-layer localization | base-model bring-up |
| — | **V11** API parity | API phase |

README letters remain stable even where operational V-order differs.

---

## 5. Cache/I/O correctness — Gate G

For identical inputs/container bytes, these pairs must preserve selected experts and numerical output:

| A | B |
|---|---|
| direct checkpoint/reference bytes | converted WASTE record bytes |
| cache disabled | cache enabled |
| direct I/O | fallback/page-cache path |
| cold cache | preloaded/hot cache |
| prefetch off | prefetch on |
| sequential prefill | chunked prefill |

Storage placement is never allowed to change model meaning. Reuse Gates C/F fixtures as initial converted-byte identity oracles.

---

## 6. Failure triage order

When downstream outputs differ, diagnose:

1. prompt/token IDs;
2. tensor binding/name/shape/transpose;
3. quantized storage/scale/indexing;
4. activation quantization / primitive arithmetic;
5. mHC/residual ordering;
6. attention norm/RoPE/QAT/compression/indexing state;
7. router IDs/order/weights;
8. expert identity/data;
9. layer-state persistence;
10. final norm/head.

Do not begin by changing tolerances.

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

Gates C/D/F and the passed Gate-E sub-seams now provide real checkpoint-backed arithmetic oracles. Ratio-128 and CSA correctness remain higher priority than broad optimization.

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
max_abs / max_rel / RMS:
exact semantic checks:
known-wrong mutations tested:
Verdict:
Evidence boundary / non-claims:
Notes:
```

A naked statement like “matches PyTorch” is not sufficient evidence.
