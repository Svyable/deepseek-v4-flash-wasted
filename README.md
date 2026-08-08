# DeepSeek V4 Flash — WASTED

> **Research + implementation handoff for running `deepseek-ai/DeepSeek-V4-Flash-0731` with WASTE-style NVMe expert streaming.**
>
> Status: **design / bootstrap only — no inference code has been ported yet.** The first implementation should optimize for correctness and observability, not speed.

The goal of this repository is to adapt the core idea behind [sqliteai/waste](https://github.com/sqliteai/waste) — keep the dense/shared trunk resident, stream only the routed MoE experts from fast storage, overlap expert I/O with compute, and use all remaining RAM as a bounded expert cache — to **DeepSeek-V4-Flash-0731**.

This is a good architectural match: DeepSeek V4 Flash is an all-MoE model with **256 routed experts per layer and only 6 routed experts active per token**, plus one shared expert. But this is **not** a model-name swap in WASTE. WASTE's current forward path is Kimi-specific. DeepSeek V4 introduces a different trunk: manifold-constrained Hyper-Connections, hybrid compressed attention, hash-routed bootstrap layers, native FP4 routed experts, FP8 trunk weights, and an attached DSpark speculative-decoding module.

The implementation strategy is therefore:

1. **Reuse WASTE's proven generic systems pieces**: bounded expert cache, direct/aligned reads, one-record-per-expert layout, RAM planning, chunked prefill ideas, read-ahead discipline, CLI/library separation, validation gates, and instrumentation.
2. **Implement a DeepSeek-V4 model path against the official reference code**, rather than trying to bend Kimi KDA/MLA code into DeepSeek shapes.
3. **Preserve DeepSeek's native quantization first**. Do not requantize the routed experts to WASTE VQ3R until the native FP4 implementation is correct and measured.
4. **Bring up the 43-layer target model without DSpark first**, prove logits/generation, then add DSpark as an optional acceleration layer.

---

## 0. Pinned sources of truth

Do not implement from blog posts or memory. Keep these exact references open while coding.

### WASTE baseline

- Repository: <https://github.com/sqliteai/waste>
- Pinned research baseline used for this plan: [`d9b919a791148b571e643d0af666bf19b4d733ab`](https://github.com/sqliteai/waste/commit/d9b919a791148b571e643d0af666bf19b4d733ab)
- Engine design: <https://github.com/sqliteai/waste/blob/main/docs/ENGINE.md>
- Container format: <https://github.com/sqliteai/waste/blob/main/docs/FORMAT.md>
- Correctness/feasibility gates: <https://github.com/sqliteai/waste/blob/main/docs/GATES.md>
- Negative results / things not to repeat: <https://github.com/sqliteai/waste/blob/main/docs/LEARNED.md>
- Current model internals: <https://github.com/sqliteai/waste/blob/main/src/model.h>
- Current converter: <https://github.com/sqliteai/waste/blob/main/tools/convert.py>

WASTE is Apache-2.0. If source is imported or adapted, retain the required license/header/NOTICE attribution.

### DeepSeek V4 Flash 0731

- Model: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- Release revision shown by Hugging Face: `9e165c3`
- Config: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json>
- Official minimal inference model: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py>
- Official kernels: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/kernel.py>
- Official checkpoint converter: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/convert.py>
- Official encoding guide: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/README.md>
- Encoding reference implementation/tests: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding>
- Technical report: <https://arxiv.org/abs/2606.19348>

The DeepSeek repository and weights are MIT-licensed. Preserve DeepSeek notices for copied reference material.

**Pin revisions in every reproducible experiment.** The converter should record the resolved Hugging Face commit SHA and the WASTE source commit in the output manifest.

---

## 1. Target model snapshot

The July 31 release is `DeepSeek-V4-Flash-0731`, the official replacement for the preview Flash checkpoint. The package includes the DSpark speculative-decoding module. Hugging Face currently reports **304B parameters** for the packaged checkpoint; the original Flash backbone is documented as **284B total / ~13B activated**. Do not hard-code either total into the runtime — the tensor inventory is authoritative.

Core config values that matter to this port:

| Property | DeepSeek-V4-Flash-0731 |
|---|---:|
| main decoder layers | **43** |
| hidden size | **4096** |
| routed experts / layer | **256** |
| activated routed experts / token / layer | **6** |
| shared experts / layer | **1** |
| MoE intermediate size | **2048** |
| hash-routed bootstrap layers | **3** |
| attention heads | **64** |
| attention head dim | **512** |
| query LoRA rank | **1024** |
| output LoRA rank | **1024** |
| sliding window | **128** |
| max context | **1,048,576** |
| Hyper-Connection multiplier | **4** |
| CSA indexer heads | **64** |
| CSA indexer head dim | **128** |
| CSA selected compressed positions | **512** |
| routed expert checkpoint dtype | **FP4 E2M1** |
| expert FP4 scale | **UE8M0, one per 32 values along K** |
| most non-expert quantized weights | **FP8 E4M3, 128×128 blocks** |
| routing scoring | **sqrt(softplus(logit))** |
| routed scaling factor | **1.5** |
| SwiGLU clamp | **10.0** |
| DSpark target main layers | **40, 41, 42** |
| DSpark block size | **5** |
| DSpark Markov rank | **256** |

`compress_ratios` for the package are:

```text
0, 0,
4, 128, 4, 128, ... alternating ...,
4,
0, 0, 0
```

For the 43 main blocks this means two initial pure sliding-window layers followed by alternating compressed-attention layers. The trailing entries are associated with the attached speculative path; derive behavior from the official reference rather than inferring solely from this list.

### Why this is attractive for WASTE

A routed DeepSeek expert is a conventional SwiGLU triplet:

```text
w1: [2048, 4096]
w3: [2048, 4096]
w2: [4096, 2048]
```

That is **25,165,824 logical weights per expert**. The official reference stores routed-expert FP4 as two E2M1 values per byte plus one UE8M0 scale per 32 logical values along K.

First-order native-FP4 storage estimate:

```text
packed weight bytes                     12,582,912
per-K32 UE8M0 scale bytes                  786,432
-------------------------------------------------
approx bytes / expert before alignment  13,369,344  ≈ 12.75 MiB

main expert records = 43 × 256 = 11,008
routed experts touched / decode token = 43 × 6 = 258
cold routed bytes / decode token ≈ 3.45 GB ≈ 3.21 GiB
```

This is **only a design estimate**. It deliberately excludes record headers/alignment and says nothing about cache hit rate. `tools/inventory.py` must replace estimates with exact checkpoint-derived numbers before we make any RAM, disk-size, or tokens/sec claim.

The important takeaway is architectural: at top-6, a token needs a tiny fraction of each layer's 256 routed experts. That is exactly the access pattern WASTE was designed to exploit.

---

## 2. What to reuse from WASTE vs. what to rewrite

### Reuse / generalize

These are the valuable WASTE primitives:

- **4 KiB-aligned independently readable expert records**.
- **One coalesced read per expert** with `w1 + w3 + w2 + scales` adjacent.
- `pread`/platform direct-I/O path and page-cache bypass behavior.
- Bounded expert cache (`ecache`) with asynchronous reader threads.
- Exact cache-vs-disk equivalence: placement changes latency, never numerics.
- Cache hints/read-ahead without making speculative routing authoritative.
- RAM budget resolver and cgroup-aware usable-memory detection.
- Chunked prefill that loads a distinct expert once and reuses it for all routed tokens in the chunk.
- Per-token/per-layer I/O, cache-hit, and timing telemetry.
- Library-first design: public C API first; CLI is only a client.
- Container validation, record headers, optional checksums, corruption diagnostics.
- Model-free/synthetic tests plus Python-oracle differential tests.
- Experimental discipline in `GATES.md` / `LEARNED.md`.

### Rewrite / add for DeepSeek

Do **not** reuse these Kimi-specific pieces as if they were generic:

- Kimi Delta Attention (`kda.*`).
- Kimi MLA latent-cache math.
- K3 AttnRes / Kimi-specific residual path.
- Kimi tokenizer/chat assumptions.
- Kimi converter tensor-name mappings.
- Kimi routing function.
- Kimi-specific WASTE config fields as the primary model contract.
- Kimi VQ expert format as the initial DeepSeek storage format.

DeepSeek needs dedicated implementations for:

1. **Manifold-Constrained Hyper-Connections (mHC)**.
2. **DeepSeek V4 hybrid attention**:
   - pure sliding-window layers;
   - compressed sparse attention (CSA, ratio 4 + learned indexer);
   - heavily compressed attention (ratio 128, dense over compressed history).
3. **DeepSeek gate semantics**, including deterministic hash routing for the first three layers.
4. **Native FP4 expert matvec/GEMM** and FP8 trunk kernels.
5. **DeepSeek V4 prompt encoding / output parsing**.
6. **DSpark**, after the base model is proven correct.

---

## 3. Proposed repository shape

Bootstrap from WASTE, then immediately separate model-independent runtime code from the DeepSeek model implementation.

```text
.
├── README.md
├── LICENSES/
│   ├── WASTE-APACHE-2.0.txt
│   └── DEEPSEEK-MIT.txt
├── src/
│   ├── waste.h                 # public API, kept model-agnostic
│   ├── waste.c
│   ├── backend.*
│   ├── ecache.*                # reuse/generalize
│   ├── memory.*                # reuse/generalize
│   ├── platform.h
│   ├── crc32.*
│   ├── format.*                # generic container reader
│   ├── quant/
│   │   ├── fp4_e2m1.*
│   │   ├── fp8_e4m3.*
│   │   └── scalar_reference.*
│   └── models/
│       └── deepseek_v4/
│           ├── config.*
│           ├── model.*
│           ├── mhc.*
│           ├── attention.*
│           ├── compressor.*
│           ├── indexer.*
│           ├── moe.*
│           ├── rope.*
│           └── dspark.*
├── cli/
├── serve/
├── tools/
│   ├── inventory.py
│   ├── convert_deepseek_v4.py
│   ├── deepseek_ref.py
│   ├── compare_tensor.py
│   ├── trace_routing.py
│   ├── simulate_cache.py
│   └── diskbench.c
├── tests/
│   ├── fixtures/
│   ├── test_fp4.c
│   ├── test_fp8.c
│   ├── test_mhc.c
│   ├── test_attention.c
│   ├── test_router.c
│   ├── test_expert.c
│   ├── test_forward.c
│   └── test_encoding.py
└── docs/
    ├── FORMAT.md
    ├── ENGINE.md
    ├── GATES.md
    ├── LEARNED.md
    └── NUMERICS.md
```

Do not spend a week perfecting this directory tree before Gate 1. It is a target separation of concerns, not a prerequisite for the first oracle comparison.

---

## 4. Bootstrap procedure

Start from this repo's `main`, import the pinned WASTE tree, and preserve this README.

A simple low-magic workflow:

```bash
git clone https://github.com/Svyable/deepseek-v4-flash-wasted.git
cd deepseek-v4-flash-wasted

git remote add upstream-waste https://github.com/sqliteai/waste.git
git fetch upstream-waste

git checkout -b bootstrap/waste-0.6.6-ish

# Copy the pinned WASTE tree into a temporary directory, excluding its README.
rm -rf /tmp/waste-bootstrap
git clone https://github.com/sqliteai/waste.git /tmp/waste-bootstrap
cd /tmp/waste-bootstrap
git checkout d9b919a791148b571e643d0af666bf19b4d733ab
cd -
rsync -a --exclude='.git' --exclude='README.md' /tmp/waste-bootstrap/ ./

# Record the exact provenance in the first implementation commit.
git add .
git commit -m 'bootstrap: import WASTE d9b919a for DeepSeek V4 port'
```

Before modifying imported source, add a short `UPSTREAM.md` that records:

```text
WASTE upstream: sqliteai/waste
Imported commit: d9b919a791148b571e643d0af666bf19b4d733ab
Imported date: <date>
Initial target: deepseek-ai/DeepSeek-V4-Flash-0731 @ 9e165c3
```

Then run the upstream model-free suite **before** the DeepSeek refactor. We want a known-good baseline:

```bash
make
make check
```

If bootstrap breaks upstream tests, fix bootstrap before model work.

---

## 5. First implementation: inventory, not inference

**The first new file should be `tools/inventory.py`.**

It should parse `model.safetensors.index.json` and safetensors headers without materializing full tensors. For every tensor, print/emit JSON containing:

```text
name
dtype
logical shape
stored bytes
layer id
subsystem
resident vs streamed candidate
quantization partner / scale tensor
DSpark vs main model
```

Classify at least:

- embeddings + LM head;
- norms;
- mHC parameters;
- attention Q/KV/O projections;
- compressors;
- CSA indexers;
- router weights/biases/hash tables;
- shared experts;
- routed experts;
- DSpark-only tensors;
- quantization scales.

Output totals such as:

```text
main routed expert bytes
shared expert bytes
attention/compressor/indexer bytes
mHC bytes
embed/head bytes
DSpark bytes
other
```

This tool is the basis for every later memory and disk claim. Do not download/convert the entire model merely to discover a mistaken tensor assumption.

### Gate 0 — inventory sanity

Pass when:

- all checkpoint tensor names are classified;
- no bytes are left in an unexplained bucket;
- dimensions agree with `config.json`;
- each routed expert has exactly one `w1`, `w2`, `w3` plus expected scale tensors;
- the first three layers expose their token-id-to-expert mapping tensors;
- DSpark tensors are cleanly distinguishable from the 43-layer target path.

If the inventory disagrees with this README, **update this README. The checkpoint wins.**

---

## 6. Container design: WASTE v0 idea, DeepSeek-specific records

Do not overwrite WASTE v0 semantics and pretend the file is compatible. Introduce a model family / format version that cannot be confused with a Kimi container.

Proposed layout:

```text
deepseek-v4-flash-0731.waste/
├── manifest.json
├── trunk.bin
├── experts-L000.bin
├── experts-L001.bin
├── ...
├── experts-L042.bin
├── dspark.bin                  # optional at first
├── tokenizer.json
├── tokenizer_config.json
├── encoding/                   # pinned reference assets/tests or generated equivalent
└── usage.waste                 # learned routing hotlist, runtime-generated
```

### Expert record invariant

One routed expert should still require **one independently readable aligned record**:

```text
[record header]
[w1 packed FP4]
[w1 UE8M0 K32 scales]
[w3 packed FP4]
[w3 UE8M0 K32 scales]
[w2 packed FP4]
[w2 UE8M0 K32 scales]
[padding to 4 KiB]
```

Suggested new format identity:

```text
FP4_E2M1_K32_UE8M0
```

Do not call this `VQ3R` or reuse a WASTE format ID silently. The bytes and arithmetic are different.

The header should include at minimum:

```text
magic
format_version
model_family
layer
expert
record_bytes
matrix offsets
scale offsets
logical shapes
payload checksum
```

As in WASTE, validate all offsets/shapes before the arithmetic sees them. Optional payload CRC on runtime misses is fine; a full offline verifier should always be able to checksum the entire container.

### Why native FP4 first

DeepSeek's checkpoint was trained/served around native FP4 routed experts. Re-encoding immediately to WASTE's Kimi-tested VQ3R creates two unknowns at once:

1. new model architecture/numerics;
2. new quantization error.

That makes debugging needlessly ambiguous. Native FP4 gives us a clean oracle target.

After full-model correctness, make WASTE VQ3R/VQ4P an **experiment**, not a foundational assumption. A 3-bit expert representation would theoretically cut routed-expert traffic ~25% versus 4-bit payloads, but Kimi's error measurements are not evidence that DeepSeek V4 tolerates it.

---

## 7. Converter plan

Create `tools/convert_deepseek_v4.py` by borrowing the resumable/shard-aware structure of WASTE's converter while following DeepSeek's official tensor naming and FP4 semantics.

### Requirements

- Accept a local Hugging Face snapshot and an output directory.
- Read shard headers lazily.
- Never require all experts or all layers in RAM.
- Convert one MoE layer/bank at a time.
- Resume safely: a completed/verified layer is skipped.
- Write to temporary files and atomically rename after fsync.
- Record source revision, config, tokenizer hashes, converter version, and format version.
- Keep routed experts in source-native FP4 for phase 1.
- Keep scale bytes in their source semantics; no round-trip through float unless a test proves byte preservation is impossible/undesirable.
- Preserve BF16/F32 tensors where the official reference intentionally uses them.
- Preserve FP8 E4M3 + UE8M0 scale structure for trunk tensors until the runtime supports it.
- Emit enough metadata that runtime shape checks do not trust tensor names alone.

### Important source-format detail

The official DeepSeek reference defines routed FP4 as:

```text
weight: logical [out, in], stored packed as [out, in/2]
scale:  [out, in/32] UE8M0
```

Most FP8 linears use E4M3 weights with 128×128 scaling blocks. The official converter can also losslessly map FP4 values into an FP8 representation for hardware that lacks FP4; our first CPU reference kernel should instead decode the packed FP4 directly so that expert bank size remains minimal.

### Conversion validation

For random experts across early/middle/late layers:

1. read source packed weight/scales;
2. write WASTE expert record;
3. read record through the C parser;
4. expand a small tile with the C scalar FP4 decoder;
5. compare against the official Python FP4 table + scale semantics;
6. compare complete expert output on fixed random vectors.

Do this **before** converting all 11,008 main expert records.

---

## 8. Numerical bring-up strategy

There must be a slow, obvious implementation before a fast one.

### Scalar reference kernels first

Implement straightforward C versions for:

- BF16 load/convert;
- FP8 E4M3 + UE8M0 scaling;
- packed FP4 E2M1 + UE8M0 K32 scaling;
- RMSNorm;
- matvec;
- SiLU + DeepSeek SwiGLU clamp;
- RoPE/YaRN;
- top-k;
- softmax/sparse attention;
- mHC Sinkhorn mixing.

Only after each primitive matches the Python oracle should NEON/AVX/AMX/Metal/CUDA paths exist.

### Do not casually reorder reductions

The official MoE reference accumulates routed expert outputs into an FP32 buffer and then adds the shared expert. Top-k routing and long chains of mHC/attention make small discrepancies compound. Keep a documented reference accumulation order and measure any reordering before adopting it.

Create `docs/NUMERICS.md` early. Every deliberate precision change belongs there with the before/after error and speed result.

---

## 9. DeepSeek mHC implementation

This is the largest conceptual difference from a conventional residual transformer.

Each block carries **4 residual streams** (`hc_mult = 4`), not one. For attention and FFN independently, the official reference:

1. flattens the four streams;
2. computes normalized learned mixing logits;
3. splits them into pre/post/combination terms using Sinkhorn-constrained mixing;
4. uses `pre` to collapse the 4 streams into one input;
5. runs attention or MoE;
6. uses `post + comb` to expand/mix back into 4 streams.

The final head performs a learned collapse before RMSNorm + LM head.

### Gate 1 — mHC parity

Before integrating attention or experts, use Python fixtures with random input + real mHC weights. C must match:

- `hc_pre` collapsed vector;
- `post` weights;
- combination matrix;
- `hc_post` output;
- final `hc_head` collapse.

Test both one token and multi-token batches. Treat Sinkhorn iteration count (`20` in config) and epsilon as model semantics, not tuning knobs.

---

## 10. Hybrid attention implementation

DeepSeek V4 does **not** use Kimi's KDA/MLA. Implement the official attention directly.

All main layers keep a recent **128-token sliding window**. Long-range memory differs by compression ratio.

### A. ratio = 0 — pure sliding-window attention

The first two main layers are the simplest attention bring-up target:

- low-rank Q path (`4096 -> 1024 -> heads × 512`);
- normalized 512-d KV projection;
- RoPE dimensions;
- 128-token ring cache;
- sparse-attention API with only window indices;
- grouped low-rank output projection.

Get these two layers correct before building compression.

### B. ratio = 128 — heavily compressed attention

The compressor turns groups of history into learned compressed KV entries. The attention then attends to:

- the uncompressed 128-token local window; plus
- the compressed long-range stream.

There is no learned top-k indexer on this path in the official minimal reference; compressed entries are cheap enough to attend densely.

### C. ratio = 4 — compressed sparse attention

This path adds the learned indexer:

- compress history by 4;
- build/index compressed key representation;
- score compressed history with the indexer;
- select **top 512** compressed positions;
- concatenate those positions with the local window indices;
- run sparse attention over only the selected set.

The indexer has its own 64 heads × 128-d representation and uses quantization behavior that must match the official reference.

### Cache representation

Correctness-first: reproduce the official reference's cache precision. Its comments explicitly note that compressed KV could use FP8, while the minimal implementation currently stores it in BF16. Match BF16 first.

Then make FP8 KV cache a measured optional optimization; DeepSeek's vLLM recipe uses FP8 KV, so there is a well-motivated later path.

### Long-context memory planner

Do not allocate 1M context by default during bring-up. Make context an explicit plan-time parameter and derive the exact cache requirement per layer from:

```text
window + ceil(context / compress_ratio)
```

plus the separate CSA indexer cache where applicable.

The planner should report each class separately:

```text
resident weights
mHC state
sliding-window KV
compressed KV
CSA indexer cache
scratch
minimum expert buffers
expert cache
DSpark (if enabled)
```

---

## 11. MoE routing semantics — implement exactly

DeepSeek V4 routing is not Kimi routing.

For each main layer:

```text
router_logits = W_router @ x_fp32
scores = sqrt(softplus(router_logits))
```

For layers **0, 1, 2**:

```text
expert_ids = tid2eid[token_id]     # deterministic hash mapping
```

For later layers:

```text
selection_scores = scores + correction_bias
expert_ids = topk(selection_scores, 6)
```

Critically, the correction bias is for **selection only**. Routing weights come from the original unbiased scores:

```text
weights = scores[selected]
weights /= sum(weights)
weights *= 1.5
```

Every layer also computes one shared expert and adds it to the routed-expert sum.

Expert activation:

```text
gate = w1(x)
up   = w3(x)
up   = clamp(up, -10, +10)
gate = min(gate, 10)
h     = silu(gate) * up
out   = w2(h)
```

### WASTE-specific opportunity: perfect early prefetch

The first 3 layers are unusually friendly to streaming. Their expert IDs are known from the token ID before their router math is needed. As soon as the next input token is known, enqueue all six records for layers 0–2.

That is **18 exact expert reads** that can start immediately — no predictor, no correctness risk.

For layers 3–42, the real router still decides. As soon as a layer's route is known, call the cache hint before doing any independent arithmetic so I/O overlaps the maximum useful work.

Only after the exact route path is stable should we experiment with WASTE-style one-layer-ahead router prediction.

### Gate 2 — router parity

On saved real hidden states:

- compare all 256 pre-activation logits;
- compare post-`sqrt(softplus)` scores;
- compare selected expert IDs exactly;
- compare routing weights;
- test hash layers separately from score-routed layers;
- include tie/near-tie fixtures.

An expert-ID mismatch is a correctness failure even if downstream logits look close on one prompt.

---

## 12. Expert streaming + cache

Once scalar FP4 expert evaluation and routing pass, connect them through WASTE's expert bank/cache.

### Required invariant

For the same token/layer/expert:

```text
expert output from freshly read record
== expert output from cache hit
```

Bit-identical is the target for the same kernel/path.

### Cache key

At minimum:

```text
(main_or_dspark, layer_id, expert_id)
```

Do not let DSpark and main-bank IDs alias if the speculative module eventually uses its own expert weights.

### Cache policy

Start with WASTE's proven LFRU behavior, but **remeasure DeepSeek routing**. Do not copy Kimi hit-rate curves into docs as projections.

Build `tools/trace_routing.py` to dump, per generated token:

```json
{"token": 123, "layer": 17, "experts": [3, 19, 41, 88, 142, 251]}
```

Then simulate cache sizes before doing large end-to-end runs.

Important questions:

- next-token expert reuse;
- concentration by `(layer, expert)`;
- hash-layer reuse;
- hit rate vs cache bytes;
- LRU vs LFRU;
- learned hotlist benefit;
- effect of exact layer-0..2 prefetch;
- whether one-layer-ahead route prediction improves wall time.

### Do not optimize for hit rate alone

WASTE measured the classic failure mode: a larger cache can improve hit rate while making the process slower if the OS starts paging resident data. The DeepSeek planner must obey a hard process budget and leave system headroom.

---

## 13. Chunked prefill

Port WASTE's most important prefill idea: **read each distinct expert once per chunk**, not once per token.

For each layer and prefill chunk:

1. compute routes for all chunk tokens;
2. collect distinct expert IDs;
3. hint/read those records;
4. reuse each expert for every token routed to it;
5. preserve the exact per-token weighted sum.

Do not begin by expanding FP4 experts to full FP32 matrices. The whole point is to avoid turning a compact I/O problem into a memory-bandwidth explosion. Implement direct quantized matmul/matvec first; only expand if a benchmark proves it wins for a particular chunk size.

Attention prefill will need its own chunking strategy because CSA indexing/compression has sequential state. Validate chunked prefill against token-at-a-time prefill at every subsystem boundary.

---

## 14. Prompt encoding is model semantics

The 0731 release intentionally ships **no Jinja chat template**. It provides a dedicated `encoding/` implementation and tests.

Do not invent a prompt format and do not treat this as a server-only concern.

The official format handles:

- `system`, `user`, `assistant`, `tool`, `latest_reminder`, and an internal-only `developer` role;
- thinking vs chat mode;
- reasoning effort `low`, `high`, `max`;
- tool calls / DSML markup;
- tool results;
- structured response schema;
- dropping or preserving prior reasoning under defined conditions;
- completion parsing.

The key special strings include:

```text
<｜begin▁of▁sentence｜>
<｜end▁of▁sentence｜>
<｜User｜>
<｜Assistant｜>
<｜latest_reminder｜>
<think>
</think>
｜DSML｜
```

### Implementation order

1. Copy the official Python encoding tests into our oracle suite (respecting MIT attribution).
2. Build a C prompt renderer or a small host-side renderer.
3. Compare rendered strings byte-for-byte.
4. Compare tokenizer IDs token-for-token.
5. Only then expose `waste chat` / OpenAI-compatible serve behavior.

As in WASTE's current tokenizer discipline, **user content must never be allowed to become structural control tokens merely because strings were concatenated incorrectly**.

---

## 15. DSpark: explicitly phase 2

The 0731 checkpoint includes DSpark and the official vLLM recipe enables it with 7 speculative tokens. The config contains one next-token-prediction stage with block size 5, target main-layer states 40/41/42, a Markov rank-256 head, and a confidence head.

Do not bring this up before the main model generates correctly.

### Phase-2 DSpark tasks

- Capture main hidden states from layers 40, 41, 42.
- Implement DSpark's projection/norm into the draft block.
- Implement its sliding-window-only attention behavior.
- Implement its MoE/HC block.
- Implement Markov logit bias.
- Implement confidence head.
- Implement draft generation.
- Implement the exact acceptance/verification algorithm from DeepSeek/vLLM/SGLang references.
- Measure accepted draft length and speedup.

Base generation with DSpark disabled must remain a supported and fully tested mode.

---

## 16. Memory planner

`waste plan MODEL` is part of the product, not an afterthought.

The DeepSeek floor should be computed from exact converted tensor sizes, not parameter-count heuristics:

```text
floor = resident trunk
      + context-dependent attention/indexer cache
      + mHC/session state
      + tokenizer/runtime metadata
      + scratch
      + minimum asynchronous expert working set
      + optional DSpark resident state
```

Everything safely left under the configured RAM ceiling can become expert cache.

### Resident-vs-streamed rule

Initial policy:

**Stream:**
- 256 routed experts per main layer.

**Resident:**
- router weights/bias/hash maps;
- one shared expert per layer;
- mHC parameters;
- attention projections;
- compressor/indexer weights;
- norms;
- DSpark only when enabled;
- small metadata/code.

**Special-case:**
- embedding/head are large but accessed differently from routed experts. Inventory them and choose deliberately: resident quantized, row-read embedding, chunked LM-head, or another measured scheme. Do not accidentally dequantize a large FP8/BF16 matrix to FP32 and call that the floor.

### Hard ceiling

Keep WASTE's philosophy:

- `--budget` bounds all engine-owned memory;
- auto-budget respects Linux cgroups, not host physical RAM;
- fail early below the computed floor;
- report exactly where bytes go;
- avoid filling all available RAM merely because it exists.

---

## 17. Correctness oracle

`tools/deepseek_ref.py` should wrap the official Python implementation and export deterministic fixtures without requiring the C engine to reverse-engineer failures from final text.

For a small set of fixed prompts/tokens, dump:

```text
embedding
per-layer HC pre output
attention-normalized input
Q/KV intermediates
compressor outputs
CSA selected indices
attention output
FFN-normalized input
router logits/scores/ids/weights
shared-expert output
selected routed-expert outputs
MoE sum
HC post output
final hidden
final logits top-N + selected full-logit fixtures
```

Also dump model state needed for incremental decode so `pos > 0` paths can be tested directly.

### Error reporting

Every differential test should print:

```text
max_abs
mean_abs
max_rel
cosine
argmax equality
first bad index
```

Do not choose one global tolerance before seeing the reference behavior. Record per-subsystem tolerances based on the intended dtype/operation, and tighten them whenever possible.

---

## 18. Validation gates

Follow WASTE's rule: **before an expensive operation, run the cheapest real test that could kill the idea.**

### Gate A — checkpoint inventory

Protects: full download/conversion assumptions.

Pass: exact byte accounting and tensor classification.

### Gate B — native FP4 decode

Protects: expert converter + all MoE work.

Pass: scalar C tile decode and one full expert agree with the official Python reference.

### Gate C — FP8 trunk linear

Protects: resident-trunk implementation.

Pass: random real linears agree with Python across representative shapes.

### Gate D — mHC

Protects: every block.

Pass: `hc_pre`, Sinkhorn split, `hc_post`, final HC head parity.

### Gate E — attention by type

Protects: full decoder integration.

Pass independently:

- ratio-0 sliding attention;
- ratio-128 compression + dense compressed attention;
- ratio-4 compressor + indexer + top-512 sparse attention;
- incremental decode state equals full-prefill reference for the same sequence.

### Gate F — routing + one MoE layer

Protects: expert banks/cache.

Pass: exact expert IDs, close routing weights, shared+routed output parity.

### Gate G — disk vs cache identity

Protects: WASTE integration.

Pass: cache off/on give identical expert/model outputs for the same kernel path.

### Gate H — one complete transformer block

Pass: all recorded block boundary tensors match the oracle.

### Gate I — 43-layer base forward

Pass: final logits match within the documented numerical budget; same next-token argmax on deterministic fixtures.

### Gate J — tokenizer/encoding

Pass: official encoding fixtures match strings and token IDs exactly.

### Gate K — generation

Pass: greedy generation matches the oracle across short prompts and multi-turn state restoration.

### Gate L — real storage

Protects: promising local-inference performance that the storage cannot deliver.

Use WASTE's real access pattern: random ~expert-sized aligned reads with multiple outstanding requests and page-cache bypass. Report GB/s, not marketing SSD bandwidth.

### Gate M — cache curve

Protects: RAM recommendations.

Trace real 0731 routing and measure/simulate hit rate across practical cache budgets.

### Gate N — DSpark

Only after K passes. DSpark-on must produce target-model-valid accepted sequences and measurable wall-clock improvement; otherwise leave it optional/off.

---

## 19. Performance metrics we will publish

Every performance number must be tied to:

```text
commit
model revision
container format + quantization
hardware
OS
RAM budget
context length
prompt/decode split
thread count / CPU affinity
storage device
cache state (cold/warm)
direct-I/O status
```

Minimum metrics:

- prefill tok/s;
- decode tok/s;
- time-to-first-token;
- expert cache hit/miss rate;
- expert bytes read/token;
- expert read GB/s;
- read queue depth / time waiting for I/O;
- FP4 expert compute time;
- shared-expert time;
- attention/compressor/indexer time;
- mHC time;
- LM-head time;
- peak RSS;
- planned vs actual engine memory;
- DSpark acceptance length/speedup when enabled.

Never report a cache hit-rate improvement as a speed improvement without wall-clock measurement.

---

## 20. Suggested PR / commit sequence

Keep each milestone reviewable and oracle-backed.

### PR 1 — bootstrap upstream

- Import WASTE at pinned commit.
- Preserve licenses/NOTICE.
- Keep upstream tests green.
- Add `UPSTREAM.md`.

### PR 2 — DeepSeek checkpoint inventory

- `tools/inventory.py`.
- Exact storage/RAM anatomy report.
- Add/update this README with measured figures.

### PR 3 — quantized scalar reference kernels

- FP4 E2M1 + UE8M0 K32.
- FP8 E4M3 + UE8M0 block scales.
- Python differential tests.

### PR 4 — DeepSeek container + converter

- New model family/format version.
- Native FP4 expert record.
- Manifest + resume + verifier.
- Convert a one-layer/subset fixture first.

### PR 5 — mHC

- Sinkhorn/mixing implementation.
- Real-weight oracle fixtures.

### PR 6 — attention

- ratio 0 first;
- compressor;
- ratio 128;
- CSA indexer + ratio 4;
- incremental cache tests.

### PR 7 — DeepSeek MoE

- hash routing;
- score routing;
- shared expert;
- native FP4 routed expert;
- one-layer parity.

### PR 8 — WASTE expert streaming integration

- bank reader + cache;
- exact early-layer prefetch;
- cache identity tests;
- memory planning.

### PR 9 — full base model

- 43 layers;
- final HC head;
- LM head;
- greedy generation;
- session state.

### PR 10 — encoding + CLI/server

- official 0731 prompt encoding parity;
- reasoning modes;
- tool calls;
- OpenAI-compatible serving.

### PR 11 — chunked prefill + profiling

- deduplicate expert reads within chunks;
- route tracing/cache simulator;
- benchmark harness;
- learned hotlist.

### PR 12 — DSpark

- speculative module;
- acceptance tests;
- benchmark on/off.

### PR 13+ — optimization experiments

Only now consider:

- SIMD FP4 kernels;
- FP8 KV cache;
- WASTE VQ3R/VQ4P expert requantization;
- one-layer-ahead route prediction;
- Metal/CUDA/offload;
- multi-NVMe bank striping;
- expert hot-set pinning;
- indexer streaming optimizations.

Each experiment needs a kill criterion and a permanent negative-result entry if it loses.

---

## 21. Immediate TODO checklist for the first engineer

Do these in order:

- [ ] Bootstrap pinned WASTE source and prove `make check` still passes.
- [ ] Add WASTE Apache-2.0 + DeepSeek MIT attribution files.
- [ ] Implement `tools/inventory.py` using safetensors headers only.
- [ ] Produce `docs/INVENTORY-0731.md` with exact byte totals.
- [ ] Build a Python oracle fixture generator around official `inference/model.py`.
- [ ] Implement scalar E2M1 FP4 unpack + UE8M0 K32 scale in C.
- [ ] Differential-test one `w1`, `w3`, and `w2` matrix tile.
- [ ] Differential-test one complete expert.
- [ ] Extend WASTE container format with an explicit DeepSeek/native-FP4 expert record.
- [ ] Convert exactly one layer and verify every record.
- [ ] Implement mHC and pass real-weight fixtures.
- [ ] Bring up the first two ratio-0 attention layers.
- [ ] Implement hash routing for layers 0–2.
- [ ] Connect one streamed MoE layer to WASTE cache and prove cache/disk identity.
- [ ] Only then move into compressed attention and full-model integration.

If you are tempted to optimize before the checkbox above is green, add the idea to `docs/RESEARCH.md` and keep going with correctness.

---

## 22. The first commands to run tomorrow

```bash
# 1. Clone our repo.
git clone https://github.com/Svyable/deepseek-v4-flash-wasted.git
cd deepseek-v4-flash-wasted

# 2. Inspect/pin WASTE.
git remote add upstream-waste https://github.com/sqliteai/waste.git
git fetch upstream-waste
git show d9b919a791148b571e643d0af666bf19b4d733ab:README.md | head

# 3. Pull only lightweight DeepSeek metadata/reference code first.
#    Do NOT make the giant weight download the first dependency.
python -m pip install 'huggingface_hub>=0.34' safetensors
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash-0731',
    revision='9e165c3',
    local_dir='reference/deepseek-v4-flash-0731',
    allow_patterns=[
        'config.json',
        'model.safetensors.index.json',
        'tokenizer*',
        'inference/*',
        'encoding/*',
        'LICENSE*',
        'README.md',
    ],
)
PY

# 4. Write inventory.py and prove the shapes before downloading weights.
python tools/inventory.py reference/deepseek-v4-flash-0731
```

If Hugging Face revision shorthand ever stops resolving, resolve the full commit SHA and pin that instead. Never silently fall back to `main` in a measurement script.

---

## 23. Non-goals for the first working version

The first version does **not** need:

- the fastest possible FP4 kernel;
- GPU support;
- 1M context by default;
- VQ3R/VQ4P requantization;
- DSpark;
- multi-user batching;
- distributed expert parallelism;
- perfect OpenAI API coverage;
- a polished installer;
- benchmark bragging rights.

It needs one thing: **the real full DeepSeek-V4-Flash-0731 backbone producing correct tokens while most routed-expert bytes live on NVMe instead of RAM.**

---

## 24. Research questions worth answering after correctness

1. **What is the real 0731 routing locality?** Six of 256 per layer is sparse, but cache value depends on reuse distribution, not sparsity alone.
2. **How much does deterministic hash routing help I/O overlap?** Layers 0–2 give us 18 exact expert reads as soon as token ID is known.
3. **What is the practical resident-trunk floor?** DeepSeek's core backbone is far smaller than K3, but we must inventory the attached module and avoid accidental dequantization inflation.
4. **Does native FP4 compute or NVMe dominate?** The answer determines whether SIMD work or read-ahead is the next lever.
5. **Can WASTE's LFRU policy beat LRU here too?** Measure; do not assume.
6. **How large must the cache be to survive a token/layer working set?** Prefetch may change the relevant lifetime just as it did in WASTE.
7. **Can the CSA indexer itself be streamed/chunked at very long context?** At 1M, top-k selection and indexer memory traffic may become a separate bottleneck.
8. **Is FP8 KV materially faster/better than BF16 KV on CPU/unified-memory targets?** The QAT path permits it; the minimal reference chooses BF16 for simplicity.
9. **Does 3-bit WASTE VQ preserve 0731 quality?** Test perplexity/logit error/agentic tasks only after native FP4 is a stable baseline.
10. **Does DSpark still help when the target is storage-bound?** Speculation adds work and expert accesses; accepted tokens per target step, not theoretical draft speed, decides the result.
11. **Would two or more NVMe devices scale random expert reads?** Bank-by-layer or striped placement may be a bigger lever than another quantization trick.
12. **Can a learned hotlist pin high-value experts without increasing paging pressure?** Evaluate against the same hard RAM ceiling.

Record failed experiments. A negative result that prevents a future week of work is part of the implementation.

---

## 25. Definition of done for v0.1

We can call the first DeepSeek WASTE port real when all of these are true:

- [ ] A reproducible converter creates a verified DeepSeek-specific WASTE container from the 0731 checkpoint.
- [ ] The 43-layer base path runs in the C library with routed experts streamed from disk.
- [ ] Cache hit vs disk miss is numerically identical for the same expert kernel.
- [ ] Layer-by-layer oracle tests explain the full numerical error budget.
- [ ] Greedy next tokens match the Python reference on a published fixture set.
- [ ] Official DeepSeek V4 message encoding tests pass.
- [ ] `waste plan` reports a real measured RAM floor and context-dependent state.
- [ ] The process respects a hard RAM budget/cgroup limit.
- [ ] A storage benchmark uses the real record size/access pattern.
- [ ] Decode benchmarks publish bytes read/token and cache hit rate alongside tok/s.
- [ ] A consumer/workstation-class machine can run the **full** model without requiring all routed experts in RAM.
- [ ] DSpark status is explicit: correct + measured, or intentionally disabled.

Then optimize.

---

## 26. Core engineering principle

WASTE's strongest idea is not a quantizer or a cache policy. It is the separation of **correctness from placement**:

> The real router chooses the real experts. The real expert bytes produce the real output. RAM versus NVMe only decides when those bytes arrive.

Keep that invariant through the DeepSeek port. It gives us a clean ladder from a slow scalar implementation to aggressive prefetch, caching, SIMD, alternate quantization, and speculative decoding without losing the ability to say exactly which optimization changed the model and which only changed its speed.

That is how we get DeepSeek V4 Flash **WASTED** without turning the project into an un-debuggable pile of approximations.
