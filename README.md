# DeepSeek V4 Flash — WASTED

> **Research + implementation handoff for running `deepseek-ai/DeepSeek-V4-Flash-0731` with WASTE-style NVMe expert streaming.**
>
> Status: **bootstrap + first DeepSeek arithmetic landed; no DeepSeek transformer forward path exists yet.** PR #3 added scalar E2M1/UE8M0/E4M3FN decode/matvec paths and exhaustive public-format tests. README Gate B / V1 is half satisfied; official DeepSeek byte/scale conventions remain blocked on reference access. Optimize for correctness and observability before speed.

The goal of this repository is to adapt the core idea behind [sqliteai/waste](https://github.com/sqliteai/waste) — keep the dense/shared trunk resident, stream only the routed MoE experts from fast storage, overlap expert I/O with compute, and use all remaining RAM as a bounded expert cache — to **DeepSeek-V4-Flash-0731**.

This is a good architectural match: DeepSeek V4 Flash is described by the current official-spec handoff as an all-MoE model with **256 routed experts per layer and 6 routed experts active per token**, plus one shared expert. But this is **not** a model-name swap in WASTE. WASTE's current forward path is Kimi-specific. DeepSeek V4 introduces a different trunk: manifold-constrained Hyper-Connections, hybrid compressed attention, hash-routed bootstrap layers, native FP4 routed experts, FP8 trunk weights, and an attached DSpark speculative-decoding module.

The implementation strategy is therefore:

1. **Reuse WASTE's proven generic systems pieces**: bounded expert cache, direct/aligned reads, one-record-per-expert layout, RAM planning, chunked prefill ideas, read-ahead discipline, CLI/library separation, validation gates, and instrumentation.
2. **Implement a DeepSeek-V4 model path against the official reference code**, rather than trying to bend Kimi KDA/MLA code into DeepSeek shapes.
3. **Preserve DeepSeek's native quantization first**. Do not requantize the routed experts to WASTE VQ3R until the native FP4 implementation is correct and measured.
4. **Bring up the 43-layer target model without DSpark first**, prove logits/generation, then add DSpark as an optional acceleration layer.

### Gate vocabulary used by this repo

This README's §18 defines **14 stable design gates A–N**. `docs/VALIDATION.md` defines operational `V0–V11` validation levels, and `ROADMAP.md` defines schedule phases. They are related but not interchangeable.

- When a README gate has a V-level, cite both: `Gate B / V1`, `Gate H / V6`, `Gate K / V9`.
- README systems/performance Gates **G/L/M/N** intentionally have no V-number.
- Operational **V7** and **V11** intentionally have no README letter.
- `docs/VALIDATION.md` §4a is the canonical concordance and maps **all 14 gates A–N**, preserving the concept and roadmap phase.
- Do not introduce a separate local “Gate 0/1/2” vocabulary. Older shorthand in this handoff is annotated to the canonical gate below.

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
- Release revision shown by Hugging Face during research: `9e165c3`
- Config: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json>
- Official minimal inference model: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py>
- Official kernels: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/kernel.py>
- Official checkpoint converter: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/convert.py>
- Official encoding guide: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/README.md>
- Encoding reference implementation/tests: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding>
- Technical report: <https://arxiv.org/abs/2606.19348>

The research handoff identifies the DeepSeek repository/weights as MIT-licensed, but PR #1 deliberately did **not** fabricate the exact publisher license text while the official repository was unreachable. Fetch and preserve the exact official license/notice before copying/adapting source; see `LICENSES/DEEPSEEK-MIT.txt.MISSING` and `docs/REFERENCE_ACCESS.md`.

**Pin revisions in every reproducible experiment.** Before generating durable fixtures, resolve the full immutable Hugging Face commit SHA. The converter should record the resolved source commit and WASTE/port commit in the output manifest.

---

## 1. Target model snapshot

The July 31 release is `DeepSeek-V4-Flash-0731`, the official replacement for the preview Flash checkpoint. The package includes the DSpark speculative-decoding module. Hugging Face research during the initial handoff reported **304B parameters** for the packaged checkpoint; the original Flash backbone is documented as **284B total / ~13B activated**. Do not hard-code either total into the runtime — **README Gate A / V0 checkpoint inventory is authoritative**.

Core config values recorded by the research handoff:

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

Treat this table as **OFFICIAL-SPEC/HANDOFF, not locally CHECKPOINT-VERIFIED** until Gate A/V0 runs against the pinned release. Native packed/scale arithmetic conventions additionally require Gate B/V1 official-reference agreement.

`compress_ratios` for the package are recorded as:

```text
0, 0,
4, 128, 4, 128, ... alternating ...,
4,
0, 0, 0
```

For the 43 main blocks this suggests two initial pure sliding-window layers followed by alternating compressed-attention layers. The trailing entries are associated with the attached speculative path in the handoff; derive behavior from the official reference rather than inferring solely from this list.

### Why this is attractive for WASTE

The current handoff models a routed DeepSeek expert as a conventional SwiGLU triplet:

```text
w1: [2048, 4096]
w3: [2048, 4096]
w2: [4096, 2048]
```

That is **25,165,824 logical weights per expert**. The current storage hypothesis is two E2M1 values per byte plus one UE8M0 scale per 32 logical values along K.

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

> `tools/inventory.py` exists and reproduces every figure above from a
> synthetic checkpoint built to this section's spec — which confirms the
> arithmetic and nothing about DeepSeek's actual weights. The checkpoint has
> still never been read in the original bootstrap environment.
> `docs/INVENTORY-0731.md` and `docs/REFERENCE_ACCESS.md` track the blocker.

The important takeaway is architectural: if Gate A/V0 confirms the top-k/expert layout, a token needs only a small fraction of each layer's routed experts. That is the access pattern WASTE is designed to exploit.

---

## 2. What to reuse from WASTE vs. what to rewrite

### Reuse / generalize

These are the valuable WASTE primitives:

- **4 KiB-aligned independently readable expert records**.
- **One coalesced read per expert** with all expert matrices/scales adjacent once Gate A/V0 proves the complete record payload.
- `pread`/platform direct-I/O path and page-cache bypass behavior.
- Bounded expert cache (`ecache`) with asynchronous reader threads.
- Exact cache-vs-disk equivalence: placement changes latency, never numerics — **README Gate G**.
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

1. **Manifold-Constrained Hyper-Connections (mHC)** — Gate D/V3.
2. **DeepSeek V4 hybrid attention** — Gate E/V5:
   - pure sliding-window layers;
   - compressed sparse attention (CSA, ratio 4 + learned indexer);
   - heavily compressed attention (ratio 128, dense over compressed history).
3. **DeepSeek routing semantics** — Gate F/V4, including deterministic/hash behavior only if Gate A/V0 proves it.
4. **Native FP4 expert and FP8 trunk arithmetic** — Gates B/C, V1/V2.
5. **DeepSeek V4 prompt encoding / output parsing** — Gate J/V10.
6. **DSpark** — Gate N, after the base model is proven correct.

---

## 3. Proposed repository shape

Bootstrap from WASTE, then incrementally separate model-independent runtime code from the DeepSeek model implementation.

```text
.
├── README.md
├── LICENSES/
│   ├── WASTE-APACHE-2.0.txt
│   └── DEEPSEEK-MIT.txt        # exact official file once retrieved
├── src/
│   ├── waste.h                 # public API, kept model-agnostic
│   ├── waste.c
│   ├── backend.*
│   ├── ecache.*                # reuse/generalize
│   ├── memory.*                # reuse/generalize
│   ├── platform.h
│   ├── crc32.*
│   ├── quant/
│   │   ├── fp4_e2m1.*          # landed in PR #3
│   │   └── fp8_e4m3.*          # landed in PR #3
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
│   ├── inventory.py            # landed in PR #1
│   ├── convert_deepseek_v4.py
│   ├── deepseek_ref.py
│   ├── compare_tensor.py
│   ├── trace_routing.py
│   ├── simulate_cache.py
│   └── diskbench.c
├── tests/
│   ├── fixtures/
│   ├── test_quant.c            # landed in PR #3
│   ├── test_mhc.c
│   ├── test_attention.c
│   ├── test_router.c
│   ├── test_expert.c
│   ├── test_forward.c
│   └── test_encoding.py
└── docs/
    ├── README.md
    ├── VALIDATION.md
    ├── NUMERICS.md
    ├── FIXTURES.md
    ├── REFERENCE_ACCESS.md
    └── ... imported + local design docs
```

Do not spend a week perfecting this directory tree before the next canonical gate. It is a target separation of concerns, not a prerequisite for oracle comparison.

---

## 4. Bootstrap procedure

This section records the original low-magic bootstrap procedure. It is **already complete** in GitHub PR #1; do not re-import WASTE over current local changes.

```bash
git clone https://github.com/Svyable/deepseek-v4-flash-wasted.git
cd deepseek-v4-flash-wasted

git remote add upstream-waste https://github.com/sqliteai/waste.git
git fetch upstream-waste

# Historical bootstrap recipe, not a command to rerun over current main:
rm -rf /tmp/waste-bootstrap
git clone https://github.com/sqliteai/waste.git /tmp/waste-bootstrap
cd /tmp/waste-bootstrap
git checkout d9b919a791148b571e643d0af666bf19b4d733ab
cd -
```

`UPSTREAM.md` records the exact import, local modifications, attribution, and baseline tests.

Verified historical baselines:

```text
before local model work: 31 passed, 0 failed, 12 skipped
after PR #1 inventory:  32 passed, 0 failed, 12 skipped
after PR #3 quant:      34 passed, 0 failed, 12 skipped
PR #3 ASan/UBSan:       33 passed, 0 failed
```

---

## 5. First implementation: inventory, not inference

`tools/inventory.py` was the first new implementation and now exists.

It parses `model.safetensors.index.json` and safetensors headers without materializing full tensors. For every tensor it can report/emit:

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

It classifies at least the expected families:

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

This tool is the basis for every later memory and disk claim. Do not download/convert the entire model merely to discover a mistaken tensor assumption.

### Local inventory sanity check — README Gate A / V0

This is the historical “Gate 0” check from the original handoff. The canonical gate is **Gate A / V0**.

Pass when:

- all checkpoint tensor names are classified;
- no bytes are left in an unexplained bucket;
- dimensions agree with `config.json`;
- each routed expert has exactly the complete official matrix/scale tensor set;
- the early/bootstrap routing representation is identified from official artifacts/reference;
- DSpark tensors are cleanly distinguishable from the base target path.

If the inventory disagrees with this README, **update this README. The checkpoint wins.**

Passing Gate A/V0 establishes checkpoint storage/tensor truth. Native packed-byte/scale arithmetic still needs Gate B/V1.

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
├── experts-L042.bin             # only if Gate A/V0 confirms the main-layer layout
├── dspark/                      # optional Gate N assets
├── tokenizer.json
├── tokenizer_config.json
├── encoding/                    # pinned assets or validated generated equivalent
└── usage.waste                 # learned routing hotlist, runtime-generated
```

### Expert record invariant

One routed expert should still require **one independently readable aligned record**. Conceptually:

```text
[record header]
[matrix A packed native payload]
[matrix A scales]
[matrix B packed native payload]
[matrix B scales]
[matrix C packed native payload]
[matrix C scales]
[verified extras, if Gate A/V0 finds them]
[padding to alignment]
```

The handoff's likely identity is native E2M1 + UE8M0 K32, but do not freeze the record until:

- **Gate A / V0** proves exact tensor set/shapes/storage geometry;
- **Gate B / V1** proves nibble/scale arithmetic conventions;
- a representative official projection reaches **Gate C / V2**.

Do not call the native format `VQ3R` or silently reuse a Kimi format ID. The bytes and arithmetic are different.

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

Re-encoding immediately to WASTE's Kimi-tested VQ3R creates two unknowns at once:

1. new model architecture/numerics;
2. new quantization error.

That makes debugging needlessly ambiguous. Native representation gives us the cleanest official-oracle target.

After full-model correctness, make WASTE VQ3R/VQ4P an **experiment**, not a foundational assumption. A 3-bit expert representation would theoretically cut routed-expert payload traffic versus 4-bit storage, but Kimi's error measurements are not evidence that DeepSeek V4 tolerates it.

See `docs/CONTAINER_V4.md` for the maintained format contract.

---

## 7. Converter plan

Create `tools/convert_deepseek_v4.py` by borrowing the resumable/shard-aware structure of WASTE's converter while following DeepSeek's official tensor mapping and native quantization semantics.

### Requirements

- Accept a local pinned official snapshot and an output directory.
- Read shard headers lazily.
- Never require all experts or all layers in RAM.
- Convert one MoE layer/bank at a time.
- Resume safely: a completed/verified layer is skipped.
- Write to temporary files and atomically rename after verification.
- Record source revision, config, tokenizer hashes, converter version, and format version.
- Keep routed experts source-native for the correctness baseline.
- Keep scale bytes/values in source semantics; no guessed reinterpretation.
- Preserve BF16/F32 tensors where the official reference intentionally uses them.
- Preserve resident FP8 structure until the runtime supports it faithfully.
- Emit enough metadata that runtime shape checks do not trust tensor names alone.

### Important source-format boundary

The research handoff predicts routed weights logically `[out, in]`, packed as `[out, in/2]`, with `[out, in/32]` scales and predominantly 128×128-block FP8 trunk tensors. PR #3 implemented public E2M1/UE8M0/E4M3FN arithmetic, but **DeepSeek-specific nibble order, scale application, and exact scale layout remain Gate B/V1 questions**. Do not convert the whole model based only on the handoff shape arithmetic.

### Conversion validation

For selected experts/tensors across early/middle/late layers:

1. read source packed weight/scales;
2. write WASTE record;
3. read record through the C parser;
4. decode a small tile with the scalar native decoder;
5. compare against an independent official-derived fixture;
6. compare a complete expert/projection output on fixed inputs.

Do this before converting the full expert population.

See `docs/CONVERSION.md` for the maintained runbook.

---

## 8. Numerical bring-up strategy

There must be a slow, obvious implementation before a fast one.

### Scalar reference kernels first

PR #3 already landed scalar/native arithmetic for:

- E2M1 decode;
- UE8M0 decode + current K32 indexing model;
- finite E4M3 (`e4m3fn`) decode + current 128×128 scale-grid model;
- FP4/FP8 scalar matvecs with `double` accumulation;
- exhaustive public-format and model-free indexing tests.

Still to implement/prove as the model path grows:

- BF16 load/convert where needed;
- RMSNorm;
- DeepSeek SwiGLU clamp;
- RoPE/YaRN;
- top-k;
- softmax/sparse attention;
- mHC Sinkhorn mixing;
- all official-reference-dependent composition.

Do not add SIMD merely because the scalar path exists. **Gate B/V1 official convention agreement and Gate C/V2 one-projection parity come first.**

### Do not casually reorder reductions

The expected MoE path and long chains of mHC/attention can amplify small discrepancies. Keep a documented reference accumulation order and measure any reordering before adopting it.

`docs/NUMERICS.md` is the maintained arithmetic contract. Every deliberate precision change belongs there with the before/after correctness and speed result.

---

## 9. DeepSeek mHC implementation

This is the largest conceptual difference from a conventional residual transformer.

Each block is described as carrying **4 residual streams** (`hc_mult = 4`), not one. The handoff says the official reference:

1. flattens the four streams;
2. computes normalized learned mixing logits;
3. splits them into pre/post/combination terms using Sinkhorn-constrained mixing;
4. uses `pre` to collapse the 4 streams into one input;
5. runs attention or MoE;
6. uses `post + comb` to expand/mix back into 4 streams.

The final head performs a learned collapse before RMSNorm + LM head.

### Local mHC check — README Gate D / V3

This is the historical “Gate 1” check from the original handoff. The canonical model-primitives gate is **Gate D / V3**.

Before integrating full attention or experts, use official fixtures with random input + real mHC weights. C must match:

- `hc_pre` collapsed vector;
- `post` weights;
- combination matrix;
- `hc_post` output;
- final `hc_head` collapse.

Test both one token and multi-token batches. Treat Sinkhorn iteration count and epsilon as model semantics, not tuning knobs.

---

## 10. Hybrid attention implementation — README Gate E / V5

DeepSeek V4 does **not** use Kimi's KDA/MLA. Implement the official attention directly.

The handoff says all main layers keep a recent **128-token sliding window**, with long-range memory varying by compression ratio.

### A. ratio = 0 — pure sliding-window attention

The initial ratio-0 layers are the simplest attention bring-up target:

- low-rank Q path;
- normalized KV projection;
- RoPE dimensions;
- sliding-window ring cache;
- sparse-attention API with only window indices;
- grouped low-rank output projection.

Get this path correct before building compression.

### B. ratio = 128 — heavily compressed attention

The handoff describes a compressor that turns groups of history into learned compressed KV entries. Attention then attends to:

- the uncompressed local window; plus
- the compressed long-range stream.

### C. ratio = 4 — compressed sparse attention

This path adds the learned indexer in the current handoff:

- compress history by 4;
- build/index compressed key representation;
- score compressed history with the indexer;
- select top compressed positions;
- concatenate those positions with the local window indices;
- run sparse attention over only the selected set.

The exact indexer representation/quantization/state semantics must come from the official reference.

### Cache representation

Correctness-first: reproduce the official reference's cache precision and update behavior. Make lower-precision state a measured optional optimization later.

### Long-context memory planner

Do not allocate maximum context by default during bring-up. Make context an explicit plan-time parameter and derive exact cache/state requirements from the official attention implementation, then report each class separately:

```text
resident weights
mHC state
sliding-window state
compressed state
CSA indexer cache
scratch
minimum expert buffers
expert cache
DSpark (if enabled)
```

---

## 11. MoE routing semantics — README Gate F / V4

DeepSeek V4 routing is not Kimi routing.

The current handoff records later-layer scoring roughly as:

```text
router_logits = W_router @ x_fp32
scores = sqrt(softplus(router_logits))
```

and describes the first three layers as deterministic/hash-routed from token ID, with later layers using correction-biased selection and unbiased routed weights.

Treat those as official-spec/handoff until Gate A/V0 and the pinned reference confirm the exact representation and sequence.

### WASTE-specific opportunity: exact early prefetch, **if Gate A/V0 confirms it**

If the first three layers' expert IDs are deterministically known from token ID as described, their selected expert records can be hinted as soon as the input token is known. Under the current top-6 hypothesis that would expose up to **18 early expert-record hints** with no route predictor.

This is an architectural opportunity, not yet a measured or checkpoint-verified fact. If Gate A/V0 shows the mapping is represented differently, update this claim before implementation.

For learned-routing layers, the real router still decides. Only after the exact route path is stable should we experiment with one-layer-ahead prediction.

### Local router check — supports README Gate F / V4

This is the historical “Gate 2” check from the original handoff. Router primitive seams may be validated during V3, but routing + one complete MoE block is **Gate F / V4**.

On saved official hidden states:

- compare all router pre-activation logits;
- compare score transform;
- compare selected expert IDs/order exactly;
- compare routing weights;
- test early/bootstrap routing separately from learned routing;
- include tie/near-tie fixtures.

An expert-ID mismatch is a correctness failure even if downstream logits look close on one prompt.

---

## 12. Expert streaming + cache — README Gate G correctness, Gates L/M performance

Once native expert evaluation and routing pass, connect them through WASTE's expert bank/cache.

### Required invariant — Gate G

For the same token/layer/expert:

```text
expert output from freshly read record
== expert output from cache hit
```

Bit-identical is the target for the same kernel/path. Prefetch and direct-I/O mode may change timing, never authoritative bytes/routing/model meaning.

### Cache key

At minimum:

```text
(main_or_dspark, layer_id, expert_id)
```

Do not let Gate N/DSpark and main-bank IDs alias if the speculative module eventually uses its own expert weights.

### Cache policy — Gate M

Start with WASTE's proven LFRU behavior as an implementation baseline, but **remeasure DeepSeek routing**. Do not copy Kimi hit-rate curves into docs as projections.

Build `tools/trace_routing.py` to dump, per generated token:

```json
{"token": 123, "layer": 17, "experts": [3, 19, 41, 88, 142, 251]}
```

Then simulate/measure cache sizes against real routing.

Important questions:

- next-token expert reuse;
- concentration by `(layer, expert)`;
- early-layer reuse;
- hit rate vs cache bytes;
- LRU vs LFRU;
- learned hotlist benefit;
- effect of exact early prefetch if Gate A/V0 confirms it;
- whether one-layer-ahead route prediction improves wall time.

### Do not optimize for hit rate alone

WASTE measured the classic failure mode: a larger cache can improve hit rate while making the process slower if the OS starts paging resident data. The DeepSeek planner must obey a hard process budget and leave system headroom.

Gate M requires a real cache curve; Gate L separately measures target storage; both depend on Gate G identity remaining true.

---

## 13. Chunked prefill

Port WASTE's most important prefill idea: **read each distinct expert once per chunk**, not once per token.

For each layer and prefill chunk:

1. compute authoritative routes for all chunk tokens;
2. collect distinct expert IDs;
3. hint/read those records;
4. reuse each expert for every token routed to it;
5. preserve the exact per-token weighted sum.

Do not begin by expanding FP4 experts to full FP32 matrices. The whole point is to avoid turning a compact I/O problem into a memory-bandwidth explosion. Implement direct quantized matmul/matvec first; only expand if a benchmark proves it wins for a particular chunk size.

Attention prefill will need its own chunking strategy because CSA indexing/compression has sequential state. Validate chunked prefill against token-at-a-time prefill at every subsystem boundary, preserving Gate G placement/optimization identity.

---

## 14. Prompt encoding is model semantics — README Gate J / V10

The 0731 release is described as shipping a dedicated `encoding/` implementation/tests rather than relying on a Jinja chat template.

Do not invent a prompt format and do not treat this as a server-only concern.

The current handoff records support for roles/modes/tools/reasoning/structured output. The exact strings, control tokens, role behavior, and parser semantics come from the pinned official `encoding/` code/tests.

### Implementation order

1. Retrieve/pin the official encoding implementation/tests and exact license.
2. Build a C or host-side renderer/parser faithful to the official semantics.
3. Compare rendered structures/strings as appropriate.
4. Compare tokenizer IDs token-for-token.
5. Pass **Gate J / V10**.
6. Only then call the OpenAI-compatible server path complete at **V11**.

As in WASTE's tokenizer discipline, **user content must never be allowed to become structural control tokens merely because strings were concatenated incorrectly**.

---

## 15. DSpark: explicitly phase 2 — README Gate N

The 0731 checkpoint includes a speculative-decoding module described in the handoff as DSpark. Its exact tensor namespace, state, proposal/acceptance algorithm, and serving semantics must come from the pinned official reference.

Do not bring this up before the main model passes:

- **Gate I / V8** final-logit parity;
- **Gate K / V9** deterministic generation;
- **Gate G** placement identity.

Gate N then owns speculative correctness + performance. `docs/DSPARK.md` defines its `N-D1…N-D5` internal subchecks, memory accounting, committed-token boundary, and benchmark requirements.

Base generation with DSpark disabled must remain a supported and fully tested mode.

---

## 16. Memory planner — Gates A/G/L/M inputs and outputs

`waste plan MODEL` is part of the product, not an afterthought.

The DeepSeek floor should be computed from exact converted tensor sizes, not parameter-count heuristics:

```text
floor = resident trunk
      + context-dependent attention/indexer state
      + mHC/session state
      + tokenizer/runtime metadata
      + scratch
      + minimum asynchronous expert working set
      + optional DSpark resident state
```

Everything safely left under the configured RAM ceiling can become expert cache.

### Resident-vs-streamed rule

Initial policy:

**Stream:** routed experts proven by Gate A/V0.

**Resident:** router weights/bias/hash maps, shared experts, mHC, attention/compressor/indexer weights, norms, metadata, and DSpark only when enabled—subject to exact Gate A/V0 inventory and later measurement.

**Special-case:** embedding/head are large but accessed differently from routed experts. Inventory them and choose deliberately; do not accidentally expand a large quantized matrix to FP32 and call that the floor.

### Hard ceiling

Keep WASTE's philosophy:

- `--budget` bounds all engine-owned memory;
- auto-budget respects Linux cgroups, not host physical RAM;
- fail early below the computed floor;
- report exactly where bytes go;
- avoid filling all available RAM merely because it exists.

`docs/MEMORY_AND_IO.md` owns the maintained formulas and Gate G/L/M methodology.

---

## 17. Correctness oracle

`tools/deepseek_ref.py` should wrap the official Python implementation and export deterministic fixtures without requiring the C engine to reverse-engineer failures from final text.

For fixed prompts/tokens, dump the smallest intermediates needed to localize the active gate:

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

All golden fixtures must follow `docs/FIXTURES.md`: expected values cannot be generated by importing the same convention/helper they are supposed to prove.

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

Do not choose one global tolerance before seeing the official behavior. Record per-subsystem tolerances based on intended dtype/operation, and tighten them whenever possible.

---

## 18. Validation gates — 14 stable design gates A–N

Follow WASTE's rule: **before an expensive operation, run the cheapest real test that could kill the idea.**

> The 14 lettered gates below are the stable design rationale. `docs/VALIDATION.md`
> is the maintained operational instance of the same ladder, with tolerances,
> fixture rules and triage order. Its §4a maps **every Gate A–N here** to an
> operational V-level or systems/performance owner, keeps the concept column,
> and maps it to a `ROADMAP.md` phase. When both labels exist, cite both in PRs.
> README Gates G/L/M/N intentionally have no V-number; V7/V11 intentionally
> have no README letter. Where operational order differs, `VALIDATION.md` wins.

### Canonical concordance

| README gate | Handoff concept | Operational owner |
|---|---|---|
| **A** | checkpoint inventory / storage truth | **V0** |
| **B** | native FP4/native-quant decode + DeepSeek convention agreement | **V1** |
| **C** | FP8/trunk quantized linear parity | **V2** |
| **D** | mHC + model primitives | **V3** |
| **E** | attention by type | **V5** |
| **F** | routing + one MoE layer | **V4** |
| **G** | disk/cache/prefetch identity | **Gate G systems correctness** |
| **H** | one complete transformer block | **V6** |
| **I** | 43-layer base forward / final logits | **V8** |
| **J** | tokenizer/encoding/parser | **V10** |
| **K** | deterministic generation | **V9** |
| **L** | real storage feasibility | **Gate L performance feasibility** |
| **M** | real cache curve | **Gate M performance feasibility** |
| **N** | DSpark speculative correctness + speed | **Gate N** |

`VALIDATION.md` §4a additionally records `V7` (multi-layer hidden-state localization) and `V11` (OpenAI-compatible API parity), which were not assigned letters in this original 14-gate handoff. It also records roadmap phases and owning docs.

The E/F and J/K V-number ordering is deliberate: the maintained test ladder can isolate MoE before full compressed attention, and can validate raw known-token greedy generation before the complete chat encoder/parser surface. **The README letters remain stable identifiers even when operational test order differs.**

### Gate A — checkpoint inventory

Protects: full download/conversion assumptions.

Pass: exact byte accounting and tensor classification from the pinned checkpoint; no unexplained main-model tensors/bytes. See `INVENTORY-0731.md`, `TENSOR_MAP.md`, `REFERENCE_ACCESS.md`.

### Gate B — native FP4/native-quant decode

Protects: expert converter + all native quantized MoE work.

Pass: public-format conformance **and** official DeepSeek packing/scale convention agreement. PR #3 satisfies only the public-format half. See `NUMERICS.md`, `FIXTURES.md`, `VALIDATION.md` V1.

### Gate C — FP8 trunk linear

Protects: resident-trunk implementation.

Pass: at least one representative official quantized trunk linear/projection agrees with the scalar C path under a justified tolerance, crossing relevant scale boundaries. See V2.

### Gate D — mHC

Protects: every block.

Pass: `hc_pre`, Sinkhorn split, `hc_post`, final HC head parity and other required model-primitives seams. See V3.

### Gate E — attention by type

Protects: full decoder integration.

Pass independently:

- ratio-0/sliding attention;
- heavily compressed attention;
- compressor + indexer sparse attention;
- incremental decode state equals full-prefill reference for the same sequence.

Operational owner: V5.

### Gate F — routing + one MoE layer

Protects: expert banks/cache integration.

Pass: exact expert IDs/order, close routing weights, shared+routed output parity for one MoE block. Operational owner: V4.

### Gate G — disk vs cache identity

Protects: WASTE integration.

Pass: cache off/on, direct/buffered read paths, and prefetch off/on preserve authoritative bytes/routing/model output for the same arithmetic path. Gate G intentionally has no V-number.

### Gate H — one complete transformer block

Pass: all recorded block boundary tensors match the oracle. Operational owner: V6.

### Gate I — 43-layer base forward

Pass: final logits match within the documented numerical budget; same next-token argmax on deterministic fixtures. Operational owner: V8, with V7 used to localize multi-layer divergence.

### Gate J — tokenizer/encoding

Pass: official encoding/parser fixtures match required structures and token IDs exactly. Operational owner: V10.

### Gate K — generation

Pass: greedy generation matches the oracle across deterministic inputs and state restoration. Operational owner: V9. User-facing chat generation ultimately requires both K/V9 and J/V10.

### Gate L — real storage

Protects: promising local-inference performance that target storage cannot deliver.

Use WASTE's real access pattern: random expert-sized aligned reads with realistic concurrency and page-cache-bypass status. Report GB/s/latency with exact device/filesystem/record provenance. Gate L intentionally has no V-number.

### Gate M — cache curve

Protects: RAM/cache recommendations.

Trace real 0731 routing and measure hit rate, read traffic, RSS/paging, and wall-clock decode across practical cache budgets. Gate M intentionally has no V-number.

### Gate N — DSpark

Only after Gate I/V8 and Gate K/V9 pass. DSpark-on must preserve official speculative correctness/committed output and produce measurable wall-clock benefit under comparable RAM/cache accounting; otherwise leave it optional/off. Gate N intentionally has no V-number.

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
README gate letter(s)
highest applicable V-level/system gate
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

Never report a cache hit-rate improvement as a speed improvement without wall-clock measurement. `docs/BENCHMARKS.md` is the authoritative ledger and records README letters and V-levels separately.

---

## 20. Suggested PR / commit sequence

Keep each milestone reviewable and oracle-backed.

> `ROADMAP.md` tracks live status against this sequence and is where a phase
> gets marked done. Read it for *what is actually finished*; read this section
> for the intended shape of each step.
>
> **“PR 1” below is not GitHub PR #1.** These are planned units of work and
> have diverged from GitHub numbering. Cite merged commit/PR numbers and the
> canonical gate, not only this planned sequence number.

### PR 1 — bootstrap upstream

- Import WASTE at pinned commit.
- Preserve licenses/NOTICE.
- Keep upstream tests green.
- Add `UPSTREAM.md`.

### PR 2 — DeepSeek checkpoint inventory — Gate A / V0

- `tools/inventory.py`.
- Exact storage/RAM anatomy report once official input is available.
- Add/update this README with measured figures.

Tooling shipped in GitHub PR #1; real Gate A/V0 input remains blocked in the original environment.

### PR 3 — quantized scalar reference kernels — Gates B/C, V1/V2

Planned scope:

- FP4 E2M1 + UE8M0 K32;
- FP8 E4M3/block-scale path;
- official differential tests.

**Actual status:** GitHub PR #3 (`91c36b8…`) landed the public-format scalar decoders/matvecs and exhaustive model-free tests. Gate B/V1 is half satisfied; official DeepSeek convention fixtures and Gate C/V2 remain blocked on reference/checkpoint access.

### PR 4 — DeepSeek container + converter

- New model family/format version.
- Native expert record after Gates A/B.
- Manifest + resume + verifier.
- Convert a one-layer/subset fixture first.

### PR 5 — mHC — Gate D / V3

- Sinkhorn/mixing implementation.
- Real-weight oracle fixtures.

### PR 6 — attention — Gate E / V5

- ratio 0 first;
- compressor;
- ratio 128;
- CSA indexer + ratio 4;
- incremental cache tests.

### PR 7 — DeepSeek MoE — Gate F / V4

- bootstrap/hash routing as proven by Gate A;
- score routing;
- shared expert;
- native routed expert;
- one-layer parity.

### PR 8 — WASTE expert streaming integration — Gate G + planner groundwork

- bank reader + cache;
- exact early-layer prefetch only if proven;
- cache identity tests;
- memory planning.

### PR 9 — full base model — Gates H/I/K, V6–V9

- complete blocks;
- multi-layer localization;
- final HC head;
- LM head;
- greedy generation;
- session state.

### PR 10 — encoding + CLI/server — Gate J / V10 + V11

- official 0731 prompt encoding parity;
- reasoning/tools only as officially supported;
- OpenAI-compatible serving.

### PR 11 — chunked prefill + profiling — reinforce Gates G/L/M

- deduplicate expert reads within chunks;
- route tracing/cache simulator;
- benchmark harness;
- learned hotlist.

### PR 12 — DSpark — Gate N

- speculative module;
- acceptance tests;
- benchmark on/off.

### PR 13+ — optimization experiments

Only after the relevant correctness gates consider:

- SIMD FP4/FP8 kernels;
- lower-precision attention state;
- WASTE VQ3R/VQ4P expert requantization;
- one-layer-ahead route prediction;
- Metal/CUDA/offload;
- multi-NVMe bank striping;
- expert hot-set pinning;
- indexer streaming optimizations.

Each experiment needs a kill criterion and a permanent negative-result entry if it loses.

---

## 21. Immediate TODO checklist for the next engineer

Do these in canonical-gate order:

- [x] Bootstrap pinned WASTE source and prove `make check` still passes.
      Imported `d9b919a` verbatim; baseline was 31 passed / 0 failed / 12 skipped before any local change. See `UPSTREAM.md`.
- [x] Preserve WASTE Apache-2.0 attribution. The exact DeepSeek license text remains intentionally unresolved until authorized official access; see `LICENSES/README.md`.
- [x] Implement `tools/inventory.py` using safetensors headers only and test it with the synthetic fixture.
- [ ] **Gate A / V0:** run the real pinned metadata/header inventory and produce `docs/INVENTORY-0731.md`/machine-readable exact totals. **Blocked on official access, not on code.**
- [x] **Gate B / V1a:** implement scalar E2M1 + UE8M0 and finite-E4M3 decode/matvec paths; exhaustive public-format/model-free tests landed in GitHub PR #3.
- [ ] **Gate B / V1b:** reconcile FP4 nibble order, scale direction/layout, and target native conventions with the pinned official reference; freeze independent fixtures.
- [ ] **Gate C / V2:** differential-test one official quantized trunk projection across a scale boundary.
- [ ] Build the Python oracle fixture generator around the pinned official `inference/` source.
- [ ] Extend WASTE container format with an explicit DeepSeek/native expert record only after Gates A/B.
- [ ] Convert exactly one layer/subset and verify every record.
- [ ] **Gate D / V3:** implement mHC and pass real-weight fixtures.
- [ ] **Gate F / V4:** implement routing/shared/routed MoE parity.
- [ ] **Gate E / V5:** bring up attention modes.
- [ ] **Gate G:** connect one streamed MoE path to WASTE cache and prove disk/cache/prefetch identity.
- [ ] Only then move into full block/base integration and performance work.

If you are tempted to optimize before the relevant correctness checkbox is green, add the idea to `docs/EXPERIMENTS.md` or the research backlog and keep going with correctness.

---

## 22. The first commands to run in an authorized environment

```bash
# 1. Clone our repo.
git clone https://github.com/Svyable/deepseek-v4-flash-wasted.git
cd deepseek-v4-flash-wasted

# 2. Verify the pinned WASTE provenance recorded by the repo.
git show d9b919a791148b571e643d0af666bf19b4d733ab:README.md | head || true

# 3. Pull only lightweight DeepSeek legal/metadata/reference assets first.
#    Do NOT make the giant weight download the first dependency.
python -m pip install 'huggingface_hub>=0.34'
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash-0731',
    revision='<FULL_RESOLVED_SHA>',
    local_dir='reference/deepseek-v4-flash-0731',
    allow_patterns=[
        'config.json',
        'generation_config.json',
        'model.safetensors.index.json',
        'tokenizer*',
        'inference/*',
        'encoding/*',
        'LICENSE*',
        'NOTICE*',
        'README.md',
    ],
)
PY

# 4. Run the existing inventory tool immediately (Gate A / V0 index-only pass).
python tools/inventory.py reference/deepseek-v4-flash-0731
```

Follow `docs/REFERENCE_ACCESS.md` for the tiered acquisition sequence. Never silently fall back to mutable `main` in a measurement/fixture script.

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

1. **What is the real 0731 routing locality?** Sparse top-k does not determine cache value; Gate M needs reuse distribution.
2. **How much does deterministic/bootstrap routing help I/O overlap?** Measure only after Gate A/V0 proves the exact mapping.
3. **What is the practical resident-trunk floor?** Gate A/V0 + real runtime representation decide it.
4. **Does native quantized compute or NVMe dominate?** The answer determines whether SIMD work or read-ahead is the next lever.
5. **Can WASTE's LFRU policy beat LRU here too?** Gate M: measure; do not assume.
6. **How large must the cache be to survive the real working set?** Gate M, with Gate G preserved.
7. **Can the CSA indexer itself be streamed/chunked at very long context?** At large context, indexer memory traffic may become a separate bottleneck.
8. **Is lower-precision attention state materially faster/better on CPU/unified-memory targets?** Compare only after the reference state semantics are correct.
9. **Does 3-bit WASTE VQ preserve 0731 quality?** Test only after native Gate I/V8 + Gate K/V9 baseline exists.
10. **Does DSpark still help when the target is storage-bound?** Gate N must count committed tokens, acceptance, RAM/cache delta, and I/O.
11. **Would two or more NVMe devices scale random expert reads?** Gate L/M experiment after the real record layout exists.
12. **Can a learned hotlist pin high-value experts without increasing paging pressure?** Evaluate against the same hard RAM ceiling.

Record failed experiments. A negative result that prevents a future week of work is part of the implementation.

---

## 25. Definition of done for v0.1

We can call the first DeepSeek WASTE port real when all of these are true:

- [ ] **Gate A / V0:** exact pinned-checkpoint tensor/storage inventory exists.
- [ ] A reproducible converter creates a verified DeepSeek-specific WASTE container from the pinned 0731 checkpoint.
- [ ] **Gate B / V1 + Gate C / V2:** native quantization conventions/projection are officially reconciled.
- [ ] **Gate H / V6 + V7 + Gate I / V8:** the complete base path runs and final logits match the oracle.
- [ ] **Gate G:** cache hit vs disk miss/prefetch mode is numerically identical for the same arithmetic path.
- [ ] **Gate K / V9:** greedy next tokens match the Python reference on a published fixture set.
- [ ] **Gate J / V10:** official DeepSeek message encoding/parser tests pass.
- [ ] `waste plan` reports a real measured RAM floor and context-dependent state.
- [ ] The process respects a hard RAM budget/cgroup limit.
- [ ] **Gate L:** a storage benchmark uses the real record size/access pattern on target storage.
- [ ] **Gate M:** decode/cache benchmarks publish real routing hit/traffic curves alongside tok/s.
- [ ] A consumer/workstation-class machine can run the **full** model without requiring all routed experts in RAM.
- [ ] **Gate N:** DSpark is either correct + positively measured or intentionally disabled.

Then optimize.

---

## 26. Core engineering principle

WASTE's strongest idea is not a quantizer or a cache policy. It is the separation of **correctness from placement**:

> The real router chooses the real experts. The real expert bytes produce the real output. RAM versus NVMe only decides when those bytes arrive.

That is **README Gate G** in one sentence.

Keep that invariant through the DeepSeek port. It gives us a clean ladder from a slow scalar implementation to aggressive prefetch, caching, SIMD, alternate quantization, and speculative decoding without losing the ability to say exactly which optimization changed the model and which only changed its speed.

That is how we get DeepSeek V4 Flash **WASTED** without turning the project into an un-debuggable pile of approximations.