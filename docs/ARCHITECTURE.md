# Architecture — DeepSeek V4 Flash on WASTE

**Status: DESIGN, grounded in the WASTE bootstrap and PR #3 scalar quantization work. No DeepSeek transformer forward pass exists yet.**

This document defines the intended system boundaries for the port. It does not freeze tensor names or byte counts; those are owned by the pinned official checkpoint and `tools/inventory.py`.

Gate terminology follows `docs/VALIDATION.md` §4a. In this document, checkpoint/storage assumptions are **Gate A / V0**, mHC/model primitives are **Gate D / V3**, routing/MoE is **Gate F / V4**, attention is **Gate E / V5**, a complete block is **Gate H / V6**, final base logits are **Gate I / V8**, and placement identity is **Gate G**.

## 1. System objective

Run the full `deepseek-ai/DeepSeek-V4-Flash-0731` base model with a WASTE-style memory/storage strategy:

```text
official HF checkpoint
        |
        v
header inventory + reference/oracle
        |
        v
DeepSeek converter ----------------------+
        |                                |
        v                                v
DeepSeek .waste container          validation fixtures
        |
        v
libwaste public API
        |
        +--> model-family loader
        |       |
        |       +--> resident trunk
        |       +--> bounded routed-expert cache
        |       +--> direct aligned expert reads from NVMe
        |
        +--> CLI
        +--> Python ctypes server --> OpenAI-compatible localhost API --> optional UI
```

The project is not trying to transform DeepSeek into Kimi. It reuses WASTE's storage/runtime machinery and supplies a DeepSeek-specific model path.

## 2. Evidence boundary inherited from PR #1 and PR #3

What is real today:

- the complete WASTE source tree is present at the pinned bootstrap revision;
- the generic cache, memory, direct-I/O, CLI, server, format parser, SIMD/backend and testing infrastructure can be inspected and reused;
- the imported WASTE baseline passed its model-free suite before DeepSeek modifications;
- `tools/inventory.py` is a tested header-only checkpoint inventory tool;
- PR #3 added scalar E2M1/UE8M0/E4M3FN decode/matvec paths with exhaustive public-format tests;
- README Gate B / V1 is only half satisfied: official DeepSeek packing/scale conventions remain unverified;
- no official 0731 checkpoint header was available to the bootstrap environment;
- no DeepSeek transformer forward path is implemented.

Therefore architecture-specific dimensions and placements remain subject to **Gate A / V0**, and native byte/scale arithmetic assumptions remain subject to **Gate B / V1**. The checkpoint/reference may force changes to this design.

## 3. Reuse boundary

### Reuse or generalize from WASTE

These are model-independent or close enough to be generalized:

| Component | Intended treatment | Reason |
|---|---|---|
| `src/waste.h` / public API discipline | reuse | library first; CLI/server are clients |
| `src/waste.c` orchestration | generalize | model dispatch must stop assuming Kimi internals |
| `src/ecache.*` | reuse/generalize | bounded expert cache is central to the port |
| async reader threads | reuse | streaming experts must overlap with compute where useful |
| `src/memory.*` | reuse/generalize | hard RAM ceiling and cgroup awareness are requirements |
| `src/platform.h` | reuse | platform-specific aligned/direct I/O already exists |
| `src/crc32.*` | reuse | expert-record verification is model-independent |
| `src/backend.*` / SIMD dispatch | reuse | new kernels register behind the same backend model |
| thread pool | reuse | model kernels need the same controlled worker pool |
| CLI | reuse/generalize | commands should stay clients of the public C API |
| `serve/` ctypes architecture | reuse/generalize | Python owns protocol/prompt logic, C owns inference |
| synthetic-container testing pattern | reuse | essential before real-weight conversion |
| gate/negative-result discipline | reuse | expensive experiments need cheap kill tests |

### Do not reuse as DeepSeek arithmetic

The following remain upstream Kimi implementation/reference material:

- KDA (`src/kda.*`);
- Kimi MLA assumptions and latent-cache math embedded in the current model path;
- K3 AttnRes/residual behavior;
- Kimi-specific tokenizer/XTML prompt encoding;
- Kimi converter name mapping;
- Kimi routing semantics;
- Kimi VQ3R/VQ4P expert format as the initial DeepSeek quantization path;
- Kimi-specific config fields as the DeepSeek model contract.

They may provide implementation patterns, but parity with Kimi is not a correctness criterion for DeepSeek.

## 4. Proposed source decomposition

Do not perform a large directory refactor before it buys a testable boundary. The target structure is:

```text
src/
  waste.h
  waste.c
  backend.*
  ecache.*
  memory.*
  platform.h
  waste_format.h / generic record utilities
  quant/
    fp4_e2m1.*
    fp8_e4m3.*
    scalar_reference.*
  models/
    deepseek_v4/
      config.*
      bind.*
      model.*
      mhc.*
      rope.*
      attention.*
      compressor.*
      indexer.*
      router.*
      moe.*
      dspark.*
```

The actual refactor should happen incrementally. A new module earns its place when there is an oracle test around its boundary.

## 5. Data placement model

The WASTE idea is useful only if placement follows use frequency.

### Resident set

Candidate resident data includes everything needed on every token or too fine-grained to stream economically:

- embeddings / output head subject to memory planning;
- norms;
- mHC parameters/state;
- attention Q/KV/O projections;
- compressors and CSA indexers;
- routers/hash-routing tables;
- shared experts;
- positional/RoPE parameters;
- quantization metadata/codebooks/scales required by resident tensors;
- recurrent or context-dependent attention state;
- scratch buffers bounded by the planner.

`tools/inventory.py` currently classifies every non-routed-expert tensor as a resident candidate. **Gate A / V0** and later profiling may refine that, but streaming arbitrary trunk tensors is not the initial design.

### Streamed set

The initial streamed population is routed MoE experts only.

An independently readable expert record should contain all bytes required to apply one selected expert without a second filesystem lookup. Exact composition is checkpoint-derived, but the design intent is:

```text
record header
matrix 1 packed weight + scale metadata
matrix 2 packed weight + scale metadata
matrix 3 packed weight + scale metadata
padding to direct-I/O alignment
```

The converter may reorder matrices on disk for execution locality, but it must preserve official arithmetic.

## 6. Runtime lifecycle

### Open / plan

1. Parse the manifest and identify the model family/version.
2. Validate every dimension/offset/record count before allocating large buffers.
3. Compute the mandatory RAM floor from exact container metadata.
4. Resolve user budget or safe automatic budget using WASTE's capacity logic.
5. Refuse a budget below the floor.
6. Load/map/decode the resident trunk according to its native stored formats.
7. Allocate bounded expert cache and I/O buffers from the remaining budget.
8. Open expert banks with requested direct-I/O/page-cache-bypass semantics.
9. Report whether bypass actually engaged.

### Prefill

Correctness baseline:

1. encode the prompt with the official DeepSeek encoding semantics;
2. process tokens in the simplest oracle-equivalent way;
3. record routes and expert reads;
4. only after parity, introduce chunked prefill that groups repeated experts across tokens.

Chunked prefill is a WASTE-proven optimization pattern, not assumed to help DeepSeek until measured.

### Decode token

Conceptual flow for one layer:

```text
input state
  -> mHC/residual preparation
  -> attention path
       -> sliding-window and/or compressed-attention machinery
       -> compressor/indexer where the layer config requires it
  -> mHC/residual merge
  -> router
       -> bootstrap/hash selection in layers where official semantics say so
       -> learned scoring/top-k elsewhere
  -> shared expert (resident)
  -> routed expert IDs + weights
       -> cache lookup
       -> async/direct read for misses
       -> native FP4 expert application
  -> combine MoE outputs
  -> block output
```

The exact order is not frozen here. The official reference/oracle controls it.

## 7. Routing and prefetch boundary — Gate A/V0 → Gate F/V4, with Gate G invariant

Routing is authoritative; prefetch is not.

If **Gate A / V0** and official reference code confirm deterministic token-ID-to-expert routing in early layers, the runtime can issue those reads as soon as the token ID is known. That is a high-confidence prefetch opportunity because it does not predict the route.

**Gate F / V4** later proves actual routing + one-MoE-layer parity.

For learned-routing layers, lookahead may be explored later, but:

- the real router still selects experts;
- prefetched-but-unused records are only wasted I/O/cache pressure;
- a prefetched value cannot substitute for an expert the real router selected;
- correctness tests with prefetch off and on must agree under **Gate G**.

## 8. Quantization architecture — Gates B/C, V1/V2

The correctness baseline preserves official native formats.

### Routed experts

Current handoff: packed FP4 E2M1 with scale metadata. PR #3 implements the scalar public-format path. **Gate B / V1** must still reconcile DeepSeek nibble order, scale application/layout, and official fixtures before the path is considered model-correct.

### Resident quantized trunk

Current handoff: predominantly FP8 E4M3 with block scaling. PR #3 implements an E4M3FN scalar path; **Gate B / V1** must verify the target convention and **Gate C / V2** must prove one official quantized trunk projection.

### Deliberately deferred

- VQ3R/VQ4P expert requantization;
- pruning;
- per-expert bit allocation;
- substitute experts;
- lossy trunk compression beyond the official release.

Those become experiments only after a native-format end-to-end baseline exists.

## 9. Container family boundary

The DeepSeek container must not masquerade as WASTE Kimi format v0.

At minimum the manifest requires:

- format/schema version;
- explicit model family, e.g. `deepseek_v4_flash`;
- pinned source repo/revision;
- converter commit;
- original config or normalized config plus source hash;
- tensor index for resident data;
- per-layer expert bank metadata;
- quantization descriptors read rather than assumed by the runtime;
- tokenizer/encoding assets or references required for a self-contained local run;
- DSpark presence/version as a separate optional component.

Details live in `CONTAINER_V4.md`.

## 10. Correctness seams and canonical gates

The implementation is intentionally separable at these boundaries:

| Seam | Canonical gate |
|---|---|
| checkpoint inventory | Gate A / V0 |
| bit-level native decode + DeepSeek convention | Gate B / V1 |
| one quantized trunk projection | Gate C / V2 |
| mHC and other model primitives | Gate D / V3 |
| routing/shared/routed MoE | Gate F / V4 |
| attention modes | Gate E / V5 |
| one full block | Gate H / V6 |
| multi-block hidden-state localization | V7 |
| final logits | Gate I / V8 |
| greedy generation | Gate K / V9 |
| prompt encoder/parser | Gate J / V10 |
| API surface | V11 |
| disk/cache/prefetch identity | Gate G |
| real target storage | Gate L |
| cache curve/recommendation | Gate M |
| DSpark | Gate N |

A failure should be localized to the earliest mismatching seam rather than debugged from bad text output.

## 11. Observability requirements

The port should retain or add telemetry for:

- resolved RAM budget and floor breakdown;
- resident bytes by subsystem;
- expert cache capacity in bytes and records;
- expert hits/misses, bytes read, and read latency;
- direct-I/O active/fallback state;
- routed experts per layer/token;
- prefetch issued/useful/wasted;
- prefill and decode timings;
- backend/kernel selection;
- context/state bytes;
- model/container/source revisions.

A benchmark without enough metadata to reproduce its storage and memory conditions does not belong in `BENCHMARKS.md`.

## 12. API architecture

Preserve WASTE's rule: **if the CLI/server needs an inference capability, put it in `waste.h` first.**

C owns:

- planning/open/close;
- tokenize/detokenize interfaces as appropriate;
- eval/generate;
- session/model state;
- stats and error details.

Python server owns:

- OpenAI request validation;
- DeepSeek conversation encoding;
- response/tool-call parsing;
- SSE framing;
- protocol-level transformations.

The UI, if added, is a separate client of the localhost API. It must not host the model inside a rerun-oriented UI process.

## 13. DSpark boundary — Gate N

DSpark is loaded and executed only after the base model path is correct. Architecturally it is an optional acceleration module, not part of the minimum ability to evaluate the 43-layer base model.

The loader/container should therefore permit:

```text
base model only
base model + DSpark assets, DSpark disabled
base model + DSpark assets, DSpark enabled
```

The first two must produce the same base decode behavior. Gate N does not start until Gate I/V8 and Gate K/V9 pass. See `DSPARK.md`.

## 14. What would invalidate this architecture

Update this document rather than forcing the implementation if **Gate A / V0** or official reference code shows that:

- routed experts cannot be represented as one self-contained record without pathological duplication;
- tensors classified as resident are actually sparsely selected at useful granularity;
- early routing is not deterministic/hash-addressable as expected;
- DSpark cannot be cleanly separated from base evaluation;
- official quantization storage/layout differs from the handoff assumptions;
- the attention state model makes the proposed RAM floor materially incomplete.

If Gate B/V1 shows current nibble/scale arithmetic conventions are wrong, update the numerics path and fixtures rather than forcing the checkpoint to agree.

The checkpoint/reference code wins. This document is a map for implementation, not a constraint on reality.