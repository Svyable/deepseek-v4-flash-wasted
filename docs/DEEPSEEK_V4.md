# DeepSeek V4 Flash 0731 — porting reference

**Status: HANDOFF/OFFICIAL-SPEC summary, not yet CHECKPOINT-VERIFIED by this repository.**

This document collects the DeepSeek-specific concepts an implementer needs without requiring them to reverse-engineer the project README every time. It is intentionally subordinate to the pinned official checkpoint and official `inference/` / `encoding/` code.

The bootstrap environment used for PR #1 could not reach Hugging Face, so the exact tensor names, stored shapes and byte totals remain pending **README Gate A / V0**.

Gate terminology follows `docs/VALIDATION.md` §4a. In particular: Gate A/V0 is checkpoint truth, Gate B/V1 native quantization semantics, Gate D/V3 mHC/primitives, Gate F/V4 routing+MoE, Gate E/V5 attention, Gate I/V8 final logits, Gate K/V9 greedy generation, Gate J/V10 encoding, and Gate N DSpark.

## 1. Target

Primary target:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
pinned revision in project handoff: 9e165c3
```

Do not silently switch to:

- the earlier `DeepSeek-V4-Flash` checkpoint;
- V4-Pro;
- a GGUF conversion;
- a third-party serving checkpoint.

A target revision change is a project event: inventory, tensor mapping, oracle fixtures and conversion provenance all need to move together.

## 2. Current architecture snapshot — Gate A / V0 pending

The existing README records the following release/config values from the research handoff. Treat them as **OFFICIAL-SPEC pending local Gate A/V0 verification**:

| Property | Current handoff value |
|---|---:|
| main decoder layers | 43 |
| hidden size | 4096 |
| routed experts / layer | 256 |
| routed experts active / token / layer | 6 |
| shared experts / layer | 1 |
| MoE intermediate size | 2048 |
| hash-routed bootstrap layers | 3 |
| attention heads | 64 |
| attention head dim | 512 |
| query LoRA rank | 1024 |
| output LoRA rank | 1024 |
| sliding window | 128 |
| maximum context | 1,048,576 |
| Hyper-Connection multiplier | 4 |
| CSA indexer heads | 64 |
| CSA indexer head dim | 128 |
| CSA selected compressed positions | 512 |
| routed expert storage | FP4 E2M1 |
| expert scale | UE8M0, K32 according to current handoff |
| most other quantized weights | FP8 E4M3, 128×128 blocks according to current handoff |
| routed scaling factor | 1.5 |
| SwiGLU clamp | 10.0 |
| DSpark target main layers | 40, 41, 42 |
| DSpark block size | 5 |
| DSpark Markov rank | 256 |

These values are implementation hypotheses until `tools/inventory.py` and the official reference establish how they appear in the released files.

## 3. Why WASTE is a plausible fit

The architectural match is sparse routed computation:

```text
many routed experts exist
only top-k experts are used for a token in a layer
```

If each selected expert can be addressed as one independent disk record, WASTE can:

- keep always-used trunk/state in RAM;
- fetch only selected routed experts;
- cache a bounded working set in remaining RAM;
- overlap future reads with current compute where routing timing permits;
- preserve exact expert bytes whether served from disk or cache.

The fit is about sparse access, not shared attention architecture. DeepSeek's trunk needs a new forward path.

## 4. Manifold-Constrained Hyper-Connections (mHC) — Gate D / V3

DeepSeek V4's residual/connectivity mechanism is not Kimi AttnRes and must be implemented from the official reference.

Porting questions the oracle must settle:

- exact input/output state shape with the configured multiplier;
- where normalization occurs;
- transforms/gates applied before attention and MoE;
- residual mixing order;
- parameter shapes and precision;
- which state is persistent versus per-layer temporary;
- where DSpark taps layer state, if applicable.

Validation requirement: isolate mHC with official intermediate fixtures under Gate D/V3 before testing a full block.

Do not translate mHC into WASTE's K3 residual structures because both “mix residual streams.” Similar purpose is not equivalent math.

## 5. Hybrid attention — Gate E / V5

The current handoff describes a schedule containing:

- pure sliding-window attention layers;
- compressed attention with one or more compression ratios;
- compressed sparse attention using a learned indexer/selected historical positions.

The README records a `compress_ratios` schedule beginning with two zero entries, then alternating `4` and `128` for much of the main stack, with trailing entries associated with the speculative path. **Do not implement from that sequence alone.** The official reference controls what each value means and when compressors/indexers update state.

The attention port should separate:

```text
projection
positional transform / RoPE
window state
compression state/update
indexer scoring + top-position selection
dense/sparse attention over the selected representation
output projection
```

This separation creates oracle seams and makes context-memory accounting possible.

## 6. Long-context state

The advertised maximum context does not mean the runtime should allocate a conventional full KV cache for every layer.

`MEMORY_AND_IO.md` requires the official reference to establish persistent state per attention mode. The implementation should allocate for the requested context and actual compressed/window representation.

Boundary tests must cover:

- first token;
- window edge;
- compressor update edges;
- sparse-selection edges;
- reset/session restore if the public API supports it;
- context-full behavior.

## 7. MoE structure — Gate F / V4

The current handoff models each routed expert as a conventional SwiGLU triplet with normalized matrix roles:

```text
w1 / gate projection
w3 / up projection
w2 / down projection
```

`tools/inventory.py` accepts both `w1/w2/w3` and common `gate_proj/up_proj/down_proj` aliases for classification. This is only a classifier convenience until the real checkpoint is read.

**Gate A / V0** must first prove every routed expert contains the complete tensor set and scale metadata required for one apply. **Gate F / V4** later proves routing + shared/routed expert behavior for one MoE block.

The shared expert is always-used and belongs in the resident trunk for the first implementation unless measurement later gives a compelling reason otherwise.

## 8. Routing — Gate A/V0 prerequisite, Gate F/V4 parity

The current handoff distinguishes two routing regimes:

### Early/bootstrap routing

The first three layers are described as hash/deterministic token-ID-to-expert routing. If the official reference confirms expert IDs are known as soon as the input token is known, WASTE can prefetch those expert records without prediction risk.

Gate A/V0/reference inspection must determine whether the mapping is:

- stored as checkpoint tensor(s);
- generated from config;
- embedded in code;
- direct expert IDs or an intermediate mapping.

Do not write a deterministic prefetch path until this is proven.

### Learned routing

Later routing is described in the README as involving `sqrt(softplus(logit))`, top-6 selection, correction/bias behavior, renormalization and a routed scaling factor.

Every step must be copied from official semantics and tested separately before Gate F/V4 can pass:

```text
raw router projection
score transform
correction used for selection versus output weighting
selected IDs/order
normalization
final route scaling
```

Selected expert IDs/order are exact validation targets.

## 9. Native routed-expert FP4 — Gate B / V1

The first correct engine preserves the official expert representation rather than requantizing it.

Current handoff expectation:

```text
E2M1 FP4 packed values
UE8M0 scale values
a scale per 32 logical values along K
```

PR #3 proves the public-format half. Gate A/V0 establishes actual checkpoint storage geometry; Gate B/V1 must still prove:

- packing nibble/bit order;
- target E2M1 convention as used by official code;
- stored versus logical matrix shape;
- scale encoding/application direction;
- scale axis/block size;
- scale indexing/layout;
- arithmetic semantics required for official parity.

Keep the scalar decoder/matvec. SIMD is a later optimization after Gate B/V1 and Gate C/V2.

## 10. Resident FP8 — Gate B/V1 then Gate C/V2

Most non-expert quantized weights are described as FP8 E4M3 with block scales in the handoff.

PR #3 implements a finite E4M3 (`e4m3fn`) scalar path. Gate B/V1 must confirm the official target tensor-family convention and scale application. Gate C/V2 then proves one official quantized trunk projection.

The runtime needs to know whether it:

- keeps the packed weights resident and decodes/applies on the fly;
- expands some small tensors;
- precomputes auxiliary kernel metadata.

That choice affects both the RAM floor and performance and must be recorded in `MEMORY_AND_IO.md`.

## 11. SwiGLU/clamp semantics — V3 primitive, Gate F/V4 composition

The current handoff records a SwiGLU clamp of `10.0`. The exact order matters:

- which branch is clamped;
- before/after activation;
- whether both gate/up values are affected;
- accumulation dtype.

Do not implement “standard SwiGLU + clamp somewhere.” Generate a one-expert oracle fixture that crosses negative, zero and clamp-boundary values. The primitive belongs in V3; one-MoE-layer composition belongs in Gate F/V4.

## 12. Encoder and response parsing — Gate J / V10

The release supplies code under `encoding/` rather than relying only on a Jinja chat template. That makes encoding code part of the model contract.

The port should:

1. retrieve the official encoder/tests at the pinned revision;
2. resolve licensing before adapting/copying source;
3. build differential tests that compare exact token IDs;
4. keep user content separate from structural/control tokens according to official semantics;
5. port incremental response parsing where the local server needs streaming/tool support.

Do not reuse Kimi XTML simply because WASTE's server already implements it.

## 13. DSpark — Gate N

DSpark is an optional speculative layer on top of the base model. Its current handoff parameters are not enough to implement it correctly.

See `DSPARK.md` for Gate N. The base path must already pass Gate I/V8 final logits and Gate K/V9 generation with DSpark absent/disabled.

## 14. Model component implementation order

After Gate A/V0/oracle availability, the maintained operational order is:

1. config + canonical tensor binding — Gate A/V0;
2. finish native convention agreement — Gate B/V1;
3. one quantized trunk projection — Gate C/V2;
4. normalization, mHC, RoPE, and other primitives — Gate D/V3;
5. routing/shared/routed MoE — Gate F/V4;
6. attention projections/modes/compression/indexer — Gate E/V5;
7. full block — Gate H/V6;
8. multi-layer localization — V7;
9. final norm/head/logits — Gate I/V8;
10. greedy generation — Gate K/V9;
11. encoder/parser — Gate J/V10;
12. API parity — V11;
13. Gate G/L/M systems/performance work;
14. optimized kernels where justified;
15. DSpark — Gate N.

This follows `VALIDATION.md` §4a. The fact that Gate F/V4 appears before Gate E/V5 is intentional operational ordering; the README letters remain stable identifiers.

## 15. Facts this document intentionally does not claim

Until the relevant canonical gates pass, this document does not establish:

- exact checkpoint tensor names — Gate A/V0;
- exact stored checkpoint bytes — Gate A/V0;
- official packed/scaled quantization conventions — Gate B/V1;
- final `.waste` container size;
- RAM floor at any context;
- real expert-record size;
- real bytes read/token;
- cache hit rate — Gate M;
- local tok/s;
- routing locality;
- DSpark acceptance or speedup — Gate N;
- numerical tolerance values.

Those results belong to the inventory, validation, memory and benchmark documents after they are measured.