# Validation — correctness gates for the DeepSeek V4 port

**Status: DESIGN. Inventory tooling is SYNTHETIC-VERIFIED; DeepSeek numerical parity is NOT STARTED.**

This project should never debug model correctness from generated prose. Validation proceeds from the smallest exact seam to final logits, and each expensive phase is protected by a cheaper test.

## 1. Correctness oracle

The primary oracle is the pinned official `DeepSeek-V4-Flash-0731` reference implementation and checkpoint.

Third-party runtimes are useful smoke tests but are not the numerical authority. WASTE's Kimi tests demonstrate a methodology, not DeepSeek expected values.

Every golden fixture generated from official code must record:

```text
model repository
resolved model revision
reference source revision/path
config hash or exact file
input token ids / positions
random seed if any
reference device
reference dtype/autocast settings
operation name
output tensor shape/dtype
fixture format version
```

If a fixture cannot answer where it came from, do not commit it.

## 2. Tolerance policy

Do not choose one global tolerance before observing the official arithmetic.

Each validation seam defines:

- exact-equality fields, such as selected expert IDs or token IDs;
- absolute error (`max_abs`);
- relative error (`max_rel`) with a documented denominator floor;
- optionally RMS/mean error for large tensors;
- argmax agreement where logits are involved;
- whether the comparison is against decoded native weights or the full official kernel.

A tolerance becomes fixed only after:

1. the scalar C implementation matches the official reference on several deterministic fixtures;
2. the observed error is explained by expected precision/accumulation differences;
3. the threshold is set above normal numerical noise but below a known wrong implementation.

Never loosen a tolerance merely to make a failing optimization pass. Diagnose the mismatch first.

## 3. Exact-equality requirements

Unless the official reference proves otherwise, these should be exact:

- tensor shapes and configured dimensions;
- tokenizer output token IDs;
- special/control-token IDs;
- router-selected expert IDs and ordering;
- deterministic/hash-routed expert IDs;
- cache-on versus cache-off bytes supplied to the arithmetic path;
- container record identity `(layer, expert)`;
- parser structure for official encoder/parser test cases;
- greedy output token IDs once final-logit parity is established closely enough to preserve argmax.

Routing weights themselves are floating-point and use numerical tolerances, but the selected set/order is a hard semantic check.

## 4. Gate ladder

### Gate V0 — checkpoint inventory

Owner: `tools/inventory.py`, `docs/INVENTORY-0731.md`, `docs/TENSOR_MAP.md`.

Pass criteria:

- all main-model names classified;
- no unexplained bytes;
- shapes/config agree;
- expert records are fully understood;
- quantization scales are associated with their owners;
- DSpark can be separated from the base path.

No numerical porting should rely on an unresolved tensor family.

### Gate V1 — bit-level native quantization decode

Test:

- construct small official-reference FP4/FP8 encoded fixtures that exercise sign, exponent/mantissa extremes, zeros and scale boundaries;
- decode them in scalar C;
- compare element-by-element with official reference decode semantics.

Pass:

- exact decoded bit-pattern agreement where the reference representation permits it, otherwise an explicitly justified numerical threshold;
- scale indexing/granularity proven at boundary sizes (including K-block transitions and nontrivial row strides).

This gate protects every later layer test.

### Gate V2 — one quantized linear projection

Test representative matrix shapes small enough for fixtures, including rows/columns that cross packing and scale-block boundaries.

Compare:

- decoded/reference weight values;
- output vector;
- accumulation behavior;
- bias if the official op has one.

Pass before adding SIMD.

### Gate V3 — model primitives

Separate oracle tests for:

- normalization;
- mHC transforms/residual path;
- RoPE;
- activation/SwiGLU and any clamp semantics;
- attention projection substeps;
- compressor;
- CSA indexer score/selection;
- router score transform;
- routing correction/bias behavior;
- top-k selection, normalization and scaling;
- shared expert;
- one routed expert.

No primitive should be tested only through a full layer.

### Gate V4 — one MoE block

Use a fixture containing:

- input hidden state;
- router output from official reference;
- selected expert IDs and weights;
- shared-expert output;
- each selected routed-expert output;
- combined MoE result.

Test both:

```text
expert bytes supplied directly
expert bytes fetched through WASTE cache/disk abstraction
```

The two C paths must agree. Placement cannot change numerics.

### Gate V5 — one attention block

Create separate fixtures for each distinct attention mode found in the official config/reference, rather than assuming one implementation covers all layers.

Capture enough intermediates to localize errors:

- Q/K/V or normalized compressed equivalents;
- positional transform inputs/outputs;
- compression state/output;
- indexer scores and selected positions where applicable;
- attention probabilities/accumulated output where practical;
- final projected attention result.

Context-boundary fixtures must include:

- position 0;
- sliding-window boundary - 1 / boundary / +1;
- compression boundary transitions;
- short prompt versus longer history;
- final supported context edge when feasible with synthetic shapes.

### Gate V6 — one full transformer layer

Compare official and C:

- layer input;
- after attention/mHC merge;
- routing IDs/weights;
- MoE output;
- layer output.

Run at least one example of every structurally distinct layer class proven by config/reference (e.g. bootstrap-routing versus learned-routing; attention-mode variants).

### Gate V7 — multi-layer hidden states

Run a short token sequence through multiple real layers and checkpoint hidden states at selected layer boundaries.

A mismatch must be attributed to the first divergent layer before continuing.

This catches state-update/order errors that isolated block fixtures miss.

### Gate V8 — final logits

Compare the final hidden state, final norm, and logits for deterministic input tokens.

Record:

- `max_abs`;
- `max_rel`;
- RMS error;
- top-1 token and top-N ordering for a small N;
- logit magnitude range.

The project may not claim model correctness without this gate on real weights.

### Gate V9 — greedy generation

With sampling disabled and the same stopping rules:

- prompt tokens must match official encoding exactly;
- generated token IDs should match token-for-token for a meaningful short sequence;
- if they diverge, return to final-logit diagnostics rather than declaring sampling noise.

### Gate V10 — encoding and parser

Port official `encoding/` behavior separately from model arithmetic.

Required:

- all official encoding tests;
- content containing strings that resemble control markers;
- tools/tool results if supported by the official encoder;
- reasoning/content region behavior if present;
- malformed/truncated model output parser tests;
- incremental/streaming parser tests if the server exposes SSE.

Do not flatten a code-based encoder into a guessed Jinja template.

### Gate V11 — OpenAI-compatible API

Test the existing WASTE server architecture with the DeepSeek encoder/model path:

- health/model listing;
- non-streaming chat completion;
- streaming chat completion;
- deterministic request parity with direct C generation;
- cancellation/client disconnect behavior;
- structured/tool fields only where supported by the DeepSeek encoding/parser contract.

API success is not a substitute for V8/V9.

## 5. Cache and I/O correctness matrix

For the same token sequence/container, compare:

| Mode A | Mode B | Requirement |
|---|---|---|
| cache disabled | cache enabled | same selected experts and numerical output |
| direct I/O | page-cache fallback | same bytes/numerics |
| cold cache | learned/preloaded cache | same numerics |
| prefetch off | prefetch on | same routing and output |
| sequential prefill | chunked prefill | agreed logits within fixed tolerance |
| one thread where supported | multiple threads | same semantics; numerical tolerance documented if reduction order differs |

Any optimization whose enable flag changes model meaning is not an optimization.

## 6. Synthetic tests versus real-model tests

### Model-free/synthetic tests should cover

- manifest parser bounds;
- expert record offsets/alignment;
- checksum/header corruption;
- cache identity and eviction behavior;
- FP4/FP8 bit operations using tiny fixtures;
- memory-plan arithmetic;
- model-family dispatch;
- tokenizer/encoder unit behavior where fixtures can be legally/self-containedly included;
- platform I/O wrappers.

These run in ordinary CI once Actions is restored.

### Real-model tests should cover

- exact tensor binding;
- official intermediate activations;
- final logits;
- generation;
- routing traces;
- performance and RAM behavior.

These may remain separate/manual because the model is too large for normal CI.

## 7. Golden fixture format

Prefer a simple auditable format over Python pickles. One option:

```text
reference/golden/<operation>/<case>/
  meta.json
  input-*.bin
  output-*.bin
```

`meta.json` declares dtype, shape, endianness and provenance. Raw binary arrays are deterministic and readable from C/Python without a dependency-heavy loader.

Do not commit large portions of model weights as fixtures. Choose tiny slices or synthetic matrices that exercise the exact format/semantics while respecting license/distribution constraints.

## 8. Failure triage order

When final logits are wrong, investigate in this order:

1. wrong prompt/token IDs;
2. wrong tensor mapping/shape/transpose;
3. quantization decode/scale indexing;
4. primitive arithmetic;
5. attention state/position update;
6. router selected IDs/order/weights;
7. expert record identity/data;
8. residual/mHC ordering;
9. layer-state persistence;
10. final norm/head.

Do not begin by changing tolerances.

## 9. Optimization acceptance rule

An optimized kernel/path lands only when:

- the scalar/reference path remains available at least for tests/debug builds;
- it passes the same fixture suite;
- real-model V8/V9 do not regress;
- the benchmark records the hardware/configuration where the speedup exists;
- memory usage does not silently violate the planner;
- failures fall back safely where a backend is optional.

## 10. Result recording template

Add a result to the relevant PR and `BENCHMARKS.md`/`EXPERIMENTS.md` with:

```text
Date:
Port commit:
Model revision:
Container/converter revision:
Hardware / OS:
Operation/gate:
Reference command:
C command:
Input fixture/prompt:
max_abs:
max_rel:
RMS:
exact semantic checks:
Verdict:
Notes:
```

A naked statement like “matches PyTorch” is not sufficient evidence.