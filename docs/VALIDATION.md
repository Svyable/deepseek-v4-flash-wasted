# Validation — correctness gates for the DeepSeek V4 port

**Status: V0 inventory tooling is SYNTHETIC-VERIFIED. V1 public-format conformance is satisfied; DeepSeek-specific V1 reference agreement and V2 remain BLOCKED on official reference/checkpoint access.**

This project should never debug model correctness from generated prose. Validation proceeds from the smallest exact seam to final logits, and each expensive phase is protected by a cheaper test.

PR #3 (`91c36b8f4168349e6893a9911a3f60075d62d973`) is the first DeepSeek arithmetic milestone: scalar E2M1/UE8M0/E4M3FN paths exist and their public-format semantics are exhaustively tested. That is meaningful progress, but it does not mean official DeepSeek byte conventions or model numerics have been verified.

See:

- `docs/NUMERICS.md` for the native-quantization arithmetic contract;
- `docs/FIXTURES.md` for fixture independence and mutation-testing rules;
- `docs/REFERENCE_ACCESS.md` for the smallest official-reference acquisition sequence.

## 1. Correctness oracle

The primary model oracle is the pinned official `DeepSeek-V4-Flash-0731` reference implementation and checkpoint.

Public numeric-format specifications are authoritative for E2M1/UE8M0/E4M3 semantics they actually define. They are not authoritative for DeepSeek-specific packing, tensor orientation, scale direction, scale layout, routing, or model operations.

Third-party runtimes are useful smoke tests but are not the numerical authority. WASTE's Kimi tests demonstrate a methodology, not DeepSeek expected values.

Every golden fixture generated from official code must record:

```text
model repository
resolved model revision
reference source revision/path
reference source hash
config hash or exact file
input token ids / positions
random seed if any
reference device
reference dtype/autocast settings
operation name
output tensor shape/dtype
fixture format version
generator commit
```

If a fixture cannot answer where it came from, do not commit it.

### 1a. Fixture independence is part of the oracle

Expected fixture values must not be generated through the same convention/helper being tested.

PR #3 proved why: the original FP4 layout test packed fixtures through the same nibble-order macro used by the decoder. Reversing that macro reversed producer and consumer together, so exhaustive code-space coverage still passed a wrong convention. The repaired test pins raw byte `0x21` independently.

Therefore:

- round-trip equality proves self-consistency, not necessarily correctness;
- silent conventions need literal or official-derived inputs/expected outputs;
- oracle generators must be independent of WASTE implementation helpers;
- high-risk semantics should be mutation-tested when practical.

`docs/FIXTURES.md` is the detailed policy.

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

1. the scalar C implementation matches the official reference on several deterministic independent fixtures;
2. the observed error is explained by expected precision/accumulation differences;
3. the threshold is set above normal numerical noise but below a known wrong implementation/mutation.

Never loosen a tolerance merely to make a failing optimization pass. Diagnose the mismatch first.

### Exact public-format tables

E2M1, UE8M0 and E4M3FN code-value tables are tested exactly, not with epsilon. Their values are dyadic and representable for the tested scalar decode contract.

This exactness does **not** determine DeepSeek nibble order or scale direction.

## 3. Exact-equality requirements

Unless the official reference proves otherwise, these should be exact:

- tensor names/shapes and configured dimensions;
- tokenizer output token IDs;
- special/control-token IDs;
- router-selected expert IDs and ordering;
- deterministic/hash-routed expert IDs;
- cache-on versus cache-off bytes supplied to the arithmetic path;
- container record identity `(layer, expert)`;
- parser structure for official encoder/parser test cases;
- discrete packing/block-index semantics once reconciled with the official reference;
- greedy output token IDs once final-logit parity is close enough to preserve argmax.

Routing weights and projection outputs are floating-point and use documented numerical tolerances; selected sets/order remain hard semantic checks.

## 4. Operational V-level ladder

The `V` labels below are operational validation levels. They are **not a replacement for README §18 Gates A–N**. §4a maps the two vocabularies completely.

### V0 — checkpoint inventory

Owners:

- `tools/inventory.py`;
- `docs/INVENTORY-0731.md`;
- `docs/TENSOR_MAP.md`;
- `docs/REFERENCE_ACCESS.md`.

Pass criteria:

- all main-model names classified;
- no unexplained bytes;
- shapes/config agree;
- expert records are fully understood;
- quantization scales are associated with their owners;
- DeepSeek-specific packing/scale conventions are located in official reference/artifacts;
- DSpark can be separated from the base path.

Current status: **tooling SYNTHETIC-VERIFIED; real official input BLOCKED in the original environment.**

No downstream model binding/storage claim should rely on an unresolved tensor family.

### V1 — bit-level native quantization decode

V1 has two halves.

#### V1a — public format conformance — PASSED

`src/quant/fp4_e2m1.*` and `src/quant/fp8_e4m3.*` implement scalar:

- E2M1 decode;
- UE8M0 decode;
- FP4 K32 scale indexing;
- finite E4M3 (`e4m3fn`) decode;
- FP8 128×128 scale-grid indexing with ragged final blocks;
- decode-row helpers;
- double-accumulating scalar matvec reference paths.

`tests/test_quant.c` checks:

- all 16 E2M1 codes;
- all 256 E4M3 byte encodings;
- all UE8M0 exponent states, including `0xFF` NaN;
- negative zero/subnormal/endpoints;
- E4M3FN maximum `448` distinction;
- FP4 K-block/row-stride boundaries;
- ragged `200 × 300` FP8 scale grid;
- scalar matvec vs separately decoded-row dot product;
- a literal raw-byte nibble-order assertion.

PR #3 reported:

```text
make check -> 34 passed, 0 failed, 12 skipped
make asan  -> 33 passed, 0 failed
```

and a clean warning-free build.

Mutation testing injected ten one-line decoder faults. The initial suite caught nine. The surviving FP4 nibble-order mutation revealed a shared-assumption fixture; after adding the literal raw-byte pin, the final suite catches all ten. See `docs/EXPERIMENTS.md` entry 3 and `docs/FIXTURES.md`.

Evidence state: **SYNTHETIC-VERIFIED / public-format conformance**.

#### V1b — official DeepSeek convention agreement — BLOCKED

The public format spec does not settle at least these current choices:

| Open question | Current local choice | Failure if wrong |
|---|---|---|
| FP4 nibble order | even logical column → low nibble | every pair of matrix columns is swapped while values remain plausible |
| FP8 scale direction | stored scale multiplies decoded E4M3 | weights can be scaled by the wrong reciprocal/squared factor without a crash |
| exact scale tensor layout/dtype | current handoff-derived geometry | wrong block/orientation silently corrupts weights |
| target E4M3 tensor-family convention | finite `e4m3fn` path implemented | wrong target convention changes large weight decode |

V1b test procedure:

1. acquire pinned official reference assets per `docs/REFERENCE_ACCESS.md`;
2. identify exact packed-byte and scale-application code;
3. create small official-derived fixtures whose expected side is independent of WASTE helpers;
4. run scalar C decode against them;
5. let current literal assertions fail if the official convention differs;
6. update implementation/docs deliberately.

V1 pass criteria:

- public-format code values pass exactly;
- scale indexing/granularity pass boundary tests;
- nibble order/scale application/layout agree with official reference;
- official fixtures replay offline with pinned provenance.

Until V1b passes, describe README Gate B / V1 only as **half satisfied**.

### V2 — one quantized linear projection

**Status: BLOCKED on official oracle/checkpoint material.**

Use a deliberately small official-derived projection fixture crossing relevant packing/scale boundaries.

Fixture should contain:

```text
raw encoded weight bytes
raw scale bytes/values
input x
selected decoded weight values
expected output y
provenance
```

Compare:

- decoded/reference weight values;
- output vector;
- accumulation behavior;
- bias if the official op has one.

Pass criteria:

- V1 is complete;
- observed output error is explained and a fixed tolerance is documented;
- at least one known wrong scale/layout/accumulation mutation falls outside the threshold;
- scalar path passes before adding SIMD.

V2 is the next arithmetic rung after reference access. Do not jump from V1a straight to a transformer layer.

### V3 — model primitives

Separate official-oracle tests for:

- normalization;
- mHC transforms/residual path;
- RoPE;
- activation/SwiGLU and clamp semantics;
- attention projection substeps;
- compressor;
- CSA indexer score/selection;
- router score transform;
- routing correction/bias behavior;
- top-k selection, normalization and scaling;
- shared expert;
- one routed expert.

No primitive should be tested only through a full layer.

### V4 — one MoE block

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

### V5 — one attention block

Create separate fixtures for each distinct attention mode found in the official config/reference, rather than assuming one implementation covers all layers.

Capture enough intermediates to localize errors:

- Q/K/V or normalized compressed equivalents;
- positional transform inputs/outputs;
- compression state/output;
- indexer scores and selected positions where applicable;
- attention probabilities/accumulated output where practical;
- final projected attention result.

Context-boundary fixtures should include:

- position 0;
- sliding-window boundary - 1 / boundary / +1;
- compression boundary transitions;
- short prompt versus longer history;
- final supported context edge when feasible with synthetic shapes.

### V6 — one full transformer layer

Compare official and C:

- layer input;
- after attention/mHC merge;
- routing IDs/weights;
- MoE output;
- layer output.

Run at least one example of every structurally distinct layer class proven by config/reference (for example bootstrap-routing versus learned-routing and attention-mode variants).

### V7 — multi-layer hidden states

Run a short token sequence through multiple real layers and checkpoint hidden states at selected layer boundaries.

A mismatch must be attributed to the first divergent layer before continuing.

This catches state-update/order errors that isolated block fixtures miss.

### V8 — final logits

Compare the final hidden state, final norm, and logits for deterministic input tokens.

Record:

- `max_abs`;
- `max_rel`;
- RMS error;
- top-1 token and top-N ordering for a small N;
- logit magnitude range.

The project may not claim model correctness without this level on real weights.

### V9 — greedy generation

With sampling disabled and the same stopping rules:

- prompt tokens must match the relevant official/known-token input contract;
- generated token IDs should match token-for-token for a meaningful short sequence;
- if they diverge, return to final-logit diagnostics rather than declaring sampling noise.

### V10 — encoding and parser

Port official `encoding/` behavior separately from model arithmetic.

Required:

- all official encoding tests;
- content containing strings that resemble control markers;
- tools/tool results if supported by the official encoder;
- reasoning/content region behavior if present;
- malformed/truncated model output parser tests;
- incremental/streaming parser tests if the server exposes SSE.

Do not flatten a code-based encoder into a guessed Jinja template.

### V11 — OpenAI-compatible API

Test the existing WASTE server architecture with the DeepSeek encoder/model path:

- health/model listing;
- non-streaming chat completion;
- streaming chat completion;
- deterministic request parity with direct C generation;
- cancellation/client disconnect behavior;
- structured/tool fields only where supported by the DeepSeek encoding/parser contract.

API success is not a substitute for V8/V9.

## 4a. Canonical concordance: README gates A–N, V-levels, and ROADMAP phases

`README.md` §18 defines **14 stable design gates, A through N**. This table maps **every one of those 14 gates** to the operational validation level or systems/performance owner used by the maintained docs. `ROADMAP.md` phases are schedule only.

Use both identifiers when a README gate has a V-level:

```text
Gate B / V1
Gate H / V6
Gate K / V9
```

For README systems/performance gates that intentionally have no V-number, cite the letter directly (`Gate G`, `Gate L`, `Gate M`, `Gate N`). Do not invent a fake V-level merely to make the table rectangular.

Conversely, this maintained ladder has two operational rungs that README §18 never gave letters: `V7` multi-layer localization and `V11` API parity. They are listed after A–N so the asymmetry is explicit rather than silently dropping either vocabulary.

| README gate | This doc / operational level | Handoff concept | ROADMAP | Owning documents |
|---|---|---|---|---|
| **A** — checkpoint inventory | **V0** | checkpoint inventory / storage truth | Phase 1 | `INVENTORY-0731.md`, `TENSOR_MAP.md`, `REFERENCE_ACCESS.md` |
| **B** — native FP4 decode | **V1** | native quantization decode + DeepSeek byte/scale convention agreement | Phase 3 | `NUMERICS.md`, `FIXTURES.md`, this document |
| **C** — FP8 trunk linear | **V2** | one official quantized trunk projection | Phase 3 | `NUMERICS.md`, this document |
| **D** — mHC | **V3** | mHC and model-primitive parity | Phase 5 | `ARCHITECTURE.md`, `DEEPSEEK_V4.md`, this document |
| **E** — attention by type | **V5** | attention-mode parity (ratio 0 / 128 / 4 + incremental state) | Phase 5 | `DEEPSEEK_V4.md`, this document |
| **F** — routing + one MoE layer | **V4** | routing/shared/routed MoE parity | Phase 5 | `ARCHITECTURE.md`, this document |
| **G** — disk vs cache identity | **Gate G systems correctness** | placement changes timing, never bytes/numerics | Phase 6 | §5 below, `MEMORY_AND_IO.md` |
| **H** — one complete transformer block | **V6** | complete block/layer parity | Phase 5 | this document |
| **I** — 43-layer base forward | **V8** | final base-model hidden/logit parity | Phase 5 | this document |
| **J** — tokenizer/encoding | **V10** | exact official prompt/token/parser semantics | Phase 7 | `API.md`, this document |
| **K** — generation | **V9** | deterministic greedy token parity | Phase 5 / 7 | this document, `API.md` |
| **L** — real storage | **Gate L performance feasibility** | real aligned expert-record I/O on target storage | Phase 6 / 8 | `MEMORY_AND_IO.md`, `BENCHMARKS.md` |
| **M** — cache curve | **Gate M performance feasibility** | routing-derived cache hit/traffic/throughput curve | Phase 6 / 8 | `MEMORY_AND_IO.md`, `BENCHMARKS.md` |
| **N** — DSpark | **Gate N speculative correctness + performance** | official speculative acceptance parity + measured wall-clock benefit | Phase 9 | `DSPARK.md`, `BENCHMARKS.md` |
| — | **V7** | multi-layer hidden-state localization | Phase 5 | this document |
| — | **V11** | OpenAI-compatible API parity | Phase 7 | `API.md`, this document |

Two ordering details are deliberate:

1. README **Gate E** (attention) maps to `V5`, while README **Gate F** (MoE) maps to `V4`. The maintained V-ladder brings up one MoE block before the full compressed-attention machinery because that seam can be isolated earlier. The README letters remain stable design identifiers; the V-order remains the operational test order.
2. README **Gate J** (encoding) maps to `V10`, while **Gate K** (generation) maps to `V9`. Raw/known-token greedy generation can establish model arithmetic before the full chat encoder/parser surface is complete; user-facing chat generation ultimately requires both.

When this document and older handoff ordering disagree about the *operational order* of validation, this document wins because it is maintained with the tests. The README letters still identify the original design gate and rationale. Neither overrides the official checkpoint/reference.

## 5. Cache and I/O correctness matrix — README Gate G

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
- exhaustive public FP4/FP8 format semantics;
- literal packing/block convention cases;
- memory-plan arithmetic;
- model-family dispatch;
- tokenizer/encoder unit behavior where fixtures can be legally/self-containedly included;
- platform I/O wrappers.

These run in ordinary CI once Actions is restored.

### Real-checkpoint/reference tests should cover

- exact tensor binding;
- DeepSeek-specific native quantization conventions;
- official intermediate activations;
- final logits;
- generation;
- routing traces;
- performance and RAM behavior.

These may remain separate/manual because the model is too large for normal CI, but small frozen oracle fixtures should replay offline in ordinary tests.

## 7. Golden fixture format

Follow `docs/FIXTURES.md`.

Prefer a simple auditable format over Python pickles. One possible layout:

```text
tests/fixtures/deepseek_v4/<operation>/<case>/
  provenance.json
  input-*.bin
  expected-*.bin
  expected.json
```

`provenance.json` declares source revision/path/hash, dtype/device policy, generator commit, shapes and endianness.

Important constraints:

- expected values are generated independently of WASTE implementation helpers;
- fixtures are frozen and replayed offline;
- convention cases include human-reviewable literal bytes/values;
- do not commit large portions of model weights;
- a synthetic fixture is never promoted to checkpoint evidence.

## 8. Failure triage order

When final logits are wrong, investigate in this order:

1. wrong prompt/token IDs;
2. wrong tensor mapping/shape/transpose;
3. wrong quantization byte convention/scale direction/indexing;
4. primitive arithmetic/accumulation;
5. attention state/position update;
6. router selected IDs/order/weights;
7. expert record identity/data;
8. residual/mHC ordering;
9. layer-state persistence;
10. final norm/head.

Do not begin by changing tolerances.

When a level fails, first ask whether its expected fixture is truly independent. A self-confirming fixture can turn a local convention bug into a much later “model is broken” symptom.

## 9. Optimization acceptance rule

An optimized kernel/path lands only when:

- the scalar/reference path remains available at least for tests/debugging;
- it passes the same independent fixture suite;
- the scalar baseline being matched has itself passed the relevant official-reference level;
- real-model V8/V9 do not regress once available;
- README Gate G remains true for placement/cache changes;
- relevant Gate L/M storage/cache measurements remain valid for performance claims;
- the benchmark records the hardware/configuration where the speedup exists;
- memory usage does not silently violate the planner;
- failures fall back safely where a backend is optional.

A SIMD path matching a scalar implementation that is still wrong about nibble order is not model correctness. This is why Gate B/V1 and Gate C/V2 precede SIMD.

## 10. Result recording template

Add a result to the relevant PR and `BENCHMARKS.md`/`EXPERIMENTS.md` with:

```text
Date:
Port commit:
Model revision:
Reference source path/hash:
Container/converter revision:
Hardware / OS:
README gate letter(s):
Operational V-level / systems gate:
Operation:
Reference command:
C command:
Input fixture/prompt:
Fixture provenance:
max_abs:
max_rel:
RMS:
exact semantic checks:
mutations/known-wrong cases tested:
Verdict:
Notes:
```

A naked statement like “matches PyTorch” or “round trips correctly” is not sufficient evidence.