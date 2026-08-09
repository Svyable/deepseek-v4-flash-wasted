# Validation — correctness gates for the DeepSeek V4 port

**Status: Gate A/V0 has header-only acquisition and inventory tooling but has not consumed the real 0731 shard headers. Gate B/V1 public-format semantics and the highest-risk DeepSeek packing/scale conventions are established from the pinned official release, with an independent F3 replay test wired into the branch. Gate C/V2 has a source-derived scalar linear preflight, but the real official projection fixture is still open.**

This project should never debug model correctness from generated prose. Validation proceeds from the smallest exact seam to final logits, and each expensive phase is protected by a cheaper test.

PR #3 (`91c36b8f4168349e6893a9911a3f60075d62d973`) established exhaustive public-format scalar quantization. PR #5 moves the next seams from handoff assumptions to pinned official-source contracts while keeping checkpoint and real-projection claims separate.

Pinned release for current official-source evidence:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

See:

- `docs/NUMERICS.md` for the native-quantization and Gate C scalar-reference contract;
- `docs/FIXTURES.md` for fixture independence and mutation-testing rules;
- `docs/OFFICIAL-0731-SOURCE.md` for pinned release-source findings;
- `docs/REFERENCE_ACCESS.md` for the smallest official-artifact acquisition sequence.

## 1. Correctness oracle

The primary model oracle is the pinned official `DeepSeek-V4-Flash-0731` reference implementation and checkpoint.

Public numeric-format specifications are authoritative for E2M1/UE8M0/E4M3 semantics they actually define. Pinned official source is authoritative for DeepSeek operation semantics it explicitly defines, such as nibble extraction, scale application, activation quantization and routing math. Exact exported tensor names/shapes/bytes remain checkpoint-header facts, and numerical projection/logit parity still requires official oracle outputs.

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

PR #5 upgrades that seam with a pinned F3 official-source fixture: raw byte `0x21` plus E8M0 scale byte `0x80` must produce logical values `[1.0, 2.0]`. The expected values are derived from the official release's low-then-high nibble expansion and multiply-scale operation, not from WASTE's pack helper.

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

1. the scalar C implementation matches the official reference on deterministic independent fixtures;
2. the observed error is explained by expected precision/accumulation differences;
3. the threshold is set above normal numerical noise but below a known wrong implementation/mutation.

Never loosen a tolerance merely to make a failing optimization pass. Diagnose the mismatch first.

### Exact public-format and discrete convention checks

E2M1, UE8M0 and E4M3FN code-value tables are tested exactly. Nibble order, scale/block index selection, bootstrap-routing IDs and other discrete conventions should also be exact once pinned by official source/artifacts.

The public format tables alone do not establish a model-specific packing convention; the pinned 0731 source now supplies that additional evidence for the current release.

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

- `tools/fetch_hf_headers.py`;
- `tools/inventory.py`;
- `docs/INVENTORY-0731.md`;
- `docs/TENSOR_MAP.md`;
- `docs/REFERENCE_ACCESS.md`.

Pass criteria:

- immutable official revision pinned;
- all main-model names classified;
- no unexplained bytes;
- shapes/config agree;
- expert records are fully understood;
- quantization scales are associated with their owners;
- DSpark can be separated from the base path.

Current status: **acquisition + inventory tooling implemented; official source/config advanced; real 48-shard header inventory NOT YET RUN.**

PR #5's fetcher uses exact HTTP Range reads and writes header-only safetensors stubs that `inventory.py` consumes without a second metadata format. It refuses servers/proxies that ignore Range rather than risking a full-shard transfer. The inventory classifier now recognizes the pinned release's `tid2eid` runtime spelling and converter `tie2eid` spelling while retaining fatal unknown-main behavior.

No downstream tensor binding/storage byte claim should rely on an unresolved tensor family.

### V1 — bit-level native quantization decode

V1 has two evidence layers.

#### V1a — public format conformance — PASSED in merged PR #3

`src/quant/fp4_e2m1.*` and `src/quant/fp8_e4m3.*` implement scalar:

- E2M1 decode;
- UE8M0 decode;
- FP4 K32 scale indexing;
- finite E4M3 (`e4m3fn`) decode;
- FP8 128×128 scale-grid indexing with ragged final blocks;
- decode-row helpers;
- double-accumulating decoded-weight matvec reference paths.

`tests/test_quant.c` checks all 16 E2M1 codes, all 256 E4M3 bytes, all UE8M0 states, boundary/index semantics and decoded-weight matvec consistency. PR #3 reported:

```text
make check -> 34 passed, 0 failed, 12 skipped
make asan  -> 33 passed, 0 failed
```

Mutation testing injected ten one-line decoder faults. The initial suite caught nine; the surviving nibble-order mutation exposed a shared-assumption fixture, and the repaired literal makes all ten fail as intended.

Evidence state: **SYNTHETIC-VERIFIED / public-format conformance**.

#### V1b — official DeepSeek convention agreement — SOURCE-VERIFIED; branch replay pending

The pinned official 0731 source resolves the PR #3 open conventions:

| Convention | Pinned release behavior | Evidence |
|---|---|---|
| FP4 nibble order | lower/even logical K index = low nibble; next = high nibble | official converter expands `[low, high]` along K |
| FP4 scale geometry | `[out, in/32]` E8M0 for logical `[out, in]` | official model binding |
| FP4 scale direction | multiply decoded block partial by weight scale | official kernel |
| `weight_scale_inv` conversion | renamed to `scale` without reciprocal | official converter |
| resident weight format | finite `torch.float8_e4m3fn` | official model/kernel |
| resident scale geometry | one weight scale per 128×128 block | official model/kernel |

`tests/fixtures/deepseek_v4/fp4_release_convention.json` freezes an independent F3 literal:

```text
packed byte 0x21
E8M0 scale 0x80 = 2.0
expected [1.0, 2.0]
```

`tests/test_release_quant_fixture.py` compiles the actual scalar FP4 implementation against that fixture and also pins `tid2eid`/`tie2eid` classifier behavior. It is wired into the existing `tests/test_inventory.py` path used by `make check`.

A narrow local reconstruction of the same scalar byte/scale path produced exactly `1 2`; however, the current private branch has not had a fresh full checkout `make check` in this execution environment. Therefore do **not** claim PR #5's integrated V1 replay as a completed branch test until that run is recorded.

V1 semantic pass criteria are now understood and represented in the branch. Gate A still owns exact real checkpoint-storage names/dtypes/shapes; do not conflate that with the source-level V1 operation contract.

### V2 — one quantized linear projection

**Status: source-derived scalar preflight IMPLEMENTED; real official projection NOT YET PASSED.**

The pinned official `linear()` first quantizes activations to E4M3 in 128-wide K blocks. With the release UE8M0 scale path:

```text
amax = max(max(abs(x_block)), 1e-4)
scale = next_power_of_two(amax / 448)
q = E4M3FN(clamp(x / scale, -448, +448))
```

FP8 GEMM then accumulates each 128-wide block after multiplying by `activation_scale * weight_scale`, with FP32 accumulation and BF16 output in the reference path.

PR #5 adds `src/quant/deepseek_v4_linear_ref.{c,h}` as a deliberately slow scalar implementation of that seam plus `tests/test_v2_linear_ref.py`. The model-free closed-form case uses two K128 blocks and expects exactly `320`; an independent local arithmetic reconstruction produced `320 320` for two output rows.

That preflight is intentionally **not V2 parity**. A V2 fixture must contain real/pinned official projection material:

```text
source tensor name + shard/header identity
raw encoded weight bytes
raw/decoded weight scales
input activation
reference activation-quantized bytes/scales where practical
expected output y
source/device/dtype/kernel provenance
```

Pass criteria:

- Gate A identifies the chosen real projection and exact storage;
- Gate B's native conventions remain satisfied;
- the scalar official-linear-shaped path matches the frozen official output under a justified fixed tolerance;
- at least one known-wrong scale/layout/activation-quantization/accumulation mutation fails;
- scalar path passes before SIMD or transformer integration.

Do not jump from the closed-form preflight to a transformer layer and call the quantized linear proven.

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

Run at least one example of every structurally distinct layer class proven by config/reference.

### V7 — multi-layer hidden states

Run a short token sequence through multiple real layers and checkpoint hidden states at selected layer boundaries. A mismatch must be attributed to the first divergent layer before continuing.

### V8 — final logits

Compare the final hidden state, final norm, and logits for deterministic input tokens. Record `max_abs`, `max_rel`, RMS error, top-1/top-N ordering and logit magnitude range. The project may not claim model correctness without this level on real weights.

### V9 — greedy generation

With sampling disabled and the same stopping rules, generated token IDs should match token-for-token for meaningful deterministic sequences. If they diverge, return to final-logit diagnostics rather than declaring sampling noise.

### V10 — encoding and parser

Port official `encoding/` behavior separately from model arithmetic. Required: official encoding tests, structure/content boundary cases, tools/tool results where supported, reasoning/content regions, malformed/truncated parser tests, and streaming parser tests where applicable. Do not flatten a code-based encoder into a guessed Jinja template.

### V11 — OpenAI-compatible API

Test health/model listing, non-streaming and streaming completion, deterministic request parity with direct C generation, cancellation, and supported structured/tool fields. API success is not a substitute for V8/V9.

## 4a. Canonical concordance: README gates A–N, V-levels, and ROADMAP phases

`README.md` §18 defines **14 stable design gates, A through N**. This table maps every gate to the maintained operational validation level or systems/performance owner. `ROADMAP.md` phases are schedule only.

Use both identifiers when both exist (`Gate B / V1`, `Gate H / V6`, `Gate K / V9`). README systems/performance Gates G/L/M/N intentionally have no V-number. Conversely, V7 and V11 have no README letter.

| README gate | This doc / operational level | Handoff concept | ROADMAP | Owning documents |
|---|---|---|---|---|
| **A** — checkpoint inventory | **V0** | checkpoint inventory / storage truth | Phase 1 | `INVENTORY-0731.md`, `TENSOR_MAP.md`, `REFERENCE_ACCESS.md` |
| **B** — native FP4 decode | **V1** | native quantization decode + DeepSeek byte/scale convention agreement | Phase 3 | `NUMERICS.md`, `FIXTURES.md`, this document |
| **C** — FP8 trunk linear | **V2** | one official quantized trunk projection | Phase 3 | `NUMERICS.md`, this document |
| **D** — mHC | **V3** | mHC and model-primitive parity | Phase 5 | `ARCHITECTURE.md`, `DEEPSEEK_V4.md`, this document |
| **E** — attention by type | **V5** | attention-mode parity | Phase 5 | `DEEPSEEK_V4.md`, this document |
| **F** — routing + one MoE layer | **V4** | routing/shared/routed MoE parity | Phase 5 | `ARCHITECTURE.md`, this document |
| **G** — disk vs cache identity | **Gate G systems correctness** | placement changes timing, never bytes/numerics | Phase 6 | §5 below, `MEMORY_AND_IO.md` |
| **H** — one complete transformer block | **V6** | complete block/layer parity | Phase 5 | this document |
| **I** — 43-layer base forward | **V8** | final base-model hidden/logit parity | Phase 5 | this document |
| **J** — tokenizer/encoding | **V10** | exact official prompt/token/parser semantics | Phase 7 | `API.md`, this document |
| **K** — generation | **V9** | deterministic greedy token parity | Phase 5 / 7 | this document, `API.md` |
| **L** — real storage | **Gate L performance feasibility** | real aligned expert-record I/O | Phase 6 / 8 | `MEMORY_AND_IO.md`, `BENCHMARKS.md` |
| **M** — cache curve | **Gate M performance feasibility** | routing-derived cache/traffic curve | Phase 6 / 8 | `MEMORY_AND_IO.md`, `BENCHMARKS.md` |
| **N** — DSpark | **Gate N speculative correctness + performance** | speculative acceptance parity + measured benefit | Phase 9 | `DSPARK.md`, `BENCHMARKS.md` |
| — | **V7** | multi-layer hidden-state localization | Phase 5 | this document |
| — | **V11** | OpenAI-compatible API parity | Phase 7 | `API.md`, this document |

Two ordering details are deliberate: Gate E maps to V5 while Gate F maps to V4 because the isolated MoE seam can be validated before all compressed-attention modes; Gate J maps to V10 while Gate K maps to V9 because known-token/raw deterministic generation can validate model arithmetic before the full chat encoder/parser surface.

When this document and older handoff ordering disagree about operational order, this document wins. README letters retain the design rationale. Neither overrides the official checkpoint/reference.

## 5. Cache and I/O correctness matrix — README Gate G

For the same token sequence/container, compare:

| Mode A | Mode B | Requirement |
|---|---|---|
| cache disabled | cache enabled | same selected experts and numerical output |
| direct I/O | page-cache fallback | same bytes/numerics |
| cold cache | learned/preloaded cache | same numerics |
| prefetch off | prefetch on | same routing and output |
| sequential prefill | chunked prefill | agreed logits within fixed tolerance |
| one thread where supported | multiple threads | same semantics; tolerance documented if reduction order differs |

Any optimization whose enable flag changes model meaning is not an optimization.

## 6. Synthetic tests versus real-model tests

### Model-free/synthetic/source-derived tests should cover

- manifest parser bounds;
- expert record offsets/alignment;
- checksum/header corruption;
- cache identity and eviction behavior;
- exhaustive public FP4/FP8 format semantics;
- literal packing/block convention cases;
- source-derived activation-quantization closed forms;
- memory-plan arithmetic;
- model-family dispatch;
- tokenizer/encoder unit behavior where fixtures can be legally/self-containedly included;
- platform I/O wrappers.

### Real-checkpoint/reference tests should cover

- exact tensor binding;
- real quantized projection outputs;
- official intermediate activations;
- final logits;
- generation;
- routing traces;
- performance and RAM behavior.

The model is too large for normal CI, but small frozen official fixtures should replay offline in ordinary tests.

## 7. Golden fixture format

Follow `docs/FIXTURES.md`. Prefer simple auditable artifacts over Python pickles:

```text
tests/fixtures/deepseek_v4/<operation>/<case>/
  provenance.json
  input-*.bin
  expected-*.bin
  expected.json
```

Expected values are independent of WASTE helpers, fixtures are frozen, convention cases remain human-reviewable, and synthetic/source-derived evidence is never promoted to checkpoint/end-to-end evidence.

## 8. Failure triage order

When final logits are wrong, investigate:

1. prompt/token IDs;
2. tensor mapping/shape/transpose;
3. quantization byte convention/scale/indexing;
4. activation quantization/primitive arithmetic/accumulation;
5. attention state/position update;
6. router IDs/order/weights;
7. expert record identity/data;
8. residual/mHC ordering;
9. layer-state persistence;
10. final norm/head.

Do not begin by changing tolerances. When a level fails, first ask whether its fixture is truly independent.

## 9. Optimization acceptance rule

An optimized path lands only when:

- the scalar/reference path remains available for tests/debugging;
- it passes the same independent fixture suite;
- the scalar baseline has passed the relevant official-reference level;
- real-model V8/V9 do not regress once available;
- README Gate G remains true for placement/cache changes;
- relevant Gate L/M measurements remain valid for performance claims;
- benchmark hardware/configuration is recorded;
- memory remains within the planner;
- optional backend failures fall back safely.

Matching an unproven scalar implementation is not model correctness. Gate B/V1 and real Gate C/V2 precede SIMD.

## 10. Result recording template

Add results to the relevant PR and `BENCHMARKS.md`/`EXPERIMENTS.md` with:

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
