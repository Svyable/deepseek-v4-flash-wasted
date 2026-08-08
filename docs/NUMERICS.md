# Numerics — native DeepSeek quantization reference contract

**Status: public E2M1/UE8M0/E4M3FN format conformance is SYNTHETIC-VERIFIED, and the highest-risk DeepSeek-specific FP4/FP8 storage conventions are now SOURCE-VERIFIED against the pinned 0731 release. Gate B/V1 has an independent pinned fixture and C replay test wired into `make check`, but this branch has not had a fresh full checkout run yet. Gate A/V0 real checkpoint headers and Gate C/V2 one real quantized projection remain open.**

This document is the arithmetic contract for the scalar native-quantization path introduced in PR #3 and extended by PR #5. Keep three evidence layers separate:

1. **public numeric format semantics** — E2M1, UE8M0, finite E4M3FN;
2. **official DeepSeek operation/storage semantics** — packing order, scale direction, block geometry, activation quantization;
3. **real checkpoint/oracle evidence** — exact exported tensors and one actual projection.

They are related, but they are not interchangeable.

Canonical gate ownership:

- **Gate A / V0** — exact checkpoint tensor names, dtypes, shapes, shards and bytes;
- **Gate B / V1** — native number-format decode plus official DeepSeek packing/scale convention agreement;
- **Gate C / V2** — one real official quantized trunk projection, including activation quantization and accumulation.

Pinned official release used for source evidence:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

See `OFFICIAL-0731-SOURCE.md` for the source-level findings and `FIXTURES.md` for fixture independence rules.

---

## 1. Implementations and evidence split

Scalar/reference code:

- `src/quant/fp4_e2m1.{c,h}` — E2M1 + UE8M0 K32 routed-expert decode/matvec;
- `src/quant/fp8_e4m3.{c,h}` — finite E4M3FN + 128x128 block-scale decode/matvec;
- `src/quant/deepseek_v4_linear_ref.{c,h}` — deliberately slow official-linear-shaped activation quantization + scaled FP8 block accumulation + BF16 output rounding;
- `tests/test_quant.c` — exhaustive public-format and model-free indexing tests;
- `tests/fixtures/deepseek_v4/fp4_release_convention.json` — pinned F3 official-source convention fixture;
- `tests/test_release_quant_fixture.py` — compiles the real scalar FP4 implementation against the pinned fixture;
- `tests/test_v2_linear_ref.py` — source-derived Gate C preflight with closed-form activation/block-scale arithmetic.

| Claim | Current evidence | Gate |
|---|---|---|
| E2M1 code values | **SYNTHETIC-VERIFIED** — all 16 codes | B / V1a |
| UE8M0 values | **SYNTHETIC-VERIFIED** — all 256 states | B / V1a |
| finite E4M3FN values | **SYNTHETIC-VERIFIED** — all 256 byte encodings | B / V1a |
| FP4 K32 indexing | **SYNTHETIC-VERIFIED** | B / V1a |
| FP8 128x128/ragged scale indexing | **SYNTHETIC-VERIFIED** | B / V1a |
| FP4 low-nibble-first along K | **OFFICIAL-SOURCE-VERIFIED** | B / V1b |
| routed FP4 scale `[out, in/32]` E8M0 | **OFFICIAL-SOURCE-VERIFIED** | B / V1b |
| stored weight scale is multiplied | **OFFICIAL-SOURCE-VERIFIED** | B / V1b |
| resident weight dtype is finite E4M3FN | **OFFICIAL-SOURCE-VERIFIED** | B / V1b |
| resident weight scales are 128x128 blocks | **OFFICIAL-SOURCE-VERIFIED** | B / V1b |
| official activation quantization rule | **OFFICIAL-SOURCE-VERIFIED**, scalar preflight implemented | C preflight |
| exact exported checkpoint storage for every tensor | **NOT YET CHECKPOINT-VERIFIED** | A / V0 |
| one real official quantized projection output | **NOT VERIFIED** | C / V2 |

A source-level fact does not replace Gate A header truth, and a source-derived closed-form projection does not replace Gate C's real fixture.

---

## 2. Scalar reference policy

Every optimized quantized path keeps an auditable scalar implementation.

The hierarchy is:

```text
public numeric specification
        ↓
independent exhaustive/literal tests
        ↓
scalar decode/reference implementation
        ↓
pinned official DeepSeek source semantics
        ↓
independent pinned source/oracle fixture
        ↓
real checkpoint/oracle projection
        ↓
optimized SIMD/backend implementation
```

Rules:

- an implementation may not generate its own expected values;
- round trips prove self-consistency, not external correctness;
- scale/block/nibble selection is a discrete semantic check and should be exact;
- optimized kernels must remain comparable to the scalar path;
- no SIMD optimization is accepted merely because it matches a scalar path that has not passed the relevant official gate.

PR #3's mutation result remains the cautionary example: exhaustive FP4 code coverage initially missed a nibble-order bug because fixture generation and decode shared the same macro.

---

## 3. E2M1

E2M1 has one sign bit, two exponent bits and one mantissa bit. The independent tests derive values from the bit definition rather than from the implementation table.

Important values:

| code | value |
|---:|---:|
| `0x0` | `+0.0` |
| `0x1` | `+0.5` |
| `0x2` | `+1.0` |
| `0x7` | `+6.0` |
| `0x8` | `-0.0` |
| `0xF` | `-6.0` |

All 16 codes are finite. Negative zero is tested by sign.

---

## 4. UE8M0

Scalar definition:

```text
scale(e) = 2^(e - 127), e = 0x00..0xFE
0xFF     = NaN
```

Landmarks:

| code | value |
|---:|---:|
| `0x7E` | `0.5` |
| `0x7F` | `1.0` |
| `0x80` | `2.0` |
| `0x00` | `2^-127` |
| `0xFE` | `2^127` |
| `0xFF` | NaN |

`2^-127` is binary32 subnormal and must stay non-zero. The implementation uses `ldexpf` rather than constructing only normal IEEE exponents.

---

## 5. Routed FP4 storage — official 0731 convention

For a logical routed-expert matrix `[out, in]`, the pinned official model binds:

```text
weight: [out, in/2]   torch.float4_e2m1fn_x2
scale:  [out, in/32]  torch.float8_e8m0fnu
```

Therefore one weight scale applies to 32 consecutive logical K values.

### Nibble order — resolved

The pinned official converter views each packed byte as unsigned, then expands it as:

```text
low  = byte & 0x0f
high = (byte >> 4) & 0x0f
stack([low, high]) along K
```

Thus:

```text
even/lower K index -> LOW nibble
odd/next K index   -> HIGH nibble
```

`WASTE_FP4_LOW_NIBBLE_IS_EVEN == 1` is no longer a handoff guess; it is source-verified for the pinned release.

### Scale direction — resolved

The official FP4 kernel operates on 32-wide K sub-blocks and accumulates:

```text
partial_dot * activation_scale * weight_scale
```

The stored E8M0 weight scale is a multiplier.

The converter also renames `weight_scale_inv` to `scale` without numerically reciprocating it. A tensor name containing `_inv` was never sufficient evidence for divide semantics; the operation is now settled by source.

### Independent pinned literal

`tests/fixtures/deepseek_v4/fp4_release_convention.json` records:

```text
packed byte = 0x21
scale byte  = 0x80 = 2.0

low nibble  0x1 -> 0.5 * 2 = 1.0
high nibble 0x2 -> 1.0 * 2 = 2.0

expected logical values = [1.0, 2.0]
```

The expected side does not invoke WASTE's packer or nibble macro. Reversing the nibble order or changing multiply to divide makes the fixture fail.

A locally reconstructed compile of the same scalar logic produced exactly `1 2`. The committed branch test still needs a full checkout run before merge.

---

## 6. Finite E4M3FN

The resident/trunk decoder uses the finite E4M3 variant (`torch.float8_e4m3fn` in the pinned source):

```text
sign = bit 7
exp  = bits 6..3
mant = bits 2..0
bias = 7
```

Only `S.1111.111` is NaN; there are no infinities. The top exponent retains finite values.

| code | value |
|---:|---:|
| `0x00` | `+0.0` |
| `0x01` | `2^-9` |
| `0x08` | `2^-6` |
| `0x38` | `1.0` |
| `0x78` | `256.0` |
| `0x7C` | `384.0` |
| `0x7E` | `448.0` |
| `0x7F` | NaN |
| `0x80` | `-0.0` |
| `0xFF` | NaN |

`448`, not `240`, is a critical format discriminator.

---

## 7. Resident FP8 block scales — official 0731 convention

For E4M3 resident weights, the pinned model uses a scale grid:

```text
scale rows = ceil(out / 128)
scale cols = ceil(in  / 128)
```

The scalar decoded-weight helper models:

```text
value(r,c) = e4m3(weight[r,c]) * scale[r/128,c/128]
```

The pinned converter does not reciprocal-transform `weight_scale_inv`, and the official FP8 GEMM multiplies activation scale by weight scale before applying each accumulated block result.

`tests/test_quant.c` already covers exact and ragged 128 boundaries with a synthetic `200 x 300` matrix. The source review upgrades the **direction and target variant** from assumptions to official-source facts; Gate A still has to show how the released checkpoint stores every particular tensor family.

---

## 8. Official activation quantization — Gate C preflight

The pinned `linear()` does not simply dequantize resident weights and multiply by the original activation. For FP8 and FP4 quantized weights it first calls the official activation quantizer.

For each activation row and each 128-wide K block:

```text
amax = max(abs(x_block))
amax = max(amax, 1e-4)
raw_scale = amax / 448
```

With the release's non-null `scale_fmt`/UE8M0 path, the scale is rounded **upward to a power of two** using the source's `fast_log2_ceil` / `fast_pow2` rule.

Then:

```text
q = E4M3FN(clamp(x / scale, -448, +448))
```

The FP8 GEMM processes 128 K values per block and accumulates:

```text
dot(q_activation, q_weight) * activation_scale * weight_scale
```

with FP32 accumulation and BF16 output in the reference path.

### Scalar implementation

`src/quant/deepseek_v4_linear_ref.{c,h}` implements a correctness-first version:

- exhaustive-search E4M3FN encoder with round-to-nearest-even tie handling;
- source-shaped power-of-two activation scale;
- 128-wide activation quantization;
- scaled FP8 block-dot accumulation;
- explicit BF16 final rounding.

The exhaustive E4M3 encoder is intentionally slow. It exists to be obvious and independently reviewable, not to run a model quickly.

### Closed-form preflight

`tests/test_v2_linear_ref.py` uses two K blocks whose normalized values are exactly representable:

```text
block 0: x = 1.0
  activation scale = 2^-8
  normalized x = 256 = E4M3 0x78
  weight scale = 1

block 1: x = 0.75
  activation scale = 2^-9
  normalized x = 384 = E4M3 0x7C
  weight scale = 2

all raw weights = E4M3 1.0 (0x38)
```

Expected result:

```text
128 * 1.0 * 1 + 128 * 0.75 * 2 = 320
```

An independent local arithmetic probe produced `320 320` for two output rows.

This proves the source-derived scalar seam is internally executable. It does **not** pass Gate C/V2 because the inputs and expected output are closed-form, not a real released projection.

---

## 9. Existing decoded-weight matvec versus official linear

`waste_fp8_matvec()` remains useful and should not be silently redefined. Its contract is:

```text
y = dequantized_weight * f32_input
```

That isolates weight decoding and block-scale indexing.

`waste_ds_v4_fp8_linear_ref()` has a stronger, different contract matching the official quantized-linear shape:

```text
f32/BF16-like input
  -> activation E4M3 quantization + scale
  -> raw E4M3 weight block dot
  -> activation_scale * weight_scale
  -> FP32 accumulation
  -> BF16-rounded output
```

The two functions answer different validation questions. Do not replace the simpler weight-decode seam with the full linear seam; both are useful when diagnosing an oracle mismatch.

---

## 10. Exactness, tolerances, and mutations

Use exact equality for:

- public format code tables;
- nibble/block index selection;
- literal source-convention fixtures;
- activation scale cases chosen to be exact powers of two;
- closed-form values exactly representable in the target format.

Gate C real-projection tolerances are chosen only after observing the official fixture's actual dtype/reduction behavior. Do not pre-author a loose threshold.

Mutation priorities remain:

- nibble reversal;
- multiply versus divide scale;
- K32 versus K128 indexing;
- scale-grid transpose;
- floor versus ceil block count;
- `ceil(log2)` versus another activation-scale rule;
- E4M3FN versus an IEEE-like E4M3 interpretation;
- premature BF16 rounding;
- accumulation/reset order.

A known-wrong mutation that survives is a fixture defect to repair before moving downstream.

---

## 11. Gate B / V1 current checklist

Semantic/evidence checklist:

- [x] E2M1 public semantics exhaustively tested.
- [x] UE8M0 public semantics exhaustively tested.
- [x] finite E4M3FN public semantics exhaustively tested.
- [x] FP4 K32 indexing model-free tested.
- [x] FP8 128x128/ragged indexing model-free tested.
- [x] scalar FP4/FP8 decoded-weight paths exist.
- [x] PR #3 mutation set is killed by the repaired fixture suite.
- [x] official pinned source confirms FP4 low-nibble-first ordering.
- [x] official pinned source confirms routed E8M0 `[out, in/32]` geometry.
- [x] official pinned source confirms weight-scale multiplication.
- [x] official pinned source confirms finite E4M3FN resident weight path and 128x128 block-scale geometry.
- [x] independent pinned F3 convention fixture committed.
- [x] committed C replay test consumes that fixture.
- [ ] fresh full PR #5 checkout run records the integrated test result.

Therefore the previous **“blocked on unknown nibble/scale semantics”** statement is obsolete. Gate B's model semantics are established; the remaining merge qualification is test execution/provenance, while real checkpoint-storage truth remains Gate A.

---

## 12. Gate C / V2 completion path

The next arithmetic gate must remain deliberately small.

Required real fixture:

```text
source model/revision
source tensor name + shard/header identity
raw E4M3 weight bytes for a bounded projection/tile
raw or decoded official weight-scale values
BF16/f32 input activation used by the official call
official activation quantized bytes + scales when practical
official expected output
reference device/dtype/kernel provenance
```

Procedure:

1. complete Gate A name/header mapping for the chosen resident projection;
2. select a representative quantized projection crossing a 128 K boundary;
3. run the pinned official reference path;
4. freeze expected artifacts independently of WASTE code;
5. replay through `deepseek_v4_linear_ref`;
6. explain the error budget and set a fixed tolerance;
7. mutation-test scale/activation/rounding seams;
8. only then expose the path to transformer code or add SIMD.

Do not jump from the closed-form source preflight to mHC/full-layer integration and call the quantized linear proven.
