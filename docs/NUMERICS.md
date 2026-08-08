# Numerics — native DeepSeek quantization reference contract

**Status: README Gate B / V1 FORMAT-CONFORMANCE HALF IS SYNTHETIC-VERIFIED. DeepSeek-specific packing/scale conventions remain BLOCKED on Gate A / V0 reference truth and the V1b official-convention check.**

This document is the arithmetic contract for the scalar native-quantization code introduced in PR #3 (`91c36b8f4168349e6893a9911a3f60075d62d973`). It exists so later SIMD, container, converter, and model work has one place to answer three different questions:

1. what the public number format means;
2. what this repository currently assumes about DeepSeek's storage convention;
3. what has actually been proved against the official DeepSeek reference.

Those are not the same thing.

Gate ownership:

- **README Gate A / V0** establishes real checkpoint tensor/storage truth that constrains this document;
- **README Gate B / V1** owns native decode + official DeepSeek packing/scale convention agreement;
- **README Gate C / V2** owns one official quantized trunk projection.

The scalar implementations are:

- `src/quant/fp4_e2m1.{c,h}` — routed-expert E2M1 + UE8M0 K32 decode/matvec;
- `src/quant/fp8_e4m3.{c,h}` — trunk E4M3FN + block-scale decode/matvec;
- `tests/test_quant.c` — exhaustive public-format conformance and layout/indexing tests.

See `docs/VALIDATION.md` §4a for the canonical A–N/V-level mapping and `docs/FIXTURES.md` for the rule that expected values must not be generated through the convention being tested.

---

## 1. Evidence split

| Claim | Current state | Gate | Evidence |
|---|---|---|---|
| E2M1 code values | **SYNTHETIC-VERIFIED** | Gate B / V1a | exhaustive 16-code test derived independently from sign/exponent/mantissa rules |
| UE8M0 values | **SYNTHETIC-VERIFIED** | Gate B / V1a | exhaustive 256-code test; `0xFF` NaN, `0x00..0xFE` powers of two |
| E4M3 finite (`e4m3fn`) values | **SYNTHETIC-VERIFIED** | Gate B / V1a | exhaustive 256-code test; top exponent remains finite except `S.1111.111` |
| FP4 K32 scale indexing | **SYNTHETIC-VERIFIED** | Gate B / V1a | boundary/row-stride tests on synthetic arrays |
| FP8 128×128 scale-grid indexing | **SYNTHETIC-VERIFIED** | Gate B / V1a | exact/ragged boundary tests including a 200×300 matrix |
| scalar matvec implementation | **SYNTHETIC-VERIFIED** | supports Gate B/C | decoded-row dot product agrees with direct scalar matvec |
| FP4 nibble order used by DeepSeek | **BLOCKED / CHECKPOINT-UNVERIFIED** | Gate A/V0 + Gate B/V1b | current assumption: even logical column is low nibble |
| meaning/direction of `weight_scale_inv` | **BLOCKED / CHECKPOINT-UNVERIFIED** | Gate A/V0 + Gate B/V1b | current assumption: stored value is multiplied into decoded E4M3 |
| exact checkpoint tensor scale dtype/layout | **BLOCKED / CHECKPOINT-UNVERIFIED** | Gate A / V0 | requires official metadata/reference |
| one real DeepSeek quantized projection | **NOT VERIFIED** | Gate C / V2 | requires official oracle/checkpoint |

The first five rows are real results. They do not promote the last four rows.

---

## 2. Scalar reference policy

Every native-quantized optimized path must have a scalar implementation that remains buildable and testable after the optimization lands.

The scalar path is intentionally not the fastest implementation:

- decode logic favors readability over table tricks that obscure the format;
- matvec accumulation is performed in `double`, then cast to `float` for output;
- no SIMD path may become the only implementation;
- the scalar path is the local implementation oracle for optimized backends only after the official DeepSeek conventions have been reconciled.

The hierarchy is:

```text
public format spec
      ↓
independent scalar conformance tests
      ↓
scalar C implementation
      ↓
Gate A/V0 checkpoint convention truth
      ↓
Gate B/V1 official DeepSeek convention reconciliation
      ↓
Gate C/V2 official projection parity
      ↓
optimized SIMD/backend implementation
```

The scalar C implementation is never allowed to become the source of expected values for tests of itself.

---

## 3. E2M1

E2M1 uses one sign bit, two exponent bits and one mantissa bit.

The repository's independent test derives each value from:

```text
sign = bit 3
exp  = bits 2..1
mant = bit 0
bias = 1
```

For `exp == 0`, the value is subnormal and has no implicit leading one. For nonzero exponent values the usual implicit leading one is present.

Important landmarks pinned by `tests/test_quant.c`:

| code | value |
|---:|---:|
| `0x0` | `+0.0` |
| `0x1` | `+0.5` |
| `0x2` | `+1.0` |
| `0x7` | `+6.0` |
| `0x8` | `-0.0` |
| `0xF` | `-6.0` |

All 16 codes are finite. E2M1 has no NaN or infinity encoding in this implementation/spec interpretation.

Negative zero is tested by sign, not merely equality with zero.

---

## 4. UE8M0

The current scalar definition is:

```text
scale(e) = 2^(e - 127), for e = 0x00..0xFE
0xFF     = NaN
```

Important landmarks:

| code | value |
|---:|---:|
| `0x7E` | `0.5` |
| `0x7F` | `1.0` |
| `0x80` | `2.0` |
| `0x00` | `2^-127` |
| `0xFE` | `2^127` |
| `0xFF` | NaN |

`2^-127` is subnormal in binary32 and must remain nonzero. An implementation that synthesizes the scale only by manipulating a normal binary32 exponent can silently flush this endpoint to zero.

### Current routed-expert scale geometry

The implementation currently treats one UE8M0 scale as covering 32 consecutive logical values along K:

```text
WASTE_UE8M0_BLOCK = 32
```

For a logical `[rows, cols]` routed-expert matrix, the current storage model is:

```text
packed weights: [rows, cols / 2] bytes
scale plane:    [rows, cols / 32] bytes
```

The dimensions must divide evenly into nibble pairs and K32 scale blocks. The scalar API refuses shapes that would require a ragged FP4 scale block rather than reading past the supplied scale plane.

**This geometry is a current port contract, not yet checkpoint-verified. README Gate A / V0 must verify the exact official representation; Gate B / V1b must verify the arithmetic convention used with it.**

---

## 5. FP4 nibble packing — highest-risk unresolved convention

Two E2M1 codes share one byte. The public number-format specification does not determine which logical column occupies which nibble.

Current repository choice:

```text
even column -> LOW nibble
odd column  -> HIGH nibble
```

This is represented by `WASTE_FP4_LOW_NIBBLE_IS_EVEN == 1`.

### Literal pin

The convention is deliberately pinned independently of the pack helper:

```text
raw byte 0x21
scale = 1.0
column 0 -> low nibble  0x1 -> 0.5
column 1 -> high nibble 0x2 -> 1.0
```

This literal is load-bearing. Do not replace it with a value produced by the same pack helper or macro used by the decoder.

If the official DeepSeek reference proves high-nibble-first packing, the literal assertion is **supposed** to fail. The correct change is to update both the implementation and the explicit documented literal after citing the official reference, not to make the fixture adapt automatically.

Why this deserves Gate B/V1b attention: the wrong nibble order does not crash. It swaps logical columns in pairs and produces plausible but invalid matrix arithmetic.

---

## 6. E4M3 finite variant (`e4m3fn`)

The trunk decoder implements the finite E4M3 variant used by the current port plan, not an IEEE-style format that reserves the entire top exponent for infinities/NaNs.

Definition used by the independent test:

```text
sign = bit 7
exp  = bits 6..3
mant = bits 2..0
bias = 7
```

Only `S.1111.111` is NaN. There are no infinities. The top exponent continues to carry finite values.

Important landmarks:

| byte | value |
|---:|---:|
| `0x00` | `+0.0` |
| `0x01` | `2^-9` (minimum positive subnormal) |
| `0x08` | `2^-6` (minimum positive normal) |
| `0x38` | `1.0` |
| `0x78` | `256.0` |
| `0x7E` | `448.0` (maximum finite positive) |
| `0x7F` | NaN |
| `0x80` | `-0.0` |
| `0xFF` | NaN |

The `448` maximum is a critical discriminator. An IEEE-style interpretation that makes the top exponent non-finite reduces the positive maximum to `240` and silently changes the largest decoded trunk weights.

---

## 7. FP8 block scales

The scalar FP8 path currently models an E4M3 byte matrix `[rows, cols]` with an f32 scale grid:

```text
scale rows = ceil(rows / 128)
scale cols = ceil(cols / 128)
value(r,c) = e4m3(weight[r,c]) * scale[r/128,c/128]
```

Unlike the current FP4 path, the final FP8 block may be ragged in either dimension.

`tests/test_quant.c` exercises a deliberately ragged `200 × 300` matrix, producing a `2 × 3` scale grid, and tests both exact 128 boundaries and the partial final blocks.

### Scale direction remains open

The current implementation multiplies by the stored scale. The DeepSeek tensor naming described by the handoff includes `weight_scale_inv`, which suggests an inverse-scale convention, but the name alone is not an arithmetic proof.

Gate A/V0 plus Gate B/V1 reference reconciliation must answer exactly:

```text
Does official inference compute:
    decoded = e4m3(raw) * weight_scale_inv
or:
    decoded = e4m3(raw) / weight_scale_inv
or an equivalent transformed convention?
```

A wrong answer can leave every value finite and plausible while shifting weights by a scale-squared factor relative to the intended representation. This must be settled before **Gate C / V2**.

---

## 8. Matvec reference behavior

Both scalar quantized matvecs follow the same semantic contract:

```text
y[r] = sum_c dequantized_weight(r,c) * x[c]
```

The accumulator is `double` and the result is stored as `float`.

The current tests validate matvec by a second local path:

1. fully decode one matrix row to f32;
2. dot that row with the input using a separate loop/double accumulator;
3. compare with the direct quantized matvec result.

This proves internal arithmetic/indexing consistency. It is **not Gate C / V2**, because both paths share the repository's current DeepSeek convention assumptions.

Gate C/V2 requires one quantized projection against the official reference.

---

## 9. Exactness and tolerance policy

### Bit-level format decode

For the public format tables, tests are exact because every represented value is a dyadic rational inside binary32 range. There is no reason to use a loose epsilon for E2M1/UE8M0/E4M3 code-value tables.

### Scale/index selection

Block selection and nibble selection are discrete semantics and should be tested exactly with values chosen so a wrong index changes the result visibly.

### Projection output

Official-reference projection comparisons in Gate C/V2 will use tolerances chosen only after observing the real reference's accumulation/dtype behavior. Do not pre-author a permissive threshold based on the scalar C path.

### Optimized kernels

An optimized backend must be evaluated against both:

- the scalar C reference for implementation regression/localization;
- the official oracle fixture for model correctness.

Matching the scalar path is necessary but not sufficient until the scalar path itself has passed Gate B/V1 reference agreement.

---

## 10. Mutation-testing requirement for silent conventions

PR #3 mutation-tested ten one-line decoder faults. Nine were caught immediately. The one survivor was the exact convention called out as highest risk: FP4 nibble order.

The reason was not lack of coverage. The test enumerated the entire E2M1 code space. The problem was shared assumption:

```text
fixture packer ─┐
                ├─ both compiled from WASTE_FP4_LOW_NIBBLE_IS_EVEN
runtime decoder ┘
```

Flipping the macro changed both producer and consumer, preserving the round trip.

The fix was the raw `0x21` literal described above. After that change all ten mutations were caught.

For any convention that can be wrong while preserving a round trip, require at least one expected input/output pair whose bytes/numbers are written independently of the implementation convention.

See `docs/FIXTURES.md` for the general rule.

---

## 11. README Gate B / V1 completion checklist

Gate B/V1 is not complete until all of these are true:

- [x] E2M1 public format semantics are exhaustively tested.
- [x] UE8M0 public format semantics are exhaustively tested.
- [x] finite E4M3 public format semantics are exhaustively tested.
- [x] FP4 K32 indexing is model-free tested.
- [x] FP8 128×128/ragged indexing is model-free tested.
- [x] scalar FP4/FP8 matvec paths exist and are model-free tested.
- [x] silent nibble-order behavior has an implementation-independent literal test.
- [x] decoder mutations used during PR #3 are all killed by the final test suite.
- [ ] official DeepSeek reference confirms FP4 nibble order.
- [ ] official DeepSeek reference confirms scale direction and scale application.
- [ ] official DeepSeek reference confirms the expected native E4M3 variant/scale layout for the target tensors.
- [ ] official oracle fixtures are committed with provenance and consumed by the C tests.

Only then should `docs/VALIDATION.md` mark Gate B/V1 passed without qualification.

---

## 12. README Gate C / V2 handoff

Once official access is restored, the next arithmetic gate should stay intentionally small:

1. obtain one real/reference quantized weight tile and its scale metadata;
2. freeze raw encoded bytes and expected decoded values independently;
3. verify Gate B/V1 nibble/scale conventions first;
4. choose a small projection crossing at least one scale boundary;
5. dump the official output with dtype/device/provenance;
6. run the scalar C matvec against exactly the same input;
7. establish a justified tolerance;
8. only then expose the primitive to the model path or add SIMD.

If the convention fixture fails, stop at step 3. Do not debug a full projection while the byte interpretation is unresolved.