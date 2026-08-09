# Numerics — native DeepSeek quantization reference contract

**Status: Gates B/V1 and C/V2 are passed at the scalar/model-semantic level for the pinned 0731 release. Gate A/V0 has already established the real checkpoint layout. The first real resident FP8 projection now uses frozen checkpoint bytes and an independent pinned-source oracle, and scalar C matches all eight outputs exactly at BF16. GPU/TileLang backend parity is explicitly not claimed.**

Pinned release:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

Evidence layers stay separate:

1. **public format semantics** — E2M1, UE8M0, finite E4M3FN;
2. **official source semantics** — packing order, block geometry, scale application, activation quantization;
3. **checkpoint evidence** — exact exported tensor/storage bytes;
4. **projection oracle evidence** — real checkpoint bytes + independently evaluated pinned source equations;
5. **backend evidence** — future optimized CPU/SIMD/GPU paths compared to the proven scalar/model-semantic seam.

See `OFFICIAL-0731-SOURCE.md`, `INVENTORY-0731.md`, `FIXTURES.md`, and `VALIDATION.md`.

---

## 1. Current gate result

| Claim | Evidence | Gate |
|---|---|---|
| E2M1 code values | exhaustive model-free tests | B/V1 |
| UE8M0 values | exhaustive model-free tests | B/V1 |
| finite E4M3FN values | exhaustive model-free tests | B/V1 |
| FP4 K32 indexing | model-free boundary tests | B/V1 |
| FP8 128x128 indexing | model-free exact/ragged tests | B/V1 |
| FP4 low-nibble-first | pinned official source + independent literal fixture | B/V1 |
| routed E8M0 `[out,in/32]` scale layout | official source + real Gate A headers | A/V0 + B/V1 |
| FP4/FP8 weight-scale multiplication | pinned official source | B/V1 |
| real resident E4M3/E8M0 projection layout | real checkpoint headers | A/V0 |
| activation K128 E4M3 quantization | pinned official source | C/V2 |
| one real quantized resident projection | real checkpoint bytes + independent source oracle + exact scalar-C BF16 replay | **C/V2 PASSED** |
| official TileLang/GPU kernel execution | not run | later backend validation |

The Gate C result is deliberately stronger than a synthetic round trip and deliberately narrower than end-to-end model parity.

---

## 2. Scalar/reference code

- `src/quant/fp4_e2m1.{c,h}` — routed E2M1 + E8M0 K32 decode/reference matvec.
- `src/quant/fp8_e4m3.{c,h}` — finite E4M3FN + 128x128 weight-scale decode/reference matvec.
- `src/quant/deepseek_v4_linear_ref.{c,h}` — official-linear-shaped K128 activation quantization, FP8 block accumulation, and BF16 output rounding.
- `tests/test_quant.c` — exhaustive public-format/model-free indexing checks.
- `tests/fixtures/deepseek_v4/fp4_release_convention.json` — pinned source convention literal.
- `tests/fixtures/deepseek_v4/v2_wq_a_real/` — frozen real Gate C projection.
- `tests/test_v2_real_projection.py` — exact scalar-C replay of the real Gate C fixture.
- `tests/test_fetch_hf_tensor_slice.py` — fail-closed bounded payload Range acquisition test.

`make check` reaches the real Gate C replay through `tests/test_release_quant_fixture.py`; no network is needed after the fixture is frozen.

---

## 3. E2M1 and E8M0

E2M1 important values:

| code | value |
|---:|---:|
| `0x0` | `+0.0` |
| `0x1` | `+0.5` |
| `0x2` | `+1.0` |
| `0x7` | `+6.0` |
| `0x8` | `-0.0` |
| `0xF` | `-6.0` |

E8M0 scalar definition:

```text
scale(e) = 2^(e - 127), e = 0x00..0xFE
0xFF     = NaN
```

Important scale codes:

| code | value |
|---:|---:|
| `0x7E` | `0.5` |
| `0x7F` | `1.0` |
| `0x80` | `2.0` |
| `0x00` | `2^-127` |
| `0xFE` | `2^127` |
| `0xFF` | NaN |

All public-format cases remain exact tests.

---

## 4. Routed FP4 — Gate B / V1 result

The real checkpoint and pinned source agree on the routed format. For logical `[out,in]`:

```text
packed weight: [out, in/2]  byte plane (checkpoint dtype I8)
scale:         [out, in/32] F8_E8M0
```

Checkpoint examples:

```text
layers.0.ffn.experts.0.w1.weight  I8        [2048,2048]
layers.0.ffn.experts.0.w1.scale   F8_E8M0  [2048,128]
layers.0.ffn.experts.0.w2.weight  I8        [4096,1024]
layers.0.ffn.experts.0.w2.scale   F8_E8M0  [4096,64]
```

The `I8` dtype is storage framing for packed nibbles; the bytes are not signed arithmetic values.

### Nibble order

Pinned official conversion expands each byte as:

```text
low  = byte & 0x0f
high = (byte >> 4) & 0x0f
stack([low, high]) along K
```

Thus even/lower K is the low nibble. The independent fixture:

```text
packed byte 0x21
E8M0 scale 0x80 = 2
expected logical values [1.0, 2.0]
```

catches both nibble reversal and multiply-vs-divide scale errors.

### Scale application

The official kernel multiplies block partials by activation scale × weight scale. The converter's `weight_scale_inv -> scale` rename does not perform a reciprocal.

Gate B/V1 is no longer an unresolved storage-convention gate for this pinned release.

---

## 5. Finite E4M3FN

The resident quantized path uses finite E4M3FN:

```text
sign = bit 7
exp  = bits 6..3
mant = bits 2..0
bias = 7
```

Only `S.1111.111` is NaN; there are no infinities.

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

`448`, not `240`, remains the important format discriminator.

---

## 6. Official quantized linear semantics

For each activation row and each K128 block, pinned source semantics are:

```text
amax = max(max(abs(x_block)), 1e-4)
scale = next_power_of_two(amax / 448)
qx = E4M3FN(clamp(x_block / scale, -448, +448))
```

For resident FP8 weights, each K128 block contributes:

```text
dot(qx, qw) * activation_scale * weight_scale
```

with FP32 accumulation and BF16 reference output.

`waste_ds_v4_fp8_linear_ref()` implements that model-semantic path separately from the simpler decoded-weight `waste_fp8_matvec()`. Both are retained because they isolate different failure classes.

---

## 7. Gate C / V2 real fixture — PASSED

Target selected from the real Gate A inventory:

```text
layers.0.attn.wq_a.weight  F8_E4M3 [1024,4096]
layers.0.attn.wq_a.scale   F8_E8M0 [8,32]
shard: model-00002-of-00048.safetensors
```

The frozen fixture contains only:

```text
32,768 B  first 8 real weight rows
    32 B  matching first real scale row
 8,192 B  deterministic BF16 input
    16 B  expected 8-element BF16 output
```

Real payload SHA-256s:

```text
weight rows  : bb329d4d1ebd10458795fca07dcdee5d78b2a90a45c770f7fc44f24f1ea24d65
weight scales: cb945324f5d26ec47059a4d4d589c9579237ff51b7e4058fe44faddcd4c1fbbe
input        : d3cf4913260c0c97c8efabf0a71b91ee2c65fdbe61cc4b617a68ca3acf91337e
expected     : 3fd1660da111fa10ea36469600386281c52926d03ee4e5133ebb9653ec2c7351
```

Expected BF16 bits:

```text
0x3e79 0xbf84 0x3f8d 0x400a 0x3ff3 0xbf9b 0x3f82 0x3ff0
```

The independent Python source oracle rejected any case whose final BF16 result differed across:

1. sequential f32 reduction;
2. reverse f32 reduction;
3. exact dyadic accumulation.

The chosen real fixture is stable across all three. Scalar C then matched all eight BF16 values **exactly**.

Validation run `31286320991` on Ubuntu 24.04.4 also reported:

```text
make check -> 32 passed, 0 failed, 13 skipped
real Gate C C replay -> PASS, exact BF16
make asan -> 31 passed, 0 failed, 14 skipped
```

Artifact zip SHA-256:

```text
7ee27b46199f2b1a1c11a3a095d127aa52f550b69c7864d5b7727d6f08a710c9
```

Fixture provenance is frozen in `tests/fixtures/deepseek_v4/v2_wq_a_real/provenance.json`.

### Evidence boundary

Gate C/V2 is considered passed for the **scalar/model-semantic projection seam**: real checkpoint bytes + pinned source equations + independent oracle + exact C replay.

It does **not** claim that the official TileLang GPU kernel was executed on this runner. Optimized CPU/SIMD/GPU backends must later match the now-proven scalar/oracle fixture; backend parity is not retroactively folded into Gate C.

---

## 8. Exactness and tolerance policy after Gate C

Gate C needed no loose numerical tolerance: the selected fixture is BF16-stable across independent reduction views and C matches exact output bits.

For downstream primitives:

- selected IDs, layout decisions and exact-format values remain exact checks;
- floating-point intermediate tolerances are introduced only when an independently generated fixture demonstrates unavoidable precision/reduction differences;
- a known-wrong mutation must remain outside the accepted threshold;
- optimized paths compare to both scalar C and independent official-source/checkpoint fixtures.

Never loosen a downstream tolerance merely because a new implementation misses the oracle.

---

## 9. Gate status and next arithmetic rung

### Gate A / V0

**PASSED / CHECKPOINT-VERIFIED.** See `INVENTORY-0731.md` and `reference/deepseek-v4-flash-0731.gate-a.json`.

### Gate B / V1

**PASSED for pinned native-format/storage semantics.** Public formats, nibble order, scale direction and real storage geometry are covered by independent tests/source/checkpoint evidence.

### Gate C / V2

**PASSED at the scalar/model-semantic level** on the frozen real `layers.0.attn.wq_a` projection described above.

### Next: Gate D / V3

The next arithmetic milestone is mHC/model primitives. Gate D should begin with an independently generated fixture for the exact Hyper-Connection/Sinkhorn operation before it is embedded inside a transformer block. Router/MoE work follows as Gate F/V4; attention modes remain Gate E/V5.
