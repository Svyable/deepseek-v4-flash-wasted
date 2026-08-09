# mHC — DeepSeek V4 Hyper-Connection Gate D result

**Status: README Gate D / the mHC portion of V3 is CHECKPOINT-VERIFIED + SOURCE-ORACLE-VERIFIED for the pinned 0731 release.**

Pinned model:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

This document records the exact scalar/model-semantic mHC seam proven in PR #7. It does not claim a complete transformer layer, attention block, or GPU kernel.

## Real layer-0 attention-HC parameters

All three tensors come from `model-00002-of-00048.safetensors`:

```text
layers.0.hc_attn_fn     F32 [24,16384]  1,572,864 B
layers.0.hc_attn_base   F32 [24]               96 B
layers.0.hc_attn_scale  F32 [3]                12 B
```

SHA-256:

```text
fn     f5c1ffdfb92df2c04ac17e9a31e38701f2b7a5cac0cd427a2df2aa3e239987fc
base   edaa695cf5de59f919415f6e71dcb35be5ad817a06fa7222ee3021d9f388adda
scale  0b0e327d2f4d1a104c53d6e0a9172cf532028383e82cdf6d70537cb83092c63f
```

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v3_mhc_real/
```

Its `provenance.json` records exact safetensors byte ranges, hashes, source contract, sparse BF16 inputs, expected output hashes, and diagnostic values.

## Pinned source equations

For `hc_mult=4`, `mix_hc=(2+4)*4=24`.

`hc_pre`:

```text
x [4,4096]
xf = flatten(x).float()                         # 16384
rsqrt = 1 / sqrt(mean(xf^2) + 1e-6)
mixes = hc_fn @ xf * rsqrt                     # 24
pre, post, comb = hc_split_sinkhorn(...)
y[d] = sum_source pre[source] * x[source,d]    # 4096
```

The checkpoint parameters are F32. The source path casts the flattened hidden state to F32 for this operation and casts the collapsed result back to the original hidden dtype afterward.

### Split + Sinkhorn

```text
pre[j]  = sigmoid(mixes[j]   * scale[0] + base[j]) + 1e-6
post[j] = 2 * sigmoid(mixes[j+4] * scale[1] + base[j+4])
```

The remaining 16 values form a row-major 4×4 matrix with axes:

```text
comb[source_hc, output_hc]
```

The official iteration order is:

1. row softmax;
2. add `hc_eps`;
3. column normalization;
4. repeat row normalization then column normalization for the remaining 19 iterations.

The algorithm therefore **ends with column normalization**. Do not append a final row normalization merely because “Sinkhorn should be doubly stochastic.” The real fixture makes that mistake visible.

### hc_post

For output HC lane `k`:

```text
out[k,d] = post[k] * branch[d]
         + sum_source comb[source,k] * residual[source,d]
```

The model-free orientation fixture deliberately uses a non-symmetric combination matrix so transposing `comb` fails immediately.

## Real fixture design

The residual input is BF16 `[4,4096]` but intentionally sparse, with two nonzero coordinates in every HC lane. This is not a shortcut around the learned parameters: each of the 24 `hc_fn` rows still consumes real checkpoint coefficients from all four HC lanes. It removes an unnecessary 16,384-term reduction-order variable from the first mHC proof.

The independent Python generator freezes:

```text
mixes[24]             F32 diagnostics
pre[4]                F32 diagnostics
post[4]               F32 diagnostics
comb[4,4]             F32 diagnostics
hc_pre output[4096]   BF16 expected
hc_post output[4,4096] BF16 expected
```

## Measured result

Real learned pre weights:

```text
0.962507725
0.00432062941
0.406412721
0.0966549814
```

Real learned post weights:

```text
0.162043542
0.00006982701598
0.000257947075
0.00308346772
```

Final combination row sums:

```text
0.924152449
1.03077358
1.07249396
0.972576029
```

Final combination column sums:

```text
0.999999115
0.999998917
0.999998891
0.999999099
```

This is expected: the final source operation is column normalization.

Scalar C versus the independent oracle:

```text
hc_pre:  exact BF16 equality at all 4,096 positions
hc_post: exact BF16 equality at all 16,384 positions
mixes/pre/post/comb diagnostic max_abs: 4.76837158e-07
```

Expected hashes:

```text
pre BF16     aa3bd08cb7bd06e6cc58f7ea093df80a341c8114a1423a4eb7d11f04112cd747
post BF16    2c56c2b2d65df15d450e6802d65f4308637bfb952e56f4e09d857859efa3ea9b
diagnostics  462f11c86d9aac3731bc25ecf827f923ff47d44ebbd8ff36f8e5ef6256e0065a
```

Validation run `31286846400`:

```text
make check                   32 passed, 0 failed, 13 skipped
model-free mHC invariants    PASS
real checkpoint mHC replay   PASS
make asan                    31 passed, 0 failed, 14 skipped
artifact SHA-256             b1de036631a958befb9a00c1a31e77692140f036f1746e390f433f8712984895
```

The real fixture was committed only after those checks passed.

## Permanent regression contract

Ordinary `make check` now reaches both:

```text
tests/test_v3_mhc_scalar.py
tests/test_v3_mhc_real.py
```

through the existing DeepSeek release-fixture aggregator.

The model-free test pins zero-logit Sinkhorn behavior and `comb[source,output]` orientation. The real test pins learned checkpoint behavior.

## Evidence boundary

This closes **README Gate D (mHC)** at the scalar/model-semantic primitive level.

It does not close every primitive grouped under the broader operational V3 bring-up. Still open before/alongside Gate F/V4 and Gate E/V5:

- router transform/selection arithmetic;
- expert SwiGLU/clamp arithmetic;
- normalization helpers used outside mHC;
- RoPE;
- attention projection/compression/indexer primitives.

It also does not claim official CUDA/TileLang execution. Optimized backends must match the frozen fixture later.

## Next

The next high-leverage seam is routing/MoE:

1. learned router using real layer-3 gate parameters;
2. hash-router IDs using real layer-0 `tid2eid`;
3. exact `sqrt(softplus)` score transform;
4. correction bias used for selection only;
5. top-6 exact IDs/order;
6. original transformed scores gathered for weights;
7. renormalize then multiply by `routed_scaling_factor=1.5`;
8. then one shared + selected routed-expert combination for Gate F/V4.
