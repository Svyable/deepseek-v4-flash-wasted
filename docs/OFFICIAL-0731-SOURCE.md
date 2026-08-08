# Official DeepSeek-V4-Flash-0731 source findings

**Evidence state: OFFICIAL-SPEC / SOURCE-VERIFIED at the immutable release commit below. Checkpoint tensor names/shapes/bytes remain Gate A/V0 until the real index/headers are consumed.**

This document records facts established directly from the official release source so they do not remain buried in research notes or depend on a moving `main` branch.

## Pinned baseline

```text
repository: deepseek-ai/DeepSeek-V4-Flash-0731
release commit: 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
release name: Release DeepSeek-V4-Flash-0731
license file: LICENSE
license: MIT
copyright: Copyright (c) 2023 DeepSeek
```

A later repository commit may update documentation or serving recipes without changing the checkpoint release. Golden fixtures, conversion manifests and model arithmetic should keep the immutable release SHA unless a project PR explicitly moves the baseline.

## Gate A/V0 facts established from source/config

The release config/reference establishes the model-level contract independently of checkpoint-header verification:

- architecture: `DeepseekV4ForCausalLM`;
- 43 main decoder layers;
- hidden size 4096;
- 256 routed experts and one shared expert per main MoE layer;
- top-6 routed experts per token;
- MoE intermediate size 2048;
- first 3 layers use the bootstrap/hash routing path;
- mHC multiplier 4 and 20 Sinkhorn iterations;
- hybrid attention with sliding window 128 and compressed/indexed modes;
- routed expert target dtype `fp4`;
- non-expert quantization config uses finite E4M3 with UE8M0 scales and 128x128 block size;
- DSpark is attached to the release but remains optional/deferred for the base WASTE path.

These are source/config facts, not a substitute for the checkpoint inventory. Gate A/V0 still has to establish the exact exported tensor names, stored shapes, dtypes, shard placement and byte totals.

## Bootstrap routing names and semantics

The official runtime uses a token-ID-to-expert table named:

```text
tid2eid
```

with logical shape:

```text
[vocab_size, num_experts_per_tok]
```

for layers satisfying:

```text
layer_id < n_hash_layers
```

The official converter also contains a source-name special case spelled:

```text
tie2eid
```

The port should therefore recognize both spellings while reconciling source/export names. This is intentionally narrow: it is not permission to add a generic catch-all for unknown router tensors.

For learned-routing layers, source semantics are:

```text
raw = router(x)
scores = sqrt(softplus(raw))
selection_scores = scores + correction_bias
ids = topk(selection_scores, top_k)
weights = scores[ids]
weights = weights / sum(weights)
weights = weights * routed_scaling_factor
```

The correction bias affects selection only; the selected routing weights come from the original transformed scores.

## Gate B/V1 routed FP4 convention — source verified

PR #3 deliberately left nibble order and scale direction unresolved. The release source settles both, and the existing scalar implementation chose the correct convention.

### Storage geometry

The official model binds a routed FP4 matrix logically as `[out, in]` and stores:

```text
weight: [out, in/2]  torch.float4_e2m1fn_x2
scale:  [out, in/32] torch.float8_e8m0fnu
```

Therefore one E8M0 scale covers 32 consecutive logical K values.

### Nibble order

The official converter views packed bytes as unsigned bytes, then expands each byte in this order:

```text
low  = byte & 0x0f
high = (byte >> 4) & 0x0f
stack([low, high]) along K
```

Therefore:

```text
even/lower logical K index -> LOW nibble
odd/next logical K index   -> HIGH nibble
```

This matches `WASTE_FP4_LOW_NIBBLE_IS_EVEN == 1`.

### Scale application

The official FP4 GEMM operates per 32 K values and accumulates each partial after multiplying by activation and weight scales. The stored E8M0 weight scale is therefore a multiplier, not a reciprocal applied by division.

The converter also renames `weight_scale_inv` to `scale` without computing a reciprocal.

The committed F3 fixture `tests/fixtures/deepseek_v4/fp4_release_convention.json` freezes this source-derived literal:

```text
packed byte: 0x21
scale byte:  0x80 = 2.0
expected logical values: [1.0, 2.0]
```

This case fails under either a nibble reversal or multiply-vs-divide scale mutation.

## Resident FP8 convention — source verified, full projection still open

The reference uses finite `torch.float8_e4m3fn` weights with one scale per 128x128 block. The converter's `weight_scale_inv -> scale` rename does not invert the numeric value, and the official GEMM multiplies by the scale.

This validates the scalar decoder's current scale direction and E4M3FN interpretation at the source-contract level.

It does **not** complete Gate C/V2. A real official quantized projection still needs the released operation's activation-quantization and accumulation behavior exercised against a frozen oracle fixture.

## Expert activation semantics

The official routed/shared expert path uses the three-matrix SwiGLU roles:

```text
gate = w1(x)
up   = w3(x)
```

With the configured clamp, source semantics clamp:

```text
up   to [-limit, +limit]
gate to maximum +limit
```

then compute:

```text
silu(gate) * up
```

followed by `w2`. Route weights are applied according to the official MoE path before/around the down-projection operation as defined by the reference; preserve its exact ordering when Gate F/V4 is implemented.

## Acquisition path now implemented locally

`tools/fetch_hf_headers.py` follows Hugging Face's documented safetensors metadata protocol:

1. fetch bytes `0-7` to obtain the little-endian JSON-header length;
2. fetch exactly bytes `8 .. 7+header_length`;
3. require HTTP `206 Partial Content`;
4. write only the native safetensors header framing locally;
5. let `tools/inventory.py` perform all tensor accounting from those stubs.

This design intentionally fails closed when a server/proxy ignores Range rather than risk converting a metadata query into a whole-shard transfer.

## Gates after this source review

| Gate | State after this PR tranche | Remaining proof |
|---|---|---|
| **A / V0** checkpoint inventory | tooling + source contract advanced; **not passed** | run the pinned index/header fetch and strict inventory; reconcile every real name/shape/byte |
| **B / V1** native quantization conventions | source semantics established; replay fixture added | run the branch test suite; keep real-header storage checks under Gate A |
| **C / V2** one quantized trunk projection | not passed | frozen official projection fixture covering activation quantization, scale boundaries and accumulation |
| **F / V4** routing/MoE | source semantics documented only | official hidden-state/router/expert fixtures and C implementation |

Do not promote source inspection into checkpoint or end-to-end evidence. It narrows the unknowns; the next gates still have to consume the released artifacts and oracle outputs.
