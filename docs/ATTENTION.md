# DeepSeek V4 attention bring-up

**Current state: Gate E / V5 is PARTIAL. The ratio-0 layer-0 attention core and the shared grouped output-projection seam are checkpoint/source verified at the scalar model-semantic level. Ratio-128 compressed attention and ratio-4 CSA/indexer attention remain open.**

Pinned release:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

This document owns the attention-specific validation contract. `docs/VALIDATION.md` remains the canonical gate ladder.

## 1. Structurally distinct attention modes

The 43-layer base stack uses three attention classes selected by `compress_ratios`:

- **ratio 0** — sliding-window-only attention; no compressor/indexer;
- **ratio 128** — compressed-history attention using the learned compressor, no CSA indexer;
- **ratio 4** — CSA: learned compressor plus learned indexer/top-k position selection.

Gate E is complete only after all three are independently verified. Shared output projection is proved separately because all three modes converge there.

## 2. Fixed 0731 geometry relevant to attention

```text
hidden size                 4096
q LoRA rank                 1024
attention heads             64
head dimension              512
KV heads                     1
KV width                    512
RoPE dimension               64
non-RoPE KV width           448
sliding window              128
KV QAT block                 64
RMS epsilon                 1e-6
base RoPE theta           10000
output groups                 8
output LoRA rank/group     1024
```

The group layout is eight heads per group:

```text
64 heads x 512
 -> 8 groups x 4096
 -> wo_a: 8 x [1024,4096]
 -> flatten 8 x 1024 = 8192
 -> wo_b: [4096,8192]
```

## 3. Source cast boundaries that are part of the model

These are not optional implementation details.

### Learned q_norm / kv_norm

The official RMSNorm module upcasts its input to f32, computes normalization and learned weight multiplication in f32, then casts back to the input BF16 dtype.

### Per-head Q normalization

After `wq_b`, the source does a separate direct tensor expression:

```text
q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)
```

This is **not** the learned RMSNorm module. The scalar reference preserves the visible BF16 tensor boundaries of square/mean/+eps/rsqrt/final multiply with f32 opmath/reduction internally. A model-free fixture is chosen so this result differs from the learned-RMSNorm path.

### RoPE

Base ratio-0 RoPE rotates only the last 64 Q/KV dimensions. The complex rotation is evaluated in f32 and copied back into the BF16 tensor. Sparse-attention output receives the inverse base RoPE on its last 64 dimensions before output projection.

### KV QAT simulation

For non-RoPE KV dimensions only:

```text
kv[..., :448]
```

is quantized/dequantized in-place to finite E4M3 with **K64** power-of-two scales. A K128 mutation is explicitly observable in the model-free fixture.

### Sparse attention

The pinned kernel contract used by the scalar reference is:

- Q/KV inputs BF16;
- dot-product scores and online-softmax state f32;
- sparse processing in 64-position blocks;
- exponential weights cast to BF16 before the value GEMM;
- accumulated value numerator f32;
- `attn_sink` contributes to the denominator only;
- denominator f32;
- output cast BF16.

## 4. Ratio-0 real checkpoint fixture — PASSED sub-seam

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v5_attn_ratio0_real/
```

Scope:

```text
2 tokens x 2 heads
layer 0
compress_ratio = 0
through inverse-RoPE sparse-attention output
wo_a/wo_b excluded from this fixture
```

The bounded payload uses real checkpoint:

- full `layers.0.attn.wq_a` + scale;
- `q_norm`;
- `wq_b` rows for heads 0–1 + matching scales;
- full `wkv` + scale;
- `kv_norm`;
- attention sinks for heads 0–1.

Token 0 is exactly the already-proven Gate-C input. Before any attention result was accepted, the standalone expected-value producer `tools/v5_attention_oracle.c` had to reproduce Gate C's first eight `wq_a` BF16 outputs bit-for-bit. The oracle includes/links no WASTE runtime helper.

Token 1 is a deterministic BF16-bit permutation/sign transform of token 0, giving a genuine position-1 RoPE/two-key case without needing another external input source.

Exact permanent comparisons:

```text
post-RoPE Q                [2,2,512] = 2,048 BF16 values
post-K64-QAT KV            [2,512]   = 1,024 BF16 values
post-inverse-RoPE attention[2,2,512] = 2,048 BF16 values
```

All **5,120 BF16 values match exactly** between the independent source-equation oracle and the WASTE scalar C reference.

Representative signatures:

```text
Q pos0/head0 first8
bf2b 3f58 bfb6 beba bee2 bfaa be58 3da2

Q pos1/head0 RoPE-tail first8
402f 3fb0 3fac bf83 bf8b 3fc3 bf46 bd40

attention pos1/head0 first8
be21 3e02 be52 be33 3de1 3f07 3e54 3e27
```

Acquisition artifact:

```text
artifact id 9032451539
sha256 ad0db9462d8acd200bac033a61cacd287e5dc2092a133880ff79a12f8850af6b
```

Fixture freeze commit:

```text
23eee6b
```

Evidence boundary: this is a source-equation CPU oracle against real checkpoint bytes, not a claim of bitwise parity with the official TileLang accelerator kernel.

## 5. Grouped output projection — PASSED shared sub-seam

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v5_attn_output_group0_real/
```

The official converter dequantizes checkpoint `wo_a` E4M3×E8M0 blocks to **BF16** before the source `einsum`. `wo_b` remains the ordinary quantized FP8 linear.

To isolate projection orientation without freezing another full 64-head attention fixture, the test constructs one structurally valid 4096-value output group from real ratio-0 token-1 heads 0/1:

- all eight 512-wide head segments are present;
- eight deterministic lanes/head are non-zero;
- the non-zero BF16 values come from the real frozen attention output;
- heads 2–7 are deterministic permutation/sign transforms and are explicitly not claimed as real attention heads.

Real checkpoint payload:

```text
wo_a group 0 rows 0:1024     F8_E4M3 [8192,4096] slice
wo_a scale rows 0:8          F8_E8M0 [64,32] slice
wo_b rows 0:8                F8_E4M3 [4096,8192] slice
wo_b scale row 0             F8_E8M0 [32,64] slice
```

Independent Python equations require sequential, reverse, and exact-rational reductions to agree at the BF16 boundary before a `wo_a` or `wo_b` expected value can be frozen.

Scalar C exact results:

```text
group latent first8
3a65 3dcb 3d2b 3d09 3cba bc02 bcab 3d9e

wo_b output first8
ba34 bce1 bd35 3bd0 3d87 3d77 bc85 bd46
```

A group-placement mutation reuses the same group latent at group index 1; it must not reproduce the group-0 `wo_b` output. This pins the `[8 groups,1024] -> 8192` flattening semantics.

Evidence boundary: this proves real checkpoint `wo_a` dequant/orientation and `wo_b` arithmetic for one full structural group. It does not claim heads 2–7 were independently generated by the real attention core, nor that all eight groups were simultaneously active.

## 6. Permanent test coverage

Ordinary `make check` replays:

- learned-vs-direct-Q normalization distinction;
- RoPE/inverse-RoPE orientation;
- K64 KV QAT and K128 mutation;
- sliding-window and incremental ring indices;
- sink-softmax and 64-position online-softmax behavior;
- real ratio-0 checkpoint Q/KV/attention fixture;
- real grouped `wo_a/wo_b` output-projection fixture.

Temporary acquisition workflows are removed before merge once these fixtures are frozen and the ordinary suite/sanitizers are green.

## 7. Next — ratio 128 compressed attention

Ratio-128 removes the CSA indexer but introduces the learned compressor. The source contract to prove next is:

```text
wkv / wgate
+ learned absolute position embedding over 128-token chunks
+ softmax-weighted pooling
+ learned RMSNorm
+ compressed-position RoPE at the chunk-start position
+ K64 QAT of non-RoPE compressed KV
+ dense attention over compressed history
```

This should be brought up before ratio-4 CSA because it isolates compressor correctness from indexer/top-k correctness.

## 8. Then — ratio 4 CSA/indexer

The final Gate-E attention mode adds:

- its own compressor path;
- 64 indexer heads × 128 dimensions;
- FP4-simulated query/KV scoring where defined by the pinned source;
- exact top-512 compressed-position selection/order;
- sparse attention over the selected compressed history.

Gate E / V5 closes only after ratio 0, ratio 128, ratio 4/CSA, and the shared output projection all have independent evidence.
