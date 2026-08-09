# DeepSeek V4 attention bring-up

**Current state: Gate E / V5 is PARTIAL. Ratio-0 attention, the shared grouped output projection, the ratio-128 learned compressor, and a ratio-128 compressed-history composition seam are checkpoint/source verified at the scalar model-semantic level. A coherent same-input ratio-128 `Attention.forward` fixture and ratio-4 CSA/indexer attention remain open.**

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
compressed RoPE theta    160000
YaRN factor                   16
YaRN original context      65536
output groups                  8
output LoRA rank/group      1024
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

The official RMSNorm module loads checkpoint BF16 weights into f32 parameters for reference convenience, computes normalization and learned weight multiplication in f32, then casts back to the input BF16 dtype. The checkpoint storage dtype and the runtime parameter dtype are distinct facts; real fixture generation must preserve the former and reproduce the latter semantically.

### Per-head Q normalization

After `wq_b`, the source does a separate direct tensor expression:

```text
q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)
```

This is **not** the learned RMSNorm module. The scalar reference preserves the visible BF16 tensor boundaries of square/mean/+eps/rsqrt/final multiply with f32 opmath/reduction internally. A model-free fixture is chosen so this result differs from the learned-RMSNorm path.

### RoPE

Ratio-0 uses base RoPE. Compressed modes use the release's YaRN-adjusted compressed RoPE. Both rotate only the final 64 Q/KV dimensions. The complex rotation is evaluated in f32 and copied back into the BF16 tensor. Sparse-attention output receives the inverse matching RoPE before output projection.

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
post-RoPE Q                 [2,2,512] = 2,048 BF16 values
post-K64-QAT KV             [2,512]   = 1,024 BF16 values
post-inverse-RoPE attention [2,2,512] = 2,048 BF16 values
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

## 6. Ratio-128 learned compressor — PASSED real-checkpoint sub-seam

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v5_compressor_ratio128_real/
```

Real checkpoint tensors at the pinned revision:

```text
layers.3.attn.compressor.wkv.weight    BF16 [512,4096]
layers.3.attn.compressor.wgate.weight  BF16 [512,4096]
layers.3.attn.compressor.ape           F32  [128,512]
layers.3.attn.compressor.norm.weight   BF16 [512]
```

The final line is an important checkpoint-derived correction. The reference module keeps RMSNorm parameters in f32 after load, but the 0731 checkpoint stores this weight in BF16. Fixture generation preserves the BF16 bytes and upcasts them only for the source-equation math.

The fixture uses two 128-token chunks with exactly one non-zero token/hidden lane per chunk. That still exercises all 512 compressor dimensions, all 128 APE/softmax positions, learned norm, position-128 YaRN and K64 QAT, while making each non-zero `wkv/wgate` output exactly one product and therefore removing projection reduction-order ambiguity before the softmax.

Independent source equations and the scalar C path match exactly at four `[2,512]` BF16 boundaries:

```text
pooled-before-norm       1,024 BF16 values
post-norm                1,024 BF16 values
post-YaRN-RoPE           1,024 BF16 values
post-K64-QAT compressed  1,024 BF16 values
TOTAL                     4,096 exact BF16 values
```

Representative signatures:

```text
pooled chunk0 first8
36ff 378c 37bc 35b9 36a2 b634 b4ec b6ee

chunk1 YaRN tail8
bbe0 bc2a 3bd4 bbcd 3c0e 3ba1 baf6 397a

compressed chunk1 first8
3b40 ba40 bbf0 3b00 3ba0 bb20 3a20 3a60
```

The frozen compressed-KV dependency has SHA-256:

```text
61dba51c0f59a6f0b93fbed1d0416e5be072f86a3f84da3725ebb18a04419fe5
```

Acquisition evidence:

```text
workflow run 31327438676
artifact id 9041948245
artifact sha256 9e1a9c146b6c7ebd8e5b9fe429781a30376c36df956260b0e7f58ed39fe817ce
fixture freeze 798c6d4296f50d5a5d76912fc9734989a1e80454
```

The model-free YaRN mutation remains load-bearing: pair 20 is used because pair 0 is invariant to both base and YaRN blending in this configuration, while pair 31's difference collapses at BF16. Dropping YaRN interpolation now fails the test.

## 7. Ratio-128 compressed-history composition — PASSED checkpoint-derived sub-seam

Frozen fixture:

```text
tests/fixtures/deepseek_v4/v5_attention_history_ratio128_real/
```

The official prefill namespace is pinned explicitly:

```text
local KV positions:       0..255
compressed KV positions: 256..257
query row:                255
visible local row:        128..255
visible compressed row:   256,257
combined top-k length:    130
```

The fixture derives a real layer-3 head-0 Q and all 256 local KV entries from bounded checkpoint slices of:

```text
wq_a / wq_a.scale
q_norm
wq_b head-0 rows / scale rows
wkv / wkv.scale
kv_norm
attn_sink head 0
```

Those checkpoint-derived Q/local-KV tensors are then composed with the independently frozen real ratio-128 compressor KV. The history oracle shares no WASTE runtime helper; the scalar C replay must reproduce the exact source index row and sparse-attention output.

Exact result:

```text
130 / 130 compressed-history indices exact
512 / 512 BF16 attention values exact
```

Representative signatures:

```text
Q position255/head0 first8
bfcf 3f44 bf8f bf7d 3e43 3e97 3f13 3fac

local KV position255 first8
3fa0 be10 bdb0 bf60 3fa0 bf00 3ea0 bea0

top-k tail
250 251 252 253 254 255 256 257

attention row255 first8
bd5b bc9a 3cc7 bde7 3c7b 3dda 3e1f 3d59
```

Acquisition evidence:

```text
workflow run 31327770254
artifact id 9042046747
artifact sha256 91890d1a4c466a11c03ce631f78d16f5325c3001dc7b4fc562c587563d11da6f
fixture freeze 71e7641cb961bbd5d5069cd5145806f52edc9f95
```

### Evidence boundary

This is deliberately **not** called a complete ratio-128 `Attention.forward` proof. The compressed KV comes from the independently frozen compressor fixture, while the Q/local-KV structural sequence is generated separately. The seam proves:

- real checkpoint Q/local-KV projection and cast conventions under the fixture's reduction-safe structural inputs;
- prefill compressed visibility boundaries and the full-seqlen compressed offset;
- local+compressed namespace composition;
- exact sparse-attention arithmetic with a real `attn_sink`.

It does **not** yet prove that one coherent input sequence produces the same compressor KV, local KV, Q and final inverse-RoPE/output in one source-equivalent forward pass.

## 8. Permanent test coverage

Ordinary `make check` is expected to enumerate each DeepSeek gate replay by name. For Gate E the permanent list must include:

- learned-vs-direct-Q normalization distinction;
- RoPE/inverse-RoPE orientation;
- K64 KV QAT and K128 mutation;
- sliding-window and incremental ring indices;
- sink-softmax and 64-position online-softmax behavior;
- real ratio-0 checkpoint Q/KV/attention fixture;
- real grouped `wo_a/wo_b` output-projection fixture;
- model-free ratio-128 pooling and pair-20 compressed-YaRN mutations;
- real ratio-128 compressor stages;
- model-free ratio-128 prefill/decode history index semantics;
- real ratio-128 compressed-history composition.

Acquisition workflows are temporary by design and are removed after fixtures freeze. They must never contain hard-coded PR readiness or merge actions.

## 9. Next — coherent ratio-128 forward, then ratio-4 CSA/indexer

The next ratio-128 fixture should use **one coherent input sequence** and prove, at minimum:

```text
same input
 -> local Q/KV
 -> ratio-128 compressor KV
 -> local + compressed index namespace
 -> sparse attention
 -> inverse compressed RoPE
```

The already-proven shared grouped output projection can then be attached without rediscovering its orientation. Closing this coherent seam removes the current composition-fixture non-claim.

After that, ratio-4 CSA adds:

- the overlapping two-half compressor behavior;
- its own 128-dimensional indexer compressor;
- Hadamard rotation and FP4 simulation where the pinned source applies them;
- 64 indexer heads × 128 dimensions;
- BF16 `weights_proj` weighting;
- exact top-512 compressed-position selection/order;
- sparse attention over the selected compressed history.

Gate E / V5 closes only after ratio 0, coherent ratio 128, ratio 4/CSA, and the shared output projection all have independent checkpoint/source evidence.
