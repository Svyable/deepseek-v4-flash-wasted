/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar ratio-128 compressor seams for DeepSeek V4 Gate E/V5.
 */
#ifndef WASTE_DEEPSEEK_V4_COMPRESSOR_REF_H
#define WASTE_DEEPSEEK_V4_COMPRESSOR_REF_H

#include <stddef.h>
#include <stdint.h>

/* Per-dimension learned gated pooling used by Compressor.forward.
 * kv/score/ape are [tokens,dim], f32. For every output dimension independently:
 *   p[t] = softmax_t(score[t,d] + ape[t,d])
 *   out[d] = sum_t kv[t,d] * p[t]
 */
int waste_ds_v4_compressor_pool_ref(const float *kv,
                                    const float *score,
                                    const float *ape,
                                    size_t tokens, size_t dim,
                                    float *out);

/* Apply the compressed-attention YaRN RoPE used when compress_ratio!=0.
 * x contains BF16-valued f32 data; only the final rope_dim values are rotated.
 * The operation evaluates frequencies in f32 and writes back to BF16. */
int waste_ds_v4_compressed_rope_ref(float *x, size_t dim, size_t rope_dim,
                                    size_t position,
                                    size_t original_seq_len,
                                    float base, float factor,
                                    float beta_fast, float beta_slow);

/* Fixed 0731 ratio-128 prefill compressor.
 *
 * x: [seqlen,4096] BF16-valued f32, seqlen a multiple of 128.
 * wkv/wgate: raw checkpoint BF16 matrices [512,4096]. The official module
 *            loads these checkpoint BF16 values into f32 Linear parameters.
 * ape:       checkpoint/source F32 [128,512].
 * norm:      checkpoint F32 [512].
 * out:       [seqlen/128,512] f32 container holding final BF16 values after
 *            pooling -> BF16 -> RMSNorm -> compressed YaRN RoPE -> K64 QAT
 *            on the first 448 non-RoPE dimensions.
 */
int waste_ds_v4_compressor_ratio128_prefill_ref(
    const float *x, size_t seqlen,
    const uint16_t *wkv_bf16,
    const uint16_t *wgate_bf16,
    const float *ape,
    const float *norm_weight,
    float norm_eps,
    float *out);

#endif /* WASTE_DEEPSEEK_V4_COMPRESSOR_REF_H */
