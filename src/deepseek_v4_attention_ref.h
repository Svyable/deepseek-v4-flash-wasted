/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar DeepSeek V4 attention primitives for Gate E/V5 bring-up.
 * Correctness/debugging seams only; not performance kernels.
 */
#ifndef WASTE_DEEPSEEK_V4_ATTENTION_REF_H
#define WASTE_DEEPSEEK_V4_ATTENTION_REF_H

#include <stddef.h>
#include <stdint.h>

/* Learned RMSNorm module: source explicitly upcasts x to f32, computes the
 * norm and f32 weight multiply, then casts the result back to BF16. */
int waste_ds_v4_rmsnorm_ref(const float *x, const float *weight,
                            size_t n, float eps, float *out);

/* Per-head Q normalization is NOT the RMSNorm module. The pinned source does
 * q *= rsqrt(q.square().mean(-1)+eps) directly on the BF16 q tensor. This
 * reference preserves the visible BF16 boundaries of square/mean/+eps/rsqrt/
 * final multiply while using f32 opmath/reduction internally. */
int waste_ds_v4_head_rmsnorm_bf16_ref(const float *x, size_t n,
                                      float eps, float *out);

int waste_ds_v4_rope_ref(float *x, size_t dim, size_t position,
                         float theta, int inverse);

int waste_ds_v4_fp8_sim_inplace_ref(float *x, size_t n, size_t block_size);

size_t waste_ds_v4_window_indices_ref(size_t window, size_t seqlen,
                                      size_t start_pos,
                                      int32_t *out, size_t out_capacity);

int waste_ds_v4_sparse_attn_head_ref(const float *q,
                                     const float *kv, size_t n_kv,
                                     const int32_t *idxs, size_t topk,
                                     size_t d, float attn_sink,
                                     float softmax_scale,
                                     float *out);

/* Ratio-0 Attention.forward seam through inverse-RoPE sparse-attention output.
 * It deliberately stops before grouped wo_a/wo_b; that output projection is a
 * subsequent Gate-E sub-seam.  This function proves the DeepSeek-specific
 * query/KV preparation and sparse attention core without mixing in CSA/HCA.
 *
 * Fixed 0731 geometry used here:
 *   input hidden 4096, q LoRA rank 1024, KV 512,
 *   head dim 512, rope dim 64, K64 KV QAT, window 128.
 *
 * q_b_weight contains exactly n_heads*512 output rows and q_b_scale contains
 * ceil(n_heads*512/128) scale rows, each with 1024/128 columns.
 * q_after_rope, kv_after_qat and attn_after_inverse_rope are optional
 * diagnostics (NULL allowed) with shapes [seqlen,n_heads,512], [seqlen,512],
 * and [seqlen,n_heads,512], respectively. */
int waste_ds_v4_attention_ratio0_prefill_ref(
    const float *x, size_t seqlen, size_t n_heads,
    const uint8_t *wq_a_weight, const uint8_t *wq_a_scale_e8m0,
    const float *q_norm_weight,
    const uint8_t *wq_b_weight, const uint8_t *wq_b_scale_e8m0,
    const uint8_t *wkv_weight, const uint8_t *wkv_scale_e8m0,
    const float *kv_norm_weight,
    const float *attn_sink,
    float rms_eps, float rope_theta,
    float *q_after_rope,
    float *kv_after_qat,
    float *attn_after_inverse_rope);

#endif /* WASTE_DEEPSEEK_V4_ATTENTION_REF_H */
