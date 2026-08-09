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

/* Official RMSNorm boundary: compute variance/normalization in f32, multiply
 * by the (checkpoint-BF16-valued) weight in f32, then cast back to BF16.
 * Passing weight=NULL is the per-head unit-weight RMS normalization used after
 * wq_b. out is an f32 container holding BF16-rounded values. */
int waste_ds_v4_rmsnorm_ref(const float *x, const float *weight,
                            size_t n, float eps, float *out);

/* Base-RoPE primitive used by ratio-0 layers. Pairs are interpreted as
 * complex (even + i*odd), rotated in f32 at `position`, and copied back to
 * the BF16 tensor boundary. inverse!=0 conjugates the frequency. */
int waste_ds_v4_rope_ref(float *x, size_t dim, size_t position,
                         float theta, int inverse);

/* In-place FP8 QAT simulation used for non-RoPE KV dims. Each block gets the
 * official power-of-two activation scale, values are E4M3FN quantized then
 * dequantized, and the result is copied back to BF16. */
int waste_ds_v4_fp8_sim_inplace_ref(float *x, size_t n, size_t block_size);

/* Reproduce get_window_topk_idxs for one batch item.
 * start_pos==0: output shape [seqlen, min(seqlen,window)].
 * start_pos>0:  output shape [1, window].
 * Returns the number of int32 entries written, or 0 on invalid arguments. */
size_t waste_ds_v4_window_indices_ref(size_t window, size_t seqlen,
                                      size_t start_pos,
                                      int32_t *out, size_t out_capacity);

/* One-head scalar replay of the pinned sparse-attention kernel.
 * q is one BF16-valued query [d]; kv is [n_kv,d] BF16-valued cache; idxs has
 * `topk` cache positions with -1 padding. The implementation preserves the
 * kernel's 64-position online-softmax blocks, BF16 cast of exp weights before
 * the value GEMM, f32 denominator, denominator-only attn sink, and final BF16
 * output boundary. */
int waste_ds_v4_sparse_attn_head_ref(const float *q,
                                     const float *kv, size_t n_kv,
                                     const int32_t *idxs, size_t topk,
                                     size_t d, float attn_sink,
                                     float softmax_scale,
                                     float *out);

#endif /* WASTE_DEEPSEEK_V4_ATTENTION_REF_H */
