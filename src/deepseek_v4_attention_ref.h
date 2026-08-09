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
 * q *= rsqrt(q.square().mean(-1)+eps) directly on the BF16 q tensor. */
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
 * It deliberately stops before grouped wo_a/wo_b; output projection is tested
 * independently below so attention-core failures remain localizable. */
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

/* One complete output-projection group.
 *
 * Official source reshapes 64 attention heads into 8 groups, each containing
 * 8*512 = 4096 BF16 values. Checkpoint wo_a is FP8 E4M3 + E8M0 scale, but the
 * official converter dequantizes it to BF16 before the einsum. For one group:
 *
 *   group_input[4096]
 *     x BF16(dequant(wo_a_group[1024,4096])) -> group_lora[1024] BF16
 *
 * The 8 group latents are flattened to 8192 then fed to quantized wo_b. This
 * bounded seam places `group_lora` at group_index and zeros all other groups,
 * then evaluates `out_rows` real wo_b rows. It proves wo_a dequant/orientation,
 * group placement, and wo_b quantized-linear semantics without pretending a
 * synthetic group vector is the full 64-head attention output.
 *
 * wo_a_scale is the group's [8,32] E8M0 block grid. wo_b_scale_rows supplies
 * one 64-entry scale-grid row when out_rows <= 128.
 */
int waste_ds_v4_attention_output_group_ref(
    const float *group_input,
    size_t group_index,
    const uint8_t *wo_a_weight,
    const uint8_t *wo_a_scale_e8m0,
    const uint8_t *wo_b_rows,
    const uint8_t *wo_b_scale_rows_e8m0,
    size_t out_rows,
    float *group_lora_out,
    float *out);

#endif /* WASTE_DEEPSEEK_V4_ATTENTION_REF_H */
