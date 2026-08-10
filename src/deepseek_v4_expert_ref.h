/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar DeepSeek V4 expert/MoE reference seams for Gate F/V4 bring-up.
 */
#ifndef WASTE_DEEPSEEK_V4_EXPERT_REF_H
#define WASTE_DEEPSEEK_V4_EXPERT_REF_H

#include <stddef.h>
#include <stdint.h>

/* Execute one routed expert with packed FP4 w1/w3 and a selectable row slice
 * of packed FP4 w2. Route weight is applied to the SwiGLU hidden before the
 * BF16 cast and w2, matching the pinned official Expert.forward path. */
int waste_ds_v4_routed_expert_ref(
    const float *x,
    size_t input_dim,
    size_t intermediate_dim,
    const uint8_t *w1,
    const uint8_t *w1_scale,
    const uint8_t *w3,
    const uint8_t *w3_scale,
    float swiglu_limit,
    float route_weight,
    const uint8_t *w2_rows,
    const uint8_t *w2_scale_rows,
    size_t out_rows,
    float *gate_out,
    float *up_out,
    float *hidden_bf16_out,
    float *out);

/* Execute the one shared expert. Shared weights are resident FP8 E4M3FN with
 * checkpoint-native E8M0 128x128 scale grids. No route weight is applied.
 * w2_scale_rows_e8m0 must contain every scale-grid row touched by out_rows:
 * ceil(out_rows/128) * (intermediate_dim/128) bytes. Gate F's <=128-row
 * fixture therefore needs one grid row; Gate H's full 4096-row replay needs
 * all 32. */
int waste_ds_v4_shared_expert_ref(
    const float *x,
    size_t input_dim,
    size_t intermediate_dim,
    const uint8_t *w1,
    const uint8_t *w1_scale_e8m0,
    const uint8_t *w3,
    const uint8_t *w3_scale_e8m0,
    float swiglu_limit,
    const uint8_t *w2_rows,
    const uint8_t *w2_scale_rows_e8m0,
    size_t out_rows,
    float *gate_out,
    float *up_out,
    float *hidden_bf16_out,
    float *out);

/* Combine one token's routed expert outputs exactly like official MoE.forward.
 * routed_out is [n_selected,out_dim] in router top-k slot order and expert_ids
 * has the matching IDs. Official code iterates experts in ascending expert ID,
 * accumulating each BF16 expert result into f32 y, then adds the BF16 shared
 * expert output and casts the final y to BF16. The result is returned as an
 * f32 container holding the BF16-rounded value.
 */
int waste_ds_v4_moe_combine_ref(const uint32_t *expert_ids,
                                const float *routed_out,
                                size_t n_selected,
                                const float *shared_out,
                                size_t out_dim,
                                float *out);

#endif /* WASTE_DEEPSEEK_V4_EXPERT_REF_H */
