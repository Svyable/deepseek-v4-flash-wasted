/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar grouped attention-output projection seam for Gate E/V5 and Gate H/V6.
 */
#ifndef WASTE_DEEPSEEK_V4_ATTENTION_OUTPUT_REF_H
#define WASTE_DEEPSEEK_V4_ATTENTION_OUTPUT_REF_H

#include <stddef.h>
#include <stdint.h>

/* Prove one of the eight DeepSeek V4 output groups independently.
 *
 * group_input: [4096] BF16-valued f32 container = 8 heads x 512.
 * wo_a_weight: checkpoint E4M3 bytes [1024,4096] for one group.
 * wo_a_scale: checkpoint E8M0 grid [8,32] for that group.
 * wo_b_rows: checkpoint E4M3 rows [out_rows,8192].
 * wo_b_scale_rows: checkpoint E8M0 scale-grid rows covering out_rows.
 * group_index: 0..7 placement in the flattened [8,1024] group latent.
 *
 * The official converter dequantizes wo_a E4M3*E8M0 to BF16 before the einsum;
 * this reference does the same, accumulates the BF16 einsum in f32, rounds the
 * 1024 group latent to BF16, inserts it at group_index (other groups zero),
 * then runs the ordinary quantized wo_b linear.
 *
 * group_lora_out is optional [1024]. out is [out_rows]. Both are f32
 * containers holding BF16-rounded values.
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

/* Complete 64-head output projection used by Gate H/V6.
 *
 * attention_heads: [64,512] BF16-valued f32, grouped as 8 groups x 8 heads.
 * wo_a_weight:     complete checkpoint [8192,4096] E4M3 tensor.
 * wo_a_scale:      complete checkpoint [64,32] E8M0 scale grid.
 * wo_b_weight:     checkpoint rows [out_rows,8192] E4M3, normally all 4096.
 * wo_b_scale:      scale-grid rows [ceil(out_rows/128),64].
 *
 * Unlike calling the one-group seam eight times, this constructs the complete
 * [8192] BF16 latent first and quantizes that vector once for wo_b, matching the
 * source operation exactly. group_lora_out is optional [8,1024].
 */
int waste_ds_v4_attention_output_full_ref(
    const float *attention_heads,
    const uint8_t *wo_a_weight,
    const uint8_t *wo_a_scale_e8m0,
    const uint8_t *wo_b_weight,
    const uint8_t *wo_b_scale_e8m0,
    size_t out_rows,
    float *group_lora_out,
    float *out);

#endif /* WASTE_DEEPSEEK_V4_ATTENTION_OUTPUT_REF_H */
