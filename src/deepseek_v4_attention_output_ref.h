/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar grouped attention-output projection seam for Gate E/V5.
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
 * wo_b_scale_rows: one checkpoint E8M0 scale-grid row [64] when out_rows<=128.
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

#endif /* WASTE_DEEPSEEK_V4_ATTENTION_OUTPUT_REF_H */
