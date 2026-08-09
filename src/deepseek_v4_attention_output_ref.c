/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_attention_output_ref.h"
#include "quant/deepseek_v4_linear_ref.h"
#include "quant/fp4_e2m1.h"
#include "quant/fp8_e4m3.h"

#include <math.h>
#include <stdlib.h>

#define GROUP_IN 4096u
#define GROUP_RANK 1024u
#define GROUPS 8u
#define FLAT_RANK (GROUPS * GROUP_RANK)
#define BLOCK 128u

static float bf16(float x)
{
    return waste_ds_v4_bf16_round_ref(x);
}

int waste_ds_v4_attention_output_group_ref(
    const float *group_input,
    size_t group_index,
    const uint8_t *wo_a_weight,
    const uint8_t *wo_a_scale_e8m0,
    const uint8_t *wo_b_rows,
    const uint8_t *wo_b_scale_rows_e8m0,
    size_t out_rows,
    float *group_lora_out,
    float *out)
{
    if (!group_input || !wo_a_weight || !wo_a_scale_e8m0 ||
        !wo_b_rows || !wo_b_scale_rows_e8m0 || !out ||
        group_index >= GROUPS || out_rows == 0 || out_rows > 128u)
        return -1;

    float *lora = malloc(GROUP_RANK * sizeof(float));
    float *flat = calloc(FLAT_RANK, sizeof(float));
    if (!lora || !flat) {
        free(lora);
        free(flat);
        return -1;
    }

    const size_t scale_cols = GROUP_IN / BLOCK;
    for (size_t r = 0; r < GROUP_RANK; r++) {
        float acc = 0.0f;
        const uint8_t *wr = wo_a_weight + r * GROUP_IN;
        const uint8_t *sr = wo_a_scale_e8m0 + (r / BLOCK) * scale_cols;
        for (size_t c = 0; c < GROUP_IN; c++) {
            float s = waste_ue8m0_decode(sr[c / BLOCK]);
            if (!isfinite(s)) {
                free(lora);
                free(flat);
                return -1;
            }
            /* Converter semantics: checkpoint FP8 is block-dequantized then
             * converted to BF16 before the einsum. */
            float w_bf16 = bf16(waste_e4m3_decode(wr[c]) * s);
            acc += bf16(group_input[c]) * w_bf16;
        }
        lora[r] = bf16(acc);
    }

    for (size_t r = 0; r < GROUP_RANK; r++)
        flat[group_index * GROUP_RANK + r] = lora[r];

    int rc = waste_ds_v4_fp8_linear_e8m0_ref(
        flat, 1, FLAT_RANK,
        wo_b_rows, wo_b_scale_rows_e8m0,
        out_rows, out);
    if (rc == 0 && group_lora_out)
        for (size_t r = 0; r < GROUP_RANK; r++)
            group_lora_out[r] = lora[r];

    free(lora);
    free(flat);
    return rc;
}
