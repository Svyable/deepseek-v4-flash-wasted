/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_expert_ref.h"
#include "quant/deepseek_v4_linear_ref.h"

#include <math.h>
#include <stdlib.h>

static float siluf_ref(float x)
{
    if (x >= 0.0f) {
        float z = expf(-x);
        return x / (1.0f + z);
    }
    float z = expf(x);
    return x * z / (1.0f + z);
}

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
    float *out)
{
    if (!x || !w1 || !w1_scale || !w3 || !w3_scale ||
        !w2_rows || !w2_scale_rows || !out ||
        input_dim == 0 || intermediate_dim == 0 || out_rows == 0 ||
        input_dim % 128u != 0 || intermediate_dim % 128u != 0 ||
        !(swiglu_limit > 0.0f) || !isfinite(swiglu_limit) ||
        !(route_weight >= 0.0f) || !isfinite(route_weight))
        return -1;

    float *gate = malloc(intermediate_dim * sizeof(float));
    float *up = malloc(intermediate_dim * sizeof(float));
    float *hidden = malloc(intermediate_dim * sizeof(float));
    if (!gate || !up || !hidden) {
        free(gate);
        free(up);
        free(hidden);
        return -1;
    }

    if (waste_ds_v4_fp4_linear_ref(
            x, 1, input_dim, w1, w1_scale, intermediate_dim, gate) != 0 ||
        waste_ds_v4_fp4_linear_ref(
            x, 1, input_dim, w3, w3_scale, intermediate_dim, up) != 0) {
        free(gate);
        free(up);
        free(hidden);
        return -1;
    }

    for (size_t i = 0; i < intermediate_dim; i++) {
        float g = gate[i];
        float u = up[i];
        if (u > swiglu_limit)
            u = swiglu_limit;
        if (u < -swiglu_limit)
            u = -swiglu_limit;
        if (g > swiglu_limit)
            g = swiglu_limit;

        gate[i] = g;
        up[i] = u;
        float h = siluf_ref(g) * u;
        h *= route_weight;
        hidden[i] = waste_ds_v4_bf16_round_ref(h);
    }

    if (waste_ds_v4_fp4_linear_ref(
            hidden, 1, intermediate_dim,
            w2_rows, w2_scale_rows, out_rows, out) != 0) {
        free(gate);
        free(up);
        free(hidden);
        return -1;
    }

    if (gate_out)
        for (size_t i = 0; i < intermediate_dim; i++) gate_out[i] = gate[i];
    if (up_out)
        for (size_t i = 0; i < intermediate_dim; i++) up_out[i] = up[i];
    if (hidden_bf16_out)
        for (size_t i = 0; i < intermediate_dim; i++) hidden_bf16_out[i] = hidden[i];

    free(gate);
    free(up);
    free(hidden);
    return 0;
}
