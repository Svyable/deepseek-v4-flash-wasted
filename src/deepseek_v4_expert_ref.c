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

static void clamp_swiglu(float *gate, float *up, float *hidden,
                         size_t n, float limit, float weight)
{
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (u > limit)
            u = limit;
        if (u < -limit)
            u = -limit;
        if (g > limit)
            g = limit;
        gate[i] = g;
        up[i] = u;
        hidden[i] = waste_ds_v4_bf16_round_ref(
            siluf_ref(g) * u * weight);
    }
}

static void copy_diagnostics(const float *gate, const float *up,
                             const float *hidden, size_t n,
                             float *gate_out, float *up_out,
                             float *hidden_out)
{
    if (gate_out)
        for (size_t i = 0; i < n; i++) gate_out[i] = gate[i];
    if (up_out)
        for (size_t i = 0; i < n; i++) up_out[i] = up[i];
    if (hidden_out)
        for (size_t i = 0; i < n; i++) hidden_out[i] = hidden[i];
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

    clamp_swiglu(gate, up, hidden, intermediate_dim,
                  swiglu_limit, route_weight);

    if (waste_ds_v4_fp4_linear_ref(
            hidden, 1, intermediate_dim,
            w2_rows, w2_scale_rows, out_rows, out) != 0) {
        free(gate);
        free(up);
        free(hidden);
        return -1;
    }

    copy_diagnostics(gate, up, hidden, intermediate_dim,
                     gate_out, up_out, hidden_bf16_out);
    free(gate);
    free(up);
    free(hidden);
    return 0;
}

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
    float *out)
{
    if (!x || !w1 || !w1_scale_e8m0 || !w3 || !w3_scale_e8m0 ||
        !w2_rows || !w2_scale_rows_e8m0 || !out ||
        input_dim == 0 || intermediate_dim == 0 || out_rows == 0 ||
        input_dim % 128u != 0 || intermediate_dim % 128u != 0 ||
        out_rows > 128u ||
        !(swiglu_limit > 0.0f) || !isfinite(swiglu_limit))
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

    if (waste_ds_v4_fp8_linear_e8m0_ref(
            x, 1, input_dim, w1, w1_scale_e8m0,
            intermediate_dim, gate) != 0 ||
        waste_ds_v4_fp8_linear_e8m0_ref(
            x, 1, input_dim, w3, w3_scale_e8m0,
            intermediate_dim, up) != 0) {
        free(gate);
        free(up);
        free(hidden);
        return -1;
    }

    clamp_swiglu(gate, up, hidden, intermediate_dim,
                  swiglu_limit, 1.0f);

    if (waste_ds_v4_fp8_linear_e8m0_ref(
            hidden, 1, intermediate_dim,
            w2_rows, w2_scale_rows_e8m0,
            out_rows, out) != 0) {
        free(gate);
        free(up);
        free(hidden);
        return -1;
    }

    copy_diagnostics(gate, up, hidden, intermediate_dim,
                     gate_out, up_out, hidden_bf16_out);
    free(gate);
    free(up);
    free(hidden);
    return 0;
}

int waste_ds_v4_moe_combine_ref(const uint32_t *expert_ids,
                                const float *routed_out,
                                size_t n_selected,
                                const float *shared_out,
                                size_t out_dim,
                                float *out)
{
    if (!expert_ids || !routed_out || !shared_out || !out ||
        n_selected == 0 || out_dim == 0)
        return -1;

    size_t *order = malloc(n_selected * sizeof(size_t));
    if (!order)
        return -1;
    for (size_t i = 0; i < n_selected; i++) {
        order[i] = i;
        for (size_t j = i; j > 0; j--) {
            size_t a = order[j - 1];
            size_t b = order[j];
            if (expert_ids[a] < expert_ids[b])
                break;
            if (expert_ids[a] == expert_ids[b]) {
                free(order);
                return -1;
            }
            order[j - 1] = b;
            order[j] = a;
        }
    }

    for (size_t d = 0; d < out_dim; d++) {
        float y = 0.0f;
        for (size_t oi = 0; oi < n_selected; oi++) {
            size_t slot = order[oi];
            y += routed_out[slot * out_dim + d];
        }
        y += shared_out[d];
        out[d] = waste_ds_v4_bf16_round_ref(y);
    }

    free(order);
    return 0;
}
