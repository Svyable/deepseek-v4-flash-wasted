/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_head_ref.h"

#include <math.h>
#include <string.h>

static uint32_t f32_bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

static float bits_f32(uint32_t u)
{
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

uint16_t waste_ds_v4_head_f32_to_bf16(float value)
{
    uint32_t u = f32_bits(value);
    uint32_t exp = u & 0x7f800000u;
    uint32_t frac = u & 0x007fffffu;
    if (exp == 0x7f800000u) {
        if (frac) u |= 0x00400000u;
        return (uint16_t)(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

float waste_ds_v4_head_bf16_to_f32(uint16_t value)
{
    return bits_f32((uint32_t)value << 16);
}

static float sigmoidf_ref(float x)
{
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

int waste_ds_v4_hc_head_ref(const float *x,
                            size_t dim,
                            const float *fn,
                            const float *scale,
                            const float *base,
                            float norm_eps,
                            float hc_eps,
                            float *out,
                            float mixes[WASTE_DS_V4_HC_HEAD_STREAMS],
                            float pre[WASTE_DS_V4_HC_HEAD_STREAMS])
{
    if (!x || !fn || !scale || !base || !out || dim == 0 ||
        !(norm_eps > 0.0f) || !(hc_eps > 0.0f))
        return -1;

    const size_t flat = WASTE_DS_V4_HC_HEAD_STREAMS * dim;
    float total = 0.0f;
    for (size_t c = 0; c < flat; c++) {
        float square = x[c] * x[c];
        total += square;
    }
    float mean = total / (float)flat;
    float rsqrt = 1.0f / sqrtf(mean + norm_eps);

    float local_mixes[WASTE_DS_V4_HC_HEAD_STREAMS];
    float local_pre[WASTE_DS_V4_HC_HEAD_STREAMS];
    for (size_t row = 0; row < WASTE_DS_V4_HC_HEAD_STREAMS; row++) {
        float acc = 0.0f;
        const float *fr = fn + row * flat;
        for (size_t c = 0; c < flat; c++) {
            float product = fr[c] * x[c];
            acc += product;
        }
        local_mixes[row] = acc * rsqrt;
        float affine = local_mixes[row] * scale[0] + base[row];
        local_pre[row] = sigmoidf_ref(affine) + hc_eps;
    }

    for (size_t d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (size_t stream = 0; stream < WASTE_DS_V4_HC_HEAD_STREAMS; stream++) {
            float product = local_pre[stream] * x[stream * dim + d];
            acc += product;
        }
        out[d] = acc;
    }

    if (mixes)
        for (size_t i = 0; i < WASTE_DS_V4_HC_HEAD_STREAMS; i++)
            mixes[i] = local_mixes[i];
    if (pre)
        for (size_t i = 0; i < WASTE_DS_V4_HC_HEAD_STREAMS; i++)
            pre[i] = local_pre[i];
    return 0;
}

int waste_ds_v4_final_rmsnorm_ref(const float *x,
                                  const float *weight,
                                  size_t dim,
                                  float eps,
                                  float *out)
{
    if (!x || !weight || !out || dim == 0 || !(eps > 0.0f))
        return -1;

    float total = 0.0f;
    for (size_t d = 0; d < dim; d++) {
        float square = x[d] * x[d];
        total += square;
    }
    float mean = total / (float)dim;
    float rsqrt = 1.0f / sqrtf(mean + eps);
    for (size_t d = 0; d < dim; d++) {
        float normalized = x[d] * rsqrt;
        out[d] = weight[d] * normalized;
    }
    return 0;
}

int waste_ds_v4_head_row_ref(const float *x,
                             const float *weight,
                             size_t dim,
                             float *logit)
{
    if (!x || !weight || !logit || dim == 0)
        return -1;
    float acc = 0.0f;
    for (size_t d = 0; d < dim; d++) {
        float product = x[d] * weight[d];
        acc += product;
    }
    *logit = acc;
    return 0;
}
