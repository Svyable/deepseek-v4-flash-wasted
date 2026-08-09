/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_attention_ref.h"
#include "quant/deepseek_v4_linear_ref.h"
#include "quant/fp8_e4m3.h"

#include <math.h>
#include <stdlib.h>

#define DS_V4_SPARSE_BLOCK 64u

static float bf16(float x)
{
    return waste_ds_v4_bf16_round_ref(x);
}

int waste_ds_v4_rmsnorm_ref(const float *x, const float *weight,
                            size_t n, float eps, float *out)
{
    if (!x || !out || n == 0 || !(eps >= 0.0f) || !isfinite(eps))
        return -1;

    float mean_sq = 0.0f;
    for (size_t i = 0; i < n; i++)
        mean_sq += x[i] * x[i];
    mean_sq /= (float)n;
    float scale = 1.0f / sqrtf(mean_sq + eps);

    for (size_t i = 0; i < n; i++) {
        float y = x[i] * scale;
        if (weight)
            y *= weight[i];
        out[i] = bf16(y);
    }
    return 0;
}

int waste_ds_v4_rope_ref(float *x, size_t dim, size_t position,
                         float theta, int inverse)
{
    if (!x || dim == 0 || (dim & 1u) != 0 || !(theta > 0.0f) || !isfinite(theta))
        return -1;

    for (size_t pair = 0; pair < dim / 2u; pair++) {
        float exponent = (float)(2u * pair) / (float)dim;
        float freq = 1.0f / powf(theta, exponent);
        float angle = (float)position * freq;
        if (inverse)
            angle = -angle;
        float c = cosf(angle);
        float s = sinf(angle);
        float re = x[2u * pair];
        float im = x[2u * pair + 1u];
        float y0 = re * c - im * s;
        float y1 = re * s + im * c;
        x[2u * pair] = bf16(y0);
        x[2u * pair + 1u] = bf16(y1);
    }
    return 0;
}

int waste_ds_v4_fp8_sim_inplace_ref(float *x, size_t n, size_t block_size)
{
    if (!x || n == 0 || block_size == 0 || n % block_size != 0)
        return -1;

    for (size_t base = 0; base < n; base += block_size) {
        float scale = waste_ds_v4_act_scale_ref(x + base, block_size);
        if (!(scale > 0.0f) || !isfinite(scale))
            return -1;
        for (size_t j = 0; j < block_size; j++) {
            float z = x[base + j] / scale;
            if (z > WASTE_E4M3_MAX)
                z = WASTE_E4M3_MAX;
            if (z < -WASTE_E4M3_MAX)
                z = -WASTE_E4M3_MAX;
            uint8_t q = waste_ds_v4_e4m3_encode_ref(z);
            x[base + j] = bf16(waste_e4m3_decode(q) * scale);
        }
    }
    return 0;
}

size_t waste_ds_v4_window_indices_ref(size_t window, size_t seqlen,
                                      size_t start_pos,
                                      int32_t *out, size_t out_capacity)
{
    if (!out || window == 0 || seqlen == 0)
        return 0;

    if (start_pos == 0) {
        size_t cols = seqlen < window ? seqlen : window;
        size_t need = seqlen * cols;
        if (out_capacity < need)
            return 0;
        for (size_t row = 0; row < seqlen; row++) {
            size_t first = row >= window - 1u ? row - window + 1u : 0u;
            for (size_t col = 0; col < cols; col++) {
                size_t idx = first + col;
                out[row * cols + col] = idx > row ? -1 : (int32_t)idx;
            }
        }
        return need;
    }

    if (out_capacity < window)
        return 0;
    if (start_pos >= window - 1u) {
        size_t pos = start_pos % window;
        size_t o = 0;
        for (size_t i = pos + 1u; i < window; i++)
            out[o++] = (int32_t)i;
        for (size_t i = 0; i <= pos; i++)
            out[o++] = (int32_t)i;
    } else {
        size_t o = 0;
        for (; o <= start_pos; o++)
            out[o] = (int32_t)o;
        for (; o < window; o++)
            out[o] = -1;
    }
    return window;
}

int waste_ds_v4_sparse_attn_head_ref(const float *q,
                                     const float *kv, size_t n_kv,
                                     const int32_t *idxs, size_t topk,
                                     size_t d, float attn_sink,
                                     float softmax_scale,
                                     float *out)
{
    if (!q || !kv || !idxs || !out || n_kv == 0 || topk == 0 || d == 0 ||
        !isfinite(attn_sink) || !isfinite(softmax_scale))
        return -1;

    float *acc = calloc(d, sizeof(float));
    if (!acc)
        return -1;

    float scores_max = -INFINITY;
    float sum_exp = 0.0f;

    for (size_t base = 0; base < topk; base += DS_V4_SPARSE_BLOCK) {
        size_t count = topk - base;
        if (count > DS_V4_SPARSE_BLOCK)
            count = DS_V4_SPARSE_BLOCK;
        float scores[DS_V4_SPARSE_BLOCK];
        float next_max = scores_max;

        for (size_t j = 0; j < count; j++) {
            int32_t idx = idxs[base + j];
            if (idx < 0) {
                scores[j] = -INFINITY;
                continue;
            }
            if ((size_t)idx >= n_kv) {
                free(acc);
                return -1;
            }
            const float *v = kv + (size_t)idx * d;
            float dot = 0.0f;
            for (size_t k = 0; k < d; k++)
                dot += q[k] * v[k];
            scores[j] = dot * softmax_scale;
            if (scores[j] > next_max)
                next_max = scores[j];
        }

        float rescale = isfinite(scores_max) ? expf(scores_max - next_max) : 0.0f;
        sum_exp *= rescale;
        for (size_t k = 0; k < d; k++)
            acc[k] *= rescale;

        for (size_t j = 0; j < count; j++) {
            if (!isfinite(scores[j]))
                continue;
            float e = expf(scores[j] - next_max);
            sum_exp += e;
            float e_bf16 = bf16(e);
            const float *v = kv + (size_t)idxs[base + j] * d;
            for (size_t k = 0; k < d; k++)
                acc[k] += e_bf16 * v[k];
        }
        scores_max = next_max;
    }

    if (!isfinite(scores_max)) {
        free(acc);
        return -1;
    }
    sum_exp += expf(attn_sink - scores_max);
    if (!(sum_exp > 0.0f) || !isfinite(sum_exp)) {
        free(acc);
        return -1;
    }
    for (size_t k = 0; k < d; k++)
        out[k] = bf16(acc[k] / sum_exp);

    free(acc);
    return 0;
}
