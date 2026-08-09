/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_attention_ref.h"
#include "quant/deepseek_v4_linear_ref.h"
#include "quant/fp8_e4m3.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DS_V4_SPARSE_BLOCK 64u
#define DS_V4_HIDDEN 4096u
#define DS_V4_Q_RANK 1024u
#define DS_V4_KV_DIM 512u
#define DS_V4_HEAD_DIM 512u
#define DS_V4_ROPE_DIM 64u
#define DS_V4_KV_QAT_DIM (DS_V4_KV_DIM - DS_V4_ROPE_DIM)
#define DS_V4_KV_QAT_BLOCK 64u
#define DS_V4_WINDOW 128u

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
        x[2u * pair] = bf16(re * c - im * s);
        x[2u * pair + 1u] = bf16(re * s + im * c);
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

static int copy_or_discard(float *dst, const float *src, size_t n)
{
    if (dst)
        memcpy(dst, src, n * sizeof(float));
    return 0;
}

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
    float *attn_after_inverse_rope)
{
    if (!x || seqlen == 0 || n_heads == 0 || n_heads > 64u ||
        !wq_a_weight || !wq_a_scale_e8m0 || !q_norm_weight ||
        !wq_b_weight || !wq_b_scale_e8m0 ||
        !wkv_weight || !wkv_scale_e8m0 || !kv_norm_weight || !attn_sink ||
        !(rms_eps >= 0.0f) || !isfinite(rms_eps) ||
        !(rope_theta > 0.0f) || !isfinite(rope_theta))
        return -1;

    const size_t q_elems = seqlen * n_heads * DS_V4_HEAD_DIM;
    const size_t kv_elems = seqlen * DS_V4_KV_DIM;
    const size_t window_cols = seqlen < DS_V4_WINDOW ? seqlen : DS_V4_WINDOW;
    const size_t index_elems = seqlen * window_cols;

    float *q_rank = malloc(seqlen * DS_V4_Q_RANK * sizeof(float));
    float *q_rank_norm = malloc(seqlen * DS_V4_Q_RANK * sizeof(float));
    float *q = malloc(q_elems * sizeof(float));
    float *kv = malloc(kv_elems * sizeof(float));
    float *kv_norm = malloc(kv_elems * sizeof(float));
    float *attn = malloc(q_elems * sizeof(float));
    int32_t *idxs = malloc(index_elems * sizeof(int32_t));
    if (!q_rank || !q_rank_norm || !q || !kv || !kv_norm || !attn || !idxs) {
        free(q_rank); free(q_rank_norm); free(q); free(kv); free(kv_norm); free(attn); free(idxs);
        return -1;
    }

    int rc = -1;
    if (waste_ds_v4_fp8_linear_e8m0_ref(
            x, seqlen, DS_V4_HIDDEN,
            wq_a_weight, wq_a_scale_e8m0, DS_V4_Q_RANK, q_rank) != 0)
        goto done;
    for (size_t t = 0; t < seqlen; t++) {
        if (waste_ds_v4_rmsnorm_ref(
                q_rank + t * DS_V4_Q_RANK, q_norm_weight,
                DS_V4_Q_RANK, rms_eps,
                q_rank_norm + t * DS_V4_Q_RANK) != 0)
            goto done;
    }
    if (waste_ds_v4_fp8_linear_e8m0_ref(
            q_rank_norm, seqlen, DS_V4_Q_RANK,
            wq_b_weight, wq_b_scale_e8m0,
            n_heads * DS_V4_HEAD_DIM, q) != 0)
        goto done;

    for (size_t t = 0; t < seqlen; t++) {
        for (size_t h = 0; h < n_heads; h++) {
            float *qh = q + (t * n_heads + h) * DS_V4_HEAD_DIM;
            float tmp[DS_V4_HEAD_DIM];
            if (waste_ds_v4_rmsnorm_ref(
                    qh, NULL, DS_V4_HEAD_DIM, rms_eps, tmp) != 0)
                goto done;
            memcpy(qh, tmp, sizeof tmp);
            if (waste_ds_v4_rope_ref(
                    qh + DS_V4_HEAD_DIM - DS_V4_ROPE_DIM,
                    DS_V4_ROPE_DIM, t, rope_theta, 0) != 0)
                goto done;
        }
    }

    if (waste_ds_v4_fp8_linear_e8m0_ref(
            x, seqlen, DS_V4_HIDDEN,
            wkv_weight, wkv_scale_e8m0, DS_V4_KV_DIM, kv) != 0)
        goto done;
    for (size_t t = 0; t < seqlen; t++) {
        float *kr = kv_norm + t * DS_V4_KV_DIM;
        if (waste_ds_v4_rmsnorm_ref(
                kv + t * DS_V4_KV_DIM, kv_norm_weight,
                DS_V4_KV_DIM, rms_eps, kr) != 0)
            goto done;
        if (waste_ds_v4_rope_ref(
                kr + DS_V4_KV_DIM - DS_V4_ROPE_DIM,
                DS_V4_ROPE_DIM, t, rope_theta, 0) != 0)
            goto done;
        if (waste_ds_v4_fp8_sim_inplace_ref(
                kr, DS_V4_KV_QAT_DIM, DS_V4_KV_QAT_BLOCK) != 0)
            goto done;
    }

    if (waste_ds_v4_window_indices_ref(
            DS_V4_WINDOW, seqlen, 0, idxs, index_elems) != index_elems)
        goto done;

    const float softmax_scale = 1.0f / sqrtf((float)DS_V4_HEAD_DIM);
    for (size_t t = 0; t < seqlen; t++) {
        const int32_t *row_idxs = idxs + t * window_cols;
        for (size_t h = 0; h < n_heads; h++) {
            float *a = attn + (t * n_heads + h) * DS_V4_HEAD_DIM;
            const float *qh = q + (t * n_heads + h) * DS_V4_HEAD_DIM;
            if (waste_ds_v4_sparse_attn_head_ref(
                    qh, kv_norm, seqlen, row_idxs, window_cols,
                    DS_V4_HEAD_DIM, attn_sink[h], softmax_scale, a) != 0)
                goto done;
            if (waste_ds_v4_rope_ref(
                    a + DS_V4_HEAD_DIM - DS_V4_ROPE_DIM,
                    DS_V4_ROPE_DIM, t, rope_theta, 1) != 0)
                goto done;
        }
    }

    copy_or_discard(q_after_rope, q, q_elems);
    copy_or_discard(kv_after_qat, kv_norm, kv_elems);
    copy_or_discard(attn_after_inverse_rope, attn, q_elems);
    rc = 0;

done:
    free(q_rank); free(q_rank_norm); free(q); free(kv); free(kv_norm); free(attn); free(idxs);
    return rc;
}
