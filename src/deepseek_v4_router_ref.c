/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_router_ref.h"

#include <math.h>
#include <string.h>

static float bits_f32(uint32_t u)
{
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

float waste_ds_v4_bf16_to_f32(uint16_t value)
{
    return bits_f32((uint32_t)value << 16);
}

float waste_ds_v4_sqrt_softplus_ref(float x)
{
    /* PyTorch F.softplus defaults beta=1, threshold=20: above threshold it
     * returns x directly rather than evaluating exp(x). */
    float softplus;
    if (x > 20.0f)
        softplus = x;
    else if (x < 0.0f)
        softplus = log1pf(expf(x));
    else
        softplus = x + log1pf(expf(-x));
    return sqrtf(softplus);
}

int waste_ds_v4_router_score_ref(const float *x,
                                 size_t dim,
                                 const uint16_t *weight_bf16,
                                 float raw[WASTE_DS_V4_ROUTER_EXPERTS],
                                 float score[WASTE_DS_V4_ROUTER_EXPERTS])
{
    if (!x || !weight_bf16 || !raw || !score || dim == 0)
        return -1;

    for (size_t expert = 0; expert < WASTE_DS_V4_ROUTER_EXPERTS; expert++) {
        const uint16_t *row = weight_bf16 + expert * dim;
        float acc = 0.0f;
        for (size_t d = 0; d < dim; d++)
            acc += waste_ds_v4_bf16_to_f32(row[d]) * x[d];
        raw[expert] = acc;
        score[expert] = waste_ds_v4_sqrt_softplus_ref(acc);
    }
    return 0;
}

static int normalize_selected(const float score[WASTE_DS_V4_ROUTER_EXPERTS],
                              const uint32_t ids[WASTE_DS_V4_ROUTER_TOPK],
                              float route_scale,
                              float weights[WASTE_DS_V4_ROUTER_TOPK])
{
    if (!(route_scale > 0.0f) || !isfinite(route_scale))
        return -1;
    float sum = 0.0f;
    for (size_t j = 0; j < WASTE_DS_V4_ROUTER_TOPK; j++) {
        if (ids[j] >= WASTE_DS_V4_ROUTER_EXPERTS)
            return -1;
        weights[j] = score[ids[j]];
        if (!(weights[j] >= 0.0f) || !isfinite(weights[j]))
            return -1;
        sum += weights[j];
    }
    if (!(sum > 0.0f) || !isfinite(sum))
        return -1;
    for (size_t j = 0; j < WASTE_DS_V4_ROUTER_TOPK; j++)
        weights[j] = weights[j] / sum * route_scale;
    return 0;
}

int waste_ds_v4_router_learned_ref(
    const float score[WASTE_DS_V4_ROUTER_EXPERTS],
    const float bias[WASTE_DS_V4_ROUTER_EXPERTS],
    float route_scale,
    uint32_t ids[WASTE_DS_V4_ROUTER_TOPK],
    float weights[WASTE_DS_V4_ROUTER_TOPK],
    float selection_scores[WASTE_DS_V4_ROUTER_EXPERTS])
{
    if (!score || !bias || !ids || !weights || !selection_scores)
        return -1;

    for (size_t i = 0; i < WASTE_DS_V4_ROUTER_EXPERTS; i++)
        selection_scores[i] = score[i] + bias[i];

    /* torch.topk(..., sorted=True) returns descending values. The official
     * fixture is required to have no exact tie at/inside the top-6 boundary;
     * this explicit lower-ID tie rule is therefore only deterministic local
     * behavior, not a claim about PyTorch's unspecified equal-value indices. */
    for (size_t j = 0; j < WASTE_DS_V4_ROUTER_TOPK; j++) {
        uint32_t best = UINT32_MAX;
        float best_value = -INFINITY;
        for (uint32_t i = 0; i < WASTE_DS_V4_ROUTER_EXPERTS; i++) {
            int already = 0;
            for (size_t k = 0; k < j; k++)
                if (ids[k] == i) {
                    already = 1;
                    break;
                }
            if (already)
                continue;
            float value = selection_scores[i];
            if (value > best_value ||
                (value == best_value && (best == UINT32_MAX || i < best))) {
                best_value = value;
                best = i;
            }
        }
        if (best == UINT32_MAX)
            return -1;
        ids[j] = best;
    }
    return normalize_selected(score, ids, route_scale, weights);
}

int waste_ds_v4_router_hash_ref(
    const float score[WASTE_DS_V4_ROUTER_EXPERTS],
    const int64_t tid2eid[WASTE_DS_V4_ROUTER_TOPK],
    float route_scale,
    uint32_t ids[WASTE_DS_V4_ROUTER_TOPK],
    float weights[WASTE_DS_V4_ROUTER_TOPK])
{
    if (!score || !tid2eid || !ids || !weights)
        return -1;
    for (size_t j = 0; j < WASTE_DS_V4_ROUTER_TOPK; j++) {
        if (tid2eid[j] < 0 || tid2eid[j] >= (int64_t)WASTE_DS_V4_ROUTER_EXPERTS)
            return -1;
        ids[j] = (uint32_t)tid2eid[j];
    }
    return normalize_selected(score, ids, route_scale, weights);
}
