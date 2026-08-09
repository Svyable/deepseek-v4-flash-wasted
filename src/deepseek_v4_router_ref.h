/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar DeepSeek V4 routing reference seam for V3 / Gate F bring-up.
 */
#ifndef WASTE_DEEPSEEK_V4_ROUTER_REF_H
#define WASTE_DEEPSEEK_V4_ROUTER_REF_H

#include <stddef.h>
#include <stdint.h>

#define WASTE_DS_V4_ROUTER_EXPERTS 256u
#define WASTE_DS_V4_ROUTER_TOPK 6u
#define WASTE_DS_V4_ROUTER_SCALE 1.5f

float waste_ds_v4_bf16_to_f32(uint16_t value);
float waste_ds_v4_sqrt_softplus_ref(float x);

/* Compute raw and transformed router scores from BF16 weights [256,dim].
 * x is the f32 expansion of the source BF16 hidden state. */
int waste_ds_v4_router_score_ref(const float *x,
                                 size_t dim,
                                 const uint16_t *weight_bf16,
                                 float raw[WASTE_DS_V4_ROUTER_EXPERTS],
                                 float score[WASTE_DS_V4_ROUTER_EXPERTS]);

/* Learned-routing path: correction bias changes selection only. The returned
 * weights are gathered from the unbiased sqrt(softplus) scores, normalized,
 * then multiplied by route_scale. */
int waste_ds_v4_router_learned_ref(
    const float score[WASTE_DS_V4_ROUTER_EXPERTS],
    const float bias[WASTE_DS_V4_ROUTER_EXPERTS],
    float route_scale,
    uint32_t ids[WASTE_DS_V4_ROUTER_TOPK],
    float weights[WASTE_DS_V4_ROUTER_TOPK],
    float selection_scores[WASTE_DS_V4_ROUTER_EXPERTS]);

/* Hash/bootstrap path: IDs are supplied by the checkpoint tid2eid row. The
 * same unbiased transformed router scores determine routing weights. */
int waste_ds_v4_router_hash_ref(
    const float score[WASTE_DS_V4_ROUTER_EXPERTS],
    const int64_t tid2eid[WASTE_DS_V4_ROUTER_TOPK],
    float route_scale,
    uint32_t ids[WASTE_DS_V4_ROUTER_TOPK],
    float weights[WASTE_DS_V4_ROUTER_TOPK]);

#endif /* WASTE_DEEPSEEK_V4_ROUTER_REF_H */
