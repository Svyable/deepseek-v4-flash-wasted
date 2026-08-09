/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar DeepSeek V4 routed-expert reference seam for Gate F/V4 bring-up.
 */
#ifndef WASTE_DEEPSEEK_V4_EXPERT_REF_H
#define WASTE_DEEPSEEK_V4_EXPERT_REF_H

#include <stddef.h>
#include <stdint.h>

/* Execute one routed expert with packed FP4 w1/w3 and a selectable row slice
 * of packed FP4 w2.
 *
 * Source semantics:
 *   gate = w1(x).float()
 *   up   = w3(x).float()
 *   up   = clamp(up, -swiglu_limit, +swiglu_limit)
 *   gate = min(gate, +swiglu_limit)       (no negative lower clamp)
 *   h    = silu(gate) * up
 *   h   *= route_weight                   (routed path only)
 *   h    = BF16(h)                        before w2
 *   out  = w2(h)
 *
 * w1/w3 are logical [intermediate,input_dim], packed [intermediate,input/2]
 * with raw E8M0 scales [intermediate,input/32]. w2_rows is a contiguous
 * output-row slice logical [out_rows,intermediate] with matching scales.
 * Outputs from each quantized linear are BF16-rounded f32 containers.
 */
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
    float *out);

#endif /* WASTE_DEEPSEEK_V4_EXPERT_REF_H */
