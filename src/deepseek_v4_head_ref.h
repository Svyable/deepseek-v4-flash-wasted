/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#ifndef WASTE_DEEPSEEK_V4_HEAD_REF_H
#define WASTE_DEEPSEEK_V4_HEAD_REF_H

#include <stddef.h>

#define WASTE_DS_V4_HC_HEAD_STREAMS 4u

/* Scalar source-derived reference for the final 4-stream HyperConnection
 * collapse. Inputs are float32 values representing the checkpoint path's BF16
 * hidden state and F32 hc_head parameters. The caller owns the official BF16
 * cast boundary after this function. Optional mixes/pre expose the four F32
 * coefficients for differential tests. */
int waste_ds_v4_hc_head_ref(const float *x,
                            size_t dim,
                            const float *fn,
                            const float *scale,
                            const float *base,
                            float norm_eps,
                            float hc_eps,
                            float *out,
                            float mixes[WASTE_DS_V4_HC_HEAD_STREAMS],
                            float pre[WASTE_DS_V4_HC_HEAD_STREAMS]);

/* Final RMSNorm scalar reference. x and weight are float32 representations of
 * BF16 checkpoint values. The official path casts the result back to x.dtype
 * (BF16 here), so the caller owns that cast boundary. */
int waste_ds_v4_final_rmsnorm_ref(const float *x,
                                  const float *weight,
                                  size_t dim,
                                  float eps,
                                  float *out);

/* One vocabulary-head row. The official 0731 ParallelHead stores the
 * parameter in float32 after loading BF16 checkpoint values and evaluates
 * F.linear(x.float(), weight), so both operands here are F32. */
int waste_ds_v4_head_row_ref(const float *x,
                             const float *weight,
                             size_t dim,
                             float *logit);

#endif
