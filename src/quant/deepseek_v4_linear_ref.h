/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Slow scalar references for the official DeepSeek V4 quantized linear paths.
 * Correctness/debugging seams only; not performance kernels.
 */
#ifndef WASTE_DEEPSEEK_V4_LINEAR_REF_H
#define WASTE_DEEPSEEK_V4_LINEAR_REF_H

#include <stddef.h>
#include <stdint.h>

uint8_t waste_ds_v4_e4m3_encode_ref(float x);

/* Official activation scale for one 128-value K block when scale_fmt=ue8m0:
 * max(abs(x), 1e-4) / 448 rounded upward to the next power of two. */
float waste_ds_v4_act_scale_ref(const float *x, size_t n);

/* Round an f32 value to the value represented by BF16, returned as f32. */
float waste_ds_v4_bf16_round_ref(float x);

/* Resident FP8 linear:
 *   activation: E4M3FN, one power-of-two scale / K128
 *   weight:     E4M3FN bytes, one decoded weight scale / 128x128 tile
 *   output:     FP32 accumulation -> BF16-rounded f32 container.
 */
int waste_ds_v4_fp8_linear_ref(const float *x,
                               size_t m, size_t k,
                               const uint8_t *weight,
                               const float *weight_scales,
                               size_t n,
                               float *y);

/* Routed-expert FP4 linear matching the pinned fp4_gemm shape:
 *   activation: E4M3FN, one power-of-two scale / K128
 *   weight:     packed E2M1, [n,k/2], low nibble then high nibble
 *   weight scale: raw E8M0 byte, one / K32 for every output row
 *   accumulation: FP32 over K32 sub-blocks, scaled by act(K128)*weight(K32)
 *   output: BF16-rounded f32 container.
 *
 * k must be a non-zero multiple of 128 (therefore also 32).
 */
int waste_ds_v4_fp4_linear_ref(const float *x,
                               size_t m, size_t k,
                               const uint8_t *packed_weight,
                               const uint8_t *weight_scales_e8m0,
                               size_t n,
                               float *y);

#endif /* WASTE_DEEPSEEK_V4_LINEAR_REF_H */
