/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Slow scalar reference for the official DeepSeek V4 quantized linear path.
 * This is a correctness seam for Gate C/V2, not a performance kernel.
 */
#ifndef WASTE_DEEPSEEK_V4_LINEAR_REF_H
#define WASTE_DEEPSEEK_V4_LINEAR_REF_H

#include <stddef.h>
#include <stdint.h>

/* Encode one finite value to E4M3FN with round-to-nearest-even semantics.
 * Inputs outside the finite range are clamped to +/-448. NaN returns a NaN
 * code. This deliberately favors an auditable exhaustive search over speed. */
uint8_t waste_ds_v4_e4m3_encode_ref(float x);

/* Official activation scale for one 128-value K block when scale_fmt=ue8m0:
 * max(abs(x), 1e-4) / 448 rounded upward to the next power of two. */
float waste_ds_v4_act_scale_ref(const float *x, size_t n);

/* Round an f32 value to the value represented by BF16, returned as f32. */
float waste_ds_v4_bf16_round_ref(float x);

/* Official-reference-shaped FP8 linear:
 *
 *   x      [m, k] f32 input (official path starts from BF16 activations)
 *   weight [n, k] raw E4M3FN bytes
 *   scales [ceil(n/128), k/128] decoded weight scale values
 *   y      [m, n] f32 container holding BF16-rounded official-style output
 *
 * Activations are quantized independently per row/per 128 K values using
 * the power-of-two UE8M0 scale rule. Each 128-wide dot-product is multiplied
 * by activation_scale * weight_scale, accumulated in f32, then BF16-rounded
 * once at output. k must be a non-zero multiple of 128.
 *
 * Returns 0 on success and -1 for invalid dimensions/pointers.
 */
int waste_ds_v4_fp8_linear_ref(const float *x,
                               size_t m, size_t k,
                               const uint8_t *weight,
                               const float *weight_scales,
                               size_t n,
                               float *y);

#endif /* WASTE_DEEPSEEK_V4_LINEAR_REF_H */
