/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * E4M3 8-bit float with 128x128 block scales — the format DeepSeek V4
 * Flash uses for most non-expert (trunk) quantized weights (README §7).
 *
 * Scalar reference only, same contract as fp4_e2m1.h.
 *
 * A trunk matrix is [rows, cols] E4M3 bytes plus a scale grid of
 * [ceil(rows/128), ceil(cols/128)]:
 *
 *     value(r, c) = e4m3(w[r][c]) * scale[r/128][c/128]
 *
 * FINITE VARIANT. This implements `e4m3fn` — the variant with no
 * infinities, where the all-ones exponent still carries normal values and
 * only mantissa 0b111 is NaN. That is what torch's `float8_e4m3fn` and the
 * pinned DeepSeek 0731 reference use, and it is why the maximum is 448
 * rather than 240.
 *
 * OFFICIAL 0731 SCALE SEMANTICS — SOURCE VERIFIED
 *
 * Pinned release:
 *   deepseek-ai/DeepSeek-V4-Flash-0731
 *   9e165c30e2704aec5d9d593cce3eebd58bbef1cb
 *
 * inference/convert.py renames checkpoint `weight_scale_inv` tensors to
 * `scale` without taking a reciprocal. inference/kernel.py's FP8 GEMM then
 * multiplies the GEMM partial by the activation scale and weight scale.
 * inference/model.py binds FP8 weights as torch.float8_e4m3fn with one
 * scale per 128x128 block; the release uses UE8M0 scale storage when
 * `scale_dtype == fp8`.
 *
 * The scalar API currently accepts the scale grid as f32 values after scale
 * decode. Gate C/V2 still needs one pinned official quantized projection to
 * validate the complete weight+activation+accumulation path.
 *
 * Unlike FP4, the block grid here is allowed to be ragged: the last row
 * and column block may be partial, which is why the scale stride is
 * computed with a ceiling rather than an exact division.
 */
#ifndef WASTE_FP8_E4M3_H
#define WASTE_FP8_E4M3_H

#include <stddef.h>
#include <stdint.h>

/* Trunk scales cover a 128x128 tile. */
#define WASTE_FP8_BLOCK 128

/* Largest finite magnitude in e4m3fn. */
#define WASTE_E4M3_MAX 448.0f

/* Decode one E4M3 byte. Returns NaN for 0x7F and 0xFF; every other code
 * is finite. There are no infinities in this variant. */
float waste_e4m3_decode(uint8_t b);
int   waste_e4m3_is_nan(uint8_t b);

/* Scale-grid dimensions for a [rows, cols] matrix. */
size_t waste_fp8_scale_rows(size_t rows);
size_t waste_fp8_scale_cols(size_t cols);
size_t waste_fp8_scale_count(size_t rows, size_t cols);

/* One element. `s` is the decoded scale grid in row-major order. */
float waste_fp8_at(const uint8_t *w, const float *s,
                   size_t cols, size_t r, size_t c);

/* One full row, dequantized to f32. `out` holds `cols` floats. */
void waste_fp8_decode_row(const uint8_t *w, const float *s,
                          size_t cols, size_t r, float *out);

/* y[r] = sum_c value(r, c) * x[c], for r in [0, rows). Accumulates in
 * double, for the same reason as the FP4 path. */
void waste_fp8_matvec(const uint8_t *w, const float *s,
                      size_t rows, size_t cols,
                      const float *x, float *y);

#endif /* WASTE_FP8_E4M3_H */
