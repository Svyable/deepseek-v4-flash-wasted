/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * E4M3 (finite variant) scalar decode with 128x128 block scales.
 * See fp8_e4m3.h for the pinned 0731 format/scale provenance.
 */

#include "fp8_e4m3.h"

#include <math.h>

int waste_e4m3_is_nan(uint8_t b)
{
    /* Only S.1111.111 is NaN. Note this is *not* "exponent is all ones":
     * 0x78 (exponent 15, mantissa 0) is the perfectly ordinary value 256. */
    return (b & 0x7Fu) == 0x7Fu;
}

float waste_e4m3_decode(uint8_t b)
{
    uint32_t sign = (uint32_t)(b >> 7);
    uint32_t exp  = (uint32_t)((b >> 3) & 0x0Fu);
    uint32_t mant = (uint32_t)(b & 0x07u);
    float mag;

    if (waste_e4m3_is_nan(b))
        return NAN;

    if (exp == 0) {
        /* Subnormal: no implicit leading 1, fixed exponent 2^(1-bias).
         * Smallest positive value is 2^-9. */
        mag = ldexpf((float)mant / 8.0f, 1 - 7);
    } else {
        /* Normal: (1 + mant/8) * 2^(exp-7). The top exponent is a normal
         * exponent here, not a NaN/Inf marker -- that is the whole point
         * of the finite variant, and it is what puts the maximum at
         * (1 + 6/8) * 2^8 = 448. */
        mag = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -mag : mag;
}

size_t waste_fp8_scale_rows(size_t rows)
{
    return (rows + WASTE_FP8_BLOCK - 1) / WASTE_FP8_BLOCK;
}

size_t waste_fp8_scale_cols(size_t cols)
{
    return (cols + WASTE_FP8_BLOCK - 1) / WASTE_FP8_BLOCK;
}

size_t waste_fp8_scale_count(size_t rows, size_t cols)
{
    return waste_fp8_scale_rows(rows) * waste_fp8_scale_cols(cols);
}

float waste_fp8_at(const uint8_t *w, const float *s,
                   size_t cols, size_t r, size_t c)
{
    size_t sc = waste_fp8_scale_cols(cols);
    float scale = s[(r / WASTE_FP8_BLOCK) * sc + (c / WASTE_FP8_BLOCK)];

    return waste_e4m3_decode(w[r * cols + c]) * scale;
}

void waste_fp8_decode_row(const uint8_t *w, const float *s,
                          size_t cols, size_t r, float *out)
{
    const uint8_t *row = w + r * cols;
    size_t sc = waste_fp8_scale_cols(cols);
    const float *srow = s + (r / WASTE_FP8_BLOCK) * sc;

    for (size_t c = 0; c < cols; c++)
        out[c] = waste_e4m3_decode(row[c]) * srow[c / WASTE_FP8_BLOCK];
}

void waste_fp8_matvec(const uint8_t *w, const float *s,
                      size_t rows, size_t cols,
                      const float *x, float *y)
{
    size_t sc = waste_fp8_scale_cols(cols);

    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * cols;
        const float *srow = s + (r / WASTE_FP8_BLOCK) * sc;
        double acc = 0.0;
        size_t c = 0;

        /* Walk block by block so the scale is applied once per tile
         * rather than once per element, matching the FP4 path's shape.
         * The final block may be partial -- the grid is ragged whenever
         * cols is not a multiple of 128, which is why this clamps instead
         * of striding blindly. */
        while (c < cols) {
            size_t end = (c / WASTE_FP8_BLOCK + 1) * WASTE_FP8_BLOCK;
            double part = 0.0;

            if (end > cols)
                end = cols;
            for (size_t i = c; i < end; i++)
                part += (double)waste_e4m3_decode(row[i]) * (double)x[i];

            acc += part * (double)srow[c / WASTE_FP8_BLOCK];
            c = end;
        }
        y[r] = (float)acc;
    }
}
