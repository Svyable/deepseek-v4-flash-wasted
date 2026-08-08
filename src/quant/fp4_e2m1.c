/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * E2M1 / UE8M0 scalar decode. See fp4_e2m1.h for the format and for the
 * two packing assumptions that Gate 0 still has to confirm.
 */

#include "fp4_e2m1.h"

#include <math.h>

/* E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa.
 *
 *   E == 0   subnormal:  (-1)^S * M * 2^-1
 *   E  > 0   normal:     (-1)^S * (1 + M/2) * 2^(E-1)
 *
 * Sixteen codes, no infinities, no NaN, max magnitude 6. Written out
 * rather than computed: the table is the format, and a reader checking
 * this against the OCP spec should not have to run the bit arithmetic in
 * their head. tests/test_quant.c derives the same values from the rule
 * above, so a typo here fails rather than propagates.
 */
static const float E2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

float waste_e2m1_decode(uint8_t code)
{
    return E2M1[code & 0x0F];
}

int waste_ue8m0_is_nan(uint8_t e)
{
    return e == 0xFF;
}

float waste_ue8m0_decode(uint8_t e)
{
    if (e == 0xFF)
        return NAN;
    /* ldexpf rather than a shift into the exponent field: the low end of
     * the range is subnormal in binary32 (2^-127 < 2^-126), and building
     * that by hand is exactly where a hand-rolled version goes wrong. */
    return ldexpf(1.0f, (int)e - 127);
}

int waste_fp4_dims_ok(size_t rows, size_t cols)
{
    if (rows == 0 || cols == 0)
        return 0;
    if (cols % WASTE_UE8M0_BLOCK != 0)
        return 0;
    /* WASTE_UE8M0_BLOCK is even, so an exact scale-block fit implies an
     * exact nibble-pair fit; assert it anyway so the invariant survives
     * someone changing the block size. */
    if (cols % WASTE_FP4_PER_BYTE != 0)
        return 0;
    return 1;
}

size_t waste_fp4_weight_bytes(size_t rows, size_t cols)
{
    return rows * (cols / WASTE_FP4_PER_BYTE);
}

size_t waste_fp4_scale_bytes(size_t rows, size_t cols)
{
    return rows * (cols / WASTE_UE8M0_BLOCK);
}

/* Pull the 4-bit code for logical column c out of a packed row. */
static uint8_t nibble_at(const uint8_t *row, size_t c)
{
    uint8_t byte = row[c / WASTE_FP4_PER_BYTE];
#if WASTE_FP4_LOW_NIBBLE_IS_EVEN
    return (c & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
#else
    return (c & 1u) ? (uint8_t)(byte & 0x0F) : (uint8_t)(byte >> 4);
#endif
}

float waste_fp4_at(const uint8_t *w, const uint8_t *s,
                   size_t cols, size_t r, size_t c)
{
    const uint8_t *row = w + r * (cols / WASTE_FP4_PER_BYTE);
    const uint8_t *srow = s + r * (cols / WASTE_UE8M0_BLOCK);

    return waste_e2m1_decode(nibble_at(row, c))
         * waste_ue8m0_decode(srow[c / WASTE_UE8M0_BLOCK]);
}

void waste_fp4_decode_row(const uint8_t *w, const uint8_t *s,
                          size_t cols, size_t r, float *out)
{
    const uint8_t *row = w + r * (cols / WASTE_FP4_PER_BYTE);
    const uint8_t *srow = s + r * (cols / WASTE_UE8M0_BLOCK);

    /* Hoist the scale across its block. Decoding it once per element
     * would be correct too, but this is the loop shape the SIMD path has
     * to match, and keeping them structurally alike makes a divergence
     * easier to localize. */
    for (size_t blk = 0; blk < cols / WASTE_UE8M0_BLOCK; blk++) {
        float scale = waste_ue8m0_decode(srow[blk]);
        size_t base = blk * WASTE_UE8M0_BLOCK;
        for (size_t i = 0; i < WASTE_UE8M0_BLOCK; i++)
            out[base + i] = waste_e2m1_decode(nibble_at(row, base + i)) * scale;
    }
}

void waste_fp4_matvec(const uint8_t *w, const uint8_t *s,
                      size_t rows, size_t cols,
                      const float *x, float *y)
{
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * (cols / WASTE_FP4_PER_BYTE);
        const uint8_t *srow = s + r * (cols / WASTE_UE8M0_BLOCK);
        double acc = 0.0;

        for (size_t blk = 0; blk < cols / WASTE_UE8M0_BLOCK; blk++) {
            double scale = (double)waste_ue8m0_decode(srow[blk]);
            size_t base = blk * WASTE_UE8M0_BLOCK;
            double part = 0.0;

            /* The scale is constant across the block, so it multiplies
             * the block's dot product once instead of every term. That is
             * not just cheaper: it keeps the per-term products in the
             * same magnitude range, which matters when the exponent is
             * near either end of UE8M0's 2^-127..2^127 span. */
            for (size_t i = 0; i < WASTE_UE8M0_BLOCK; i++)
                part += (double)waste_e2m1_decode(nibble_at(row, base + i))
                      * (double)x[base + i];

            acc += part * scale;
        }
        y[r] = (float)acc;
    }
}
