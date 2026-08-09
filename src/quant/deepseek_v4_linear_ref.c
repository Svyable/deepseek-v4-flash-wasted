/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_linear_ref.h"
#include "fp8_e4m3.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define DS_V4_ACT_BLOCK 128u

static uint32_t f32_bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

static float bits_f32(uint32_t u)
{
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

uint8_t waste_ds_v4_e4m3_encode_ref(float x)
{
    if (isnan(x))
        return 0x7Fu;
    if (x == 0.0f)
        return signbit(x) ? 0x80u : 0x00u;
    if (x > WASTE_E4M3_MAX)
        x = WASTE_E4M3_MAX;
    if (x < -WASTE_E4M3_MAX)
        x = -WASTE_E4M3_MAX;

    const int neg = signbit(x) != 0;
    double best_err = DBL_MAX;
    uint8_t best = neg ? 0x80u : 0x00u;

    /* Reference-only exhaustive search. Restrict candidates to x's sign so
     * very small negative values round to -0 rather than +0. On an exact
     * midpoint, E4M3 round-to-nearest-even chooses a code whose significand
     * least-significant bit is zero, which is the encoded low bit here. */
    for (unsigned mag = 0; mag < 0x80u; mag++) {
        uint8_t code = (uint8_t)(mag | (neg ? 0x80u : 0u));
        if ((code & 0x7Fu) == 0x7Fu)
            continue;
        double value = (double)waste_e4m3_decode(code);
        double err = fabs((double)x - value);
        if (err < best_err ||
            (err == best_err && (code & 1u) == 0u && (best & 1u) != 0u)) {
            best_err = err;
            best = code;
        }
    }
    return best;
}

float waste_ds_v4_act_scale_ref(const float *x, size_t n)
{
    if (!x || n == 0)
        return NAN;

    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax)
            amax = a;
    }
    if (amax < 1e-4f)
        amax = 1e-4f;

    /* Pinned 0731 kernel fast_round_scale(amax, 1/448):
     * ceil(log2(amax/448)) synthesized as an exact power of two from the
     * binary32 exponent. The lower clamp keeps the ratio normal. */
    float ratio = amax * (1.0f / WASTE_E4M3_MAX);
    uint32_t bits = f32_bits(ratio);
    int exp = (int)((bits >> 23) & 0xFFu) - 127;
    if ((bits & 0x7FFFFFu) != 0u)
        exp++;
    return ldexpf(1.0f, exp);
}

float waste_ds_v4_bf16_round_ref(float x)
{
    uint32_t u = f32_bits(x);
    uint32_t exp = u & 0x7F800000u;
    uint32_t frac = u & 0x007FFFFFu;

    /* Keep infinities/NaNs non-finite without allowing the rounding add to
     * wrap an exponent. Payload exactness is not part of the model contract. */
    if (exp == 0x7F800000u) {
        if (frac != 0u)
            u |= 0x00400000u;  /* quiet NaN */
        return bits_f32(u & 0xFFFF0000u);
    }

    /* Round-to-nearest-even at the BF16 cut: add 0x7fff plus the retained
     * least-significant bit, then clear the discarded 16 bits. */
    u += 0x7FFFu + ((u >> 16) & 1u);
    return bits_f32(u & 0xFFFF0000u);
}

int waste_ds_v4_fp8_linear_ref(const float *x,
                               size_t m, size_t k,
                               const uint8_t *weight,
                               const float *weight_scales,
                               size_t n,
                               float *y)
{
    if (!x || !weight || !weight_scales || !y ||
        m == 0 || n == 0 || k == 0 || k % DS_V4_ACT_BLOCK != 0)
        return -1;

    const size_t k_blocks = k / DS_V4_ACT_BLOCK;
    const size_t n_scale_rows = (n + WASTE_FP8_BLOCK - 1) / WASTE_FP8_BLOCK;
    (void)n_scale_rows;  /* documents the expected grid; index below uses n/128 */

    for (size_t mi = 0; mi < m; mi++) {
        const float *xr = x + mi * k;
        for (size_t ni = 0; ni < n; ni++) {
            float accum = 0.0f;
            const uint8_t *wr = weight + ni * k;

            for (size_t kb = 0; kb < k_blocks; kb++) {
                const float *xb = xr + kb * DS_V4_ACT_BLOCK;
                const uint8_t *wb = wr + kb * DS_V4_ACT_BLOCK;
                float act_scale = waste_ds_v4_act_scale_ref(xb, DS_V4_ACT_BLOCK);
                float weight_scale =
                    weight_scales[(ni / WASTE_FP8_BLOCK) * k_blocks + kb];
                float part = 0.0f;

                for (size_t j = 0; j < DS_V4_ACT_BLOCK; j++) {
                    float normalized = xb[j] / act_scale;
                    if (normalized > WASTE_E4M3_MAX)
                        normalized = WASTE_E4M3_MAX;
                    if (normalized < -WASTE_E4M3_MAX)
                        normalized = -WASTE_E4M3_MAX;
                    uint8_t qa = waste_ds_v4_e4m3_encode_ref(normalized);
                    float a = waste_e4m3_decode(qa);
                    float b = waste_e4m3_decode(wb[j]);
                    part += a * b;
                }

                accum += part * act_scale * weight_scale;
            }
            y[mi * n + ni] = waste_ds_v4_bf16_round_ref(accum);
        }
    }
    return 0;
}
