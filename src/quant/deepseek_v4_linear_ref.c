/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_linear_ref.h"
#include "fp4_e2m1.h"
#include "fp8_e4m3.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DS_V4_ACT_BLOCK 128u
#define DS_V4_FP4_WEIGHT_BLOCK 32u

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

    if (exp == 0x7F800000u) {
        if (frac != 0u)
            u |= 0x00400000u;
        return bits_f32(u & 0xFFFF0000u);
    }

    u += 0x7FFFu + ((u >> 16) & 1u);
    return bits_f32(u & 0xFFFF0000u);
}

static uint8_t quantize_act(const float *block, size_t lane, float scale)
{
    float normalized = block[lane] / scale;
    if (normalized > WASTE_E4M3_MAX)
        normalized = WASTE_E4M3_MAX;
    if (normalized < -WASTE_E4M3_MAX)
        normalized = -WASTE_E4M3_MAX;
    return waste_ds_v4_e4m3_encode_ref(normalized);
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
    uint8_t *qx = malloc(m * k);
    float *act_scales = malloc(m * k_blocks * sizeof(float));
    if (!qx || !act_scales) {
        free(qx);
        free(act_scales);
        return -1;
    }

    /* Activation quantization is independent of the output row. The original
     * scalar reference recomputed it inside every row; precomputing it once
     * preserves the exact byte/scale values while making attention-sized
     * projection fixtures practical. */
    for (size_t mi = 0; mi < m; mi++) {
        const float *xr = x + mi * k;
        for (size_t kb = 0; kb < k_blocks; kb++) {
            const float *xb = xr + kb * DS_V4_ACT_BLOCK;
            float s = waste_ds_v4_act_scale_ref(xb, DS_V4_ACT_BLOCK);
            if (!(s > 0.0f) || !isfinite(s)) {
                free(qx);
                free(act_scales);
                return -1;
            }
            act_scales[mi * k_blocks + kb] = s;
            for (size_t j = 0; j < DS_V4_ACT_BLOCK; j++)
                qx[mi * k + kb * DS_V4_ACT_BLOCK + j] = quantize_act(xb, j, s);
        }
    }

    for (size_t mi = 0; mi < m; mi++) {
        const uint8_t *qr = qx + mi * k;
        const float *as = act_scales + mi * k_blocks;
        for (size_t ni = 0; ni < n; ni++) {
            float accum = 0.0f;
            const uint8_t *wr = weight + ni * k;
            for (size_t kb = 0; kb < k_blocks; kb++) {
                const uint8_t *qb = qr + kb * DS_V4_ACT_BLOCK;
                const uint8_t *wb = wr + kb * DS_V4_ACT_BLOCK;
                float weight_scale =
                    weight_scales[(ni / WASTE_FP8_BLOCK) * k_blocks + kb];
                float part = 0.0f;
                for (size_t j = 0; j < DS_V4_ACT_BLOCK; j++)
                    part += waste_e4m3_decode(qb[j]) * waste_e4m3_decode(wb[j]);
                accum += part * as[kb] * weight_scale;
            }
            y[mi * n + ni] = waste_ds_v4_bf16_round_ref(accum);
        }
    }

    free(qx);
    free(act_scales);
    return 0;
}

int waste_ds_v4_fp8_linear_e8m0_ref(const float *x,
                                    size_t m, size_t k,
                                    const uint8_t *weight,
                                    const uint8_t *weight_scales_e8m0,
                                    size_t n,
                                    float *y)
{
    if (!x || !weight || !weight_scales_e8m0 || !y ||
        m == 0 || n == 0 || k == 0 || k % DS_V4_ACT_BLOCK != 0)
        return -1;

    const size_t k_blocks = k / DS_V4_ACT_BLOCK;
    const size_t scale_rows = (n + WASTE_FP8_BLOCK - 1u) / WASTE_FP8_BLOCK;
    const size_t count = scale_rows * k_blocks;
    float *decoded = malloc(count * sizeof(float));
    if (!decoded)
        return -1;

    for (size_t i = 0; i < count; i++) {
        decoded[i] = waste_ue8m0_decode(weight_scales_e8m0[i]);
        if (!isfinite(decoded[i])) {
            free(decoded);
            return -1;
        }
    }

    int rc = waste_ds_v4_fp8_linear_ref(x, m, k, weight, decoded, n, y);
    free(decoded);
    return rc;
}

static uint8_t packed_fp4_code(const uint8_t *row, size_t logical_k)
{
    uint8_t byte = row[logical_k / 2u];
    return (logical_k & 1u) ? (uint8_t)(byte >> 4)
                            : (uint8_t)(byte & 0x0Fu);
}

int waste_ds_v4_fp4_linear_ref(const float *x,
                               size_t m, size_t k,
                               const uint8_t *packed_weight,
                               const uint8_t *weight_scales_e8m0,
                               size_t n,
                               float *y)
{
    if (!x || !packed_weight || !weight_scales_e8m0 || !y ||
        m == 0 || n == 0 || k == 0 ||
        k % DS_V4_ACT_BLOCK != 0 || k % DS_V4_FP4_WEIGHT_BLOCK != 0)
        return -1;

    const size_t weight_blocks = k / DS_V4_FP4_WEIGHT_BLOCK;

    for (size_t mi = 0; mi < m; mi++) {
        const float *xr = x + mi * k;
        for (size_t ni = 0; ni < n; ni++) {
            const uint8_t *wr = packed_weight + ni * (k / 2u);
            const uint8_t *sr = weight_scales_e8m0 + ni * weight_blocks;
            float accum = 0.0f;

            for (size_t wb = 0; wb < weight_blocks; wb++) {
                size_t base = wb * DS_V4_FP4_WEIGHT_BLOCK;
                size_t ab = base / DS_V4_ACT_BLOCK;
                const float *act_block = xr + ab * DS_V4_ACT_BLOCK;
                float act_scale = waste_ds_v4_act_scale_ref(
                    act_block, DS_V4_ACT_BLOCK);
                float weight_scale = waste_ue8m0_decode(sr[wb]);
                if (!isfinite(weight_scale))
                    return -1;
                float part = 0.0f;

                for (size_t j = 0; j < DS_V4_FP4_WEIGHT_BLOCK; j++) {
                    size_t logical_k = base + j;
                    size_t act_lane = logical_k - ab * DS_V4_ACT_BLOCK;
                    float a = waste_e4m3_decode(
                        quantize_act(act_block, act_lane, act_scale));
                    float b = waste_e2m1_decode(
                        packed_fp4_code(wr, logical_k));
                    part += a * b;
                }
                accum += part * act_scale * weight_scale;
            }
            y[mi * n + ni] = waste_ds_v4_bf16_round_ref(accum);
        }
    }
    return 0;
}
