/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_compressor_ref.h"
#include "deepseek_v4_attention_ref.h"
#include "quant/deepseek_v4_linear_ref.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DS_HIDDEN 4096u
#define DS_KV 512u
#define DS_RATIO 128u
#define DS_ROPE 64u
#define DS_NOPE (DS_KV - DS_ROPE)
#define DS_QAT_BLOCK 64u
#define DS_ORIGINAL_SEQ 65536u
#define DS_COMPRESS_BASE 160000.0f
#define DS_YARN_FACTOR 16.0f
#define DS_BETA_FAST 32.0f
#define DS_BETA_SLOW 1.0f

static float bf16(float x)
{
    return waste_ds_v4_bf16_round_ref(x);
}

static float bf16_from_bits(uint16_t b)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)b << 16;
    return v.f;
}

int waste_ds_v4_compressor_pool_ref(const float *kv,
                                    const float *score,
                                    const float *ape,
                                    size_t tokens, size_t dim,
                                    float *out)
{
    if (!kv || !score || !ape || !out || tokens == 0 || dim == 0)
        return -1;

    for (size_t d = 0; d < dim; d++) {
        float maxv = -INFINITY;
        for (size_t t = 0; t < tokens; t++) {
            float z = score[t * dim + d] + ape[t * dim + d];
            if (!isfinite(z))
                return -1;
            if (z > maxv)
                maxv = z;
        }
        float denom = 0.0f;
        float numer = 0.0f;
        for (size_t t = 0; t < tokens; t++) {
            float z = score[t * dim + d] + ape[t * dim + d];
            float e = expf(z - maxv);
            denom += e;
            numer += kv[t * dim + d] * e;
        }
        if (!(denom > 0.0f) || !isfinite(denom))
            return -1;
        out[d] = numer / denom;
    }
    return 0;
}

static float correction_dim(float rotations, size_t dim, float base,
                            size_t original_seq_len)
{
    const float two_pi = 6.28318530717958647692f;
    return (float)dim * logf((float)original_seq_len /
                             (rotations * two_pi)) /
           (2.0f * logf(base));
}

static float yarn_freq(size_t pair, size_t rope_dim,
                       size_t original_seq_len, float base, float factor,
                       float beta_fast, float beta_slow)
{
    float freq = 1.0f / powf(base, (float)(2u * pair) / (float)rope_dim);
    float lowf = floorf(correction_dim(beta_fast, rope_dim, base,
                                      original_seq_len));
    float highf = ceilf(correction_dim(beta_slow, rope_dim, base,
                                      original_seq_len));
    if (lowf < 0.0f) lowf = 0.0f;
    if (highf > (float)rope_dim - 1.0f) highf = (float)rope_dim - 1.0f;
    /* Source applies linear_ramp_factor(low, high, dim/2), then smooth = 1-ramp.
     * freqs = freqs/factor*(1-smooth) + freqs*smooth. */
    float high = highf;
    float low = lowf;
    if (low == high)
        high += 0.001f;
    float ramp = ((float)pair - low) / (high - low);
    if (ramp < 0.0f) ramp = 0.0f;
    if (ramp > 1.0f) ramp = 1.0f;
    float smooth = 1.0f - ramp;
    return freq / factor * (1.0f - smooth) + freq * smooth;
}

int waste_ds_v4_compressed_rope_ref(float *x, size_t dim, size_t rope_dim,
                                    size_t position,
                                    size_t original_seq_len,
                                    float base, float factor,
                                    float beta_fast, float beta_slow)
{
    if (!x || dim == 0 || rope_dim == 0 || rope_dim > dim ||
        (rope_dim & 1u) != 0 || original_seq_len == 0 ||
        !(base > 0.0f) || !(factor > 0.0f))
        return -1;

    float *tail = x + dim - rope_dim;
    for (size_t pair = 0; pair < rope_dim / 2u; pair++) {
        float freq = yarn_freq(pair, rope_dim, original_seq_len,
                               base, factor, beta_fast, beta_slow);
        float angle = (float)position * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float re = tail[2u * pair];
        float im = tail[2u * pair + 1u];
        tail[2u * pair] = bf16(re * c - im * s);
        tail[2u * pair + 1u] = bf16(re * s + im * c);
    }
    return 0;
}

static int all_zero_bf16(const float *x)
{
    for (size_t i = 0; i < DS_HIDDEN; i++)
        if (x[i] != 0.0f)
            return 0;
    return 1;
}

static void bf16_linear_512(const float *x, const uint16_t *w, float *out)
{
    if (all_zero_bf16(x)) {
        memset(out, 0, DS_KV * sizeof(float));
        return;
    }
    for (size_t r = 0; r < DS_KV; r++) {
        const uint16_t *wr = w + r * DS_HIDDEN;
        float acc = 0.0f;
        for (size_t k = 0; k < DS_HIDDEN; k++)
            acc += x[k] * bf16_from_bits(wr[k]);
        out[r] = acc;
    }
}

int waste_ds_v4_compressor_ratio128_prefill_ref(
    const float *x, size_t seqlen,
    const uint16_t *wkv_bf16,
    const uint16_t *wgate_bf16,
    const float *ape,
    const float *norm_weight,
    float norm_eps,
    float *out)
{
    if (!x || !wkv_bf16 || !wgate_bf16 || !ape || !norm_weight || !out ||
        seqlen == 0 || seqlen % DS_RATIO != 0 ||
        !(norm_eps >= 0.0f) || !isfinite(norm_eps))
        return -1;

    float *kv = malloc(DS_RATIO * DS_KV * sizeof(float));
    float *score = malloc(DS_RATIO * DS_KV * sizeof(float));
    float *pooled = malloc(DS_KV * sizeof(float));
    float *pooled_bf16 = malloc(DS_KV * sizeof(float));
    if (!kv || !score || !pooled || !pooled_bf16) {
        free(kv); free(score); free(pooled); free(pooled_bf16);
        return -1;
    }

    const size_t chunks = seqlen / DS_RATIO;
    int rc = -1;
    for (size_t chunk = 0; chunk < chunks; chunk++) {
        const float *xc = x + chunk * DS_RATIO * DS_HIDDEN;
        for (size_t t = 0; t < DS_RATIO; t++) {
            bf16_linear_512(xc + t * DS_HIDDEN, wkv_bf16,
                            kv + t * DS_KV);
            bf16_linear_512(xc + t * DS_HIDDEN, wgate_bf16,
                            score + t * DS_KV);
        }

        if (waste_ds_v4_compressor_pool_ref(
                kv, score, ape, DS_RATIO, DS_KV, pooled) != 0)
            goto done;

        /* Source explicitly casts pooled f32 to model dtype before RMSNorm. */
        for (size_t d = 0; d < DS_KV; d++)
            pooled_bf16[d] = bf16(pooled[d]);
        float *dst = out + chunk * DS_KV;
        if (waste_ds_v4_rmsnorm_ref(
                pooled_bf16, norm_weight, DS_KV, norm_eps, dst) != 0)
            goto done;

        /* Prefill takes freqs_cis[::ratio], so chunk i uses token position i*128. */
        if (waste_ds_v4_compressed_rope_ref(
                dst, DS_KV, DS_ROPE, chunk * DS_RATIO,
                DS_ORIGINAL_SEQ, DS_COMPRESS_BASE, DS_YARN_FACTOR,
                DS_BETA_FAST, DS_BETA_SLOW) != 0)
            goto done;

        if (waste_ds_v4_fp8_sim_inplace_ref(
                dst, DS_NOPE, DS_QAT_BLOCK) != 0)
            goto done;
    }

    rc = 0;
done:
    free(kv); free(score); free(pooled); free(pooled_bf16);
    return rc;
}
