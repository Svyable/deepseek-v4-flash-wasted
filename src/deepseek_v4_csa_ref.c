/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_csa_ref.h"
#include "deepseek_v4_attention_ref.h"
#include "deepseek_v4_compressor_ref.h"
#include "quant/deepseek_v4_linear_ref.h"
#include "quant/fp4_e2m1.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DS_HIDDEN 4096u
#define DS_RATIO4 4u
#define DS_ROPE 64u
#define DS_INDEX_HEADS 64u
#define DS_INDEX_DIM 128u
#define DS_Q_RANK 1024u
#define DS_INDEX_OUT (DS_INDEX_HEADS * DS_INDEX_DIM)
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

static int is_pow2(size_t n)
{
    return n != 0 && (n & (n - 1u)) == 0;
}

int waste_ds_v4_hadamard_bf16_ref(float *x, size_t n)
{
    if (!x || !is_pow2(n))
        return -1;

    for (size_t width = 1; width < n; width <<= 1u) {
        size_t span = width << 1u;
        for (size_t base = 0; base < n; base += span) {
            for (size_t j = 0; j < width; j++) {
                float a = x[base + j];
                float b = x[base + width + j];
                x[base + j] = a + b;
                x[base + width + j] = a - b;
            }
        }
    }

    float scale = 1.0f / sqrtf((float)n);
    for (size_t i = 0; i < n; i++)
        x[i] = bf16(x[i] * scale);
    return 0;
}

static float fp4_scale_ref(const float *x, size_t n)
{
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax)
            amax = a;
    }

    /* Official kernel: max(amax, 6*2^-126), then fast_round_scale(amax,1/6).
     * The lower bound ensures amax/6 is a normal binary32 value. */
    float floorv = ldexpf(6.0f, -126);
    if (amax < floorv)
        amax = floorv;
    float ratio = amax * (1.0f / 6.0f);
    union { float f; uint32_t u; } v = { ratio };
    int exp = (int)((v.u >> 23) & 0xFFu) - 127;
    if (v.u & 0x7FFFFFu)
        exp++;
    return ldexpf(1.0f, exp);
}

static uint8_t e2m1_encode_ref(float x)
{
    static const float pos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int neg = signbit(x) != 0;
    float a = fabsf(x);
    if (a > 6.0f)
        a = 6.0f;

    uint8_t best = 0;
    float best_err = fabsf(a - pos[0]);
    for (uint8_t code = 1; code < 8; code++) {
        float err = fabsf(a - pos[code]);
        if (err < best_err || (err == best_err &&
                               (code & 1u) == 0u && (best & 1u) != 0u)) {
            best = code;
            best_err = err;
        }
    }
    return (uint8_t)(best | (neg ? 0x08u : 0u));
}

int waste_ds_v4_fp4_sim_inplace_ref(float *x, size_t n, size_t block_size)
{
    if (!x || n == 0 || block_size == 0 || n % block_size != 0)
        return -1;

    for (size_t base = 0; base < n; base += block_size) {
        float scale = fp4_scale_ref(x + base, block_size);
        for (size_t j = 0; j < block_size; j++) {
            float z = x[base + j] / scale;
            if (z < -6.0f) z = -6.0f;
            if (z >  6.0f) z =  6.0f;
            uint8_t code = e2m1_encode_ref(z);
            x[base + j] = bf16(waste_e2m1_decode(code) * scale);
        }
    }
    return 0;
}

static int all_zero(const float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (x[i] != 0.0f)
            return 0;
    return 1;
}

static void bf16_linear_f32(const float *x, const uint16_t *w,
                            size_t rows, size_t cols, float *out)
{
    if (all_zero(x, cols)) {
        memset(out, 0, rows * sizeof(*out));
        return;
    }
    for (size_t r = 0; r < rows; r++) {
        const uint16_t *wr = w + r * cols;
        float acc = 0.0f;
        for (size_t k = 0; k < cols; k++)
            acc += x[k] * bf16_from_bits(wr[k]);
        out[r] = acc;
    }
}

static int overlap_pool(const float *kv, const float *score,
                        const float *ape, size_t chunks,
                        size_t d, float *out)
{
    size_t two_d = 2u * d;
    for (size_t c = 0; c < chunks; c++) {
        for (size_t j = 0; j < d; j++) {
            float logits[8];
            float vals[8];
            for (size_t t = 0; t < DS_RATIO4; t++) {
                if (c == 0) {
                    logits[t] = -INFINITY;
                    vals[t] = 0.0f;
                } else {
                    size_t p = ((c - 1u) * DS_RATIO4 + t) * two_d + j;
                    logits[t] = score[p] + ape[t * two_d + j];
                    vals[t] = kv[p];
                }
                size_t p = (c * DS_RATIO4 + t) * two_d + d + j;
                logits[DS_RATIO4 + t] = score[p] + ape[t * two_d + d + j];
                vals[DS_RATIO4 + t] = kv[p];
            }

            float maxv = -INFINITY;
            for (size_t t = 0; t < 8; t++)
                if (logits[t] > maxv) maxv = logits[t];
            float den = 0.0f;
            float num = 0.0f;
            for (size_t t = 0; t < 8; t++) {
                if (!isfinite(logits[t]))
                    continue;
                float e = expf(logits[t] - maxv);
                den += e;
                num += vals[t] * e;
            }
            if (!(den > 0.0f) || !isfinite(den))
                return -1;
            out[c * d + j] = bf16(num / den);
        }
    }
    return 0;
}

int waste_ds_v4_csa_compressor_prefill_ref(
    const float *x, size_t seqlen,
    const uint16_t *wkv_bf16,
    const uint16_t *wgate_bf16,
    const float *ape,
    const float *norm_weight,
    size_t head_dim,
    int rotate,
    float *pooled_bf16_out,
    float *norm_out,
    float *rope_out,
    float *out)
{
    if (!x || !wkv_bf16 || !wgate_bf16 || !ape || !norm_weight || !out ||
        seqlen == 0 || seqlen % DS_RATIO4 != 0 || head_dim < DS_ROPE ||
        (rotate && !is_pow2(head_dim)))
        return -1;

    size_t rows = 2u * head_dim;
    size_t chunks = seqlen / DS_RATIO4;
    float *kv = malloc(seqlen * rows * sizeof(*kv));
    float *score = malloc(seqlen * rows * sizeof(*score));
    float *pooled = malloc(chunks * head_dim * sizeof(*pooled));
    if (!kv || !score || !pooled) {
        free(kv); free(score); free(pooled);
        return -1;
    }

    int rc = -1;
    for (size_t t = 0; t < seqlen; t++) {
        const float *xt = x + t * DS_HIDDEN;
        bf16_linear_f32(xt, wkv_bf16, rows, DS_HIDDEN, kv + t * rows);
        bf16_linear_f32(xt, wgate_bf16, rows, DS_HIDDEN, score + t * rows);
    }
    if (overlap_pool(kv, score, ape, chunks, head_dim, pooled) != 0)
        goto done;

    if (pooled_bf16_out)
        memcpy(pooled_bf16_out, pooled, chunks * head_dim * sizeof(*pooled));

    for (size_t c = 0; c < chunks; c++) {
        float *dst = out + c * head_dim;
        if (waste_ds_v4_rmsnorm_ref(pooled + c * head_dim, norm_weight,
                                    head_dim, 1e-6f, dst) != 0)
            goto done;
        if (norm_out)
            memcpy(norm_out + c * head_dim, dst, head_dim * sizeof(*dst));

        if (waste_ds_v4_compressed_rope_ref(
                dst, head_dim, DS_ROPE, c * DS_RATIO4,
                DS_ORIGINAL_SEQ, DS_COMPRESS_BASE, DS_YARN_FACTOR,
                DS_BETA_FAST, DS_BETA_SLOW) != 0)
            goto done;
        if (rope_out)
            memcpy(rope_out + c * head_dim, dst, head_dim * sizeof(*dst));

        if (rotate) {
            if (waste_ds_v4_hadamard_bf16_ref(dst, head_dim) != 0 ||
                waste_ds_v4_fp4_sim_inplace_ref(dst, head_dim, 32) != 0)
                goto done;
        } else {
            size_t nope = head_dim - DS_ROPE;
            if (waste_ds_v4_fp8_sim_inplace_ref(dst, nope, 64) != 0)
                goto done;
        }
    }

    rc = 0;
done:
    free(kv); free(score); free(pooled);
    return rc;
}

int waste_ds_v4_csa_indexer_query_ref(
    const float *qr,
    const uint8_t *wq_b_fp8,
    const uint8_t *wq_b_scale_e8m0,
    size_t position,
    float *out)
{
    if (!qr || !wq_b_fp8 || !wq_b_scale_e8m0 || !out)
        return -1;
    if (waste_ds_v4_fp8_linear_e8m0_ref(
            qr, 1, DS_Q_RANK, wq_b_fp8, wq_b_scale_e8m0,
            DS_INDEX_OUT, out) != 0)
        return -1;
    for (size_t h = 0; h < DS_INDEX_HEADS; h++) {
        float *q = out + h * DS_INDEX_DIM;
        if (waste_ds_v4_compressed_rope_ref(
                q, DS_INDEX_DIM, DS_ROPE, position,
                DS_ORIGINAL_SEQ, DS_COMPRESS_BASE, DS_YARN_FACTOR,
                DS_BETA_FAST, DS_BETA_SLOW) != 0 ||
            waste_ds_v4_hadamard_bf16_ref(q, DS_INDEX_DIM) != 0 ||
            waste_ds_v4_fp4_sim_inplace_ref(q, DS_INDEX_DIM, 32) != 0)
            return -1;
    }
    return 0;
}

int waste_ds_v4_csa_indexer_weights_ref(
    const float *x,
    const uint16_t *weights_bf16,
    float *out64)
{
    if (!x || !weights_bf16 || !out64)
        return -1;
    bf16_linear_f32(x, weights_bf16, DS_INDEX_HEADS, DS_HIDDEN, out64);
    float scale = 1.0f / sqrtf((float)(DS_INDEX_DIM * DS_INDEX_HEADS));
    for (size_t h = 0; h < DS_INDEX_HEADS; h++)
        out64[h] = bf16(bf16(out64[h]) * scale);
    return 0;
}

int waste_ds_v4_csa_index_scores_ref(
    const float *q,
    const float *compressed_kv,
    size_t n_compressed,
    const float *weights,
    float *scores)
{
    if (!q || !compressed_kv || !weights || !scores || n_compressed == 0)
        return -1;

    for (size_t t = 0; t < n_compressed; t++) {
        const float *kv = compressed_kv + t * DS_INDEX_DIM;
        float acc = 0.0f;
        for (size_t h = 0; h < DS_INDEX_HEADS; h++) {
            const float *qh = q + h * DS_INDEX_DIM;
            float dot = 0.0f;
            for (size_t d = 0; d < DS_INDEX_DIM; d++)
                dot += qh[d] * kv[d];
            dot = bf16(dot);
            if (dot < 0.0f)
                dot = 0.0f;
            float product = bf16(dot * weights[h]);
            acc += product;
        }
        scores[t] = bf16(acc);
    }
    return 0;
}

struct ranked_score {
    float score;
    size_t index;
    int valid;
};

static int rank_better(const struct ranked_score *a,
                       const struct ranked_score *b)
{
    if (a->valid != b->valid)
        return a->valid > b->valid;
    if (a->valid && a->score != b->score)
        return a->score > b->score;
    return a->index < b->index;
}

size_t waste_ds_v4_csa_topk_row_ref(
    const float *scores,
    size_t n_compressed,
    size_t visible,
    size_t offset,
    int32_t *out,
    size_t out_capacity)
{
    if (!scores || !out || n_compressed == 0)
        return 0;
    size_t k = n_compressed < 512u ? n_compressed : 512u;
    if (out_capacity < k)
        return 0;
    if (visible > n_compressed)
        visible = n_compressed;

    struct ranked_score *ranked = malloc(n_compressed * sizeof(*ranked));
    if (!ranked)
        return 0;
    for (size_t i = 0; i < n_compressed; i++) {
        if (!isfinite(scores[i])) {
            free(ranked);
            return 0;
        }
        ranked[i].score = scores[i];
        ranked[i].index = i;
        ranked[i].valid = i < visible;
    }

    /* O(n*k) selection is intentional: top-k reference, not runtime kernel. */
    for (size_t pos = 0; pos < k; pos++) {
        size_t best = pos;
        for (size_t j = pos + 1; j < n_compressed; j++)
            if (rank_better(&ranked[j], &ranked[best]))
                best = j;
        struct ranked_score tmp = ranked[pos];
        ranked[pos] = ranked[best];
        ranked[best] = tmp;
        out[pos] = ranked[pos].valid
                 ? (int32_t)(offset + ranked[pos].index)
                 : -1;
    }
    free(ranked);
    return k;
}
