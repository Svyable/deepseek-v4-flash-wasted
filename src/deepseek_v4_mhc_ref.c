/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_mhc_ref.h"

#include <math.h>

static float sigmoidf_ref(float x)
{
    /* Stable enough for the small learned mHC logits and mirrors sigmoid's
     * mathematical contract without importing a framework dependency. */
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

int waste_ds_v4_hc_split_sinkhorn(const float mixes[WASTE_DS_V4_HC_MIX],
                                  const float scale[3],
                                  const float base[WASTE_DS_V4_HC_MIX],
                                  unsigned sinkhorn_iters,
                                  float eps,
                                  float pre[WASTE_DS_V4_HC],
                                  float post[WASTE_DS_V4_HC],
                                  float comb[WASTE_DS_V4_HC_COMB])
{
    if (!mixes || !scale || !base || !pre || !post || !comb ||
        sinkhorn_iters == 0 || !(eps > 0.0f))
        return -1;

    for (size_t j = 0; j < WASTE_DS_V4_HC; j++) {
        pre[j] = sigmoidf_ref(mixes[j] * scale[0] + base[j]) + eps;
        post[j] = 2.0f * sigmoidf_ref(
            mixes[j + WASTE_DS_V4_HC] * scale[1]
            + base[j + WASTE_DS_V4_HC]);
    }

    /* The official kernel stores comb[j,k] with j as the residual/source HC
     * and k as the output HC. Start with row softmax + eps. */
    for (size_t j = 0; j < WASTE_DS_V4_HC; j++) {
        float row_max = -INFINITY;
        for (size_t k = 0; k < WASTE_DS_V4_HC; k++) {
            size_t p = j * WASTE_DS_V4_HC + k;
            float v = mixes[2 * WASTE_DS_V4_HC + p] * scale[2]
                    + base[2 * WASTE_DS_V4_HC + p];
            comb[p] = v;
            if (v > row_max)
                row_max = v;
        }
        float sum = 0.0f;
        for (size_t k = 0; k < WASTE_DS_V4_HC; k++) {
            size_t p = j * WASTE_DS_V4_HC + k;
            comb[p] = expf(comb[p] - row_max);
            sum += comb[p];
        }
        for (size_t k = 0; k < WASTE_DS_V4_HC; k++) {
            size_t p = j * WASTE_DS_V4_HC + k;
            comb[p] = comb[p] / sum + eps;
        }
    }

    /* First column normalization. */
    for (size_t k = 0; k < WASTE_DS_V4_HC; k++) {
        float sum = 0.0f;
        for (size_t j = 0; j < WASTE_DS_V4_HC; j++)
            sum += comb[j * WASTE_DS_V4_HC + k];
        for (size_t j = 0; j < WASTE_DS_V4_HC; j++)
            comb[j * WASTE_DS_V4_HC + k] /= sum + eps;
    }

    /* Official kernel performs sinkhorn_iters-1 additional row/column pairs. */
    for (unsigned iter = 1; iter < sinkhorn_iters; iter++) {
        for (size_t j = 0; j < WASTE_DS_V4_HC; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < WASTE_DS_V4_HC; k++)
                sum += comb[j * WASTE_DS_V4_HC + k];
            for (size_t k = 0; k < WASTE_DS_V4_HC; k++)
                comb[j * WASTE_DS_V4_HC + k] /= sum + eps;
        }
        for (size_t k = 0; k < WASTE_DS_V4_HC; k++) {
            float sum = 0.0f;
            for (size_t j = 0; j < WASTE_DS_V4_HC; j++)
                sum += comb[j * WASTE_DS_V4_HC + k];
            for (size_t j = 0; j < WASTE_DS_V4_HC; j++)
                comb[j * WASTE_DS_V4_HC + k] /= sum + eps;
        }
    }
    return 0;
}

int waste_ds_v4_hc_pre_ref(const float *x,
                           size_t dim,
                           const float *fn,
                           const float scale[3],
                           const float base[WASTE_DS_V4_HC_MIX],
                           float norm_eps,
                           unsigned sinkhorn_iters,
                           float hc_eps,
                           float *y,
                           float *mixes_out,
                           float *pre_out,
                           float *post_out,
                           float *comb_out)
{
    if (!x || !fn || !scale || !base || !y || dim == 0 ||
        !(norm_eps > 0.0f))
        return -1;

    const size_t flat = WASTE_DS_V4_HC * dim;
    float square_sum = 0.0f;
    for (size_t i = 0; i < flat; i++)
        square_sum += x[i] * x[i];
    float rsqrt = 1.0f / sqrtf(square_sum / (float)flat + norm_eps);

    float mixes[WASTE_DS_V4_HC_MIX];
    for (size_t r = 0; r < WASTE_DS_V4_HC_MIX; r++) {
        float acc = 0.0f;
        const float *row = fn + r * flat;
        for (size_t c = 0; c < flat; c++)
            acc += row[c] * x[c];
        mixes[r] = acc * rsqrt;
    }

    float pre[WASTE_DS_V4_HC];
    float post[WASTE_DS_V4_HC];
    float comb[WASTE_DS_V4_HC_COMB];
    if (waste_ds_v4_hc_split_sinkhorn(
            mixes, scale, base, sinkhorn_iters, hc_eps,
            pre, post, comb) != 0)
        return -1;

    for (size_t d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (size_t j = 0; j < WASTE_DS_V4_HC; j++)
            acc += pre[j] * x[j * dim + d];
        y[d] = acc;
    }

    if (mixes_out)
        for (size_t i = 0; i < WASTE_DS_V4_HC_MIX; i++)
            mixes_out[i] = mixes[i];
    if (pre_out)
        for (size_t i = 0; i < WASTE_DS_V4_HC; i++)
            pre_out[i] = pre[i];
    if (post_out)
        for (size_t i = 0; i < WASTE_DS_V4_HC; i++)
            post_out[i] = post[i];
    if (comb_out)
        for (size_t i = 0; i < WASTE_DS_V4_HC_COMB; i++)
            comb_out[i] = comb[i];
    return 0;
}

int waste_ds_v4_hc_post_ref(const float *branch,
                            const float *residual,
                            size_t dim,
                            const float post[WASTE_DS_V4_HC],
                            const float comb[WASTE_DS_V4_HC_COMB],
                            float *y)
{
    if (!branch || !residual || !post || !comb || !y || dim == 0)
        return -1;

    /* Source expression:
     *   post[k] * branch[d] + sum_j comb[j,k] * residual[j,d]
     * torch.sum(..., dim=2) sums the source-HC dimension. */
    for (size_t k = 0; k < WASTE_DS_V4_HC; k++) {
        for (size_t d = 0; d < dim; d++) {
            float acc = post[k] * branch[d];
            for (size_t j = 0; j < WASTE_DS_V4_HC; j++)
                acc += comb[j * WASTE_DS_V4_HC + k]
                     * residual[j * dim + d];
            y[k * dim + d] = acc;
        }
    }
    return 0;
}
