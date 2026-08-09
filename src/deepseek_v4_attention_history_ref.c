/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_attention_history_ref.h"
#include "deepseek_v4_attention_ref.h"

#include <stdlib.h>
#include <string.h>

size_t waste_ds_v4_compress_indices_ref(size_t ratio, size_t seqlen,
                                        size_t start_pos, size_t offset,
                                        int32_t *out, size_t out_capacity)
{
    if (!out || ratio == 0 || seqlen == 0)
        return 0;

    if (start_pos > 0) {
        size_t cols = (start_pos + 1u) / ratio;
        if (out_capacity < cols)
            return 0;
        for (size_t c = 0; c < cols; c++)
            out[c] = (int32_t)(offset + c);
        return cols;
    }

    size_t cols = seqlen / ratio;
    size_t need = seqlen * cols;
    if (out_capacity < need)
        return 0;
    for (size_t row = 0; row < seqlen; row++) {
        size_t visible = (row + 1u) / ratio;
        for (size_t c = 0; c < cols; c++)
            out[row * cols + c] = c >= visible ? -1 : (int32_t)(offset + c);
    }
    return need;
}

size_t waste_ds_v4_ratio_history_topk_row_ref(size_t window, size_t ratio,
                                              size_t seqlen,
                                              size_t start_pos,
                                              size_t query_row,
                                              int32_t *out,
                                              size_t out_capacity)
{
    if (!out || window == 0 || ratio == 0 || seqlen == 0)
        return 0;

    if (start_pos == 0) {
        if (query_row >= seqlen)
            return 0;
        size_t local_cols = seqlen < window ? seqlen : window;
        size_t comp_cols = seqlen / ratio;
        size_t need = local_cols + comp_cols;
        if (out_capacity < need)
            return 0;

        size_t local_total = seqlen * local_cols;
        size_t comp_total = seqlen * comp_cols;
        int32_t *local = malloc(local_total * sizeof(*local));
        int32_t *comp = comp_total ? malloc(comp_total * sizeof(*comp)) : NULL;
        if (!local || (comp_total && !comp)) {
            free(local);
            free(comp);
            return 0;
        }
        if (waste_ds_v4_window_indices_ref(window, seqlen, 0,
                                           local, local_total) != local_total) {
            free(local);
            free(comp);
            return 0;
        }
        if (comp_total &&
            waste_ds_v4_compress_indices_ref(ratio, seqlen, 0, seqlen,
                                             comp, comp_total) != comp_total) {
            free(local);
            free(comp);
            return 0;
        }
        memcpy(out, local + query_row * local_cols,
               local_cols * sizeof(*out));
        if (comp_cols)
            memcpy(out + local_cols, comp + query_row * comp_cols,
                   comp_cols * sizeof(*out));
        free(local);
        free(comp);
        return need;
    }

    if (query_row != 0)
        return 0;
    size_t comp_cols = (start_pos + 1u) / ratio;
    size_t need = window + comp_cols;
    if (out_capacity < need)
        return 0;
    if (waste_ds_v4_window_indices_ref(window, seqlen, start_pos,
                                       out, window) != window)
        return 0;
    if (comp_cols &&
        waste_ds_v4_compress_indices_ref(ratio, seqlen, start_pos, window,
                                         out + window, comp_cols) != comp_cols)
        return 0;
    return need;
}

int waste_ds_v4_ratio_history_sparse_attn_head_ref(
    const float *q,
    const float *local_kv, size_t n_local,
    const float *compressed_kv, size_t n_compressed,
    size_t window, size_t ratio,
    size_t seqlen, size_t start_pos, size_t query_row,
    size_t d, float attn_sink, float softmax_scale,
    float *out)
{
    if (!q || !local_kv || !compressed_kv || !out ||
        window == 0 || ratio == 0 || seqlen == 0 || d == 0)
        return -1;

    size_t need_comp;
    if (start_pos == 0) {
        if (n_local != seqlen)
            return -1;
        need_comp = seqlen / ratio;
    } else {
        if (n_local != window || query_row != 0)
            return -1;
        need_comp = (start_pos + 1u) / ratio;
    }
    if (n_compressed < need_comp)
        return -1;

    size_t n_kv = n_local + n_compressed;
    if (n_kv > SIZE_MAX / d)
        return -1;
    float *kv = malloc(n_kv * d * sizeof(*kv));
    if (!kv)
        return -1;
    memcpy(kv, local_kv, n_local * d * sizeof(*kv));
    memcpy(kv + n_local * d, compressed_kv,
           n_compressed * d * sizeof(*kv));

    size_t topk_cap = (start_pos == 0 ? (seqlen < window ? seqlen : window) : window)
                    + (start_pos == 0 ? seqlen / ratio : (start_pos + 1u) / ratio);
    int32_t *idxs = malloc(topk_cap * sizeof(*idxs));
    if (!idxs) {
        free(kv);
        return -1;
    }
    size_t topk = waste_ds_v4_ratio_history_topk_row_ref(
        window, ratio, seqlen, start_pos, query_row, idxs, topk_cap);
    if (topk == 0) {
        free(idxs);
        free(kv);
        return -1;
    }

    int rc = waste_ds_v4_sparse_attn_head_ref(
        q, kv, n_kv, idxs, topk, d, attn_sink, softmax_scale, out);
    free(idxs);
    free(kv);
    return rc;
}
