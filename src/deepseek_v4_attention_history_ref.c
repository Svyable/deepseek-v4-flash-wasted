/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_attention_history_ref.h"
#include "deepseek_v4_attention_ref.h"

#include <stdlib.h>
#include <string.h>

size_t waste_ds_v4_combine_history_indices_ref(
    const int32_t *compressed_idxs, size_t n_compressed_idxs,
    const int32_t *local_idxs, size_t n_local_idxs,
    size_t n_compress_cache,
    int32_t *out, size_t out_capacity)
{
    if ((!compressed_idxs && n_compressed_idxs) ||
        (!local_idxs && n_local_idxs) || !out ||
        out_capacity < n_compressed_idxs + n_local_idxs)
        return 0;

    size_t o = 0;
    for (size_t i = 0; i < n_compressed_idxs; i++) {
        int32_t idx = compressed_idxs[i];
        if (idx < -1 || (idx >= 0 && (size_t)idx >= n_compress_cache))
            return 0;
        out[o++] = idx;
    }
    for (size_t i = 0; i < n_local_idxs; i++) {
        int32_t idx = local_idxs[i];
        if (idx < -1)
            return 0;
        if (idx < 0)
            out[o++] = -1;
        else
            out[o++] = (int32_t)(n_compress_cache + (size_t)idx);
    }
    return o;
}

int waste_ds_v4_compressed_history_head_ref(
    const float *q,
    const float *compressed_kv, size_t n_compress_cache,
    const float *local_kv, size_t n_local_cache,
    const int32_t *compressed_idxs, size_t n_compressed_idxs,
    const int32_t *local_idxs, size_t n_local_idxs,
    size_t d, float attn_sink, float softmax_scale,
    float *out)
{
    if (!q || !out || d == 0 ||
        (!compressed_kv && n_compress_cache) ||
        (!local_kv && n_local_cache) ||
        (!compressed_idxs && n_compressed_idxs) ||
        (!local_idxs && n_local_idxs) ||
        n_compress_cache + n_local_cache == 0 ||
        n_compressed_idxs + n_local_idxs == 0)
        return -1;

    const size_t n_cache = n_compress_cache + n_local_cache;
    const size_t n_idxs = n_compressed_idxs + n_local_idxs;
    float *kv = malloc(n_cache * d * sizeof(float));
    int32_t *idx = malloc(n_idxs * sizeof(int32_t));
    if (!kv || !idx) {
        free(kv);
        free(idx);
        return -1;
    }

    if (n_compress_cache)
        memcpy(kv, compressed_kv, n_compress_cache * d * sizeof(float));
    if (n_local_cache)
        memcpy(kv + n_compress_cache * d,
               local_kv, n_local_cache * d * sizeof(float));

    if (waste_ds_v4_combine_history_indices_ref(
            compressed_idxs, n_compressed_idxs,
            local_idxs, n_local_idxs,
            n_compress_cache, idx, n_idxs) != n_idxs) {
        free(kv);
        free(idx);
        return -1;
    }

    /* Validate the local namespace before offsetting it. */
    for (size_t i = 0; i < n_local_idxs; i++) {
        if (local_idxs[i] >= 0 && (size_t)local_idxs[i] >= n_local_cache) {
            free(kv);
            free(idx);
            return -1;
        }
    }

    int rc = waste_ds_v4_sparse_attn_head_ref(
        q, kv, n_cache, idx, n_idxs, d,
        attn_sink, softmax_scale, out);
    free(kv);
    free(idx);
    return rc;
}
