/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Ratio-128 compressed-history + local-window composition seams for Gate E/V5.
 */
#ifndef WASTE_DEEPSEEK_V4_ATTENTION_HISTORY_REF_H
#define WASTE_DEEPSEEK_V4_ATTENTION_HISTORY_REF_H

#include <stddef.h>
#include <stdint.h>

/* Concatenate the two source index namespaces used by compressed attention.
 * Compressed-cache indices come first unchanged. Local-cache indices follow
 * and are offset by n_compress_cache; -1 padding remains -1.
 * Returns number of entries written, or 0 on invalid args/capacity. */
size_t waste_ds_v4_combine_history_indices_ref(
    const int32_t *compressed_idxs, size_t n_compressed_idxs,
    const int32_t *local_idxs, size_t n_local_idxs,
    size_t n_compress_cache,
    int32_t *out, size_t out_capacity);

/* One-head compressed-history attention composition.
 *
 * combined KV storage is exactly:
 *   [compressed_kv[0:n_compress_cache], local_kv[0:n_local_cache]]
 *
 * combined indices are built by waste_ds_v4_combine_history_indices_ref and
 * passed to the already-proven sink sparse-attention reference.
 * All q/KV values are BF16-valued f32 containers. Output is BF16-valued f32.
 */
int waste_ds_v4_compressed_history_head_ref(
    const float *q,
    const float *compressed_kv, size_t n_compress_cache,
    const float *local_kv, size_t n_local_cache,
    const int32_t *compressed_idxs, size_t n_compressed_idxs,
    const int32_t *local_idxs, size_t n_local_idxs,
    size_t d, float attn_sink, float softmax_scale,
    float *out);

#endif /* WASTE_DEEPSEEK_V4_ATTENTION_HISTORY_REF_H */
