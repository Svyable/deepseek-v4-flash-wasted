/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar source-contract references for DeepSeek V4 compressed-history
 * attention. Correctness/debugging seams only; not performance kernels.
 */
#ifndef WASTE_DEEPSEEK_V4_ATTENTION_HISTORY_REF_H
#define WASTE_DEEPSEEK_V4_ATTENTION_HISTORY_REF_H

#include <stddef.h>
#include <stdint.h>

/* Official get_compress_topk_idxs() semantics.
 *
 * Prefill (start_pos == 0): returns seqlen * (seqlen/ratio) row-major
 * indices. A compressed entry c is visible to query row r iff
 * c < floor((r+1)/ratio); valid entries are offset+c, others are -1.
 *
 * Decode (start_pos > 0): returns floor((start_pos+1)/ratio) entries,
 * all valid and offset by the caller-provided KV-cache boundary.
 */
size_t waste_ds_v4_compress_indices_ref(size_t ratio, size_t seqlen,
                                        size_t start_pos, size_t offset,
                                        int32_t *out, size_t out_capacity);

/* Build one query row of the exact combined top-k index list used by
 * Attention.forward for non-CSA compressed attention:
 *
 *   [sliding-window indices, compressed-history indices]
 *
 * In prefill, compressed entries are appended after the full current KV
 * tensor (offset=seqlen). In decode, the cache layout is
 * [window-ring, compressed-cache], so offset=window.
 */
size_t waste_ds_v4_ratio_history_topk_row_ref(size_t window, size_t ratio,
                                              size_t seqlen,
                                              size_t start_pos,
                                              size_t query_row,
                                              int32_t *out,
                                              size_t out_capacity);

/* Convenience composition seam for one attention head. local_kv and
 * compressed_kv are concatenated exactly as the source-visible namespace for
 * the selected phase, the corresponding top-k row is generated, then the
 * already-proven sink sparse-attention primitive is applied.
 *
 * Prefill requires n_local == seqlen and n_compressed >= seqlen/ratio.
 * Decode requires n_local == window and n_compressed >=
 * floor((start_pos+1)/ratio).
 */
int waste_ds_v4_ratio_history_sparse_attn_head_ref(
    const float *q,
    const float *local_kv, size_t n_local,
    const float *compressed_kv, size_t n_compressed,
    size_t window, size_t ratio,
    size_t seqlen, size_t start_pos, size_t query_row,
    size_t d, float attn_sink, float softmax_scale,
    float *out);

#endif /* WASTE_DEEPSEEK_V4_ATTENTION_HISTORY_REF_H */
