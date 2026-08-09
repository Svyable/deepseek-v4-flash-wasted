/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar correctness references for DeepSeek V4 ratio-4 CSA/indexer.
 * These are diagnostic/model-semantic seams, not optimized kernels.
 */
#ifndef WASTE_DEEPSEEK_V4_CSA_REF_H
#define WASTE_DEEPSEEK_V4_CSA_REF_H

#include <stddef.h>
#include <stdint.h>

/* Normalized Walsh-Hadamard transform, equivalent to
 * fast_hadamard_transform.hadamard_transform(x, scale=n^-0.5).
 * Input/output are BF16-valued f32; arithmetic is f32 and the final vector is
 * written back through the BF16 boundary. n must be a power of two. */
int waste_ds_v4_hadamard_bf16_ref(float *x, size_t n);

/* Official in-place FP4 activation simulation used by the CSA indexer.
 * Each K32 block chooses a power-of-two scale ceil_pow2(amax/6), quantizes to
 * finite E2M1 with RN-even, dequantizes, and writes BF16 values back in place.
 */
int waste_ds_v4_fp4_sim_inplace_ref(float *x, size_t n, size_t block_size);

/* Source overlap_transform + per-dimension softmax pool, isolated from the
 * projections. `kv` and `score` are [chunks,4,2*head_dim] f32 projection
 * outputs; APE is [4,2*head_dim]. Output is [chunks,head_dim] BF16-valued.
 * For chunk c, slots 0..3 come from chunk c-1's first half (unavailable/-inf
 * for c=0), while slots 4..7 come from chunk c's second half. */
int waste_ds_v4_csa_overlap_pool_ref(
    const float *kv,
    const float *score,
    const float *ape,
    size_t chunks,
    size_t head_dim,
    float *out);

/* Ratio-4 overlapping compressor prefill. The checkpoint wkv/wgate matrices
 * are BF16 [2*head_dim,4096] and are semantically loaded as f32 linears.
 * APE is F32 [4,2*head_dim], norm_weight is checkpoint BF16 upcast to f32.
 *
 * overlap semantics for chunk c:
 *   slots 0..3 = previous chunk's FIRST head_dim half (unavailable for c=0)
 *   slots 4..7 = current chunk's SECOND head_dim half
 * followed by per-dimension softmax pooling across the 8 slots.
 *
 * The pooled result is explicitly cast to BF16, RMS-normalized, receives
 * compressed YaRN at absolute positions c*4, then:
 *   rotate=0: K64 FP8 simulation on non-RoPE dims (main attention compressor)
 *   rotate=1: normalized Hadamard + K32 FP4 simulation over all dims
 *             (Indexer compressor)
 *
 * Optional diagnostics are [seqlen/4,head_dim] BF16-valued f32 arrays.
 */
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
    float *out);

/* Indexer query path from the already-computed main q-rank vector `qr`:
 * FP8 wq_b [64*128,1024] -> [64,128], compressed YaRN on each head's final
 * 64 dims, normalized Hadamard, K32 FP4 simulation. `out` is [64,128]. */
int waste_ds_v4_csa_indexer_query_ref(
    const float *qr,
    const uint8_t *wq_b_fp8,
    const uint8_t *wq_b_scale_e8m0,
    size_t position,
    float *out);

/* BF16 weights_proj [64,4096] followed by the source scalar multiplier
 * 1/sqrt(index_head_dim) * 1/sqrt(index_n_heads). Output is BF16-valued. */
int waste_ds_v4_csa_indexer_weights_ref(
    const float *x,
    const uint16_t *weights_bf16,
    float *out64);

/* Source-equation Indexer score for one query row.
 * q is [64,128], compressed_kv [n_compressed,128], weights [64].
 * The visible tensor boundaries are BF16: q/kv, einsum dot result, weights,
 * per-head product, and final head-sum output. */
int waste_ds_v4_csa_index_scores_ref(
    const float *q,
    const float *compressed_kv,
    size_t n_compressed,
    const float *weights,
    float *scores);

/* Apply prefill causality and torch.topk-style descending selection for one
 * row. `visible` is floor((row+1)/4); k is min(512,n_compressed).
 * Invalid selections are written as -1 after ranking; valid indices receive
 * `offset`. Scores are assumed finite BF16 values. Ties use lower source index
 * only to keep the scalar reference deterministic; real fixtures reject ties.
 */
size_t waste_ds_v4_csa_topk_row_ref(
    const float *scores,
    size_t n_compressed,
    size_t visible,
    size_t offset,
    int32_t *out,
    size_t out_capacity);

#endif /* WASTE_DEEPSEEK_V4_CSA_REF_H */
