/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Scalar DeepSeek V4 Hyper-Connection reference seam.
 * Correctness/debugging code for Gate D/V3; not a performance kernel.
 */
#ifndef WASTE_DEEPSEEK_V4_MHC_REF_H
#define WASTE_DEEPSEEK_V4_MHC_REF_H

#include <stddef.h>

#define WASTE_DS_V4_HC 4u
#define WASTE_DS_V4_HC_MIX ((2u + WASTE_DS_V4_HC) * WASTE_DS_V4_HC) /* 24 */
#define WASTE_DS_V4_HC_COMB (WASTE_DS_V4_HC * WASTE_DS_V4_HC)       /* 16 */

/* Pinned 0731 hc_split_sinkhorn semantics for hc_mult=4.
 * comb is row-major [source_hc][output_hc]. */
int waste_ds_v4_hc_split_sinkhorn(const float mixes[WASTE_DS_V4_HC_MIX],
                                  const float scale[3],
                                  const float base[WASTE_DS_V4_HC_MIX],
                                  unsigned sinkhorn_iters,
                                  float eps,
                                  float pre[WASTE_DS_V4_HC],
                                  float post[WASTE_DS_V4_HC],
                                  float comb[WASTE_DS_V4_HC_COMB]);

/* Official Block.hc_pre model-semantic seam for one [hc,dim] state.
 * x is the f32 expansion of the original BF16 hidden state.
 * fn is [24, hc*dim] row-major F32 checkpoint data.
 * y is f32 before the source path's final BF16 cast.
 * mixes/pre/post/comb are optional diagnostic outputs (may be NULL). */
int waste_ds_v4_hc_pre_ref(const float *x,
                           size_t dim,
                           const float *fn,
                           const float scale[3],
                           const float base[WASTE_DS_V4_HC_MIX],
                           float norm_eps,
                           unsigned sinkhorn_iters,
                           float hc_eps,
                           float *y,
                           float *mixes,
                           float *pre,
                           float *post,
                           float *comb);

/* Official Block.hc_post seam for one token.
 * branch is [dim], residual is [hc,dim], comb is [source][output].
 * y is [hc,dim] f32 before the source path's final type_as(branch) cast. */
int waste_ds_v4_hc_post_ref(const float *branch,
                            const float *residual,
                            size_t dim,
                            const float post[WASTE_DS_V4_HC],
                            const float comb[WASTE_DS_V4_HC_COMB],
                            float *y);

#endif /* WASTE_DEEPSEEK_V4_MHC_REF_H */
