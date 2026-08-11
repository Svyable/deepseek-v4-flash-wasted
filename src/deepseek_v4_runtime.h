/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
/*
 * deepseek_v4_runtime.h — bind a validated DeepSeek manifest to WASTE's
 * existing resident-kernel and expert-cache runtime seams.
 *
 * This is deliberately below model execution. It can locate/apply one
 * resident FP8 plane and fetch/bind one routed FP4 expert record, but it does
 * not expose a transformer step or generation entry point. The family
 * manifest's unconditional stepping refusal remains the capability gate.
 */
#ifndef WASTE_DEEPSEEK_V4_RUNTIME_H
#define WASTE_DEEPSEEK_V4_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_manifest.h"
#include "ecache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct waste_ds_v4_runtime {
    /* Copy the validated description so caller mutation cannot change offsets
     * after a cache entry or resident view has already been accepted. */
    waste_ds_v4_manifest manifest;
    const uint8_t *trunk;
    size_t trunk_bytes;

    /* Optional routed-record path. A NULL fetch callback leaves the runtime
     * resident-only; routed_record then refuses rather than inventing I/O. */
    waste_ecache cache;
    waste_fetch_fn fetch;
    void *fetch_user;
    uint8_t *miss_buf;
    int routed_ready;
} waste_ds_v4_runtime;

/* Bind already-validated manifest/trunk bytes. `fetch` uses the normal WASTE
 * cache callback contract and may be NULL for a resident-only binding.
 * `cache_bytes` is ignored when fetch is NULL. Returns 0 on success. */
int waste_ds_v4_runtime_init(waste_ds_v4_runtime *runtime,
                             const waste_ds_v4_manifest *manifest,
                             const void *trunk,
                             size_t trunk_bytes,
                             size_t cache_bytes,
                             int cache_policy,
                             waste_fetch_fn fetch,
                             void *fetch_user);

void waste_ds_v4_runtime_free(waste_ds_v4_runtime *runtime);

/* Fetch one opaque routed expert record through waste_ecache, then bind its
 * six native FP4/E8M0 planes through the manifest map. The returned pointers
 * have the same lifetime as waste_ecache_get: consume them before requesting
 * another routed record from this runtime. */
int waste_ds_v4_runtime_routed_record(waste_ds_v4_runtime *runtime,
                                      int layer,
                                      int expert,
                                      waste_ds_v4_routed_record_view *out);

/* Apply one resident E4M3/E8M0 matrix through the normal backend dispatch.
 * Matrix dimensions come from the manifest plane, not from caller guesses.
 * `x` is [m, cols], `y` is [m, rows]. */
int waste_ds_v4_runtime_resident_linear(waste_ds_v4_runtime *runtime,
                                        size_t resident_index,
                                        const float *x,
                                        size_t m,
                                        float *y);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_DEEPSEEK_V4_RUNTIME_H */
