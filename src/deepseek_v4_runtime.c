/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_runtime.h"

#include <string.h>

#include "waste_backend.h"

static int manifest_ready(const waste_ds_v4_manifest *manifest)
{
    if (!manifest ||
        strcmp(manifest->family, WASTE_DS_V4_FAMILY) != 0 ||
        strcmp(manifest->revision, WASTE_DS_V4_0731_REVISION) != 0 ||
        manifest->manifest_version != WASTE_DS_V4_MANIFEST_VERSION ||
        manifest->resident_count == 0 ||
        manifest->resident_count > WASTE_DS_V4_MAX_RESIDENT_PLANES ||
        manifest->trunk_bytes == 0 ||
        manifest->routed_map.record_bytes == 0)
        return 0;

    if (waste_ds_v4_gate_a_manifest_validate(&manifest->gate_a) != 0 ||
        waste_ds_v4_routed_record_map_validate(&manifest->routed_map,
                                                &manifest->routed_layout) != 0)
        return 0;
    return 1;
}

int waste_ds_v4_runtime_init(waste_ds_v4_runtime *runtime,
                             const waste_ds_v4_manifest *manifest,
                             const void *trunk,
                             size_t trunk_bytes,
                             size_t cache_bytes,
                             int cache_policy,
                             waste_fetch_fn fetch,
                             void *fetch_user)
{
    if (!runtime)
        return -1;
    memset(runtime, 0, sizeof *runtime);

    if (!manifest_ready(manifest) || !trunk ||
        (uint64_t)trunk_bytes < manifest->trunk_bytes)
        return -1;

    runtime->manifest = *manifest;
    runtime->trunk = (const uint8_t *)trunk;
    runtime->trunk_bytes = trunk_bytes;

    if (!fetch)
        return 0; /* resident-only binding is a valid, narrower capability */

    runtime->fetch = fetch;
    runtime->fetch_user = fetch_user;
    const size_t rec_bytes = runtime->manifest.routed_map.record_bytes;
    if (waste_ecache_init(&runtime->cache, cache_bytes, rec_bytes,
                          cache_policy) != 0) {
        memset(runtime, 0, sizeof *runtime);
        return -1;
    }

    if (runtime->cache.n_slots == 0) {
        runtime->miss_buf = (uint8_t *)waste_dio_alloc(rec_bytes);
        if (!runtime->miss_buf) {
            waste_ecache_free(&runtime->cache);
            memset(runtime, 0, sizeof *runtime);
            return -1;
        }
    }
    runtime->routed_ready = 1;
    return 0;
}

void waste_ds_v4_runtime_free(waste_ds_v4_runtime *runtime)
{
    if (!runtime)
        return;
    if (runtime->routed_ready)
        waste_ecache_free(&runtime->cache);
    waste_dio_free(runtime->miss_buf);
    memset(runtime, 0, sizeof *runtime);
}

static int routed_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    waste_ds_v4_runtime *runtime = (waste_ds_v4_runtime *)user;
    if (!runtime || !runtime->routed_ready || !runtime->fetch || !dst ||
        layer < 0 || (uint32_t)layer >= runtime->manifest.gate_a.main_layers ||
        expert < 0 ||
        (uint32_t)expert >= runtime->manifest.gate_a.routed_experts_per_layer)
        return -1;
    return runtime->fetch(runtime->fetch_user, layer, expert, dst);
}

int waste_ds_v4_runtime_routed_record(waste_ds_v4_runtime *runtime,
                                      int layer,
                                      int expert,
                                      waste_ds_v4_routed_record_view *out)
{
    if (out)
        memset(out, 0, sizeof *out);
    if (!runtime || !out || !runtime->routed_ready || !runtime->fetch ||
        layer < 0 || (uint32_t)layer >= runtime->manifest.gate_a.main_layers ||
        expert < 0 ||
        (uint32_t)expert >= runtime->manifest.gate_a.routed_experts_per_layer)
        return -1;

    const size_t rec_bytes = runtime->manifest.routed_map.record_bytes;
    const uint8_t *record = NULL;
    if (runtime->cache.n_slots > 0) {
        record = waste_ecache_get(&runtime->cache, layer, expert,
                                  routed_fetch, runtime);
    } else {
        runtime->cache.misses++;
        runtime->cache.bytes_read += rec_bytes;
        if (routed_fetch(runtime, layer, expert, runtime->miss_buf) == 0)
            record = runtime->miss_buf;
    }
    if (!record)
        return -1;

    return waste_ds_v4_manifest_bind_routed_record(
        &runtime->manifest, record, rec_bytes, out);
}

int waste_ds_v4_runtime_resident_linear(waste_ds_v4_runtime *runtime,
                                        size_t resident_index,
                                        const float *x,
                                        size_t m,
                                        float *y)
{
    if (!runtime || !x || !y || m == 0 ||
        resident_index >= runtime->manifest.resident_count)
        return -1;

    const uint8_t *weights = NULL;
    const uint8_t *scales = NULL;
    if (waste_ds_v4_manifest_resident_plane(
            &runtime->manifest, resident_index,
            runtime->trunk, runtime->trunk_bytes,
            &weights, &scales) != 0)
        return -1;

    waste_backend_init(WASTE_BE_AUTO);
    if (!waste_k.ds_v4_fp8_linear_e8m0)
        return -1;

    const waste_ds_v4_fp8_plane_layout *layout =
        &runtime->manifest.resident[resident_index].layout;
    return waste_k.ds_v4_fp8_linear_e8m0(
        x, m, layout->cols, weights, scales, layout->rows, y);
}
