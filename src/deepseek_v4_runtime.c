/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_runtime.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include "platform.h"
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

static int routed_map_equal(const waste_ds_v4_routed_record_map *a,
                            const waste_ds_v4_routed_record_map *b)
{
    return a && b &&
           a->record_bytes == b->record_bytes &&
           a->w1_offset == b->w1_offset &&
           a->w1_scale_offset == b->w1_scale_offset &&
           a->w3_offset == b->w3_offset &&
           a->w3_scale_offset == b->w3_scale_offset &&
           a->w2_offset == b->w2_offset &&
           a->w2_scale_offset == b->w2_scale_offset;
}

void waste_ds_v4_positional_source_free(waste_ds_v4_positional_source *source)
{
    if (!source)
        return;
    free(source->record_offsets);
    free(source->banks);
    memset(source, 0, sizeof *source);
}

int waste_ds_v4_positional_source_init(
    waste_ds_v4_positional_source *source,
    const waste_ds_v4_manifest *manifest,
    const waste_ds_v4_positional_bank *banks,
    size_t bank_count)
{
    if (!source)
        return -1;
    memset(source, 0, sizeof *source);

    if (!manifest_ready(manifest) || !banks ||
        bank_count != (size_t)manifest->gate_a.main_layers)
        return -1;

    const size_t experts = (size_t)manifest->gate_a.routed_experts_per_layer;
    const size_t rec_bytes = manifest->routed_map.record_bytes;
    if (experts == 0 || rec_bytes == 0 ||
        (uint64_t)rec_bytes > (uint64_t)INT64_MAX ||
        bank_count > SIZE_MAX / sizeof *source->banks ||
        experts > SIZE_MAX / bank_count)
        return -1;

    const size_t offset_count = bank_count * experts;
    if (offset_count > SIZE_MAX / sizeof *source->record_offsets)
        return -1;

    waste_ds_v4_positional_bank *bank_copy =
        (waste_ds_v4_positional_bank *)calloc(bank_count, sizeof *bank_copy);
    uint64_t *offset_copy =
        (uint64_t *)malloc(offset_count * sizeof *offset_copy);
    if (!bank_copy || !offset_copy) {
        free(offset_copy);
        free(bank_copy);
        return -1;
    }

    const uint64_t rec64 = (uint64_t)rec_bytes;
    for (size_t layer = 0; layer < bank_count; layer++) {
        const waste_ds_v4_positional_bank *in = &banks[layer];
        if (!in->read_at || !in->record_offsets ||
            in->record_count != experts || in->bytes == 0 ||
            in->bytes > (uint64_t)INT64_MAX) {
            free(offset_copy);
            free(bank_copy);
            return -1;
        }

        uint64_t *dst = offset_copy + layer * experts;
        for (size_t expert = 0; expert < experts; expert++) {
            const uint64_t off = in->record_offsets[expert];
            /* Subtraction form avoids wrapping on a malicious near-UINT64_MAX
             * offset. The signed-range check keeps every later waste_pread
             * offset representable on Windows as well as POSIX. */
            if (off > in->bytes || rec64 > in->bytes - off ||
                off > (uint64_t)INT64_MAX ||
                rec64 > (uint64_t)INT64_MAX - off) {
                free(offset_copy);
                free(bank_copy);
                return -1;
            }
            dst[expert] = off;
        }

        bank_copy[layer] = *in;
        bank_copy[layer].record_offsets = dst;
        bank_copy[layer].record_count = experts;
    }

    source->banks = bank_copy;
    source->record_offsets = offset_copy;
    source->bank_count = bank_count;
    source->record_bytes = rec_bytes;
    source->experts_per_layer = (uint32_t)experts;
    source->routed_map = manifest->routed_map;
    return 0;
}

int waste_ds_v4_positional_fetch(void *user,
                                 int layer,
                                 int expert,
                                 uint8_t *dst)
{
    waste_ds_v4_positional_source *source =
        (waste_ds_v4_positional_source *)user;
    if (!source || !source->banks || !source->record_offsets || !dst ||
        source->record_bytes == 0 || layer < 0 || expert < 0 ||
        (size_t)layer >= source->bank_count ||
        (uint32_t)expert >= source->experts_per_layer)
        return -1;

    waste_ds_v4_positional_bank *bank = &source->banks[layer];
    if (!bank->read_at || !bank->record_offsets ||
        bank->record_count != source->experts_per_layer)
        return -1;

    const uint64_t base = bank->record_offsets[expert];
    size_t done = 0;
    while (done < source->record_bytes) {
        const size_t left = source->record_bytes - done;
        const int64_t got = bank->read_at(bank->read_user,
                                          dst + done,
                                          left,
                                          base + (uint64_t)done);
        if (got <= 0 || (uint64_t)got > (uint64_t)left)
            return -1;
        done += (size_t)got;
    }
    return 0;
}

int64_t waste_ds_v4_fd_read_at(void *user,
                               void *dst,
                               size_t n,
                               uint64_t off)
{
    if (!user || !dst || off > (uint64_t)INT64_MAX)
        return -1;
    const int fd = *(const int *)user;
    if (fd < 0)
        return -1;
    return waste_pread(fd, dst, n, (int64_t)off);
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

int waste_ds_v4_runtime_init_positional(
    waste_ds_v4_runtime *runtime,
    const waste_ds_v4_manifest *manifest,
    const void *trunk,
    size_t trunk_bytes,
    size_t cache_bytes,
    int cache_policy,
    waste_ds_v4_positional_source *source)
{
    if (!manifest_ready(manifest) || !source || !source->banks ||
        source->bank_count != (size_t)manifest->gate_a.main_layers ||
        source->experts_per_layer != manifest->gate_a.routed_experts_per_layer ||
        source->record_bytes != manifest->routed_map.record_bytes ||
        !routed_map_equal(&source->routed_map, &manifest->routed_map))
        return -1;

    return waste_ds_v4_runtime_init(runtime, manifest, trunk, trunk_bytes,
                                    cache_bytes, cache_policy,
                                    waste_ds_v4_positional_fetch, source);
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

/* ---- explicit native-file ownership ----------------------------------- */

static int fd_read_exact_at(int fd, void *dst, size_t n, uint64_t off)
{
    if (fd < 0 || !dst || off > (uint64_t)INT64_MAX ||
        (uint64_t)n > (uint64_t)INT64_MAX - off)
        return -1;

    uint8_t *p = (uint8_t *)dst;
    size_t done = 0;
    while (done < n) {
        const size_t left = n - done;
        const int64_t got = waste_ds_v4_fd_read_at(
            &fd, p + done, left, off + (uint64_t)done);
        if (got <= 0 || (uint64_t)got > (uint64_t)left)
            return -1;
        done += (size_t)got;
    }
    return 0;
}

void waste_ds_v4_file_runtime_close(waste_ds_v4_file_runtime *file_runtime)
{
    if (!file_runtime)
        return;

    /* Runtime/cache state may still point through source->banks[].read_user
     * into bank_fds, so tear down strictly from the leaves outward. */
    waste_ds_v4_runtime_free(&file_runtime->runtime);
    waste_ds_v4_positional_source_free(&file_runtime->source);

    if (file_runtime->bank_fds) {
        for (size_t i = 0; i < file_runtime->bank_count; i++)
            if (file_runtime->bank_fds[i] >= 0)
                close(file_runtime->bank_fds[i]);
    }
    free(file_runtime->bank_fds);
    waste_dio_free(file_runtime->trunk);
    memset(file_runtime, 0, sizeof *file_runtime);
}

int waste_ds_v4_file_runtime_open(
    waste_ds_v4_file_runtime *out,
    const waste_ds_v4_manifest *manifest,
    const waste_ds_v4_file_open_spec *spec)
{
    if (!out)
        return -1;
    memset(out, 0, sizeof *out);

    if (!manifest_ready(manifest) || !spec || !spec->trunk_path ||
        !*spec->trunk_path || !spec->banks ||
        spec->bank_count != (size_t)manifest->gate_a.main_layers ||
        manifest->trunk_bytes > (uint64_t)SIZE_MAX ||
        manifest->trunk_bytes > (uint64_t)INT64_MAX ||
        spec->bank_count > SIZE_MAX / sizeof *out->bank_fds)
        return -1;

    const size_t experts = (size_t)manifest->gate_a.routed_experts_per_layer;
    for (size_t i = 0; i < spec->bank_count; i++) {
        if (!spec->banks[i].path || !*spec->banks[i].path ||
            !spec->banks[i].record_offsets ||
            spec->banks[i].record_count != experts)
            return -1;
    }

    int trunk_fd = -1;
    waste_ds_v4_positional_bank *pos_banks = NULL;

    trunk_fd = open(spec->trunk_path, O_RDONLY | WASTE_O_BINARY);
    if (trunk_fd < 0)
        goto fail;

    const int64_t trunk_size = waste_file_size(trunk_fd);
    if (trunk_size < 0 ||
        (uint64_t)trunk_size != manifest->trunk_bytes)
        goto fail;

    out->trunk = (uint8_t *)waste_dio_alloc((size_t)manifest->trunk_bytes);
    if (!out->trunk ||
        fd_read_exact_at(trunk_fd, out->trunk,
                         (size_t)manifest->trunk_bytes, 0) != 0)
        goto fail;

    close(trunk_fd);
    trunk_fd = -1;

    out->bank_fds = (int *)malloc(spec->bank_count * sizeof *out->bank_fds);
    pos_banks = (waste_ds_v4_positional_bank *)calloc(
        spec->bank_count, sizeof *pos_banks);
    if (!out->bank_fds || !pos_banks)
        goto fail;

    out->bank_count = spec->bank_count;
    for (size_t i = 0; i < out->bank_count; i++)
        out->bank_fds[i] = -1;

    for (size_t i = 0; i < out->bank_count; i++) {
        const waste_ds_v4_file_bank_spec *in = &spec->banks[i];
        const int fd = open(in->path, O_RDONLY | WASTE_O_BINARY);
        if (fd < 0)
            goto fail;
        out->bank_fds[i] = fd;

        const int64_t bytes = waste_file_size(fd);
        if (bytes <= 0)
            goto fail;

        pos_banks[i].read_at = waste_ds_v4_fd_read_at;
        pos_banks[i].read_user = &out->bank_fds[i];
        pos_banks[i].bytes = (uint64_t)bytes;
        pos_banks[i].record_offsets = in->record_offsets;
        pos_banks[i].record_count = in->record_count;
    }

    if (waste_ds_v4_positional_source_init(
            &out->source, manifest, pos_banks, out->bank_count) != 0)
        goto fail;

    free(pos_banks);
    pos_banks = NULL;

    if (waste_ds_v4_runtime_init_positional(
            &out->runtime, manifest, out->trunk,
            (size_t)manifest->trunk_bytes,
            spec->cache_bytes, spec->cache_policy, &out->source) != 0)
        goto fail;

    return 0;

fail:
    if (trunk_fd >= 0)
        close(trunk_fd);
    free(pos_banks);
    waste_ds_v4_file_runtime_close(out);
    return -1;
}
