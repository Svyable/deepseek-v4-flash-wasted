/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Resource-owning file layer for the DeepSeek runtime substrate. Paths and
 * expert offsets are caller/evidence supplied: this file deliberately does not
 * invent a container filename, record stride, ordering, header, or alignment.
 */
#include "deepseek_v4_runtime.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

#ifdef _WIN32
#include <io.h>
#endif

static void ds_close_fd(int fd)
{
    if (fd < 0)
        return;
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

static int read_exact_fd(int fd, void *dst, size_t n)
{
    uint8_t *p = (uint8_t *)dst;
    size_t done = 0;
    while (done < n) {
        const size_t left = n - done;
        const int64_t got = waste_pread(fd, p + done, left, (int64_t)done);
        if (got <= 0 || (uint64_t)got > (uint64_t)left)
            return -1;
        done += (size_t)got;
    }
    return 0;
}

void waste_ds_v4_file_runtime_close(waste_ds_v4_file_runtime *fr)
{
    if (!fr)
        return;

    /* Runtime/cache state may still call through source.read_user, so release
     * it before freeing the positional source or closing any bank handle. */
    waste_ds_v4_runtime_free(&fr->runtime);
    waste_ds_v4_positional_source_free(&fr->source);

    if (fr->bank_fds) {
        for (size_t i = 0; i < fr->bank_count; i++)
            ds_close_fd(fr->bank_fds[i]);
    }
    free(fr->bank_fds);
    waste_dio_free(fr->trunk);
    memset(fr, 0, sizeof *fr);
}

int waste_ds_v4_file_runtime_open(
    waste_ds_v4_file_runtime *out,
    const waste_ds_v4_manifest *manifest,
    const waste_ds_v4_file_open_spec *spec)
{
    if (!out)
        return -1;
    memset(out, 0, sizeof *out);

    if (!manifest || !spec || !spec->trunk_path || !*spec->trunk_path ||
        !spec->banks ||
        spec->bank_count != (size_t)manifest->gate_a.main_layers ||
        manifest->trunk_bytes == 0 || manifest->trunk_bytes > SIZE_MAX)
        return -1;

    const size_t trunk_bytes = (size_t)manifest->trunk_bytes;

    /* Buffered/random-access is intentional here. Direct-I/O alignment is a
     * later evidence/performance contract; this ownership layer must not infer
     * it from the imported Kimi format. */
    const int trunk_fd = waste_open_stream(spec->trunk_path, 0);
    if (trunk_fd < 0)
        return -1;

    const int64_t trunk_file_bytes = waste_file_size(trunk_fd);
    if (trunk_file_bytes < 0 ||
        (uint64_t)trunk_file_bytes != manifest->trunk_bytes) {
        ds_close_fd(trunk_fd);
        return -1;
    }

    out->trunk = (uint8_t *)waste_dio_alloc(trunk_bytes);
    if (!out->trunk || read_exact_fd(trunk_fd, out->trunk, trunk_bytes) != 0) {
        ds_close_fd(trunk_fd);
        waste_ds_v4_file_runtime_close(out);
        return -1;
    }
    ds_close_fd(trunk_fd); /* resident bytes are final; no second trunk copy */

    if (spec->bank_count > SIZE_MAX / sizeof *out->bank_fds) {
        waste_ds_v4_file_runtime_close(out);
        return -1;
    }
    out->bank_fds = (int *)malloc(spec->bank_count * sizeof *out->bank_fds);
    waste_ds_v4_positional_bank *pos = (waste_ds_v4_positional_bank *)calloc(
        spec->bank_count, sizeof *pos);
    if (!out->bank_fds || !pos) {
        free(pos);
        waste_ds_v4_file_runtime_close(out);
        return -1;
    }
    out->bank_count = spec->bank_count;
    for (size_t i = 0; i < out->bank_count; i++)
        out->bank_fds[i] = -1;

    for (size_t layer = 0; layer < spec->bank_count; layer++) {
        const waste_ds_v4_file_bank_spec *in = &spec->banks[layer];
        if (!in->path || !*in->path || !in->record_offsets ||
            in->record_count !=
                (size_t)manifest->gate_a.routed_experts_per_layer) {
            free(pos);
            waste_ds_v4_file_runtime_close(out);
            return -1;
        }

        const int fd = waste_open_stream(in->path, 0);
        if (fd < 0) {
            free(pos);
            waste_ds_v4_file_runtime_close(out);
            return -1;
        }
        out->bank_fds[layer] = fd;

        const int64_t bytes = waste_file_size(fd);
        if (bytes <= 0) {
            free(pos);
            waste_ds_v4_file_runtime_close(out);
            return -1;
        }

        pos[layer].read_at = waste_ds_v4_fd_read_at;
        pos[layer].read_user = &out->bank_fds[layer];
        pos[layer].bytes = (uint64_t)bytes;
        pos[layer].record_offsets = in->record_offsets;
        pos[layer].record_count = in->record_count;
    }

    if (waste_ds_v4_positional_source_init(
            &out->source, manifest, pos, spec->bank_count) != 0) {
        free(pos);
        waste_ds_v4_file_runtime_close(out);
        return -1;
    }
    free(pos);

    if (waste_ds_v4_runtime_init_positional(
            &out->runtime, manifest, out->trunk, trunk_bytes,
            spec->cache_bytes, spec->cache_policy, &out->source) != 0) {
        waste_ds_v4_file_runtime_close(out);
        return -1;
    }

    return 0;
}
