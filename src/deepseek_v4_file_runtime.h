/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
/*
 * deepseek_v4_file_runtime.h — resource ownership for already-resolved
 * DeepSeek runtime files.
 *
 * This layer deliberately does not resolve a family directory. Paths and
 * expert offsets are evidence/caller supplied, so no filename, record header,
 * stride, ordering, alignment, or direct-I/O convention is invented here.
 */
#ifndef WASTE_DEEPSEEK_V4_FILE_RUNTIME_H
#define WASTE_DEEPSEEK_V4_FILE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct waste_ds_v4_file_bank_spec {
    const char *path;
    const uint64_t *record_offsets;
    size_t record_count;
} waste_ds_v4_file_bank_spec;

typedef struct waste_ds_v4_file_open_spec {
    const char *trunk_path;
    const waste_ds_v4_file_bank_spec *banks;
    size_t bank_count;
    size_t cache_bytes;
    int cache_policy;
} waste_ds_v4_file_open_spec;

typedef struct waste_ds_v4_file_runtime {
    waste_ds_v4_runtime runtime;
    waste_ds_v4_positional_source source;
    uint8_t *trunk;
    int *bank_fds;
    size_t bank_count;
} waste_ds_v4_file_runtime;

/* Open an already-resolved file set. The validated manifest is revalidated at
 * this runtime boundary because parsed C structs can be mutated after parse.
 *
 * `trunk_path`, bank paths, and caller offset arrays are borrowed only during
 * this call. On success the object owns the resident bytes and every native
 * bank descriptor, while the positional source owns a deep copy of the expert
 * offsets. The resident file must be exactly manifest->trunk_bytes; routed bank
 * limits come from the files' actual OS-reported sizes.
 *
 * Returns 0 on success. Every failure leaves `out` fully zeroed. */
int waste_ds_v4_file_runtime_open(
    waste_ds_v4_file_runtime *out,
    const waste_ds_v4_manifest *manifest,
    const waste_ds_v4_file_open_spec *spec);

/* Safe after success, after a failed open attempt, and after an earlier close.
 * Teardown is runtime/cache -> positional source -> bank fds -> trunk bytes so
 * no callback can outlive the object it references. */
void waste_ds_v4_file_runtime_close(waste_ds_v4_file_runtime *file_runtime);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_DEEPSEEK_V4_FILE_RUNTIME_H */
