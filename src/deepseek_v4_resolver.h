/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
/*
 * deepseek_v4_resolver.h — derive a DeepSeek file set from container metadata.
 *
 * `deepseek_v4_file_runtime.h` owns already-resolved paths and offsets and
 * deliberately refuses to invent them. This is the layer that produces them,
 * and the rule that shapes it is the same one: **every fact is read from the
 * container's own declaration, never constructed from a convention.**
 *
 * Concretely, the resolver does not build `trunk.bin`, does not build
 * `experts-L%03u.bin`, does not assume a bank per layer is ordered by its
 * position in an array, and does not assume a uniform record stride. It reads
 * a path, reads a declared layer number, and reads either an explicit
 * per-expert offset table or an explicitly declared base/stride pair. A
 * container that arranges its experts some other way stays expressible; a
 * container that declares nothing gets refused rather than guessed at.
 *
 * What the resolver *does* own is the validation that can be done before
 * touching the filesystem, which is the pattern the file-runtime layer
 * established:
 *
 * - paths are relative and cannot escape the container root;
 * - every main layer is declared exactly once, with no gap and no duplicate;
 * - each bank declares exactly `routed_experts_per_layer` records;
 * - records within a bank do not overlap each other;
 * - every offset stays inside WASTE's signed-64-bit positional-I/O range.
 *
 * Actual file sizes are still the OS's answer, not the container's claim:
 * bounds against real bytes remain `waste_ds_v4_file_runtime_open`'s job.
 *
 * This layer opens no file and enables no execution.
 */
#ifndef WASTE_DEEPSEEK_V4_RESOLVER_H
#define WASTE_DEEPSEEK_V4_RESOLVER_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_file_runtime.h"
#include "deepseek_v4_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum waste_ds_v4_resolve_status {
    WASTE_DS_V4_RESOLVE_OK = 0,
    WASTE_DS_V4_RESOLVE_E_ARG,       /* NULL/oversize/not NUL-terminated     */
    WASTE_DS_V4_RESOLVE_E_MANIFEST,  /* manifest failed runtime revalidation */
    WASTE_DS_V4_RESOLVE_E_JSON,      /* unparseable, or no usable `files`    */
    WASTE_DS_V4_RESOLVE_E_PATH,      /* absolute, escaping, or malformed     */
    WASTE_DS_V4_RESOLVE_E_TOPOLOGY,  /* layer coverage/duplication wrong     */
    WASTE_DS_V4_RESOLVE_E_OFFSETS,   /* count, overlap, or range wrong       */
    WASTE_DS_V4_RESOLVE_E_MEMORY,
    WASTE_DS_V4_RESOLVE_E_OPEN     /* declaration was sound; the I/O was not */
} waste_ds_v4_resolve_status;

/* Owned result. `spec` is ready to hand straight to the file-runtime open;
 * its borrowed pointers reference this object, so it must outlive that call
 * (which copies what it needs) and then be freed. */
typedef struct waste_ds_v4_resolved_files {
    char *trunk_path;
    char **bank_paths;
    uint64_t *record_offsets;              /* bank_count * experts, flat     */
    waste_ds_v4_file_bank_spec *banks;
    size_t bank_count;
    size_t experts_per_layer;
    waste_ds_v4_file_open_spec spec;
} waste_ds_v4_resolved_files;

/* Resolve the `files` section of a container document against a validated
 * manifest. `root_dir` may be NULL or "" for paths relative to the process's
 * working directory; otherwise every resolved path is `root_dir` + "/" + the
 * declared relative path. `json`/`len` are the same NUL-terminated document
 * the manifest was parsed from.
 *
 * `cache_bytes`/`cache_policy` are carried into `out->spec` unchanged; the
 * resolver has no opinion about cache sizing.
 *
 * Returns WASTE_DS_V4_RESOLVE_OK, or a status with `out` fully zeroed. */
waste_ds_v4_resolve_status waste_ds_v4_resolve_files(
    waste_ds_v4_resolved_files *out,
    const waste_ds_v4_manifest *manifest,
    const char *root_dir,
    const char *json,
    size_t len,
    size_t cache_bytes,
    int cache_policy);

/* Safe on a zeroed struct, a failed resolve, and a second call. */
void waste_ds_v4_resolved_files_free(waste_ds_v4_resolved_files *files);

/* Human-readable, stable, never NULL. */
const char *waste_ds_v4_resolve_strerror(waste_ds_v4_resolve_status status);

/* Resolve, open, and release the resolved description in one step, so no
 * caller has to reason about how long the borrowed path/offset pointers stay
 * valid. On success `out` owns every resource exactly as if
 * `waste_ds_v4_file_runtime_open` had been called directly, and is closed with
 * `waste_ds_v4_file_runtime_close`.
 *
 * A failed open returns WASTE_DS_V4_RESOLVE_E_OPEN, so one value still tells
 * the caller whether it succeeded. `open_rc`, when non-NULL, additionally
 * receives the file-runtime return code. */
waste_ds_v4_resolve_status waste_ds_v4_resolver_open(
    waste_ds_v4_file_runtime *out,
    const waste_ds_v4_manifest *manifest,
    const char *root_dir,
    const char *json,
    size_t len,
    size_t cache_bytes,
    int cache_policy,
    int *open_rc);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_DEEPSEEK_V4_RESOLVER_H */
