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

/* One positional storage read. The contract intentionally mirrors pread:
 * return the number of bytes copied, 0 for EOF, or a negative value on error.
 * Short positive reads are legal; the source layer loops until the exact
 * expert record has landed. `off` is absolute within the named bank.
 *
 * Keeping this callback below filenames/fds is deliberate. The real 0731
 * container evidence has not frozen bank names, record headers, ordering or
 * alignment. A caller may use ordinary files, direct I/O, memory, a test fault
 * injector, or another positional medium without changing expert identity. */
typedef int64_t (*waste_ds_v4_read_at_fn)(void *user,
                                          void *dst,
                                          size_t n,
                                          uint64_t off);

/* Caller-supplied description of one routed bank. There must be exactly one
 * spec per manifest layer and exactly one explicit offset per routed expert.
 * No ordering is inferred from the expert id. */
typedef struct waste_ds_v4_positional_bank {
    waste_ds_v4_read_at_fn read_at;
    void *read_user;
    uint64_t bytes;
    const uint64_t *record_offsets;
    size_t record_count;
} waste_ds_v4_positional_bank;

/* Validated, owned copy of the per-layer positional index. The read_user
 * objects themselves stay caller-owned and must outlive this source and any
 * runtime using it. The routed map is copied too so a source cannot later be
 * paired with another equally-sized record layout and silently misbind planes. */
typedef struct waste_ds_v4_positional_source {
    waste_ds_v4_positional_bank *banks;
    uint64_t *record_offsets;
    size_t bank_count;
    size_t record_bytes;
    uint32_t experts_per_layer;
    waste_ds_v4_routed_record_map routed_map;
} waste_ds_v4_positional_source;

/* Validate and deep-copy every bank descriptor and expert offset. This freezes
 * placement identity after open: mutating the caller's offset table cannot
 * redirect a later cache miss. Offsets must fit both the declared bank size
 * and WASTE's signed-64-bit positional-I/O range. */
int waste_ds_v4_positional_source_init(
    waste_ds_v4_positional_source *source,
    const waste_ds_v4_manifest *manifest,
    const waste_ds_v4_positional_bank *banks,
    size_t bank_count);

void waste_ds_v4_positional_source_free(waste_ds_v4_positional_source *source);

/* waste_fetch_fn-compatible adapter used by waste_ecache. It performs an
 * exact-length read at the explicit [layer][expert] offset; EOF, zero progress,
 * an over-reporting reader, or any read error is a hard failure. */
int waste_ds_v4_positional_fetch(void *user,
                                 int layer,
                                 int expert,
                                 uint8_t *dst);

/* Small adapter for callers backed by a native WASTE file descriptor. `user`
 * points to an int fd. The source still requires the file's declared byte size
 * and explicit expert offsets; this helper does not invent either. */
int64_t waste_ds_v4_fd_read_at(void *user,
                               void *dst,
                               size_t n,
                               uint64_t off);

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

/* Same runtime binding, but with the validated positional source above. This
 * prevents a family-open caller from accidentally pairing a source built for a
 * different layer/expert/record geometry or routed-plane map with this
 * manifest. The source must outlive the runtime. */
int waste_ds_v4_runtime_init_positional(
    waste_ds_v4_runtime *runtime,
    const waste_ds_v4_manifest *manifest,
    const void *trunk,
    size_t trunk_bytes,
    size_t cache_bytes,
    int cache_policy,
    waste_ds_v4_positional_source *source);

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

/* ---- explicit file ownership -------------------------------------------
 *
 * This is still below the DeepSeek family/container naming layer. Paths and
 * expert offsets are supplied explicitly by the caller; nothing here assumes
 * `trunk.bin`, one bank per layer, uniform stride, a record header, or a bank
 * ordering. The purpose is narrower and production-facing: own native file
 * handles, validate their actual sizes, load the declared resident trunk once
 * into its final aligned allocation, build the positional source, and unwind
 * every partially-open resource on failure. */
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

/* Open an already-resolved file set. `manifest` is already parsed/validated;
 * this function checks the resident file is exactly `manifest->trunk_bytes`,
 * opens every supplied bank, asks the OS for its actual byte length, and lets
 * waste_ds_v4_positional_source_init prove every declared expert extent fits.
 *
 * Paths and caller offset arrays are only borrowed during this call. On
 * success the source owns a copy of the offsets and file_runtime owns all file
 * descriptors and resident bytes. Returns 0 on success and leaves `out` fully
 * zeroed on every failure. */
int waste_ds_v4_file_runtime_open(
    waste_ds_v4_file_runtime *out,
    const waste_ds_v4_manifest *manifest,
    const waste_ds_v4_file_open_spec *spec);

/* Safe after a successful open or a failed open attempt. Runtime/cache state is
 * released before the source and bank descriptors it may reference, then every
 * native bank handle and the aligned resident allocation are released. */
void waste_ds_v4_file_runtime_close(waste_ds_v4_file_runtime *file_runtime);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_DEEPSEEK_V4_RUNTIME_H */
