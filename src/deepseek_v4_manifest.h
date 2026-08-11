/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
/*
 * deepseek_v4_manifest.h — fail-closed parser for a DeepSeek-family manifest.
 *
 * `deepseek_v4_container_contract.h` says what the geometry must be. This says
 * how a container declares it, and refuses every declaration that does not
 * match. The two are deliberately separate: the contract is checkpoint-derived
 * and frozen, the manifest is untrusted input.
 *
 * Three things this parser exists to prevent.
 *
 * 1. **Family confusion.** A DeepSeek container and a Kimi v0 container must
 *    never be readable as each other. `family` is mandatory and exact; a
 *    manifest without it is refused rather than guessed at, and
 *    `waste_ds_v4_manifest_is_family` lets a dispatching loader ask which
 *    parser to use without either parser having to tolerate the other's
 *    schema.
 *
 * 2. **Geometry drift.** Every Gate-A number is compared against the pinned
 *    0731 contract, not merely range-checked. A container claiming 42 layers
 *    or a 2049-wide MoE intermediate is not a smaller model, it is a mistake.
 *
 * 3. **Silent overlap.** Resident FP8 planes and the six routed planes are
 *    located by manifest-supplied byte offsets. Offsets are not prescribed —
 *    plane order, headers and alignment padding stay open — but they must fit
 *    their declared region and must not overlap each other. An overlapping map
 *    reads plausible numbers out of the wrong tensor, which is exactly the
 *    failure a checksum does not catch.
 *
 * Nothing here enables execution. `waste_ds_v4_manifest_step_refused` is
 * unconditional and stays that way until the numerical gates close: a parsed
 * manifest is a validated description, not a runnable model.
 */
#ifndef WASTE_DEEPSEEK_V4_MANIFEST_H
#define WASTE_DEEPSEEK_V4_MANIFEST_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_container_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact, mandatory. A manifest that omits this is refused, never assumed. */
#define WASTE_DS_V4_FAMILY "deepseek-v4-flash"
#define WASTE_DS_V4_MANIFEST_VERSION 1u

/* Bounded so a manifest cannot size an allocation. The 0731 trunk needs far
 * fewer than this per container; raising it is a deliberate edit, not
 * something a container can ask for. */
#define WASTE_DS_V4_MAX_RESIDENT_PLANES 128u
#define WASTE_DS_V4_MAX_NAME 96u

typedef enum waste_ds_v4_manifest_status {
    WASTE_DS_V4_MANIFEST_OK = 0,
    WASTE_DS_V4_MANIFEST_E_ARG,        /* NULL/oversize/embedded NUL          */
    WASTE_DS_V4_MANIFEST_E_JSON,       /* unparseable, or root is not object  */
    WASTE_DS_V4_MANIFEST_E_FAMILY,     /* missing or foreign family           */
    WASTE_DS_V4_MANIFEST_E_VERSION,    /* unsupported manifest_version        */
    WASTE_DS_V4_MANIFEST_E_REVISION,   /* not the pinned 0731 revision        */
    WASTE_DS_V4_MANIFEST_E_GEOMETRY,   /* Gate-A geometry drifted             */
    WASTE_DS_V4_MANIFEST_E_ROUTED_MAP, /* routed record map invalid           */
    WASTE_DS_V4_MANIFEST_E_RESIDENT,   /* resident FP8 descriptors invalid    */
    WASTE_DS_V4_MANIFEST_E_GENERATION  /* manifest asserts a refused capability */
} waste_ds_v4_manifest_status;

/* One resident FP8 tile-quantized plane: E4M3 weights plus an E8M0 scale grid,
 * both located inside the trunk region by manifest-supplied offsets. */
typedef struct waste_ds_v4_resident_plane {
    char name[WASTE_DS_V4_MAX_NAME];
    waste_ds_v4_fp8_plane_layout layout;
    uint64_t weight_offset;
    uint64_t scale_offset;
} waste_ds_v4_resident_plane;

typedef struct waste_ds_v4_manifest {
    char family[32];
    char revision[80];
    uint32_t manifest_version;

    waste_ds_v4_gate_a_manifest gate_a;

    /* Derived from Gate-A geometry, not read from the manifest: the manifest
     * supplies where the planes are, never how large they are. */
    waste_ds_v4_routed_payload_layout routed_layout;
    waste_ds_v4_routed_record_map routed_map;

    uint64_t trunk_bytes;
    size_t resident_count;
    waste_ds_v4_resident_plane resident[WASTE_DS_V4_MAX_RESIDENT_PLANES];

    /* Always 0. Present so callers read a value rather than a comment. */
    int generation_enabled;
} waste_ds_v4_manifest;

/* Cheap family probe for a dispatching loader: 1 when `json` is an object
 * whose `family` is exactly WASTE_DS_V4_FAMILY, 0 otherwise (including
 * unparseable input). Does no geometry validation. */
int waste_ds_v4_manifest_is_family(const char *json, size_t len);

/* Parse and validate. `json` must be NUL-terminated with strlen(json) == len.
 * `out` is fully written on success and zeroed on failure, so a caller cannot
 * act on a partially validated manifest. */
waste_ds_v4_manifest_status waste_ds_v4_manifest_parse(
    const char *json, size_t len, waste_ds_v4_manifest *out);

/* Human-readable, stable, never NULL. */
const char *waste_ds_v4_manifest_strerror(waste_ds_v4_manifest_status status);

/* Bind one routed cache record through the manifest's map. Thin wrapper that
 * keeps callers from reaching past the manifest for the layout. */
int waste_ds_v4_manifest_bind_routed_record(
    const waste_ds_v4_manifest *manifest,
    const void *record,
    size_t record_bytes,
    waste_ds_v4_routed_record_view *out);

/* Locate one resident plane's bytes inside a mapped trunk region. Refuses
 * anything the parse already refused, and refuses a region shorter than the
 * manifest's declared trunk. */
int waste_ds_v4_manifest_resident_plane(
    const waste_ds_v4_manifest *manifest,
    size_t index,
    const void *trunk,
    size_t trunk_bytes,
    const uint8_t **weights_out,
    const uint8_t **scales_out);

/* Unconditional. Returns non-zero and, when `why` is non-NULL, points it at a
 * static reason string. Every step/generation entry point for this family goes
 * through here until Gate I/V8 and Gate K/V9 close. */
int waste_ds_v4_manifest_step_refused(const char **why);

#ifdef __cplusplus
}
#endif

#endif
