/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_manifest.h"

#include "deepseek_v4_json_strict.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* span bookkeeping                                                    */
/* ------------------------------------------------------------------ */

typedef struct ds_span {
    uint64_t begin;
    uint64_t end;
} ds_span;

static int ds_span_make(uint64_t begin, uint64_t bytes, uint64_t region,
                        ds_span *out)
{
    if (!out || bytes == 0 || begin > region || bytes > region - begin)
        return -1;
    out->begin = begin;
    out->end = begin + bytes;
    return 0;
}

static int ds_span_overlaps(ds_span a, ds_span b)
{
    return a.begin < b.end && b.begin < a.end;
}

/* ------------------------------------------------------------------ */
/* public entry points                                                 */
/* ------------------------------------------------------------------ */

const char *waste_ds_v4_manifest_strerror(waste_ds_v4_manifest_status status)
{
    switch (status) {
    case WASTE_DS_V4_MANIFEST_OK:
        return "ok";
    case WASTE_DS_V4_MANIFEST_E_ARG:
        return "manifest buffer is missing, oversize, or not NUL-terminated";
    case WASTE_DS_V4_MANIFEST_E_JSON:
        return "manifest is not a JSON object";
    case WASTE_DS_V4_MANIFEST_E_FAMILY:
        return "manifest does not declare the deepseek-v4-flash family";
    case WASTE_DS_V4_MANIFEST_E_VERSION:
        return "unsupported manifest_version";
    case WASTE_DS_V4_MANIFEST_E_REVISION:
        return "manifest revision is not the pinned 0731 revision";
    case WASTE_DS_V4_MANIFEST_E_GEOMETRY:
        return "manifest geometry does not match the Gate-A contract";
    case WASTE_DS_V4_MANIFEST_E_ROUTED_MAP:
        return "routed record map is truncated, overlapping, or malformed";
    case WASTE_DS_V4_MANIFEST_E_RESIDENT:
        return "resident FP8 descriptors are invalid, overlapping, or out of range";
    case WASTE_DS_V4_MANIFEST_E_GENERATION:
        return "manifest asserts a stepping/generation capability that is refused";
    }
    return "unknown manifest status";
}

int waste_ds_v4_manifest_is_family(const char *json, size_t len)
{
    if (!dsjs_input_ok(json, len))
        return 0;
    js_doc d;
    if (js_parse(&d, json) != 0) {
        js_free(&d);
        return 0;
    }
    int ok = 0;
    if (d.n > 0 && d.tok[0].type == JS_OBJ) {
        char family[32];
        if (dsjs_member_str(&d, 0, "family", family, sizeof family) == 0)
            ok = strcmp(family, WASTE_DS_V4_FAMILY) == 0;
    }
    js_free(&d);
    return ok;
}

/* `true` for either of these is a claim we refuse to honour, and so is any
 * shape other than an absent key or a literal `false` — a string "no" or an
 * object must not read as "not enabled". */
static int ds_capability_refused(const js_doc *d, int root, const char *key)
{
    int tok = js_get(d, root, key);
    if (tok < 0)
        return 0;                       /* absent: fine */
    if (tok >= d->n || d->tok[tok].type != JS_BOOL)
        return 1;
    int len = d->tok[tok].end - d->tok[tok].start;
    return !(len == 5 && memcmp(d->src + d->tok[tok].start, "false", 5) == 0);
}

static waste_ds_v4_manifest_status ds_parse_geometry(
    const js_doc *d, int root, waste_ds_v4_gate_a_manifest *gate_a)
{
    int geo = js_get(d, root, "geometry");
    if (geo < 0 || geo >= d->n || d->tok[geo].type != JS_OBJ)
        return WASTE_DS_V4_MANIFEST_E_GEOMETRY;

    memset(gate_a, 0, sizeof *gate_a);
    if (dsjs_member_u32(d, geo, "main_layers", &gate_a->main_layers) != 0 ||
        dsjs_member_u32(d, geo, "hidden_size", &gate_a->hidden_size) != 0 ||
        dsjs_member_u32(d, geo, "routed_experts_per_layer",
                      &gate_a->routed_experts_per_layer) != 0 ||
        dsjs_member_u32(d, geo, "shared_experts_per_layer",
                      &gate_a->shared_experts_per_layer) != 0 ||
        dsjs_member_u32(d, geo, "routed_experts_per_token",
                      &gate_a->routed_experts_per_token) != 0 ||
        dsjs_member_u32(d, geo, "moe_intermediate_size",
                      &gate_a->moe_intermediate_size) != 0 ||
        dsjs_member_u32(d, geo, "bootstrap_hash_layers",
                      &gate_a->bootstrap_hash_layers) != 0 ||
        dsjs_member_u32(d, geo, "shards", &gate_a->shards) != 0 ||
        dsjs_member_u32(d, geo, "tensors", &gate_a->tensors) != 0 ||
        dsjs_member_u64(d, geo, "payload_bytes", &gate_a->payload_bytes) != 0 ||
        dsjs_member_u32(d, geo, "routed_records", &gate_a->routed_records) != 0 ||
        dsjs_member_u32(d, geo, "routed_payload_bytes_per_record",
                      &gate_a->routed_payload_bytes_per_record) != 0)
        return WASTE_DS_V4_MANIFEST_E_GEOMETRY;

    /* Compared against the pinned contract, not merely range-checked. */
    if (waste_ds_v4_gate_a_manifest_validate(gate_a) != 0)
        return WASTE_DS_V4_MANIFEST_E_GEOMETRY;
    return WASTE_DS_V4_MANIFEST_OK;
}

static waste_ds_v4_manifest_status ds_parse_routed(
    const js_doc *d, int root, waste_ds_v4_manifest *out)
{
    /* Sizes come from Gate-A geometry; the manifest supplies only offsets. */
    if (waste_ds_v4_routed_payload_layout_init(
            out->gate_a.hidden_size, out->gate_a.moe_intermediate_size,
            &out->routed_layout) != 0)
        return WASTE_DS_V4_MANIFEST_E_GEOMETRY;
    if (out->routed_layout.payload_bytes !=
        (size_t)out->gate_a.routed_payload_bytes_per_record)
        return WASTE_DS_V4_MANIFEST_E_GEOMETRY;

    int rec = js_get(d, root, "routed_record");
    if (rec < 0 || rec >= d->n || d->tok[rec].type != JS_OBJ)
        return WASTE_DS_V4_MANIFEST_E_ROUTED_MAP;

    memset(&out->routed_map, 0, sizeof out->routed_map);
    if (dsjs_member_size(d, rec, "record_bytes", &out->routed_map.record_bytes) != 0 ||
        dsjs_member_size(d, rec, "w1_offset", &out->routed_map.w1_offset) != 0 ||
        dsjs_member_size(d, rec, "w1_scale_offset",
                       &out->routed_map.w1_scale_offset) != 0 ||
        dsjs_member_size(d, rec, "w3_offset", &out->routed_map.w3_offset) != 0 ||
        dsjs_member_size(d, rec, "w3_scale_offset",
                       &out->routed_map.w3_scale_offset) != 0 ||
        dsjs_member_size(d, rec, "w2_offset", &out->routed_map.w2_offset) != 0 ||
        dsjs_member_size(d, rec, "w2_scale_offset",
                       &out->routed_map.w2_scale_offset) != 0)
        return WASTE_DS_V4_MANIFEST_E_ROUTED_MAP;

    if (waste_ds_v4_routed_record_map_validate(&out->routed_map,
                                               &out->routed_layout) != 0)
        return WASTE_DS_V4_MANIFEST_E_ROUTED_MAP;
    return WASTE_DS_V4_MANIFEST_OK;
}

static waste_ds_v4_manifest_status ds_parse_resident(
    const js_doc *d, int root, waste_ds_v4_manifest *out)
{
    int trunk = js_get(d, root, "trunk");
    if (trunk < 0 || trunk >= d->n || d->tok[trunk].type != JS_OBJ)
        return WASTE_DS_V4_MANIFEST_E_RESIDENT;
    if (dsjs_member_u64(d, trunk, "bytes", &out->trunk_bytes) != 0 ||
        out->trunk_bytes == 0)
        return WASTE_DS_V4_MANIFEST_E_RESIDENT;

    int planes = js_get(d, trunk, "resident");
    if (planes < 0 || planes >= d->n || d->tok[planes].type != JS_ARR)
        return WASTE_DS_V4_MANIFEST_E_RESIDENT;
    int count = js_size(d, planes);
    if (count <= 0 || (uint32_t)count > WASTE_DS_V4_MAX_RESIDENT_PLANES)
        return WASTE_DS_V4_MANIFEST_E_RESIDENT;

    /* Two spans per plane — weights and the E8M0 scale grid — all checked
     * against each other. A scale grid overlapping another plane's weights
     * decodes to plausible numbers, which is exactly why it is checked here
     * rather than left to a payload checksum. */
    ds_span spans[2u * WASTE_DS_V4_MAX_RESIDENT_PLANES];
    size_t span_count = 0;

    for (int i = 0; i < count; i++) {
        int item = js_at(d, planes, i);
        if (item < 0 || item >= d->n || d->tok[item].type != JS_OBJ)
            return WASTE_DS_V4_MANIFEST_E_RESIDENT;
        waste_ds_v4_resident_plane *plane = &out->resident[i];
        memset(plane, 0, sizeof *plane);

        size_t rows = 0, cols = 0;
        if (dsjs_member_str(d, item, "name", plane->name, sizeof plane->name) != 0 ||
            dsjs_member_size(d, item, "rows", &rows) != 0 ||
            dsjs_member_size(d, item, "cols", &cols) != 0 ||
            dsjs_member_u64(d, item, "weight_offset", &plane->weight_offset) != 0 ||
            dsjs_member_u64(d, item, "scale_offset", &plane->scale_offset) != 0)
            return WASTE_DS_V4_MANIFEST_E_RESIDENT;

        /* Tile geometry is derived and refused, never taken on trust: this is
         * where a 127-row plane or a non-128-multiple width dies. */
        if (waste_ds_v4_fp8_plane_layout_init(rows, cols, &plane->layout) != 0)
            return WASTE_DS_V4_MANIFEST_E_RESIDENT;

        /* Two planes may not share a name: the binding is by name, and a
         * duplicate silently shadows whichever the loader looks up second. */
        for (int j = 0; j < i; j++)
            if (strcmp(out->resident[j].name, plane->name) == 0)
                return WASTE_DS_V4_MANIFEST_E_RESIDENT;

        if (ds_span_make(plane->weight_offset, (uint64_t)plane->layout.weight_bytes,
                         out->trunk_bytes, &spans[span_count]) != 0)
            return WASTE_DS_V4_MANIFEST_E_RESIDENT;
        span_count++;
        if (ds_span_make(plane->scale_offset, (uint64_t)plane->layout.scale_bytes,
                         out->trunk_bytes, &spans[span_count]) != 0)
            return WASTE_DS_V4_MANIFEST_E_RESIDENT;
        span_count++;
    }

    for (size_t i = 0; i < span_count; i++)
        for (size_t j = i + 1; j < span_count; j++)
            if (ds_span_overlaps(spans[i], spans[j]))
                return WASTE_DS_V4_MANIFEST_E_RESIDENT;

    out->resident_count = (size_t)count;
    return WASTE_DS_V4_MANIFEST_OK;
}

waste_ds_v4_manifest_status waste_ds_v4_manifest_parse(
    const char *json, size_t len, waste_ds_v4_manifest *out)
{
    if (!out)
        return WASTE_DS_V4_MANIFEST_E_ARG;
    memset(out, 0, sizeof *out);
    if (!dsjs_input_ok(json, len))
        return WASTE_DS_V4_MANIFEST_E_ARG;

    js_doc d;
    if (js_parse(&d, json) != 0) {
        js_free(&d);
        return WASTE_DS_V4_MANIFEST_E_JSON;
    }

    waste_ds_v4_manifest_status status = WASTE_DS_V4_MANIFEST_OK;
    if (d.n <= 0 || d.tok[0].type != JS_OBJ) {
        status = WASTE_DS_V4_MANIFEST_E_JSON;
        goto done;
    }

    /* Family first, and exactly. Everything below reads DeepSeek keys out of
     * this object, so a Kimi v0 manifest must be refused before, not after,
     * its `version` field is mistaken for ours. */
    if (dsjs_member_str(&d, 0, "family", out->family, sizeof out->family) != 0 ||
        strcmp(out->family, WASTE_DS_V4_FAMILY) != 0) {
        status = WASTE_DS_V4_MANIFEST_E_FAMILY;
        goto done;
    }
    if (dsjs_member_u32(&d, 0, "manifest_version", &out->manifest_version) != 0 ||
        out->manifest_version != WASTE_DS_V4_MANIFEST_VERSION) {
        status = WASTE_DS_V4_MANIFEST_E_VERSION;
        goto done;
    }
    if (dsjs_member_str(&d, 0, "revision", out->revision, sizeof out->revision) != 0 ||
        strcmp(out->revision, WASTE_DS_V4_0731_REVISION) != 0) {
        status = WASTE_DS_V4_MANIFEST_E_REVISION;
        goto done;
    }

    /* A container does not get to turn stepping on by declaring it. */
    if (ds_capability_refused(&d, 0, "generation") ||
        ds_capability_refused(&d, 0, "stepping")) {
        status = WASTE_DS_V4_MANIFEST_E_GENERATION;
        goto done;
    }

    if ((status = ds_parse_geometry(&d, 0, &out->gate_a)) != WASTE_DS_V4_MANIFEST_OK)
        goto done;
    if ((status = ds_parse_routed(&d, 0, out)) != WASTE_DS_V4_MANIFEST_OK)
        goto done;
    if ((status = ds_parse_resident(&d, 0, out)) != WASTE_DS_V4_MANIFEST_OK)
        goto done;

    out->generation_enabled = 0;

done:
    js_free(&d);
    if (status != WASTE_DS_V4_MANIFEST_OK)
        memset(out, 0, sizeof *out);   /* never hand back a half-validated view */
    return status;
}

int waste_ds_v4_manifest_bind_routed_record(
    const waste_ds_v4_manifest *manifest,
    const void *record,
    size_t record_bytes,
    waste_ds_v4_routed_record_view *out)
{
    /* A failed parse zeroes the struct, so a matching family is what proves
     * this manifest was validated rather than merely declared on the stack. */
    if (!manifest || strcmp(manifest->family, WASTE_DS_V4_FAMILY) != 0)
        return -1;
    return waste_ds_v4_routed_record_view_bind(
        record, record_bytes, &manifest->routed_map, &manifest->routed_layout, out);
}

int waste_ds_v4_manifest_resident_plane(
    const waste_ds_v4_manifest *manifest,
    size_t index,
    const void *trunk,
    size_t trunk_bytes,
    const uint8_t **weights_out,
    const uint8_t **scales_out)
{
    if (!manifest || !trunk || !weights_out || !scales_out ||
        strcmp(manifest->family, WASTE_DS_V4_FAMILY) != 0 ||
        index >= manifest->resident_count)
        return -1;
    /* A region shorter than the manifest declared makes every offset below
     * unverified, so refuse rather than re-derive bounds from the caller. */
    if ((uint64_t)trunk_bytes < manifest->trunk_bytes)
        return -1;

    const waste_ds_v4_resident_plane *plane = &manifest->resident[index];
    ds_span weights, scales;
    if (ds_span_make(plane->weight_offset, (uint64_t)plane->layout.weight_bytes,
                     manifest->trunk_bytes, &weights) != 0 ||
        ds_span_make(plane->scale_offset, (uint64_t)plane->layout.scale_bytes,
                     manifest->trunk_bytes, &scales) != 0)
        return -1;

    const uint8_t *base = (const uint8_t *)trunk;
    *weights_out = base + plane->weight_offset;
    *scales_out = base + plane->scale_offset;
    return 0;
}

int waste_ds_v4_manifest_step_refused(const char **why)
{
    if (why)
        *why = "DeepSeek V4 stepping/generation is refused until Gate I/V8 "
               "final logits and Gate K/V9 greedy generation pass";
    return -1;
}
