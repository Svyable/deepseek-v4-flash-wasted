/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_container_contract.h"
#include "quant/fp4_e2m1.h"

#include <limits.h>
#include <string.h>

static int size_add(size_t a, size_t b, size_t *out)
{
    if (!out || a > SIZE_MAX - b)
        return -1;
    *out = a + b;
    return 0;
}

static int manifest_equal(const waste_ds_v4_gate_a_manifest *a,
                          const waste_ds_v4_gate_a_manifest *b)
{
    return a->main_layers == b->main_layers &&
           a->hidden_size == b->hidden_size &&
           a->routed_experts_per_layer == b->routed_experts_per_layer &&
           a->shared_experts_per_layer == b->shared_experts_per_layer &&
           a->routed_experts_per_token == b->routed_experts_per_token &&
           a->moe_intermediate_size == b->moe_intermediate_size &&
           a->bootstrap_hash_layers == b->bootstrap_hash_layers &&
           a->shards == b->shards &&
           a->tensors == b->tensors &&
           a->payload_bytes == b->payload_bytes &&
           a->routed_records == b->routed_records &&
           a->routed_payload_bytes_per_record ==
               b->routed_payload_bytes_per_record;
}

void waste_ds_v4_gate_a_0731_manifest(waste_ds_v4_gate_a_manifest *out)
{
    if (!out)
        return;
    memset(out, 0, sizeof(*out));
    out->main_layers = WASTE_DS_V4_0731_MAIN_LAYERS;
    out->hidden_size = WASTE_DS_V4_0731_HIDDEN;
    out->routed_experts_per_layer = WASTE_DS_V4_0731_ROUTED_EXPERTS;
    out->shared_experts_per_layer = WASTE_DS_V4_0731_SHARED_EXPERTS;
    out->routed_experts_per_token = WASTE_DS_V4_0731_TOPK;
    out->moe_intermediate_size = WASTE_DS_V4_0731_MOE_INTERMEDIATE;
    out->bootstrap_hash_layers = WASTE_DS_V4_0731_BOOTSTRAP_HASH_LAYERS;
    out->shards = WASTE_DS_V4_0731_SHARDS;
    out->tensors = WASTE_DS_V4_0731_TENSORS;
    out->payload_bytes = WASTE_DS_V4_0731_PAYLOAD_BYTES;
    out->routed_records = WASTE_DS_V4_0731_ROUTED_RECORDS;
    out->routed_payload_bytes_per_record =
        WASTE_DS_V4_0731_ROUTED_PAYLOAD_BYTES;
}

int waste_ds_v4_gate_a_manifest_validate(
    const waste_ds_v4_gate_a_manifest *manifest)
{
    if (!manifest)
        return -1;
    waste_ds_v4_gate_a_manifest expected;
    waste_ds_v4_gate_a_0731_manifest(&expected);
    if (!manifest_equal(manifest, &expected))
        return -1;
    if ((uint64_t)manifest->main_layers *
            (uint64_t)manifest->routed_experts_per_layer !=
        manifest->routed_records)
        return -1;
    if (manifest->routed_experts_per_token >
        manifest->routed_experts_per_layer)
        return -1;
    return 0;
}

static int fp4_plane(size_t rows, size_t cols,
                     waste_ds_v4_fp4_plane_layout *out)
{
    if (!out || !waste_fp4_dims_ok(rows, cols))
        return -1;
    size_t weight = waste_fp4_weight_bytes(rows, cols);
    size_t scales = waste_fp4_scale_bytes(rows, cols);
    if (!weight || !scales)
        return -1;
    out->rows = rows;
    out->cols = cols;
    out->packed_weight_bytes = weight;
    out->e8m0_scale_bytes = scales;
    return 0;
}

int waste_ds_v4_routed_payload_layout_init(
    size_t hidden_size,
    size_t intermediate_size,
    waste_ds_v4_routed_payload_layout *out)
{
    if (!out || hidden_size == 0 || intermediate_size == 0)
        return -1;
    memset(out, 0, sizeof(*out));
    if (fp4_plane(intermediate_size, hidden_size, &out->w1) != 0 ||
        fp4_plane(intermediate_size, hidden_size, &out->w3) != 0 ||
        fp4_plane(hidden_size, intermediate_size, &out->w2) != 0)
        return -1;

    size_t total = 0;
    const waste_ds_v4_fp4_plane_layout *planes[3] = {
        &out->w1, &out->w3, &out->w2,
    };
    for (size_t i = 0; i < 3; i++) {
        size_t plane_bytes = 0;
        if (size_add(planes[i]->packed_weight_bytes,
                     planes[i]->e8m0_scale_bytes,
                     &plane_bytes) != 0 ||
            size_add(total, plane_bytes, &total) != 0)
            return -1;
    }
    out->payload_bytes = total;
    return 0;
}

int waste_ds_v4_fp8_plane_layout_init(
    size_t rows,
    size_t cols,
    waste_ds_v4_fp8_plane_layout *out)
{
    if (!out || rows == 0 || cols == 0 ||
        rows % 128u != 0 || cols % 128u != 0 ||
        rows > SIZE_MAX / cols)
        return -1;
    memset(out, 0, sizeof(*out));
    out->rows = rows;
    out->cols = cols;
    out->weight_bytes = rows * cols;
    out->scale_rows = rows / 128u;
    out->scale_cols = cols / 128u;
    if (out->scale_rows > SIZE_MAX / out->scale_cols)
        return -1;
    out->scale_bytes = out->scale_rows * out->scale_cols;
    return 0;
}

typedef struct span {
    size_t begin;
    size_t end;
} span;

static int make_span(size_t begin, size_t bytes, size_t record_bytes, span *out)
{
    if (!out || bytes == 0 || begin > record_bytes ||
        bytes > record_bytes - begin)
        return -1;
    out->begin = begin;
    out->end = begin + bytes;
    return 0;
}

static int spans_overlap(span a, span b)
{
    return a.begin < b.end && b.begin < a.end;
}

int waste_ds_v4_routed_record_map_validate(
    const waste_ds_v4_routed_record_map *map,
    const waste_ds_v4_routed_payload_layout *layout)
{
    if (!map || !layout || map->record_bytes < layout->payload_bytes)
        return -1;

    span spans[6];
    if (make_span(map->w1_offset, layout->w1.packed_weight_bytes,
                  map->record_bytes, &spans[0]) != 0 ||
        make_span(map->w1_scale_offset, layout->w1.e8m0_scale_bytes,
                  map->record_bytes, &spans[1]) != 0 ||
        make_span(map->w3_offset, layout->w3.packed_weight_bytes,
                  map->record_bytes, &spans[2]) != 0 ||
        make_span(map->w3_scale_offset, layout->w3.e8m0_scale_bytes,
                  map->record_bytes, &spans[3]) != 0 ||
        make_span(map->w2_offset, layout->w2.packed_weight_bytes,
                  map->record_bytes, &spans[4]) != 0 ||
        make_span(map->w2_scale_offset, layout->w2.e8m0_scale_bytes,
                  map->record_bytes, &spans[5]) != 0)
        return -1;

    for (size_t i = 0; i < 6; i++)
        for (size_t j = i + 1; j < 6; j++)
            if (spans_overlap(spans[i], spans[j]))
                return -1;
    return 0;
}

int waste_ds_v4_routed_record_view_bind(
    const void *record,
    size_t record_bytes,
    const waste_ds_v4_routed_record_map *map,
    const waste_ds_v4_routed_payload_layout *layout,
    waste_ds_v4_routed_record_view *out)
{
    if (!record || !map || !layout || !out || record_bytes != map->record_bytes ||
        waste_ds_v4_routed_record_map_validate(map, layout) != 0)
        return -1;
    const uint8_t *base = (const uint8_t *)record;
    out->w1 = base + map->w1_offset;
    out->w1_scale = base + map->w1_scale_offset;
    out->w3 = base + map->w3_offset;
    out->w3_scale = base + map->w3_scale_offset;
    out->w2 = base + map->w2_offset;
    out->w2_scale = base + map->w2_scale_offset;
    return 0;
}
