/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#ifndef WASTE_DEEPSEEK_V4_CONTAINER_CONTRACT_H
#define WASTE_DEEPSEEK_V4_CONTAINER_CONTRACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WASTE_DS_V4_0731_REVISION \
    "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

#define WASTE_DS_V4_0731_MAIN_LAYERS 43u
#define WASTE_DS_V4_0731_HIDDEN 4096u
#define WASTE_DS_V4_0731_ROUTED_EXPERTS 256u
#define WASTE_DS_V4_0731_SHARED_EXPERTS 1u
#define WASTE_DS_V4_0731_TOPK 6u
#define WASTE_DS_V4_0731_MOE_INTERMEDIATE 2048u
#define WASTE_DS_V4_0731_BOOTSTRAP_HASH_LAYERS 3u
#define WASTE_DS_V4_0731_SHARDS 48u
#define WASTE_DS_V4_0731_TENSORS 72317u
#define WASTE_DS_V4_0731_PAYLOAD_BYTES UINT64_C(166878536440)
#define WASTE_DS_V4_0731_ROUTED_RECORDS 11008u
#define WASTE_DS_V4_0731_ROUTED_PAYLOAD_BYTES 13369344u

typedef struct waste_ds_v4_fp4_plane_layout {
    size_t rows;
    size_t cols;
    size_t packed_weight_bytes;
    size_t e8m0_scale_bytes;
} waste_ds_v4_fp4_plane_layout;

typedef struct waste_ds_v4_routed_payload_layout {
    waste_ds_v4_fp4_plane_layout w1;
    waste_ds_v4_fp4_plane_layout w3;
    waste_ds_v4_fp4_plane_layout w2;
    size_t payload_bytes;
} waste_ds_v4_routed_payload_layout;

typedef struct waste_ds_v4_fp8_plane_layout {
    size_t rows;
    size_t cols;
    size_t weight_bytes;
    size_t scale_rows;
    size_t scale_cols;
    size_t scale_bytes;
} waste_ds_v4_fp8_plane_layout;

typedef struct waste_ds_v4_gate_a_manifest {
    uint32_t main_layers;
    uint32_t hidden_size;
    uint32_t routed_experts_per_layer;
    uint32_t shared_experts_per_layer;
    uint32_t routed_experts_per_token;
    uint32_t moe_intermediate_size;
    uint32_t bootstrap_hash_layers;
    uint32_t shards;
    uint32_t tensors;
    uint64_t payload_bytes;
    uint32_t routed_records;
    uint32_t routed_payload_bytes_per_record;
} waste_ds_v4_gate_a_manifest;

/* Manifest-provided byte mapping for one opaque cache record.  Offsets are not
 * prescribed by this contract: a future container may choose plane ordering,
 * add a header, and add alignment padding.  Validation only requires the six
 * proven native tensor planes to fit without overlap and with exact sizes. */
typedef struct waste_ds_v4_routed_record_map {
    size_t record_bytes;
    size_t w1_offset;
    size_t w1_scale_offset;
    size_t w3_offset;
    size_t w3_scale_offset;
    size_t w2_offset;
    size_t w2_scale_offset;
} waste_ds_v4_routed_record_map;

typedef struct waste_ds_v4_routed_record_view {
    const uint8_t *w1;
    const uint8_t *w1_scale;
    const uint8_t *w3;
    const uint8_t *w3_scale;
    const uint8_t *w2;
    const uint8_t *w2_scale;
} waste_ds_v4_routed_record_view;

void waste_ds_v4_gate_a_0731_manifest(waste_ds_v4_gate_a_manifest *out);
int waste_ds_v4_gate_a_manifest_validate(
    const waste_ds_v4_gate_a_manifest *manifest);

int waste_ds_v4_routed_payload_layout_init(
    size_t hidden_size,
    size_t intermediate_size,
    waste_ds_v4_routed_payload_layout *out);

int waste_ds_v4_fp8_plane_layout_init(
    size_t rows,
    size_t cols,
    waste_ds_v4_fp8_plane_layout *out);

/* Validate a manifest-supplied map against the native routed payload geometry.
 * Extra record bytes are allowed for future headers/padding; plane overlap or
 * truncation is always refused. */
int waste_ds_v4_routed_record_map_validate(
    const waste_ds_v4_routed_record_map *map,
    const waste_ds_v4_routed_payload_layout *layout);

/* Bind one opaque expert-cache record to the six native plane pointers after
 * validating both the map and caller-provided record length. */
int waste_ds_v4_routed_record_view_bind(
    const void *record,
    size_t record_bytes,
    const waste_ds_v4_routed_record_map *map,
    const waste_ds_v4_routed_payload_layout *layout,
    waste_ds_v4_routed_record_view *out);

#ifdef __cplusplus
}
#endif

#endif
