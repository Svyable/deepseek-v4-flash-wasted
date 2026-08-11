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

/* Pinned Gate-A/V0 source identity.  This is a checkpoint contract, not a
 * public WASTE container version.  The future file parser must prove its
 * manifest describes this geometry before it can bind any runtime object. */
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

/* One native quantized linear plane.  ``rows`` and ``cols`` are logical BF16
 * matrix dimensions; the checkpoint stores E2M1 as packed nibbles and one raw
 * E8M0 exponent byte per K32 group.  No dequantized shadow representation is
 * part of this contract. */
typedef struct waste_ds_v4_fp4_plane_layout {
    size_t rows;
    size_t cols;
    size_t packed_weight_bytes;
    size_t e8m0_scale_bytes;
} waste_ds_v4_fp4_plane_layout;

/* Routed expert payload only.  No on-disk header or plane ordering is frozen
 * here: Gate A explicitly measured 12.75 MiB of tensor payload per record and
 * excluded future WASTE headers/alignment.  A later container revision may
 * choose offsets while preserving these native planes byte-for-byte. */
typedef struct waste_ds_v4_routed_payload_layout {
    waste_ds_v4_fp4_plane_layout w1;
    waste_ds_v4_fp4_plane_layout w3;
    waste_ds_v4_fp4_plane_layout w2;
    size_t payload_bytes;
} waste_ds_v4_routed_payload_layout;

/* Resident FP8 linears use row-major E4M3 bytes plus a raw E8M0 scale grid,
 * one scale byte per 128x128 logical tile. */
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

/* Fill the one checkpoint manifest that current real-weight evidence supports. */
void waste_ds_v4_gate_a_0731_manifest(waste_ds_v4_gate_a_manifest *out);

/* Exact validation: a future loader must refuse rather than infer a different
 * model geometry under the same family discriminator. */
int waste_ds_v4_gate_a_manifest_validate(
    const waste_ds_v4_gate_a_manifest *manifest);

/* Native routed-expert payload geometry.  The current checkpoint has
 * w1/w3 [intermediate, hidden] and w2 [hidden, intermediate]. */
int waste_ds_v4_routed_payload_layout_init(
    size_t hidden_size,
    size_t intermediate_size,
    waste_ds_v4_routed_payload_layout *out);

/* Native resident FP8 geometry.  Rows/cols must map exactly onto the K128
 * E8M0 scale grid used by the proven reference linear. */
int waste_ds_v4_fp8_plane_layout_init(
    size_t rows,
    size_t cols,
    waste_ds_v4_fp8_plane_layout *out);

#ifdef __cplusplus
}
#endif

#endif
