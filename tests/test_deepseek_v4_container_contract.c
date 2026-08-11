/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "../src/deepseek_v4_container_contract.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_gate_a_manifest(void)
{
    waste_ds_v4_gate_a_manifest manifest;
    waste_ds_v4_gate_a_0731_manifest(&manifest);
    assert(waste_ds_v4_gate_a_manifest_validate(&manifest) == 0);
    assert(manifest.main_layers == 43u);
    assert(manifest.hidden_size == 4096u);
    assert(manifest.routed_experts_per_layer == 256u);
    assert(manifest.routed_experts_per_token == 6u);
    assert(manifest.moe_intermediate_size == 2048u);
    assert(manifest.routed_records == 11008u);
    assert(manifest.routed_payload_bytes_per_record == 13369344u);
    assert(manifest.payload_bytes == UINT64_C(166878536440));

    waste_ds_v4_gate_a_manifest bad = manifest;
    bad.hidden_size++;
    assert(waste_ds_v4_gate_a_manifest_validate(&bad) != 0);
    bad = manifest;
    bad.routed_records--;
    assert(waste_ds_v4_gate_a_manifest_validate(&bad) != 0);
    assert(waste_ds_v4_gate_a_manifest_validate(NULL) != 0);
}

static void assert_fp4_plane(const waste_ds_v4_fp4_plane_layout *plane,
                             size_t rows, size_t cols)
{
    assert(plane->rows == rows);
    assert(plane->cols == cols);
    assert(plane->packed_weight_bytes == 4194304u);
    assert(plane->e8m0_scale_bytes == 262144u);
}

static void test_routed_payload(void)
{
    waste_ds_v4_routed_payload_layout layout;
    assert(waste_ds_v4_routed_payload_layout_init(4096u, 2048u, &layout) == 0);
    assert_fp4_plane(&layout.w1, 2048u, 4096u);
    assert_fp4_plane(&layout.w3, 2048u, 4096u);
    assert_fp4_plane(&layout.w2, 4096u, 2048u);
    assert(layout.payload_bytes == 13369344u);
    assert(layout.payload_bytes == WASTE_DS_V4_0731_ROUTED_PAYLOAD_BYTES);
    assert(layout.payload_bytes % 4096u == 0u);

    assert(waste_ds_v4_routed_payload_layout_init(4095u, 2048u, &layout) != 0);
    assert(waste_ds_v4_routed_payload_layout_init(4096u, 2047u, &layout) != 0);
    assert(waste_ds_v4_routed_payload_layout_init(4096u, 2048u, NULL) != 0);
}

static void test_resident_fp8(void)
{
    waste_ds_v4_fp8_plane_layout layout;
    assert(waste_ds_v4_fp8_plane_layout_init(1024u, 4096u, &layout) == 0);
    assert(layout.weight_bytes == 4194304u);
    assert(layout.scale_rows == 8u);
    assert(layout.scale_cols == 32u);
    assert(layout.scale_bytes == 256u);

    assert(waste_ds_v4_fp8_plane_layout_init(1023u, 4096u, &layout) != 0);
    assert(waste_ds_v4_fp8_plane_layout_init(1024u, 4095u, &layout) != 0);
    assert(waste_ds_v4_fp8_plane_layout_init(1024u, 4096u, NULL) != 0);
}

int main(void)
{
    test_gate_a_manifest();
    test_routed_payload();
    test_resident_fp8();
    puts("PASS DeepSeek V4 fail-closed container contract");
    puts("Gate A: 43 layers / hidden 4096 / 256 routed / top6");
    puts("routed expert native payload: 13,369,344 bytes (no header/order frozen)");
    puts("resident FP8 example: 1024x4096 E4M3 + 8x32 raw E8M0 scales");
    return 0;
}
