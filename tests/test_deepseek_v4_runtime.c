/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Model-free coverage for the runtime seam introduced after PR #21:
 * validated resident planes go through the normal backend dispatch, while
 * opaque routed records go through waste_ecache before the manifest binds
 * their six native planes. No transformer stepping is enabled here.
 */
#include "../src/deepseek_v4_runtime.h"
#include "../src/quant/deepseek_v4_linear_ref.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W_BYTES 4194304u
#define S_BYTES 262144u
#define RECORD  13369344u

#define TRUNK       32768u
#define RES_W_OFF    4096u
#define RES_S_OFF   24576u
#define RES_W_BYTES 16384u

static const char *GOOD =
"{\n"
"  \"family\": \"deepseek-v4-flash\",\n"
"  \"manifest_version\": 1,\n"
"  \"revision\": \"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\",\n"
"  \"geometry\": {\n"
"    \"main_layers\": 43,\n"
"    \"hidden_size\": 4096,\n"
"    \"routed_experts_per_layer\": 256,\n"
"    \"shared_experts_per_layer\": 1,\n"
"    \"routed_experts_per_token\": 6,\n"
"    \"moe_intermediate_size\": 2048,\n"
"    \"bootstrap_hash_layers\": 3,\n"
"    \"shards\": 48,\n"
"    \"tensors\": 72317,\n"
"    \"payload_bytes\": 166878536440,\n"
"    \"routed_records\": 11008,\n"
"    \"routed_payload_bytes_per_record\": 13369344\n"
"  },\n"
"  \"routed_record\": {\n"
"    \"record_bytes\": 13369344,\n"
"    \"w1_offset\": 0,\n"
"    \"w3_offset\": 4194304,\n"
"    \"w2_offset\": 8388608,\n"
"    \"w1_scale_offset\": 12582912,\n"
"    \"w3_scale_offset\": 12845056,\n"
"    \"w2_scale_offset\": 13107200\n"
"  },\n"
"  \"trunk\": {\n"
"    \"bytes\": 32768,\n"
"    \"resident\": [\n"
"      {\"name\": \"synthetic.resident\", \"rows\": 128, \"cols\": 128,\n"
"       \"weight_offset\": 4096, \"scale_offset\": 24576}\n"
"    ]\n"
"  }\n"
"}\n";

static waste_ds_v4_manifest parse_good(void)
{
    waste_ds_v4_manifest m;
    assert(waste_ds_v4_manifest_parse(GOOD, strlen(GOOD), &m) ==
           WASTE_DS_V4_MANIFEST_OK);
    return m;
}

typedef struct fake_bank {
    size_t record_bytes;
    int calls;
} fake_bank;

static int fake_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    fake_bank *bank = (fake_bank *)user;
    assert(bank != NULL && dst != NULL);
    assert(layer >= 0 && layer < 43);
    assert(expert >= 0 && expert < 256);
    memset(dst, 0, bank->record_bytes);

    /* One unmistakable byte at the beginning of every validated plane. */
    dst[0] = 0x11;
    dst[W_BYTES] = 0x22;
    dst[2u * W_BYTES] = 0x33;
    dst[3u * W_BYTES] = 0x44;
    dst[3u * W_BYTES + S_BYTES] = 0x55;
    dst[3u * W_BYTES + 2u * S_BYTES] = 0x66;
    bank->calls++;
    return 0;
}

static void assert_sentinels(const waste_ds_v4_routed_record_view *view)
{
    assert(view->w1[0] == 0x11);
    assert(view->w3[0] == 0x22);
    assert(view->w2[0] == 0x33);
    assert(view->w1_scale[0] == 0x44);
    assert(view->w3_scale[0] == 0x55);
    assert(view->w2_scale[0] == 0x66);
}

static void test_routed_cache_binding(void)
{
    waste_ds_v4_manifest manifest = parse_good();
    uint8_t *trunk = (uint8_t *)calloc(1, TRUNK);
    assert(trunk != NULL);

    fake_bank bank = { RECORD, 0 };
    waste_ds_v4_runtime runtime;
    assert(waste_ds_v4_runtime_init(&runtime, &manifest, trunk, TRUNK,
                                    RECORD, 0, fake_fetch, &bank) == 0);
    assert(runtime.routed_ready == 1);
    assert(runtime.cache.n_slots == 1);

    waste_ds_v4_routed_record_view view;
    assert(waste_ds_v4_runtime_routed_record(&runtime, 4, 7, &view) == 0);
    assert_sentinels(&view);
    assert(bank.calls == 1);

    /* Same key is a cache hit: placement changes I/O, not record bytes. */
    assert(waste_ds_v4_runtime_routed_record(&runtime, 4, 7, &view) == 0);
    assert_sentinels(&view);
    assert(bank.calls == 1);
    assert(runtime.cache.hits == 1);

    /* Untrusted ids are refused before reaching the storage callback. */
    assert(waste_ds_v4_runtime_routed_record(&runtime, -1, 7, &view) != 0);
    assert(waste_ds_v4_runtime_routed_record(&runtime, 43, 7, &view) != 0);
    assert(waste_ds_v4_runtime_routed_record(&runtime, 4, -1, &view) != 0);
    assert(waste_ds_v4_runtime_routed_record(&runtime, 4, 256, &view) != 0);
    assert(bank.calls == 1);

    /* One-slot cache must fetch a different expert and bind the same layout. */
    assert(waste_ds_v4_runtime_routed_record(&runtime, 4, 8, &view) == 0);
    assert_sentinels(&view);
    assert(bank.calls == 2);
    waste_ds_v4_runtime_free(&runtime);

    /* A zero cache budget uses the same callback/map but reads every time. */
    bank.calls = 0;
    assert(waste_ds_v4_runtime_init(&runtime, &manifest, trunk, TRUNK,
                                    0, 0, fake_fetch, &bank) == 0);
    assert(runtime.cache.n_slots == 0 && runtime.miss_buf != NULL);
    assert(waste_ds_v4_runtime_routed_record(&runtime, 2, 9, &view) == 0);
    assert(waste_ds_v4_runtime_routed_record(&runtime, 2, 9, &view) == 0);
    assert(bank.calls == 2);
    assert(runtime.cache.misses == 2);
    waste_ds_v4_runtime_free(&runtime);
    free(trunk);
}

static void test_resident_backend_binding(void)
{
    waste_ds_v4_manifest manifest = parse_good();
    uint8_t *trunk = (uint8_t *)calloc(1, TRUNK);
    assert(trunk != NULL);

    const uint8_t half = waste_ds_v4_e4m3_encode_ref(0.5f);
    memset(trunk + RES_W_OFF, half, RES_W_BYTES);
    trunk[RES_S_OFF] = 127u; /* UE8M0 2^(127-127) = 1 */

    float x[128], want[128], got[128];
    for (size_t i = 0; i < 128u; i++) x[i] = 1.0f;

    const uint8_t *weights = NULL, *scales = NULL;
    assert(waste_ds_v4_manifest_resident_plane(
               &manifest, 0, trunk, TRUNK, &weights, &scales) == 0);
    assert(weights == trunk + RES_W_OFF && scales == trunk + RES_S_OFF);
    assert(waste_ds_v4_fp8_linear_e8m0_ref(
               x, 1, 128, weights, scales, 128, want) == 0);

    /* No routed fetch callback is needed to bind/use resident weights. */
    waste_ds_v4_runtime runtime;
    assert(waste_ds_v4_runtime_init(&runtime, &manifest, trunk, TRUNK,
                                    0, 0, NULL, NULL) == 0);
    assert(runtime.routed_ready == 0);
    assert(waste_ds_v4_runtime_resident_linear(&runtime, 0, x, 1, got) == 0);
    assert(memcmp(got, want, sizeof got) == 0);
    assert(got[0] != 0.0f);

    waste_ds_v4_routed_record_view view;
    assert(waste_ds_v4_runtime_routed_record(&runtime, 0, 0, &view) != 0);

    /* Binding arithmetic must not accidentally turn on execution capability. */
    assert(waste_ds_v4_manifest_step_refused(NULL) != 0);
    waste_ds_v4_runtime_free(&runtime);
    free(trunk);
}

static void test_fail_closed_init(void)
{
    waste_ds_v4_manifest manifest = parse_good();
    uint8_t trunk[TRUNK] = {0};
    waste_ds_v4_runtime runtime;

    assert(waste_ds_v4_runtime_init(NULL, &manifest, trunk, sizeof trunk,
                                    0, 0, NULL, NULL) != 0);
    assert(waste_ds_v4_runtime_init(&runtime, &manifest, trunk, TRUNK - 1u,
                                    0, 0, NULL, NULL) != 0);

    waste_ds_v4_manifest blank;
    memset(&blank, 0, sizeof blank);
    assert(waste_ds_v4_runtime_init(&runtime, &blank, trunk, sizeof trunk,
                                    0, 0, NULL, NULL) != 0);
}

int main(void)
{
    test_routed_cache_binding();
    test_resident_backend_binding();
    test_fail_closed_init();
    puts("PASS DeepSeek V4 runtime binding");
    return 0;
}
