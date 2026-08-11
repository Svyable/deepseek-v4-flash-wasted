/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Model-free coverage for the DeepSeek runtime seam: validated resident planes
 * go through the normal backend dispatch, opaque routed records go through
 * waste_ecache, and explicit positional storage identities are exact-read and
 * fail-closed. No transformer stepping is enabled here.
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

#define LAYERS  43u
#define EXPERTS 256u

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

typedef struct fake_reader {
    size_t max_chunk;
    int calls;
    int zero_after;
    int over_report;
    uint64_t first_off;
} fake_reader;

static void write_generated_sentinels(uint8_t *dst,
                                      size_t n,
                                      uint64_t off)
{
    static const uint64_t pos[] = {
        0,
        W_BYTES,
        2u * W_BYTES,
        3u * W_BYTES,
        3u * W_BYTES + S_BYTES,
        3u * W_BYTES + 2u * S_BYTES
    };
    static const uint8_t value[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };

    memset(dst, 0, n);
    const uint64_t rel = off % (uint64_t)RECORD;
    for (size_t i = 0; i < sizeof pos / sizeof pos[0]; i++) {
        if (pos[i] >= rel && pos[i] - rel < (uint64_t)n)
            dst[(size_t)(pos[i] - rel)] = value[i];
    }
}

static int64_t generated_read_at(void *user,
                                 void *dst,
                                 size_t n,
                                 uint64_t off)
{
    fake_reader *reader = (fake_reader *)user;
    assert(reader != NULL && dst != NULL);
    if (reader->calls == 0)
        reader->first_off = off;
    reader->calls++;

    if (reader->zero_after > 0 && reader->calls > reader->zero_after)
        return 0;
    if (reader->over_report)
        return n == SIZE_MAX ? -1 : (int64_t)(n + 1u);

    size_t take = n;
    if (reader->max_chunk > 0 && take > reader->max_chunk)
        take = reader->max_chunk;
    write_generated_sentinels((uint8_t *)dst, take, off);
    return (int64_t)take;
}

static uint64_t *make_explicit_offsets(waste_ds_v4_positional_bank *banks,
                                       fake_reader *reader)
{
    uint64_t *offsets = (uint64_t *)malloc(
        (size_t)LAYERS * EXPERTS * sizeof *offsets);
    assert(offsets != NULL);

    const uint64_t bank_bytes = (uint64_t)RECORD * EXPERTS;
    for (size_t layer = 0; layer < LAYERS; layer++) {
        uint64_t *row = offsets + layer * EXPERTS;
        for (size_t expert = 0; expert < EXPERTS; expert++) {
            /* 37 is odd, therefore this is a permutation modulo 256. The
             * physical order is intentionally not expert-id order. */
            const size_t slot = (expert * 37u + layer * 11u) & 255u;
            row[expert] = (uint64_t)slot * RECORD;
        }
        banks[layer].read_at = generated_read_at;
        banks[layer].read_user = reader;
        banks[layer].bytes = bank_bytes;
        banks[layer].record_offsets = row;
        banks[layer].record_count = EXPERTS;
    }
    return offsets;
}

static void test_positional_source_binding(void)
{
    waste_ds_v4_manifest manifest = parse_good();
    uint8_t *trunk = (uint8_t *)calloc(1, TRUNK);
    assert(trunk != NULL);

    fake_reader reader;
    memset(&reader, 0, sizeof reader);
    reader.max_chunk = 65537u; /* deliberately not a record/page divisor */

    waste_ds_v4_positional_bank banks[LAYERS];
    memset(banks, 0, sizeof banks);
    uint64_t *offsets = make_explicit_offsets(banks, &reader);
    const uint64_t expected = offsets[4u * EXPERTS + 7u];

    waste_ds_v4_positional_source source;
    assert(waste_ds_v4_positional_source_init(
               &source, &manifest, banks, LAYERS) == 0);
    assert(source.bank_count == LAYERS);
    assert(source.experts_per_layer == EXPERTS);
    assert(source.record_bytes == RECORD);

    /* Placement identity is frozen at init, not borrowed from caller memory. */
    offsets[4u * EXPERTS + 7u] = expected == 0 ? RECORD : 0;

    /* Same record size is not enough. Swapping two valid plane offsets would
     * produce plausible but wrong arithmetic if a source could be rebound to
     * an equally-sized manifest with a different record map. */
    waste_ds_v4_manifest reordered = manifest;
    const size_t tmp = reordered.routed_map.w1_offset;
    reordered.routed_map.w1_offset = reordered.routed_map.w3_offset;
    reordered.routed_map.w3_offset = tmp;
    assert(waste_ds_v4_routed_record_map_validate(
               &reordered.routed_map, &reordered.routed_layout) == 0);

    waste_ds_v4_runtime runtime;
    assert(waste_ds_v4_runtime_init_positional(
               &runtime, &reordered, trunk, TRUNK, RECORD, 0, &source) != 0);
    assert(waste_ds_v4_runtime_init_positional(
               &runtime, &manifest, trunk, TRUNK, RECORD, 0, &source) == 0);

    waste_ds_v4_routed_record_view view;
    assert(waste_ds_v4_runtime_routed_record(&runtime, 4, 7, &view) == 0);
    assert_sentinels(&view);
    assert(reader.first_off == expected);
    assert(reader.calls > 1); /* exact-read loop consumed legal short reads */

    const int calls = reader.calls;
    assert(waste_ds_v4_runtime_routed_record(&runtime, 4, 7, &view) == 0);
    assert_sentinels(&view);
    assert(reader.calls == calls); /* cache placement preserves exact bytes */

    waste_ds_v4_runtime_free(&runtime);
    waste_ds_v4_positional_source_free(&source);
    free(offsets);
    free(trunk);
}

static void test_positional_source_refusals(void)
{
    waste_ds_v4_manifest manifest = parse_good();
    fake_reader reader;
    memset(&reader, 0, sizeof reader);
    reader.max_chunk = 4096u;

    waste_ds_v4_positional_bank banks[LAYERS];
    memset(banks, 0, sizeof banks);
    uint64_t *offsets = make_explicit_offsets(banks, &reader);
    waste_ds_v4_positional_source source;

    assert(waste_ds_v4_positional_source_init(
               &source, &manifest, banks, LAYERS - 1u) != 0);

    waste_ds_v4_read_at_fn saved_read = banks[3].read_at;
    banks[3].read_at = NULL;
    assert(waste_ds_v4_positional_source_init(
               &source, &manifest, banks, LAYERS) != 0);
    banks[3].read_at = saved_read;

    const size_t saved_count = banks[3].record_count;
    banks[3].record_count = EXPERTS - 1u;
    assert(waste_ds_v4_positional_source_init(
               &source, &manifest, banks, LAYERS) != 0);
    banks[3].record_count = saved_count;

    const uint64_t saved_off = offsets[3u * EXPERTS + 9u];
    offsets[3u * EXPERTS + 9u] = banks[3].bytes;
    assert(waste_ds_v4_positional_source_init(
               &source, &manifest, banks, LAYERS) != 0);
    offsets[3u * EXPERTS + 9u] = saved_off;

    assert(waste_ds_v4_positional_source_init(
               &source, &manifest, banks, LAYERS) == 0);

    uint8_t *record = (uint8_t *)malloc(RECORD);
    assert(record != NULL);
    assert(waste_ds_v4_positional_fetch(&source, -1, 0, record) != 0);
    assert(waste_ds_v4_positional_fetch(&source, 0, -1, record) != 0);
    assert(waste_ds_v4_positional_fetch(&source, LAYERS, 0, record) != 0);
    assert(waste_ds_v4_positional_fetch(&source, 0, EXPERTS, record) != 0);

    reader.calls = 0;
    reader.zero_after = 1;
    assert(waste_ds_v4_positional_fetch(&source, 0, 0, record) != 0);

    reader.calls = 0;
    reader.zero_after = 0;
    reader.over_report = 1;
    assert(waste_ds_v4_positional_fetch(&source, 0, 0, record) != 0);

    reader.over_report = 0;
    free(record);
    waste_ds_v4_positional_source_free(&source);
    free(offsets);
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
    test_positional_source_binding();
    test_positional_source_refusals();
    test_resident_backend_binding();
    test_fail_closed_init();
    puts("PASS DeepSeek V4 runtime binding");
    return 0;
}
