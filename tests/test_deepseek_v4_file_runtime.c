/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Actual-filesystem coverage for the resource-owning DeepSeek runtime layer.
 * This is still model-free: paths and offsets are explicit synthetic evidence.
 */
#include "../src/deepseek_v4_file_runtime.h"
#include "../src/quant/deepseek_v4_linear_ref.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W_BYTES 4194304u
#define S_BYTES 262144u
#define RECORD  13369344u
#define TRUNK   32768u
#define LAYERS  43u
#define EXPERTS 256u

#define RES_W_OFF    4096u
#define RES_S_OFF   24576u
#define RES_W_BYTES 16384u

#define TRUNK_PATH   "test_ds_v4_file_trunk.bin"
#define BANK_PATH    "test_ds_v4_file_bank.bin"
#define ALT_PATH     "test_ds_v4_file_alt.bin"
#define MISSING_PATH "definitely-not-a-deepseek-bank.bin"

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

static waste_ds_v4_manifest manifest_good(void)
{
    waste_ds_v4_manifest m;
    assert(waste_ds_v4_manifest_parse(GOOD, strlen(GOOD), &m) ==
           WASTE_DS_V4_MANIFEST_OK);
    return m;
}

static void write_trunk(const char *path, size_t n)
{
    uint8_t *buf = (uint8_t *)calloc(n ? n : 1u, 1u);
    assert(buf != NULL);
    if (n >= RES_W_OFF + RES_W_BYTES) {
        const uint8_t half = waste_ds_v4_e4m3_encode_ref(0.5f);
        memset(buf + RES_W_OFF, half, RES_W_BYTES);
    }
    if (n > RES_S_OFF)
        buf[RES_S_OFF] = 127u; /* E8M0 2^(127-127) = 1 */

    FILE *f = fopen(path, "wb");
    assert(f != NULL);
    assert(fwrite(buf, 1, n, f) == n);
    assert(fclose(f) == 0);
    free(buf);
}

static void poke_byte(const char *path, size_t off, uint8_t value)
{
    assert(off <= (size_t)LONG_MAX);
    FILE *f = fopen(path, "r+b");
    assert(f != NULL);
    assert(fseek(f, (long)off, SEEK_SET) == 0);
    assert(fwrite(&value, 1, 1, f) == 1);
    assert(fclose(f) == 0);
}

static void write_bank(const char *path, size_t n)
{
    FILE *f = fopen(path, "wb");
    assert(f != NULL);
    if (n) {
        assert(n - 1u <= (size_t)LONG_MAX);
        assert(fseek(f, (long)(n - 1u), SEEK_SET) == 0);
        assert(fputc(0, f) != EOF); /* sparse where the filesystem supports it */
    }
    assert(fclose(f) == 0);

    static const size_t pos[] = {
        0,
        W_BYTES,
        2u * W_BYTES,
        3u * W_BYTES,
        3u * W_BYTES + S_BYTES,
        3u * W_BYTES + 2u * S_BYTES
    };
    static const uint8_t val[] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
    for (size_t i = 0; i < sizeof pos / sizeof pos[0]; i++)
        if (pos[i] < n)
            poke_byte(path, pos[i], val[i]);
}

static void make_files(void)
{
    write_trunk(TRUNK_PATH, TRUNK);
    write_bank(BANK_PATH, RECORD);
}

static void clear_files(void)
{
    (void)remove(TRUNK_PATH);
    (void)remove(BANK_PATH);
    (void)remove(ALT_PATH);
    (void)remove(MISSING_PATH);
}

static void fill_specs(waste_ds_v4_file_bank_spec banks[LAYERS],
                       uint64_t offsets[EXPERTS])
{
    memset(offsets, 0, EXPERTS * sizeof *offsets);
    for (size_t i = 0; i < LAYERS; i++) {
        banks[i].path = BANK_PATH;
        banks[i].record_offsets = offsets;
        banks[i].record_count = EXPERTS;
    }
}

static void assert_closed(const waste_ds_v4_file_runtime *fr)
{
    assert(fr->runtime.trunk == NULL);
    assert(fr->runtime.routed_ready == 0);
    assert(fr->source.banks == NULL);
    assert(fr->source.record_offsets == NULL);
    assert(fr->trunk == NULL);
    assert(fr->bank_fds == NULL);
    assert(fr->bank_count == 0);
}

static void assert_record_sentinels(const waste_ds_v4_routed_record_view *view)
{
    assert(view->w1[0] == 0x11);
    assert(view->w3[0] == 0x22);
    assert(view->w2[0] == 0x33);
    assert(view->w1_scale[0] == 0x44);
    assert(view->w3_scale[0] == 0x55);
    assert(view->w2_scale[0] == 0x66);
}

static void test_file_open_and_fetch(void)
{
    waste_ds_v4_manifest manifest = manifest_good();
    waste_ds_v4_file_bank_spec banks[LAYERS];
    uint64_t offsets[EXPERTS];
    fill_specs(banks, offsets);

    waste_ds_v4_file_open_spec spec = {
        TRUNK_PATH, banks, LAYERS, 0, 0
    };
    waste_ds_v4_file_runtime fr;
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) == 0);
    assert(fr.bank_count == LAYERS);
    assert(fr.runtime.trunk == fr.trunk);
    assert(fr.runtime.trunk_bytes == TRUNK);
    assert(fr.trunk[RES_W_OFF] == waste_ds_v4_e4m3_encode_ref(0.5f));
    assert(fr.trunk[RES_S_OFF] == 127u);

    waste_ds_v4_routed_record_view view;
    assert(waste_ds_v4_runtime_routed_record(&fr.runtime, 12, 231, &view) == 0);
    assert_record_sentinels(&view);

    /* Offset/path metadata is borrowed only during open. Zero cache forces the
     * second request through the owned descriptor/source rather than hiding the
     * lifetime property behind a cache hit. */
    offsets[231] = RECORD;
    banks[12].path = MISSING_PATH;
    assert(waste_ds_v4_runtime_routed_record(&fr.runtime, 12, 231, &view) == 0);
    assert_record_sentinels(&view);
    assert(fr.runtime.cache.misses == 2);

    /* Resident arithmetic consumes the actual file-loaded trunk through the
     * same backend dispatch proved by the lower-level runtime test. */
    float x[128], y[128];
    for (size_t i = 0; i < 128u; i++)
        x[i] = 1.0f;
    assert(waste_ds_v4_runtime_resident_linear(&fr.runtime, 0, x, 1, y) == 0);
    assert(y[0] != 0.0f);
    assert(waste_ds_v4_manifest_step_refused(NULL) != 0);

    waste_ds_v4_file_runtime_close(&fr);
    assert_closed(&fr);
    waste_ds_v4_file_runtime_close(&fr); /* idempotent after zeroing */
    assert_closed(&fr);

    /* On Windows these removals also detect leaked CRT file handles. */
    assert(remove(BANK_PATH) == 0);
    assert(remove(TRUNK_PATH) == 0);
    make_files();
}

static void test_fail_closed_files(void)
{
    waste_ds_v4_manifest manifest = manifest_good();
    waste_ds_v4_file_bank_spec banks[LAYERS];
    uint64_t offsets[EXPERTS];
    fill_specs(banks, offsets);
    waste_ds_v4_file_open_spec spec = {
        TRUNK_PATH, banks, LAYERS, 0, 0
    };
    waste_ds_v4_file_runtime fr;

    assert(waste_ds_v4_file_runtime_open(NULL, &manifest, &spec) != 0);

    /* A valid manifest describes exactly TRUNK bytes; trailing or truncated
     * resident data is a different artifact and is refused before banks open. */
    write_trunk(ALT_PATH, TRUNK - 1u);
    spec.trunk_path = ALT_PATH;
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    write_trunk(ALT_PATH, TRUNK + 1u);
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    spec.trunk_path = TRUNK_PATH;

    /* Malformed ownership metadata is rejected before opening resources. */
    const size_t saved_count = banks[8].record_count;
    banks[8].record_count = EXPERTS - 1u;
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    banks[8].record_count = saved_count;

    /* Actual bank bytes, not a manifest/caller size guess, bound every extent. */
    write_bank(BANK_PATH, RECORD - 1u);
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    write_bank(BANK_PATH, RECORD);

    offsets[99] = 1u; /* exact-size bank: record would extend one byte past EOF */
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    offsets[99] = 0;

    /* Missing layer 17 exercises unwind after earlier native handles opened. */
    const char *saved_path = banks[17].path;
    banks[17].path = MISSING_PATH;
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    banks[17].path = saved_path;

    /* On Windows deletion fails if any earlier bank handle leaked. Recreate the
     * one shared synthetic bank after proving partial-open teardown. */
    assert(remove(BANK_PATH) == 0);
    write_bank(BANK_PATH, RECORD);

    /* Post-parse mutation is refused through the shared runtime validator. */
    waste_ds_v4_manifest mutated = manifest;
    mutated.gate_a.main_layers = 42;
    assert(waste_ds_v4_file_runtime_open(&fr, &mutated, &spec) != 0);
    assert_closed(&fr);

    /* The same object remains reusable after all failed opens. */
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) == 0);
    waste_ds_v4_file_runtime_close(&fr);
    assert_closed(&fr);
}

int main(void)
{
    clear_files();
    make_files();
    test_file_open_and_fetch();
    test_fail_closed_files();
    clear_files();
    puts("PASS DeepSeek V4 file runtime ownership");
    return 0;
}
