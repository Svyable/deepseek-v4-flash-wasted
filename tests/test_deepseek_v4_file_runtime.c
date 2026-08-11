/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Actual-filesystem coverage for the resource-owning DeepSeek runtime layer.
 * This is still model-free: paths and offsets are explicit synthetic evidence.
 */
#include "../src/deepseek_v4_runtime.h"

#include <assert.h>
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

#define TRUNK_PATH "test_ds_v4_file_trunk.bin"
#define BANK_PATH  "test_ds_v4_file_bank.bin"
#define SHORT_PATH "test_ds_v4_file_short.bin"

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

static void write_zeros(const char *path, size_t n)
{
    FILE *f = fopen(path, "wb");
    assert(f != NULL);
    uint8_t zero[65536] = {0};
    while (n) {
        const size_t take = n < sizeof zero ? n : sizeof zero;
        assert(fwrite(zero, 1, take, f) == take);
        n -= take;
    }
    assert(fclose(f) == 0);
}

static void poke_byte(const char *path, long off, uint8_t value)
{
    FILE *f = fopen(path, "r+b");
    assert(f != NULL);
    assert(fseek(f, off, SEEK_SET) == 0);
    assert(fwrite(&value, 1, 1, f) == 1);
    assert(fclose(f) == 0);
}

static void make_files(void)
{
    write_zeros(TRUNK_PATH, TRUNK);
    poke_byte(TRUNK_PATH, 4096, 0x7e);
    poke_byte(TRUNK_PATH, 24576, 0x7f);

    write_zeros(BANK_PATH, RECORD);
    poke_byte(BANK_PATH, 0, 0x11);
    poke_byte(BANK_PATH, W_BYTES, 0x22);
    poke_byte(BANK_PATH, 2u * W_BYTES, 0x33);
    poke_byte(BANK_PATH, 3u * W_BYTES, 0x44);
    poke_byte(BANK_PATH, 3u * W_BYTES + S_BYTES, 0x55);
    poke_byte(BANK_PATH, 3u * W_BYTES + 2u * S_BYTES, 0x66);
}

static void clear_files(void)
{
    remove(TRUNK_PATH);
    remove(BANK_PATH);
    remove(SHORT_PATH);
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
    assert(fr->source.banks == NULL);
    assert(fr->trunk == NULL);
    assert(fr->bank_fds == NULL);
    assert(fr->bank_count == 0);
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
    assert(fr.trunk[4096] == 0x7e);
    assert(fr.trunk[24576] == 0x7f);

    waste_ds_v4_routed_record_view view;
    assert(waste_ds_v4_runtime_routed_record(&fr.runtime, 12, 231, &view) == 0);
    assert(view.w1[0] == 0x11);
    assert(view.w3[0] == 0x22);
    assert(view.w2[0] == 0x33);
    assert(view.w1_scale[0] == 0x44);
    assert(view.w3_scale[0] == 0x55);
    assert(view.w2_scale[0] == 0x66);

    /* Offset arrays are copied by the positional source after file open. */
    offsets[231] = RECORD;
    assert(waste_ds_v4_runtime_routed_record(&fr.runtime, 12, 231, &view) == 0);
    assert(view.w1[0] == 0x11);

    waste_ds_v4_file_runtime_close(&fr);
    assert_closed(&fr);
    /* Idempotent close is part of partial-failure cleanup safety. */
    waste_ds_v4_file_runtime_close(&fr);
    assert_closed(&fr);
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

    /* A valid manifest describes exactly TRUNK bytes; trailing/truncated data
     * is a different artifact and is refused before any bank is opened. */
    write_zeros(SHORT_PATH, TRUNK - 1u);
    spec.trunk_path = SHORT_PATH;
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    spec.trunk_path = TRUNK_PATH;

    /* Missing layer 17 exercises unwind after earlier native handles opened. */
    const char *saved = banks[17].path;
    banks[17].path = "definitely-not-a-deepseek-bank.bin";
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    banks[17].path = saved;

    /* Actual bank bytes, not a manifest guess, bound every expert extent. */
    offsets[99] = RECORD;
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) != 0);
    assert_closed(&fr);
    offsets[99] = 0;

    /* Post-parse mutation is refused before file-driven allocation/I/O. */
    waste_ds_v4_manifest mutated = manifest;
    mutated.gate_a.main_layers = 42;
    assert(waste_ds_v4_file_runtime_open(&fr, &mutated, &spec) != 0);
    assert_closed(&fr);

    /* The object remains reusable after every failed partial open. */
    assert(waste_ds_v4_file_runtime_open(&fr, &manifest, &spec) == 0);
    waste_ds_v4_file_runtime_close(&fr);
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
