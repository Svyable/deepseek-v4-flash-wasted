/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * The DeepSeek-family manifest parser, tested the way the container format
 * is: one manifest that must parse, and a mutation for every silent seam that
 * would otherwise read plausible numbers out of the wrong bytes.
 *
 * Model-free. No checkpoint or container is touched.
 */
#include "../src/deepseek_v4_manifest.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Geometry from the pinned 0731 contract; offsets are this test's choice, and
 * deliberately not in plane order — the contract allows any non-overlapping
 * arrangement, and a parser that only accepts w1/w3/w2 order would be
 * enforcing a layout nobody has frozen. */
#define W_BYTES 4194304u   /* one FP4 plane's packed weights   */
#define S_BYTES 262144u    /* one FP4 plane's E8M0 scales      */
#define RECORD  13369344u

/* Two resident FP8 planes, both 128-multiples in each dimension. */
#define R0_W 4194304u      /* 1024 x 4096 E4M3                 */
#define R0_S 256u          /*    8 x   32 E8M0                 */
#define R1_W 8388608u      /* 2048 x 4096                      */
#define R1_S 512u          /*   16 x   32                      */
#define TRUNK 33554432u

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
"    \"bytes\": 33554432,\n"
"    \"resident\": [\n"
"      {\"name\": \"layers.0.attn.wq_a\", \"rows\": 1024, \"cols\": 4096,\n"
"       \"weight_offset\": 0, \"scale_offset\": 16777216},\n"
"      {\"name\": \"layers.0.attn.wo\", \"rows\": 2048, \"cols\": 4096,\n"
"       \"weight_offset\": 4194304, \"scale_offset\": 16777472}\n"
"    ]\n"
"  }\n"
"}\n";

static waste_ds_v4_manifest_status parse(const char *json,
                                         waste_ds_v4_manifest *out)
{
    return waste_ds_v4_manifest_parse(json, strlen(json), out);
}

/* Replace the first occurrence of `find` with `with`, so each mutation below
 * is one visible edit to the good manifest rather than a second copy of it. */
static char *mutate(const char *json, const char *find, const char *with)
{
    const char *at = strstr(json, find);
    assert(at != NULL && "mutation target must exist in the good manifest");
    size_t head = (size_t)(at - json);
    size_t flen = strlen(find), wlen = strlen(with);
    size_t total = strlen(json) - flen + wlen;
    char *out = (char *)malloc(total + 1);
    assert(out != NULL);
    memcpy(out, json, head);
    memcpy(out + head, with, wlen);
    strcpy(out + head + wlen, at + flen);
    return out;
}

static void refuses(const char *find, const char *with,
                    waste_ds_v4_manifest_status want, const char *label)
{
    char *json = mutate(GOOD, find, with);
    waste_ds_v4_manifest m;
    waste_ds_v4_manifest_status got = parse(json, &m);
    if (got != want) {
        fprintf(stderr, "FAIL %s: status %d (%s), expected %d (%s)\n",
                label, (int)got, waste_ds_v4_manifest_strerror(got),
                (int)want, waste_ds_v4_manifest_strerror(want));
        abort();
    }
    /* A refused manifest must be zeroed, not half-filled: a caller that
     * ignores the status must not find a usable-looking family string. */
    assert(m.family[0] == '\0');
    assert(m.resident_count == 0);
    assert(m.trunk_bytes == 0);
    assert(m.routed_map.record_bytes == 0);
    free(json);
}

static void test_good(void)
{
    waste_ds_v4_manifest m;
    waste_ds_v4_manifest_status st = parse(GOOD, &m);
    assert(st == WASTE_DS_V4_MANIFEST_OK);
    assert(strcmp(m.family, WASTE_DS_V4_FAMILY) == 0);
    assert(strcmp(m.revision, WASTE_DS_V4_0731_REVISION) == 0);
    assert(m.manifest_version == WASTE_DS_V4_MANIFEST_VERSION);

    assert(m.gate_a.main_layers == 43u);
    assert(m.gate_a.hidden_size == 4096u);
    assert(m.gate_a.payload_bytes == UINT64_C(166878536440));
    assert(waste_ds_v4_gate_a_manifest_validate(&m.gate_a) == 0);

    /* Sizes are derived from geometry, never read from the manifest. */
    assert(m.routed_layout.payload_bytes == RECORD);
    assert(m.routed_layout.w1.packed_weight_bytes == W_BYTES);
    assert(m.routed_layout.w1.e8m0_scale_bytes == S_BYTES);
    assert(m.routed_map.record_bytes == RECORD);

    assert(m.trunk_bytes == TRUNK);
    assert(m.resident_count == 2u);
    assert(strcmp(m.resident[0].name, "layers.0.attn.wq_a") == 0);
    assert(m.resident[0].layout.weight_bytes == R0_W);
    assert(m.resident[0].layout.scale_bytes == R0_S);
    assert(m.resident[0].layout.scale_rows == 8u && m.resident[0].layout.scale_cols == 32u);
    assert(m.resident[1].layout.weight_bytes == R1_W);
    assert(m.resident[1].layout.scale_bytes == R1_S);

    /* Parsing describes a container; it never enables one. */
    assert(m.generation_enabled == 0);
    const char *why = NULL;
    assert(waste_ds_v4_manifest_step_refused(&why) != 0);
    assert(why != NULL && *why != '\0');
    assert(waste_ds_v4_manifest_step_refused(NULL) != 0);
}

static void test_binding(void)
{
    waste_ds_v4_manifest m;
    assert(parse(GOOD, &m) == WASTE_DS_V4_MANIFEST_OK);

    unsigned char *record = (unsigned char *)calloc(1, RECORD);
    assert(record != NULL);
    waste_ds_v4_routed_record_view view;
    assert(waste_ds_v4_manifest_bind_routed_record(&m, record, RECORD, &view) == 0);
    assert(view.w1 == record + 0);
    assert(view.w3 == record + W_BYTES);
    assert(view.w2 == record + 2u * W_BYTES);
    assert(view.w1_scale == record + 3u * W_BYTES);
    assert(view.w3_scale == record + 3u * W_BYTES + S_BYTES);
    assert(view.w2_scale == record + 3u * W_BYTES + 2u * S_BYTES);
    /* A record of the wrong length is refused, not clamped. */
    assert(waste_ds_v4_manifest_bind_routed_record(&m, record, RECORD - 1u, &view) != 0);
    assert(waste_ds_v4_manifest_bind_routed_record(&m, NULL, RECORD, &view) != 0);
    free(record);

    unsigned char *trunk = (unsigned char *)calloc(1, TRUNK);
    assert(trunk != NULL);
    const uint8_t *w = NULL, *s = NULL;
    assert(waste_ds_v4_manifest_resident_plane(&m, 0, trunk, TRUNK, &w, &s) == 0);
    assert(w == trunk + 0 && s == trunk + 16777216u);
    assert(waste_ds_v4_manifest_resident_plane(&m, 1, trunk, TRUNK, &w, &s) == 0);
    assert(w == trunk + 4194304u && s == trunk + 16777472u);
    /* Past the end of the declared planes, and a region smaller than the
     * manifest declared, are both refusals rather than reads. */
    assert(waste_ds_v4_manifest_resident_plane(&m, 2, trunk, TRUNK, &w, &s) != 0);
    assert(waste_ds_v4_manifest_resident_plane(&m, 0, trunk, TRUNK - 1u, &w, &s) != 0);
    free(trunk);

    /* A zeroed struct is not a parsed manifest, and must not bind. */
    waste_ds_v4_manifest blank;
    memset(&blank, 0, sizeof blank);
    waste_ds_v4_routed_record_view v2;
    unsigned char one[16] = {0};
    assert(waste_ds_v4_manifest_bind_routed_record(&blank, one, sizeof one, &v2) != 0);
    assert(waste_ds_v4_manifest_bind_routed_record(NULL, one, sizeof one, &v2) != 0);
    assert(waste_ds_v4_manifest_resident_plane(&blank, 0, one, sizeof one, &w, &s) != 0);
}

static void test_family_probe(void)
{
    assert(waste_ds_v4_manifest_is_family(GOOD, strlen(GOOD)) == 1);

    /* The reason this probe exists: a Kimi v0 manifest must be identifiable
     * as not-ours without either parser tolerating the other's schema. */
    static const char *KIMI =
        "{\"version\": 6, \"arch\": \"KimiLinearForCausalLM\", \"layers\": 93}";
    assert(waste_ds_v4_manifest_is_family(KIMI, strlen(KIMI)) == 0);
    waste_ds_v4_manifest m;
    assert(parse(KIMI, &m) == WASTE_DS_V4_MANIFEST_E_FAMILY);

    static const char *NEARLY = "{\"family\": \"deepseek-v4-flash-0731\"}";
    assert(waste_ds_v4_manifest_is_family(NEARLY, strlen(NEARLY)) == 0);
    assert(parse(NEARLY, &m) == WASTE_DS_V4_MANIFEST_E_FAMILY);

    assert(waste_ds_v4_manifest_is_family("not json", 8) == 0);
    assert(waste_ds_v4_manifest_is_family(NULL, 4) == 0);
    assert(waste_ds_v4_manifest_is_family("{}", 2) == 0);
}

static void test_input_and_shape(void)
{
    waste_ds_v4_manifest m;
    assert(waste_ds_v4_manifest_parse(NULL, 4, &m) == WASTE_DS_V4_MANIFEST_E_ARG);
    assert(waste_ds_v4_manifest_parse(GOOD, 0, &m) == WASTE_DS_V4_MANIFEST_E_ARG);
    assert(waste_ds_v4_manifest_parse(GOOD, strlen(GOOD), NULL) ==
           WASTE_DS_V4_MANIFEST_E_ARG);
    /* A declared length that disagrees with the buffer is refused rather than
     * trusted; js.h reads to the NUL and would otherwise parse past it. */
    assert(waste_ds_v4_manifest_parse(GOOD, strlen(GOOD) - 1u, &m) ==
           WASTE_DS_V4_MANIFEST_E_ARG);

    static const char EMBEDDED[] = "{\"family\":\0\"deepseek-v4-flash\"}";
    assert(waste_ds_v4_manifest_parse(EMBEDDED, sizeof EMBEDDED - 1u, &m) ==
           WASTE_DS_V4_MANIFEST_E_ARG);

    assert(parse("[]", &m) == WASTE_DS_V4_MANIFEST_E_JSON);
    assert(parse("\"deepseek-v4-flash\"", &m) == WASTE_DS_V4_MANIFEST_E_JSON);
    assert(parse("{\"family\": ", &m) == WASTE_DS_V4_MANIFEST_E_JSON);
}

static void test_identity_mutations(void)
{
    refuses("\"family\": \"deepseek-v4-flash\"", "\"fam\": \"deepseek-v4-flash\"",
            WASTE_DS_V4_MANIFEST_E_FAMILY, "family key absent");
    refuses("\"deepseek-v4-flash\"", "\"kimi-linear\"",
            WASTE_DS_V4_MANIFEST_E_FAMILY, "foreign family");
    refuses("\"manifest_version\": 1", "\"manifest_version\": 2",
            WASTE_DS_V4_MANIFEST_E_VERSION, "future manifest version");
    refuses("\"manifest_version\": 1", "\"manifest_version\": \"1\"",
            WASTE_DS_V4_MANIFEST_E_VERSION, "stringly-typed version");
    /* One hex digit of the pinned revision. This is the mutation that decides
     * whether "pinned" means anything. */
    refuses("9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
            "9e165c30e2704aec5d9d593cce3eebd58bbef1cc",
            WASTE_DS_V4_MANIFEST_E_REVISION, "one-digit revision drift");
    refuses("\"revision\": \"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\"",
            "\"rev\": \"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\"",
            WASTE_DS_V4_MANIFEST_E_REVISION, "revision key absent");
}

static void test_capability_mutations(void)
{
    refuses("\"manifest_version\": 1,",
            "\"manifest_version\": 1, \"generation\": true,",
            WASTE_DS_V4_MANIFEST_E_GENERATION, "manifest claims generation");
    refuses("\"manifest_version\": 1,",
            "\"manifest_version\": 1, \"stepping\": true,",
            WASTE_DS_V4_MANIFEST_E_GENERATION, "manifest claims stepping");
    /* Not-a-boolean must not read as "not enabled". */
    refuses("\"manifest_version\": 1,",
            "\"manifest_version\": 1, \"generation\": \"false\",",
            WASTE_DS_V4_MANIFEST_E_GENERATION, "stringly-typed capability");
    refuses("\"manifest_version\": 1,",
            "\"manifest_version\": 1, \"generation\": {},",
            WASTE_DS_V4_MANIFEST_E_GENERATION, "object-typed capability");

    /* An explicit `false` is the one accepted form. */
    char *json = mutate(GOOD, "\"manifest_version\": 1,",
                        "\"manifest_version\": 1, \"generation\": false,");
    waste_ds_v4_manifest m;
    assert(parse(json, &m) == WASTE_DS_V4_MANIFEST_OK);
    assert(m.generation_enabled == 0);
    free(json);
}

static void test_geometry_mutations(void)
{
    refuses("\"main_layers\": 43", "\"main_layers\": 42",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "layer count drift");
    refuses("\"hidden_size\": 4096", "\"hidden_size\": 4095",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "hidden size drift");
    refuses("\"routed_experts_per_token\": 6", "\"routed_experts_per_token\": 8",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "top-k drift");
    refuses("\"moe_intermediate_size\": 2048", "\"moe_intermediate_size\": 2176",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "MoE intermediate drift");
    refuses("\"routed_records\": 11008", "\"routed_records\": 11007",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "record count drift");
    refuses("\"payload_bytes\": 166878536440", "\"payload_bytes\": 166878536441",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "payload byte drift");
    refuses("\"geometry\": {", "\"geometry\": [",
            WASTE_DS_V4_MANIFEST_E_JSON, "geometry is not an object");
    refuses("\"main_layers\": 43,", "",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "geometry key absent");

    /* Numbers that size or locate bytes are read strictly: a sign, a decimal
     * point, an exponent or a leading zero is a manifest we do not understand.
     * atof() would have turned every one of these into 4096. */
    refuses("\"hidden_size\": 4096", "\"hidden_size\": 4096.0",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "non-integer dimension");
    refuses("\"hidden_size\": 4096", "\"hidden_size\": 4.096e3",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "exponent dimension");
    refuses("\"hidden_size\": 4096", "\"hidden_size\": -4096",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "negative dimension");
    refuses("\"hidden_size\": 4096", "\"hidden_size\": 04096",
            WASTE_DS_V4_MANIFEST_E_GEOMETRY, "leading-zero dimension");
}

static void test_routed_map_mutations(void)
{
    /* The map is manifest-supplied, so overlap is the failure it must catch:
     * w3 starting one byte early makes both planes decode plausibly. */
    refuses("\"w3_offset\": 4194304", "\"w3_offset\": 4194303",
            WASTE_DS_V4_MANIFEST_E_ROUTED_MAP, "routed weight planes overlap");
    refuses("\"w2_scale_offset\": 13107200", "\"w2_scale_offset\": 13107199",
            WASTE_DS_V4_MANIFEST_E_ROUTED_MAP, "routed scale planes overlap");
    /* One byte short of holding the last plane. */
    refuses("\"w2_scale_offset\": 13107200", "\"w2_scale_offset\": 13107201",
            WASTE_DS_V4_MANIFEST_E_ROUTED_MAP, "last plane runs past the record");
    refuses("\"record_bytes\": 13369344", "\"record_bytes\": 13369343",
            WASTE_DS_V4_MANIFEST_E_ROUTED_MAP, "record smaller than the payload");
    refuses("\"w1_offset\": 0,", "",
            WASTE_DS_V4_MANIFEST_E_ROUTED_MAP, "routed offset absent");
    refuses("\"routed_record\": {", "\"routed_record\": [",
            WASTE_DS_V4_MANIFEST_E_JSON, "routed_record is not an object");

    /* A larger record is allowed: headers and alignment padding stay open,
     * which is the whole reason offsets are supplied rather than prescribed. */
    char *json = mutate(GOOD, "\"record_bytes\": 13369344",
                        "\"record_bytes\": 13373440");
    waste_ds_v4_manifest m;
    assert(parse(json, &m) == WASTE_DS_V4_MANIFEST_OK);
    assert(m.routed_map.record_bytes == 13373440u);
    free(json);

    /* And so is a different plane order, for the same reason. */
    json = mutate(GOOD, "\"w1_offset\": 0,\n    \"w3_offset\": 4194304,",
                  "\"w1_offset\": 4194304,\n    \"w3_offset\": 0,");
    assert(parse(json, &m) == WASTE_DS_V4_MANIFEST_OK);
    assert(m.routed_map.w1_offset == 4194304u && m.routed_map.w3_offset == 0u);
    free(json);
}

static void test_resident_mutations(void)
{
    refuses("\"rows\": 1024", "\"rows\": 1000",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "resident rows not a 128 multiple");
    refuses("\"cols\": 4096", "\"cols\": 4095",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "resident cols not a 128 multiple");
    refuses("\"rows\": 1024", "\"rows\": 0",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "zero-row resident plane");
    /* The second plane's weights starting one byte early overlaps the first;
     * both would still decode as E4M3. */
    refuses("\"weight_offset\": 4194304", "\"weight_offset\": 4194303",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "resident weight planes overlap");
    /* A scale grid landing inside another plane's weights is the same class
     * of bug and is 256 bytes wide, so nothing else would notice. */
    refuses("\"scale_offset\": 16777216", "\"scale_offset\": 1024",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "resident scale grid inside weights");
    refuses("\"scale_offset\": 16777472", "\"scale_offset\": 33554432",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "resident plane past the trunk");
    refuses("\"bytes\": 33554432", "\"bytes\": 16777216",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "trunk too small for its planes");
    refuses("\"bytes\": 33554432", "\"bytes\": 0",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "zero-byte trunk");
    refuses("\"name\": \"layers.0.attn.wo\"", "\"name\": \"layers.0.attn.wq_a\"",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "duplicate resident plane name");
    refuses("\"name\": \"layers.0.attn.wo\"", "\"name\": \"\"",
            WASTE_DS_V4_MANIFEST_E_RESIDENT, "empty resident plane name");
    refuses("\"resident\": [", "\"resident\": {",
            WASTE_DS_V4_MANIFEST_E_JSON, "resident is not an array");
    refuses("\"trunk\": {", "\"trunk\": [",
            WASTE_DS_V4_MANIFEST_E_JSON, "trunk is not an object");
}

int main(void)
{
    test_good();
    test_binding();
    test_family_probe();
    test_input_and_shape();
    test_identity_mutations();
    test_capability_mutations();
    test_geometry_mutations();
    test_routed_map_mutations();
    test_resident_mutations();
    printf("PASS DeepSeek V4 family manifest parser: "
           "geometry bound, offsets validated, stepping refused\n");
    return 0;
}
