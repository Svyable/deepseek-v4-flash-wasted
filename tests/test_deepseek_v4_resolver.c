/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * The evidence-backed family resolver.
 *
 * Resolution reads a container's own declaration and opens nothing, so almost
 * all of this runs with no filesystem at all: the document is built in memory
 * by `build_files`, and each mutation is one field of `files_opts` away from
 * the good one. That keeps a refusal attributable to the thing that changed.
 *
 * Two properties carry most of the weight. The declared *stride* form and the
 * explicit *table* form must materialize byte-identical offsets, because the
 * whole point of accepting both is that they mean the same thing. And a bank's
 * layer must come from its declaration rather than its position in the array,
 * which is checked by declaring all 43 banks in reverse and requiring the same
 * result.
 *
 * What is deliberately not here: an end-to-end open against a real, disjoint
 * bank set. One honest bank is 256 x 12.75 MiB = 3.42 GB and the full set is
 * ~147 GB, so a fabricated one would be either sparse-file trickery or a lie
 * about record layout. Real-file ownership is already covered one layer down
 * in tests/test_deepseek_v4_file_runtime.c; opening a real DeepSeek container
 * waits for real container bytes.
 */
#include "../src/deepseek_v4_resolver.h"

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAYERS  43u
#define EXPERTS 256u
#define RECORD  13369344ull

static const char *HEAD =
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
"       \"weight_offset\": 0, \"scale_offset\": 16777216}\n"
"    ]\n"
"  }";

/* ------------------------------------------------------------------ */
/* document builder                                                    */
/* ------------------------------------------------------------------ */

typedef struct sb {
    char *buf;
    size_t len, cap;
} sb;

static void sb_add(sb *s, const char *text)
{
    size_t n = strlen(text);
    if (s->len + n + 1u > s->cap) {
        size_t want = (s->cap ? s->cap * 2u : 4096u);
        while (want < s->len + n + 1u)
            want *= 2u;
        char *p = (char *)realloc(s->buf, want);
        assert(p != NULL);
        s->buf = p;
        s->cap = want;
    }
    memcpy(s->buf + s->len, text, n);
    s->len += n;
    s->buf[s->len] = 0;
}

static void sb_addf(sb *s, const char *fmt, ...)
{
    char tmp[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    assert(n > 0 && (size_t)n < sizeof tmp);
    sb_add(s, tmp);
}

typedef struct files_opts {
    const char *resident;      /* NULL for the default                      */
    const char *bank0_path;    /* NULL for the default                      */
    int use_stride;            /* declare base+stride instead of a table    */
    int omit_files;            /* no `files` section at all                 */
    int files_not_object;      /* `files` present but an array              */
    int banks_not_array;       /* `banks` present but an object             */
    int bank_delta;            /* declare LAYERS + delta banks              */
    int duplicate_layer;       /* declare layer 0 twice                     */
    int layer_out_of_range;    /* declare layer 43                          */
    int omit_layer_key;        /* bank 0 declares no `layer`                */
    int overlap;               /* bank 0 record 1 overlaps record 0         */
    int short_table;           /* bank 0 declares EXPERTS - 1 offsets       */
    int table_not_array;       /* bank 0 `record_offsets` is a number       */
    int both_forms;            /* bank 0 declares a table *and* a stride    */
    int half_stride;           /* bank 0 declares first_offset only         */
    int no_offsets;            /* bank 0 declares neither form              */
    unsigned long long stride; /* 0 for RECORD                              */
    int reverse;               /* emit banks in reverse array order         */
} files_opts;

static void emit_bank(sb *s, unsigned layer, const files_opts *o, int first)
{
    const int is0 = (layer == 0);
    unsigned long long stride = o->stride ? o->stride : RECORD;

    sb_add(s, first ? "\n      {" : ",\n      {");
    if (!(is0 && o->omit_layer_key))
        sb_addf(s, "\"layer\": %u, ",
                (is0 && o->layer_out_of_range) ? LAYERS
                /* Layer 0's bank claims layer 1's slot instead, so one layer
                 * is declared twice and another not at all. */
                : ((is0 && o->duplicate_layer) ? 1u : layer));
    const char *path = (is0 && o->bank0_path) ? o->bank0_path : NULL;
    if (path)
        sb_addf(s, "\"path\": \"%s\"", path);
    else
        sb_addf(s, "\"path\": \"banks/experts-%03u.bin\"", layer);

    if (is0 && o->no_offsets) {
        sb_add(s, "}");
        return;
    }
    if (is0 && o->half_stride) {
        sb_add(s, ", \"first_offset\": 0}");
        return;
    }
    if (is0 && o->table_not_array) {
        sb_add(s, ", \"record_offsets\": 0}");
        return;
    }

    if (o->use_stride && !(is0 && o->both_forms)) {
        sb_addf(s, ", \"first_offset\": 0, \"record_stride\": %llu}", stride);
        return;
    }

    sb_add(s, ", \"record_offsets\": [");
    unsigned n = EXPERTS - ((is0 && o->short_table) ? 1u : 0u);
    for (unsigned i = 0; i < n; i++) {
        unsigned long long off = (unsigned long long)i * RECORD;
        /* One record placed one byte inside its predecessor: plausible bytes,
         * no error anywhere downstream, and the reason this check exists. */
        if (is0 && o->overlap && i == 1u)
            off = RECORD - 1ull;
        sb_addf(s, "%s%llu", i ? ", " : "", off);
    }
    sb_add(s, "]");
    if (is0 && o->both_forms)
        sb_addf(s, ", \"first_offset\": 0, \"record_stride\": %llu", stride);
    sb_add(s, "}");
}

/* Caller frees. `*len_out` is strlen of the result. */
static char *build_files(const files_opts *o, size_t *len_out)
{
    sb s = {0};
    sb_add(&s, HEAD);
    if (!o->omit_files) {
        if (o->files_not_object) {
            sb_add(&s, ",\n  \"files\": []");
        } else {
            sb_add(&s, ",\n  \"files\": {\n");
            sb_addf(&s, "    \"resident\": \"%s\",\n",
                    o->resident ? o->resident : "trunk.bin");
            if (o->banks_not_array) {
                sb_add(&s, "    \"banks\": {}");
            } else {
                sb_add(&s, "    \"banks\": [");
                long count = (long)LAYERS + o->bank_delta;
                for (long k = 0; k < count; k++) {
                    unsigned layer = (unsigned)(o->reverse ? count - 1 - k : k);
                    emit_bank(&s, layer, o, k == 0);
                }
                sb_add(&s, "\n    ]");
            }
            sb_add(&s, "\n  }");
        }
    }
    sb_add(&s, "\n}\n");
    if (len_out)
        *len_out = s.len;
    return s.buf;
}

static waste_ds_v4_manifest manifest_from(const char *json, size_t len)
{
    waste_ds_v4_manifest m;
    waste_ds_v4_manifest_status st = waste_ds_v4_manifest_parse(json, len, &m);
    if (st != WASTE_DS_V4_MANIFEST_OK) {
        fprintf(stderr, "FAIL manifest parse: %s\n",
                waste_ds_v4_manifest_strerror(st));
        abort();
    }
    return m;
}

/* ------------------------------------------------------------------ */

static void assert_zeroed(const waste_ds_v4_resolved_files *f)
{
    assert(f->trunk_path == NULL);
    assert(f->bank_paths == NULL);
    assert(f->banks == NULL);
    assert(f->record_offsets == NULL);
    assert(f->bank_count == 0);
    assert(f->spec.trunk_path == NULL);
    assert(f->spec.bank_count == 0);
}

static void refuses(const files_opts *o, waste_ds_v4_resolve_status want,
                    const char *label)
{
    size_t len = 0;
    char *json = build_files(o, &len);
    waste_ds_v4_manifest m = manifest_from(json, len);
    waste_ds_v4_resolved_files files;
    waste_ds_v4_resolve_status got =
        waste_ds_v4_resolve_files(&files, &m, "root", json, len, 0, 0);
    if (got != want) {
        fprintf(stderr, "FAIL %s: status %d (%s), expected %d (%s)\n",
                label, (int)got, waste_ds_v4_resolve_strerror(got),
                (int)want, waste_ds_v4_resolve_strerror(want));
        abort();
    }
    /* A refused resolve must leave nothing usable behind, and must be safe to
     * free again. */
    assert_zeroed(&files);
    waste_ds_v4_resolved_files_free(&files);
    assert_zeroed(&files);
    free(json);
}

static void test_good_table_form(void)
{
    files_opts o = {0};
    size_t len = 0;
    char *json = build_files(&o, &len);
    waste_ds_v4_manifest m = manifest_from(json, len);

    waste_ds_v4_resolved_files f;
    assert(waste_ds_v4_resolve_files(&f, &m, "root", json, len, 4096u, 1) ==
           WASTE_DS_V4_RESOLVE_OK);

    assert(f.bank_count == LAYERS);
    assert(f.experts_per_layer == EXPERTS);
    assert(strcmp(f.trunk_path, "root/trunk.bin") == 0);
    assert(f.spec.trunk_path == f.trunk_path);
    assert(f.spec.banks == f.banks);
    assert(f.spec.bank_count == LAYERS);
    /* Carried through untouched: the resolver has no cache opinion. */
    assert(f.spec.cache_bytes == 4096u);
    assert(f.spec.cache_policy == 1);

    for (unsigned layer = 0; layer < LAYERS; layer++) {
        char want[64];
        snprintf(want, sizeof want, "root/banks/experts-%03u.bin", layer);
        assert(strcmp(f.banks[layer].path, want) == 0);
        assert(f.banks[layer].record_count == EXPERTS);
        for (unsigned i = 0; i < EXPERTS; i++)
            assert(f.banks[layer].record_offsets[i] == (uint64_t)i * RECORD);
    }

    waste_ds_v4_resolved_files_free(&f);
    assert_zeroed(&f);
    waste_ds_v4_resolved_files_free(&f);      /* idempotent */
    waste_ds_v4_resolved_files_free(NULL);
    free(json);
}

/* The two declared forms must mean the same thing, or accepting both is a way
 * to describe two different containers with one contract. */
static void test_stride_matches_table(void)
{
    files_opts table = {0};
    files_opts stride = {0};
    stride.use_stride = 1;

    size_t a_len = 0, b_len = 0;
    char *a_json = build_files(&table, &a_len);
    char *b_json = build_files(&stride, &b_len);
    waste_ds_v4_manifest a_m = manifest_from(a_json, a_len);
    waste_ds_v4_manifest b_m = manifest_from(b_json, b_len);

    waste_ds_v4_resolved_files a, b;
    assert(waste_ds_v4_resolve_files(&a, &a_m, "root", a_json, a_len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);
    assert(waste_ds_v4_resolve_files(&b, &b_m, "root", b_json, b_len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);

    assert(a.bank_count == b.bank_count);
    assert(memcmp(a.record_offsets, b.record_offsets,
                  (size_t)LAYERS * EXPERTS * sizeof *a.record_offsets) == 0);
    for (unsigned layer = 0; layer < LAYERS; layer++)
        assert(strcmp(a.banks[layer].path, b.banks[layer].path) == 0);

    waste_ds_v4_resolved_files_free(&a);
    waste_ds_v4_resolved_files_free(&b);
    free(a_json);
    free(b_json);
}

/* Expert identity comes from the declared layer, never from array position. */
static void test_declaration_order_is_not_identity(void)
{
    files_opts forward = {0};
    files_opts reversed = {0};
    reversed.reverse = 1;

    size_t a_len = 0, b_len = 0;
    char *a_json = build_files(&forward, &a_len);
    char *b_json = build_files(&reversed, &b_len);
    waste_ds_v4_manifest a_m = manifest_from(a_json, a_len);
    waste_ds_v4_manifest b_m = manifest_from(b_json, b_len);

    waste_ds_v4_resolved_files a, b;
    assert(waste_ds_v4_resolve_files(&a, &a_m, "root", a_json, a_len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);
    assert(waste_ds_v4_resolve_files(&b, &b_m, "root", b_json, b_len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);

    for (unsigned layer = 0; layer < LAYERS; layer++)
        assert(strcmp(a.banks[layer].path, b.banks[layer].path) == 0);

    waste_ds_v4_resolved_files_free(&a);
    waste_ds_v4_resolved_files_free(&b);
    free(a_json);
    free(b_json);
}

static void test_root_forms(void)
{
    files_opts o = {0};
    size_t len = 0;
    char *json = build_files(&o, &len);
    waste_ds_v4_manifest m = manifest_from(json, len);
    waste_ds_v4_resolved_files f;

    /* No root, empty root, and a root with a trailing separator all resolve,
     * and none of them produces a doubled separator. */
    assert(waste_ds_v4_resolve_files(&f, &m, NULL, json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);
    assert(strcmp(f.trunk_path, "trunk.bin") == 0);
    waste_ds_v4_resolved_files_free(&f);

    assert(waste_ds_v4_resolve_files(&f, &m, "", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);
    assert(strcmp(f.trunk_path, "trunk.bin") == 0);
    waste_ds_v4_resolved_files_free(&f);

    assert(waste_ds_v4_resolve_files(&f, &m, "root///", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);
    assert(strcmp(f.trunk_path, "root/trunk.bin") == 0);
    waste_ds_v4_resolved_files_free(&f);

    free(json);
}

static void test_arguments(void)
{
    files_opts o = {0};
    size_t len = 0;
    char *json = build_files(&o, &len);
    waste_ds_v4_manifest m = manifest_from(json, len);
    waste_ds_v4_resolved_files f;

    assert(waste_ds_v4_resolve_files(NULL, &m, "root", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_E_ARG);
    assert(waste_ds_v4_resolve_files(&f, &m, "root", NULL, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_E_ARG);
    assert(waste_ds_v4_resolve_files(&f, &m, "root", json, 0, 0, 0) ==
           WASTE_DS_V4_RESOLVE_E_ARG);
    /* A declared length that disagrees with the buffer is refused rather than
     * trusted: js.h reads to the NUL and would parse past it. */
    assert(waste_ds_v4_resolve_files(&f, &m, "root", json, len - 1u, 0, 0) ==
           WASTE_DS_V4_RESOLVE_E_ARG);
    assert(waste_ds_v4_resolve_files(&f, NULL, "root", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_E_MANIFEST);

    /* The same front door the file-runtime open uses: a parsed manifest is an
     * ordinary struct, and mutating it after parse must not survive here. */
    waste_ds_v4_manifest mutated = m;
    mutated.gate_a.routed_experts_per_layer = 128u;
    assert(waste_ds_v4_resolve_files(&f, &mutated, "root", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_E_MANIFEST);
    mutated = m;
    mutated.family[0] = 'X';
    assert(waste_ds_v4_resolve_files(&f, &mutated, "root", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_E_MANIFEST);

    free(json);
}

static void test_path_mutations(void)
{
    /* Every one of these becomes an open() argument, so each is a refusal
     * rather than something to normalize. */
    static const char *bad[] = {
        "/etc/passwd",                     /* absolute                       */
        "../../etc/passwd",                /* escapes the root               */
        "banks/../../outside.bin",         /* escapes after a good component */
        "banks/./experts.bin",             /* "." component                  */
        "banks//experts.bin",              /* empty component                */
        "banks/experts.bin/",              /* trailing separator             */
        "C:/windows/system32/x.bin",       /* drive-absolute                 */
        "C:relative.bin",                  /* drive-relative                 */
        "banks\\experts.bin",              /* backslash separator            */
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        files_opts o = {0};
        o.resident = bad[i];
        refuses(&o, WASTE_DS_V4_RESOLVE_E_PATH, "bad resident path");
        files_opts b = {0};
        b.bank0_path = bad[i];
        refuses(&b, WASTE_DS_V4_RESOLVE_E_PATH, "bad bank path");
    }

    files_opts empty = {0};
    empty.resident = "";
    refuses(&empty, WASTE_DS_V4_RESOLVE_E_PATH, "empty resident path");

    /* A nested but non-escaping path is legitimate and must still resolve. */
    files_opts nested = {0};
    nested.resident = "weights/resident/trunk.bin";
    size_t len = 0;
    char *json = build_files(&nested, &len);
    waste_ds_v4_manifest m = manifest_from(json, len);
    waste_ds_v4_resolved_files f;
    assert(waste_ds_v4_resolve_files(&f, &m, "root", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);
    assert(strcmp(f.trunk_path, "root/weights/resident/trunk.bin") == 0);
    waste_ds_v4_resolved_files_free(&f);
    free(json);
}

static void test_topology_mutations(void)
{
    files_opts o;

    memset(&o, 0, sizeof o); o.omit_files = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_JSON, "no files section");
    memset(&o, 0, sizeof o); o.files_not_object = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_JSON, "files is not an object");
    memset(&o, 0, sizeof o); o.banks_not_array = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_TOPOLOGY, "banks is not an array");

    memset(&o, 0, sizeof o); o.bank_delta = -1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_TOPOLOGY, "one bank short of the layers");
    memset(&o, 0, sizeof o); o.bank_delta = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_TOPOLOGY, "one bank too many");

    /* A duplicate claim leaves another layer undeclared; both halves of that
     * must be refused, and the duplicate is the one a gap check alone would
     * miss. */
    memset(&o, 0, sizeof o); o.duplicate_layer = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_TOPOLOGY, "two banks claim one layer");
    memset(&o, 0, sizeof o); o.layer_out_of_range = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_TOPOLOGY, "layer past the last main layer");
    memset(&o, 0, sizeof o); o.omit_layer_key = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_TOPOLOGY, "bank declares no layer");
}

static void test_offset_mutations(void)
{
    files_opts o;

    memset(&o, 0, sizeof o); o.overlap = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "records overlap by one byte");
    memset(&o, 0, sizeof o); o.short_table = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "one offset short of the experts");
    memset(&o, 0, sizeof o); o.table_not_array = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "record_offsets is not an array");
    memset(&o, 0, sizeof o); o.no_offsets = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "bank declares neither form");
    memset(&o, 0, sizeof o); o.half_stride = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "stride form missing its stride");
    memset(&o, 0, sizeof o); o.both_forms = 1;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "bank declares both forms");

    /* A stride below the record size overlaps every neighbour by construction,
     * and one byte is enough. */
    memset(&o, 0, sizeof o); o.use_stride = 1; o.stride = RECORD - 1ull;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "stride one byte under the record");
    memset(&o, 0, sizeof o); o.use_stride = 1; o.stride = 1ull;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "degenerate one-byte stride");

    /* Past the signed-64-bit positional-I/O range WASTE's pread seam accepts
     * on POSIX and Windows alike. */
    memset(&o, 0, sizeof o); o.use_stride = 1;
    o.stride = 9223372036854775807ull / 8ull;
    refuses(&o, WASTE_DS_V4_RESOLVE_E_OFFSETS, "stride walks past INT64_MAX");

    /* A stride above the record size is legitimate: padding and alignment are
     * exactly what is not frozen yet. */
    memset(&o, 0, sizeof o); o.use_stride = 1; o.stride = RECORD + 4096ull;
    size_t len = 0;
    char *json = build_files(&o, &len);
    waste_ds_v4_manifest m = manifest_from(json, len);
    waste_ds_v4_resolved_files f;
    assert(waste_ds_v4_resolve_files(&f, &m, "root", json, len, 0, 0) ==
           WASTE_DS_V4_RESOLVE_OK);
    assert(f.banks[0].record_offsets[1] == RECORD + 4096ull);
    waste_ds_v4_resolved_files_free(&f);
    free(json);
}

/* The strict number rules the manifest parser owns must also hold here: these
 * are offsets, and atof() would turn every one of them into something. */
static void test_strict_numbers(void)
{
    static const char *bad[] = {"0.0", "1e3", "-1", "007"};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        files_opts o = {0};
        size_t len = 0;
        char *json = build_files(&o, &len);
        /* Replace the first table entry, which is the literal "0" that opens
         * every bank's record_offsets array. */
        const char *at = strstr(json, "\"record_offsets\": [0,");
        assert(at != NULL);
        size_t head = (size_t)(at - json) + strlen("\"record_offsets\": [");
        sb s = {0};
        char keep = json[head];
        json[head] = 0;
        sb_add(&s, json);
        json[head] = keep;
        sb_add(&s, bad[i]);
        sb_add(&s, json + head + 1u);

        waste_ds_v4_manifest m = manifest_from(s.buf, s.len);
        waste_ds_v4_resolved_files f;
        waste_ds_v4_resolve_status got =
            waste_ds_v4_resolve_files(&f, &m, "root", s.buf, s.len, 0, 0);
        if (got != WASTE_DS_V4_RESOLVE_E_OFFSETS) {
            fprintf(stderr, "FAIL loose offset %s accepted as %d\n",
                    bad[i], (int)got);
            abort();
        }
        free(s.buf);
        free(json);
    }
}

/* The one-step entry point must not leak the resolved description, must not
 * attempt an open when the declaration is refused, and must report an I/O
 * failure as such rather than as a bad declaration. */
static void test_resolver_open(void)
{
    files_opts bad = {0};
    bad.overlap = 1;
    size_t len = 0;
    char *json = build_files(&bad, &len);
    waste_ds_v4_manifest m = manifest_from(json, len);

    waste_ds_v4_file_runtime fr;
    int open_rc = 12345;
    assert(waste_ds_v4_resolver_open(&fr, &m, "root", json, len, 0, 0,
                                     &open_rc) ==
           WASTE_DS_V4_RESOLVE_E_OFFSETS);
    /* Untouched: a refused declaration never reaches the filesystem. */
    assert(open_rc == 0);
    assert(fr.bank_fds == NULL && fr.bank_count == 0 && fr.trunk == NULL);
    waste_ds_v4_file_runtime_close(&fr);
    free(json);

    /* A sound declaration whose files do not exist is an open failure, and
     * every partially acquired resource still unwinds. */
    files_opts good = {0};
    json = build_files(&good, &len);
    m = manifest_from(json, len);
    open_rc = 0;
    assert(waste_ds_v4_resolver_open(&fr, &m, "no-such-container-root", json,
                                     len, 0, 0, &open_rc) ==
           WASTE_DS_V4_RESOLVE_E_OPEN);
    assert(open_rc != 0);
    assert(fr.bank_fds == NULL && fr.bank_count == 0 && fr.trunk == NULL);
    assert(fr.source.banks == NULL);
    waste_ds_v4_file_runtime_close(&fr);
    waste_ds_v4_file_runtime_close(&fr);      /* idempotent */

    /* open_rc is optional. */
    assert(waste_ds_v4_resolver_open(&fr, &m, "no-such-container-root", json,
                                     len, 0, 0, NULL) ==
           WASTE_DS_V4_RESOLVE_E_OPEN);
    assert(waste_ds_v4_resolver_open(NULL, &m, "root", json, len, 0, 0, NULL) ==
           WASTE_DS_V4_RESOLVE_E_ARG);
    free(json);

    /* Resolution describes a container; it never enables one. */
    assert(waste_ds_v4_manifest_step_refused(NULL) != 0);
}

int main(void)
{
    test_good_table_form();
    test_stride_matches_table();
    test_declaration_order_is_not_identity();
    test_root_forms();
    test_arguments();
    test_path_mutations();
    test_topology_mutations();
    test_offset_mutations();
    test_strict_numbers();
    test_resolver_open();
    printf("PASS DeepSeek V4 family resolver: %u banks x %u records resolved "
           "from declaration, paths confined, records disjoint\n",
           LAYERS, EXPERTS);
    return 0;
}
