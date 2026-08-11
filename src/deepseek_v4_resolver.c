/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
#include "deepseek_v4_resolver.h"

#include "deepseek_v4_json_strict.h"
#include "deepseek_v4_runtime.h"

#include <stdlib.h>
#include <string.h>

const char *waste_ds_v4_resolve_strerror(waste_ds_v4_resolve_status status)
{
    switch (status) {
    case WASTE_DS_V4_RESOLVE_OK:
        return "ok";
    case WASTE_DS_V4_RESOLVE_E_ARG:
        return "resolver arguments are missing, oversize, or not NUL-terminated";
    case WASTE_DS_V4_RESOLVE_E_MANIFEST:
        return "manifest failed runtime revalidation before resolution";
    case WASTE_DS_V4_RESOLVE_E_JSON:
        return "container document has no usable files section";
    case WASTE_DS_V4_RESOLVE_E_PATH:
        return "a declared path is absolute, escapes the container root, or is malformed";
    case WASTE_DS_V4_RESOLVE_E_TOPOLOGY:
        return "declared bank layers do not cover every main layer exactly once";
    case WASTE_DS_V4_RESOLVE_E_OFFSETS:
        return "declared expert offsets are the wrong count, overlap, or are out of range";
    case WASTE_DS_V4_RESOLVE_E_MEMORY:
        return "out of memory while resolving the file set";
    case WASTE_DS_V4_RESOLVE_E_OPEN:
        return "resolved file set could not be opened";
    }
    return "unknown resolver status";
}

/* ------------------------------------------------------------------ */
/* path safety                                                         */
/*                                                                     */
/* A manifest is untrusted input, and these strings become open()       */
/* arguments. The rule is deliberately narrow: a declared path is a     */
/* relative path made of ordinary components under the container root,  */
/* and anything else is refused rather than normalized. Normalizing is  */
/* how "a/../../etc/passwd" becomes a bug report.                       */
/* ------------------------------------------------------------------ */

static int path_rel_ok(const char *p, size_t n)
{
    if (!p || n == 0)
        return 0;
    /* Absolute, in either separator convention, and Windows drive-relative
     * ("C:x") — which is neither absolute nor safely relative. */
    if (p[0] == '/' || p[0] == '\\')
        return 0;
    if (n >= 2 && p[1] == ':')
        return 0;
    if (memchr(p, '\\', n) != NULL)
        return 0;

    size_t start = 0;
    for (size_t i = 0; i <= n; i++) {
        if (i != n && p[i] != '/')
            continue;
        size_t len = i - start;
        /* Empty covers a leading, trailing or doubled separator. "." and ".."
         * are refused outright: one is noise, the other is an escape. */
        if (len == 0)
            return 0;
        if (len == 1 && p[start] == '.')
            return 0;
        if (len == 2 && p[start] == '.' && p[start + 1] == '.')
            return 0;
        start = i + 1;
    }
    return 1;
}

/* Join `root` and a bounded relative path into a fresh NUL-terminated string. */
static char *path_join(const char *root, const char *rel, size_t rel_len)
{
    size_t root_len = (root && *root) ? strlen(root) : 0;
    /* A trailing separator on the root would otherwise produce "dir//file",
     * which opens fine but reads badly in every error message. */
    while (root_len > 0 && root[root_len - 1] == '/')
        root_len--;

    size_t total = rel_len + (root_len ? root_len + 1u : 0u);
    if (total < rel_len)
        return NULL;
    char *out = (char *)malloc(total + 1u);
    if (!out)
        return NULL;
    size_t at = 0;
    if (root_len) {
        memcpy(out, root, root_len);
        at = root_len;
        out[at++] = '/';
    }
    memcpy(out + at, rel, rel_len);
    out[at + rel_len] = 0;
    return out;
}

static waste_ds_v4_resolve_status resolve_member_path(
    const js_doc *d, int obj, const char *key, const char *root, char **out)
{
    const char *text = NULL;
    size_t len = 0;
    if (dsjs_member_text(d, obj, key, &text, &len) != 0)
        return WASTE_DS_V4_RESOLVE_E_PATH;
    if (!path_rel_ok(text, len))
        return WASTE_DS_V4_RESOLVE_E_PATH;
    char *joined = path_join(root, text, len);
    if (!joined)
        return WASTE_DS_V4_RESOLVE_E_MEMORY;
    *out = joined;
    return WASTE_DS_V4_RESOLVE_OK;
}

/* ------------------------------------------------------------------ */
/* offsets                                                             */
/* ------------------------------------------------------------------ */

/* Every record must be representable as a signed 64-bit positional read, the
 * same range waste_pread accepts on POSIX and Windows alike. */
static int offset_in_io_range(uint64_t off, uint64_t rec)
{
    return off <= (uint64_t)INT64_MAX && rec <= (uint64_t)INT64_MAX - off;
}

/* Records inside one bank may be arranged in any order — Gate A has not frozen
 * one — but they may not overlap. Two experts sharing bytes means one expert's
 * weights silently serve another, which produces plausible numbers and no
 * error anywhere downstream. The positional source bounds each record against
 * the bank but deliberately does not compare records to each other, so this is
 * the layer that has to.
 *
 * O(n^2) over 256 records is ~32k comparisons per bank. Sorting would be
 * asymptotically better and materially harder to read for no measurable gain
 * at a size the contract fixes. */
static waste_ds_v4_resolve_status offsets_disjoint(const uint64_t *offsets,
                                                   size_t count,
                                                   uint64_t rec)
{
    for (size_t i = 0; i < count; i++)
        for (size_t j = i + 1; j < count; j++) {
            uint64_t a = offsets[i], b = offsets[j];
            if (a < b + rec && b < a + rec)
                return WASTE_DS_V4_RESOLVE_E_OFFSETS;
        }
    return WASTE_DS_V4_RESOLVE_OK;
}

/* One bank declares its records either as an explicit table or as a base and
 * stride. Exactly one form, never both and never neither: a bank carrying both
 * is ambiguous, and silently preferring one is how the other stops being read
 * without anybody noticing. */
static waste_ds_v4_resolve_status resolve_bank_offsets(
    const js_doc *d, int bank, size_t experts, uint64_t rec, uint64_t *dst)
{
    const int table = js_get(d, bank, "record_offsets");
    const int has_table = table >= 0;
    const int has_base = js_get(d, bank, "first_offset") >= 0;
    const int has_stride = js_get(d, bank, "record_stride") >= 0;

    /* Exactly one form. A bank carrying both is ambiguous, and a bank
     * carrying half of the stride form is a typo that must not fall through
     * to "no offsets declared". */
    if (has_table == (has_base || has_stride))
        return WASTE_DS_V4_RESOLVE_E_OFFSETS;
    if (!has_table && !(has_base && has_stride))
        return WASTE_DS_V4_RESOLVE_E_OFFSETS;

    if (has_table) {
        /* Present but not an array is malformed, not absent — otherwise it
         * would read as "no table" after we already rejected the stride form
         * for being absent. */
        if (table >= d->n || d->tok[table].type != JS_ARR)
            return WASTE_DS_V4_RESOLVE_E_OFFSETS;
        if ((size_t)js_size(d, table) != experts)
            return WASTE_DS_V4_RESOLVE_E_OFFSETS;
        for (size_t i = 0; i < experts; i++) {
            uint64_t off = 0;
            if (dsjs_u64(d, js_at(d, table, (int)i), &off) != 0 ||
                !offset_in_io_range(off, rec))
                return WASTE_DS_V4_RESOLVE_E_OFFSETS;
            dst[i] = off;
        }
        return offsets_disjoint(dst, experts, rec);
    }

    uint64_t base = 0, stride = 0;
    if (dsjs_member_u64(d, bank, "first_offset", &base) != 0 ||
        dsjs_member_u64(d, bank, "record_stride", &stride) != 0)
        return WASTE_DS_V4_RESOLVE_E_OFFSETS;
    /* A stride below the record size overlaps by construction. Checking it
     * here rather than relying on offsets_disjoint keeps the refusal specific,
     * and costs nothing. */
    if (stride < rec)
        return WASTE_DS_V4_RESOLVE_E_OFFSETS;

    uint64_t off = base;
    for (size_t i = 0; i < experts; i++) {
        if (!offset_in_io_range(off, rec))
            return WASTE_DS_V4_RESOLVE_E_OFFSETS;
        dst[i] = off;
        if (i + 1 < experts) {
            if (off > UINT64_MAX - stride)
                return WASTE_DS_V4_RESOLVE_E_OFFSETS;
            off += stride;
        }
    }
    /* A uniform stride at or above the record size cannot overlap, so the
     * pairwise pass is redundant here — but running it means the two declared
     * forms are held to one rule rather than two, and the test that feeds the
     * same bank both ways is comparing like with like. */
    return offsets_disjoint(dst, experts, rec);
}

/* ------------------------------------------------------------------ */

void waste_ds_v4_resolved_files_free(waste_ds_v4_resolved_files *files)
{
    if (!files)
        return;
    if (files->bank_paths) {
        for (size_t i = 0; i < files->bank_count; i++)
            free(files->bank_paths[i]);
        free(files->bank_paths);
    }
    free(files->banks);
    free(files->record_offsets);
    free(files->trunk_path);
    memset(files, 0, sizeof *files);
}

waste_ds_v4_resolve_status waste_ds_v4_resolve_files(
    waste_ds_v4_resolved_files *out,
    const waste_ds_v4_manifest *manifest,
    const char *root_dir,
    const char *json,
    size_t len,
    size_t cache_bytes,
    int cache_policy)
{
    if (!out)
        return WASTE_DS_V4_RESOLVE_E_ARG;
    memset(out, 0, sizeof *out);
    if (!dsjs_input_ok(json, len))
        return WASTE_DS_V4_RESOLVE_E_ARG;
    /* Same front door the file-runtime open uses: a parsed manifest is an
     * ordinary C struct and can be mutated between parse and resolve. */
    if (waste_ds_v4_runtime_manifest_validate(manifest) != 0)
        return WASTE_DS_V4_RESOLVE_E_MANIFEST;

    const size_t layers = (size_t)manifest->gate_a.main_layers;
    const size_t experts = (size_t)manifest->gate_a.routed_experts_per_layer;
    const uint64_t rec = (uint64_t)manifest->routed_map.record_bytes;
    if (layers == 0 || experts == 0 || rec == 0 ||
        experts > SIZE_MAX / layers)
        return WASTE_DS_V4_RESOLVE_E_MANIFEST;

    js_doc d;
    if (js_parse(&d, json) != 0) {
        js_free(&d);
        return WASTE_DS_V4_RESOLVE_E_JSON;
    }

    waste_ds_v4_resolve_status status = WASTE_DS_V4_RESOLVE_OK;
    unsigned char *seen = NULL;

    if (d.n <= 0 || d.tok[0].type != JS_OBJ) {
        status = WASTE_DS_V4_RESOLVE_E_JSON;
        goto done;
    }
    int files = js_get(&d, 0, "files");
    if (files < 0 || files >= d.n || d.tok[files].type != JS_OBJ) {
        status = WASTE_DS_V4_RESOLVE_E_JSON;
        goto done;
    }
    int banks = js_get(&d, files, "banks");
    if (banks < 0 || banks >= d.n || d.tok[banks].type != JS_ARR) {
        status = WASTE_DS_V4_RESOLVE_E_TOPOLOGY;
        goto done;
    }
    /* Exactly one bank per main layer. A container with more or fewer is not
     * a container this manifest describes. */
    if ((size_t)js_size(&d, banks) != layers) {
        status = WASTE_DS_V4_RESOLVE_E_TOPOLOGY;
        goto done;
    }

    if ((status = resolve_member_path(&d, files, "resident", root_dir,
                                      &out->trunk_path)) !=
        WASTE_DS_V4_RESOLVE_OK)
        goto done;

    out->bank_paths = (char **)calloc(layers, sizeof *out->bank_paths);
    out->banks = (waste_ds_v4_file_bank_spec *)calloc(layers, sizeof *out->banks);
    out->record_offsets =
        (uint64_t *)calloc(layers * experts, sizeof *out->record_offsets);
    seen = (unsigned char *)calloc(layers, 1);
    if (!out->bank_paths || !out->banks || !out->record_offsets || !seen) {
        status = WASTE_DS_V4_RESOLVE_E_MEMORY;
        goto done;
    }
    /* Set before any early exit so the free path knows how many path slots to
     * release; calloc leaves the unfilled ones NULL, which free() accepts. */
    out->bank_count = layers;
    out->experts_per_layer = experts;

    for (size_t i = 0; i < layers; i++) {
        int bank = js_at(&d, banks, (int)i);
        if (bank < 0 || bank >= d.n || d.tok[bank].type != JS_OBJ) {
            status = WASTE_DS_V4_RESOLVE_E_TOPOLOGY;
            goto done;
        }
        /* The layer a bank serves is declared, not taken from its position in
         * the array. Array order is presentation; expert identity is not. */
        uint32_t layer = 0;
        if (dsjs_member_u32(&d, bank, "layer", &layer) != 0 ||
            (size_t)layer >= layers || seen[layer]) {
            status = WASTE_DS_V4_RESOLVE_E_TOPOLOGY;
            goto done;
        }
        seen[layer] = 1;

        if ((status = resolve_member_path(&d, bank, "path", root_dir,
                                          &out->bank_paths[layer])) !=
            WASTE_DS_V4_RESOLVE_OK)
            goto done;

        uint64_t *dst = out->record_offsets + (size_t)layer * experts;
        if ((status = resolve_bank_offsets(&d, bank, experts, rec, dst)) !=
            WASTE_DS_V4_RESOLVE_OK)
            goto done;

        out->banks[layer].path = out->bank_paths[layer];
        out->banks[layer].record_offsets = dst;
        out->banks[layer].record_count = experts;
    }

    /* `seen` is exhaustive by construction — `layers` distinct values drawn
     * from [0, layers) — so this cannot fail here. It is checked anyway
     * because that reasoning depends on the duplicate test above, and a future
     * edit to it would otherwise silently allow a gap. */
    for (size_t i = 0; i < layers; i++)
        if (!seen[i]) {
            status = WASTE_DS_V4_RESOLVE_E_TOPOLOGY;
            goto done;
        }

    out->spec.trunk_path = out->trunk_path;
    out->spec.banks = out->banks;
    out->spec.bank_count = layers;
    out->spec.cache_bytes = cache_bytes;
    out->spec.cache_policy = cache_policy;

done:
    free(seen);
    js_free(&d);
    if (status != WASTE_DS_V4_RESOLVE_OK)
        waste_ds_v4_resolved_files_free(out);
    return status;
}

waste_ds_v4_resolve_status waste_ds_v4_resolver_open(
    waste_ds_v4_file_runtime *out,
    const waste_ds_v4_manifest *manifest,
    const char *root_dir,
    const char *json,
    size_t len,
    size_t cache_bytes,
    int cache_policy,
    int *open_rc)
{
    if (open_rc)
        *open_rc = 0;
    if (!out)
        return WASTE_DS_V4_RESOLVE_E_ARG;
    memset(out, 0, sizeof *out);

    waste_ds_v4_resolved_files files;
    waste_ds_v4_resolve_status status = waste_ds_v4_resolve_files(
        &files, manifest, root_dir, json, len, cache_bytes, cache_policy);
    if (status != WASTE_DS_V4_RESOLVE_OK)
        return status;

    /* The open copies every path and offset it needs, so the resolved
     * description is released here rather than handed to the caller to
     * outlive. That is the whole reason this entry point exists. */
    int rc = waste_ds_v4_file_runtime_open(out, manifest, &files.spec);
    waste_ds_v4_resolved_files_free(&files);
    if (open_rc)
        *open_rc = rc;
    return rc == 0 ? WASTE_DS_V4_RESOLVE_OK : WASTE_DS_V4_RESOLVE_E_OPEN;
}
