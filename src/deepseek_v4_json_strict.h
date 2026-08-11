/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
/*
 * deepseek_v4_json_strict.h — strict scalar readers over src/json.h.
 *
 * `js_num` goes through `atof`, which accepts "4096.0", "4.096e3", "-4096" and
 * "04096" and hands back 4096 for every one of them. Every number in a
 * DeepSeek manifest or resolver document sizes or locates bytes, so each is
 * read from its raw text and refused unless it is a plain non-negative integer
 * that fits.
 *
 * This lives in its own header because the family manifest parser and the
 * container resolver both read untrusted numbers, and a second copy of these
 * rules is a second place for them to drift. The manifest's mutation suite
 * (tests/test_deepseek_v4_manifest.c) is what holds them.
 *
 * Header-only and `static inline`, matching src/json.h.
 */
#ifndef WASTE_DEEPSEEK_V4_JSON_STRICT_H
#define WASTE_DEEPSEEK_V4_JSON_STRICT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "json.h"

/* Bounded so a malformed document cannot make us read an unbounded buffer.
 * Real 0731 documents need a few tens of KiB; an explicit per-expert offset
 * table for 43 banks of 256 records is still comfortably inside this. */
#define WASTE_DS_JSON_MAX_BYTES (16u * 1024u * 1024u)

/* A NUL-terminated buffer whose declared length agrees with its contents.
 * js.h reads to the NUL, so a length that disagrees would silently parse past
 * what the caller believes it handed over. */
static inline int dsjs_input_ok(const char *json, size_t len)
{
    return json && len > 0 && len <= WASTE_DS_JSON_MAX_BYTES &&
           json[len] == '\0' && memchr(json, '\0', len) == NULL;
}

static inline int dsjs_u64(const js_doc *d, int tok, uint64_t *out)
{
    if (!d || tok < 0 || tok >= d->n || d->tok[tok].type != JS_NUM || !out)
        return -1;
    int begin = d->tok[tok].start, end = d->tok[tok].end;
    if (end <= begin || end - begin > 20)
        return -1;
    uint64_t v = 0;
    for (int i = begin; i < end; i++) {
        char c = d->src[i];
        if (c < '0' || c > '9')
            return -1;               /* sign, '.', exponent: all refused */
        uint64_t digit = (uint64_t)(c - '0');
        if (v > (UINT64_MAX - digit) / 10u)
            return -1;
        v = v * 10u + digit;
    }
    /* Leading zeros are not JSON, and are how "007" sneaks past a reader that
     * only checks the digits. */
    if (end - begin > 1 && d->src[begin] == '0')
        return -1;
    *out = v;
    return 0;
}

static inline int dsjs_u32(const js_doc *d, int tok, uint32_t *out)
{
    uint64_t v = 0;
    if (!out || dsjs_u64(d, tok, &v) != 0 || v > UINT32_MAX)
        return -1;
    *out = (uint32_t)v;
    return 0;
}

static inline int dsjs_size(const js_doc *d, int tok, size_t *out)
{
    uint64_t v = 0;
    if (!out || dsjs_u64(d, tok, &v) != 0)
        return -1;
#if SIZE_MAX < UINT64_MAX
    if (v > (uint64_t)SIZE_MAX)
        return -1;
#endif
    *out = (size_t)v;
    return 0;
}

static inline int dsjs_member_u32(const js_doc *d, int obj, const char *key,
                                  uint32_t *out)
{
    return dsjs_u32(d, js_get(d, obj, key), out);
}

static inline int dsjs_member_u64(const js_doc *d, int obj, const char *key,
                                  uint64_t *out)
{
    return dsjs_u64(d, js_get(d, obj, key), out);
}

static inline int dsjs_member_size(const js_doc *d, int obj, const char *key,
                                   size_t *out)
{
    return dsjs_size(d, js_get(d, obj, key), out);
}

/* Locate a string member without copying. Returns 0 and sets `text`/`len` to a
 * range inside the document, or -1 when absent, empty, or not a string.
 *
 * js.h does not unescape, so a value containing a backslash would be compared
 * or used in its escaped form. Refuse rather than act on something that is not
 * what the writer meant — for a filesystem path that is a security property,
 * not a nicety. */
static inline int dsjs_member_text(const js_doc *d, int obj, const char *key,
                                   const char **text, size_t *len)
{
    int tok = js_get(d, obj, key);
    if (!d || tok < 0 || tok >= d->n || d->tok[tok].type != JS_STR ||
        !text || !len)
        return -1;
    size_t n = (size_t)(d->tok[tok].end - d->tok[tok].start);
    const char *p = d->src + d->tok[tok].start;
    if (n == 0 || memchr(p, '\\', n) != NULL)
        return -1;
    *text = p;
    *len = n;
    return 0;
}

/* Copy a string member, refusing truncation: a value that does not fit is a
 * document we do not understand, not a value to shorten. */
static inline int dsjs_member_str(const js_doc *d, int obj, const char *key,
                                  char *buf, size_t cap)
{
    const char *text = NULL;
    size_t len = 0;
    if (!buf || cap == 0 || dsjs_member_text(d, obj, key, &text, &len) != 0 ||
        len >= cap)
        return -1;
    memcpy(buf, text, len);
    buf[len] = 0;
    return 0;
}

#endif /* WASTE_DEEPSEEK_V4_JSON_STRICT_H */
