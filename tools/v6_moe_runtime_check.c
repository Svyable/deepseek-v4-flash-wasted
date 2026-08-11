/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Gate H acquisition checker. Unlike tools/v4_moe_oracle.c this program
 * deliberately links the WASTE scalar reference implementation. Acquisition
 * runs the standalone oracle first, then this checker must reproduce every
 * one of the 4096 BF16 outputs from the same real checkpoint slices.
 */
#include "deepseek_v4_expert_ref.h"
#include "quant/deepseek_v4_linear_ref.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT 4096u
#define INTER 2048u
#define OUT_ROWS 4096u
#define TOPK 6u

static uint32_t f32_bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

static uint16_t bf16_bits(float x)
{
    uint32_t u = f32_bits(x);
    uint32_t exp = u & 0x7f800000u;
    uint32_t frac = u & 0x007fffffu;
    if (exp == 0x7f800000u) {
        if (frac) u |= 0x00400000u;
        return (uint16_t)(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static float bf16_float(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

static void *read_exact(const char *path, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        return NULL;
    }
    void *p = malloc(n ? n : 1u);
    if (!p) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(p, 1, n, f);
    int extra = fgetc(f);
    fclose(f);
    if (got != n || extra != EOF) {
        fprintf(stderr, "%s: expected %zu bytes, got %zu%s\n",
                path, n, got, extra == EOF ? "" : "+");
        free(p);
        return NULL;
    }
    return p;
}

static int load_input(const char *path, float *x)
{
    uint16_t *bits = read_exact(path, INPUT * sizeof(uint16_t));
    if (!bits) return -1;
    for (size_t i = 0; i < INPUT; i++) x[i] = bf16_float(bits[i]);
    free(bits);
    return 0;
}

static int compare_expected(const float *out, const char *path)
{
    uint16_t *want = read_exact(path, OUT_ROWS * sizeof(uint16_t));
    if (!want) return -1;
    for (size_t i = 0; i < OUT_ROWS; i++) {
        uint16_t got = bf16_bits(out[i]);
        if (got != want[i]) {
            fprintf(stderr, "output row %zu: got %04x expected %04x\n",
                    i, (unsigned)got, (unsigned)want[i]);
            free(want);
            return 1;
        }
    }
    free(want);
    return 0;
}

static int routed(int argc, char **argv)
{
    if (argc != 11) return -1;
    float route = strtof(argv[9], NULL);
    float *x = malloc(INPUT * sizeof(float));
    float *out = malloc(OUT_ROWS * sizeof(float));
    uint8_t *w1 = read_exact(argv[3], INTER * (INPUT / 2u));
    uint8_t *s1 = read_exact(argv[4], INTER * (INPUT / 32u));
    uint8_t *w3 = read_exact(argv[5], INTER * (INPUT / 2u));
    uint8_t *s3 = read_exact(argv[6], INTER * (INPUT / 32u));
    uint8_t *w2 = read_exact(argv[7], OUT_ROWS * (INTER / 2u));
    uint8_t *s2 = read_exact(argv[8], OUT_ROWS * (INTER / 32u));
    int rc = 1;
    if (!x || !out || !w1 || !s1 || !w3 || !s3 || !w2 || !s2 ||
        load_input(argv[2], x)) goto done;
    if (waste_ds_v4_routed_expert_ref(
            x, INPUT, INTER, w1, s1, w3, s3, 10.0f, route,
            w2, s2, OUT_ROWS, NULL, NULL, NULL, out) != 0) {
        fprintf(stderr, "WASTE routed expert rejected the full-row input\n");
        goto done;
    }
    rc = compare_expected(out, argv[10]);
done:
    free(x); free(out); free(w1); free(s1); free(w3); free(s3); free(w2); free(s2);
    return rc;
}

static int shared(int argc, char **argv)
{
    if (argc != 10) return -1;
    float *x = malloc(INPUT * sizeof(float));
    float *out = malloc(OUT_ROWS * sizeof(float));
    uint8_t *w1 = read_exact(argv[3], INTER * INPUT);
    uint8_t *s1 = read_exact(argv[4], (INTER / 128u) * (INPUT / 128u));
    uint8_t *w3 = read_exact(argv[5], INTER * INPUT);
    uint8_t *s3 = read_exact(argv[6], (INTER / 128u) * (INPUT / 128u));
    uint8_t *w2 = read_exact(argv[7], OUT_ROWS * INTER);
    uint8_t *s2 = read_exact(argv[8],
        ((OUT_ROWS + 127u) / 128u) * (INTER / 128u));
    int rc = 1;
    if (!x || !out || !w1 || !s1 || !w3 || !s3 || !w2 || !s2 ||
        load_input(argv[2], x)) goto done;
    if (waste_ds_v4_shared_expert_ref(
            x, INPUT, INTER, w1, s1, w3, s3, 10.0f,
            w2, s2, OUT_ROWS, NULL, NULL, NULL, out) != 0) {
        fprintf(stderr, "WASTE shared expert rejected the full-row input\n");
        goto done;
    }
    rc = compare_expected(out, argv[9]);
done:
    free(x); free(out); free(w1); free(s1); free(w3); free(s3); free(w2); free(s2);
    return rc;
}

static int combine(int argc, char **argv)
{
    if (argc != 6) return -1;
    uint32_t *ids = read_exact(argv[2], TOPK * sizeof(uint32_t));
    uint16_t *routed_bits = read_exact(argv[3], TOPK * OUT_ROWS * sizeof(uint16_t));
    uint16_t *shared_bits = read_exact(argv[4], OUT_ROWS * sizeof(uint16_t));
    uint16_t *want = read_exact(argv[5], OUT_ROWS * sizeof(uint16_t));
    float *routed = malloc(TOPK * OUT_ROWS * sizeof(float));
    float *shared_out = malloc(OUT_ROWS * sizeof(float));
    float *out = malloc(OUT_ROWS * sizeof(float));
    int rc = 1;
    if (!ids || !routed_bits || !shared_bits || !want ||
        !routed || !shared_out || !out) goto done;
    for (size_t i = 0; i < TOPK * OUT_ROWS; i++)
        routed[i] = bf16_float(routed_bits[i]);
    for (size_t i = 0; i < OUT_ROWS; i++) shared_out[i] = bf16_float(shared_bits[i]);
    if (waste_ds_v4_moe_combine_ref(ids, routed, TOPK, shared_out, OUT_ROWS, out) != 0) {
        fprintf(stderr, "WASTE MoE combine rejected the full branch\n");
        goto done;
    }
    for (size_t i = 0; i < OUT_ROWS; i++) {
        uint16_t got = bf16_bits(out[i]);
        if (got != want[i]) {
            fprintf(stderr, "combined row %zu: got %04x expected %04x\n",
                    i, (unsigned)got, (unsigned)want[i]);
            goto done;
        }
    }
    rc = 0;
done:
    free(ids); free(routed_bits); free(shared_bits); free(want);
    free(routed); free(shared_out); free(out);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "routed") == 0) {
        int rc = routed(argc, argv);
        if (rc < 0)
            fprintf(stderr, "usage: %s routed input w1 s1 w3 s3 w2 s2 route expected\n", argv[0]);
        return rc < 0 ? 2 : rc;
    }
    if (argc > 1 && strcmp(argv[1], "shared") == 0) {
        int rc = shared(argc, argv);
        if (rc < 0)
            fprintf(stderr, "usage: %s shared input w1 s1 w3 s3 w2 s2 expected\n", argv[0]);
        return rc < 0 ? 2 : rc;
    }
    if (argc > 1 && strcmp(argv[1], "combine") == 0) {
        int rc = combine(argc, argv);
        if (rc < 0)
            fprintf(stderr, "usage: %s combine ids routed shared expected\n", argv[0]);
        return rc < 0 ? 2 : rc;
    }
    fprintf(stderr, "expected routed, shared, or combine mode\n");
    return 2;
}
