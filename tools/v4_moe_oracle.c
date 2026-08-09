/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * Standalone fixture oracle for the fixed layer-3 Gate F representative.
 *
 * IMPORTANT: this file intentionally does not include or link any WASTE model
 * or quantization implementation.  It is the expected-value producer for the
 * frozen fixture, so sharing decoder/linear helpers with src/ would recreate
 * the self-confirming-fixture bug found in PR #3.
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT 4096u
#define INTER 2048u
#define OUT_ROWS 8u
#define ACT_BLOCK 128u
#define FP4_BLOCK 32u
#define FP8_BLOCK 128u

static uint32_t f32_bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float bits_f32(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }

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

static float bf16_float(uint16_t b) { return bits_f32((uint32_t)b << 16); }

static float e2m1(uint8_t c)
{
    static const float t[16] = {
        0, .5f, 1, 1.5f, 2, 3, 4, 6,
        -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6
    };
    return t[c & 15u];
}

static float e4m3(uint8_t b)
{
    uint32_t sign = b >> 7;
    uint32_t exp = (b >> 3) & 15u;
    uint32_t mant = b & 7u;
    if ((b & 0x7fu) == 0x7fu) return NAN;
    float mag = exp == 0 ? ldexpf((float)mant / 8.0f, -6)
                         : ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -mag : mag;
}

static float e8m0(uint8_t b)
{
    if (b == 0xffu) return NAN;
    return ldexpf(1.0f, (int)b - 127);
}

static uint8_t e4m3_encode(float x)
{
    if (isnan(x)) return 0x7fu;
    if (x == 0.0f) return signbit(x) ? 0x80u : 0u;
    if (x > 448.0f) x = 448.0f;
    if (x < -448.0f) x = -448.0f;
    int neg = signbit(x) != 0;
    float best_err = INFINITY;
    uint8_t best = neg ? 0x80u : 0u;
    for (unsigned mag = 0; mag < 0x7fu; mag++) {
        uint8_t code = (uint8_t)(mag | (neg ? 0x80u : 0u));
        float err = fabsf(x - e4m3(code));
        if (err < best_err ||
            (err == best_err && !(code & 1u) && (best & 1u))) {
            best_err = err;
            best = code;
        }
    }
    return best;
}

static float act_scale(const float *x, size_t n)
{
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-4f) amax = 1e-4f;
    float ratio = amax * (1.0f / 448.0f);
    uint32_t u = f32_bits(ratio);
    int e = (int)((u >> 23) & 0xffu) - 127;
    if (u & 0x7fffffu) e++;
    return ldexpf(1.0f, e);
}

static int quant_act(const float *x, size_t k, uint8_t *q, float *scales)
{
    if (k % ACT_BLOCK) return -1;
    for (size_t base = 0; base < k; base += ACT_BLOCK) {
        float s = act_scale(x + base, ACT_BLOCK);
        scales[base / ACT_BLOCK] = s;
        for (size_t j = 0; j < ACT_BLOCK; j++) {
            float z = x[base + j] / s;
            if (z > 448.0f) z = 448.0f;
            if (z < -448.0f) z = -448.0f;
            q[base + j] = e4m3_encode(z);
        }
    }
    return 0;
}

static int fp4_linear(const float *x, size_t k,
                      const uint8_t *w, const uint8_t *s,
                      size_t rows, float *out)
{
    if (k % ACT_BLOCK || k % FP4_BLOCK) return -1;
    uint8_t *qx = malloc(k);
    float *as = malloc((k / ACT_BLOCK) * sizeof(float));
    if (!qx || !as) { free(qx); free(as); return -1; }
    if (quant_act(x, k, qx, as)) { free(qx); free(as); return -1; }
    size_t packed_row = k / 2u;
    size_t scale_row = k / FP4_BLOCK;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *wr = w + r * packed_row;
        const uint8_t *sr = s + r * scale_row;
        float acc = 0.0f;
        for (size_t wb = 0; wb < scale_row; wb++) {
            size_t base = wb * FP4_BLOCK;
            float part = 0.0f;
            for (size_t j = 0; j < FP4_BLOCK; j++) {
                size_t kk = base + j;
                uint8_t byte = wr[kk / 2u];
                uint8_t code = (kk & 1u) ? (byte >> 4) : (byte & 15u);
                part += e4m3(qx[kk]) * e2m1(code);
            }
            float ws = e8m0(sr[wb]);
            if (!isfinite(ws)) { free(qx); free(as); return -1; }
            acc += part * as[base / ACT_BLOCK] * ws;
        }
        out[r] = bf16_float(bf16_bits(acc));
    }
    free(qx); free(as); return 0;
}

static int fp8_linear(const float *x, size_t k,
                      const uint8_t *w, const uint8_t *s,
                      size_t rows, float *out)
{
    if (k % ACT_BLOCK) return -1;
    uint8_t *qx = malloc(k);
    float *as = malloc((k / ACT_BLOCK) * sizeof(float));
    if (!qx || !as) { free(qx); free(as); return -1; }
    if (quant_act(x, k, qx, as)) { free(qx); free(as); return -1; }
    size_t kb = k / FP8_BLOCK;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *wr = w + r * k;
        const uint8_t *sr = s + (r / FP8_BLOCK) * kb;
        float acc = 0.0f;
        for (size_t b = 0; b < kb; b++) {
            size_t base = b * FP8_BLOCK;
            float part = 0.0f;
            for (size_t j = 0; j < FP8_BLOCK; j++)
                part += e4m3(qx[base + j]) * e4m3(wr[base + j]);
            float ws = e8m0(sr[b]);
            if (!isfinite(ws)) { free(qx); free(as); return -1; }
            acc += part * as[b] * ws;
        }
        out[r] = bf16_float(bf16_bits(acc));
    }
    free(qx); free(as); return 0;
}

static float silu(float x)
{
    if (x >= 0.0f) return x / (1.0f + expf(-x));
    float z = expf(x);
    return x * z / (1.0f + z);
}

static void *read_exact(const char *path, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s: %s\n", path, strerror(errno)); return NULL; }
    void *p = malloc(n ? n : 1);
    if (!p) { fclose(f); return NULL; }
    size_t got = fread(p, 1, n, f);
    int extra = fgetc(f);
    fclose(f);
    if (got != n || extra != EOF) {
        fprintf(stderr, "%s: expected %zu bytes, got %zu%s\n",
                path, n, got, extra == EOF ? "" : "+");
        free(p); return NULL;
    }
    return p;
}

static int write_exact(const char *path, const void *p, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = fwrite(p, 1, n, f) == n && fclose(f) == 0;
    return ok ? 0 : -1;
}

static int load_input(const char *path, float *x)
{
    uint16_t *b = read_exact(path, INPUT * sizeof(uint16_t));
    if (!b) return -1;
    for (size_t i = 0; i < INPUT; i++) x[i] = bf16_float(b[i]);
    free(b); return 0;
}

static int routed(int argc, char **argv)
{
    if (argc != 12) return -1;
    float route = strtof(argv[9], NULL);
    float *x = malloc(INPUT * sizeof(float));
    float *gate = malloc(INTER * sizeof(float));
    float *up = malloc(INTER * sizeof(float));
    float *hidden = malloc(INTER * sizeof(float));
    uint8_t *w1 = read_exact(argv[3], INTER * (INPUT/2));
    uint8_t *s1 = read_exact(argv[4], INTER * (INPUT/32));
    uint8_t *w3 = read_exact(argv[5], INTER * (INPUT/2));
    uint8_t *s3 = read_exact(argv[6], INTER * (INPUT/32));
    uint8_t *w2 = read_exact(argv[7], OUT_ROWS * (INTER/2));
    uint8_t *s2 = read_exact(argv[8], OUT_ROWS * (INTER/32));
    uint16_t out_bits[OUT_ROWS];
    float out[OUT_ROWS];
    if (!x || !gate || !up || !hidden || !w1 || !s1 || !w3 || !s3 || !w2 || !s2 ||
        load_input(argv[2], x) ||
        fp4_linear(x, INPUT, w1, s1, INTER, gate) ||
        fp4_linear(x, INPUT, w3, s3, INTER, up)) goto bad;
    for (size_t i = 0; i < INTER; i++) {
        float g = gate[i] > 10.0f ? 10.0f : gate[i];
        float u = up[i] > 10.0f ? 10.0f : (up[i] < -10.0f ? -10.0f : up[i]);
        hidden[i] = bf16_float(bf16_bits(silu(g) * u * route));
    }
    if (fp4_linear(hidden, INTER, w2, s2, OUT_ROWS, out)) goto bad;
    for (size_t i = 0; i < OUT_ROWS; i++) out_bits[i] = bf16_bits(out[i]);
    if (write_exact(argv[10], out_bits, sizeof out_bits)) goto bad;
    free(x); free(gate); free(up); free(hidden); free(w1); free(s1); free(w3); free(s3); free(w2); free(s2);
    return 0;
bad:
    free(x); free(gate); free(up); free(hidden); free(w1); free(s1); free(w3); free(s3); free(w2); free(s2);
    return 1;
}

static int shared(int argc, char **argv)
{
    if (argc != 14) return -1;
    float *x = malloc(INPUT * sizeof(float));
    float *gate = malloc(INTER * sizeof(float));
    float *up = malloc(INTER * sizeof(float));
    float *hidden = malloc(INTER * sizeof(float));
    uint8_t *w1 = read_exact(argv[3], INTER * INPUT);
    uint8_t *s1 = read_exact(argv[4], (INTER/128) * (INPUT/128));
    uint8_t *w3 = read_exact(argv[5], INTER * INPUT);
    uint8_t *s3 = read_exact(argv[6], (INTER/128) * (INPUT/128));
    uint8_t *w2 = read_exact(argv[7], OUT_ROWS * INTER);
    uint8_t *s2 = read_exact(argv[8], (INTER/128));
    uint16_t *hidden_bits = malloc(INTER * sizeof(uint16_t));
    uint16_t out_bits[OUT_ROWS];
    float out[OUT_ROWS];
    if (!x || !gate || !up || !hidden || !w1 || !s1 || !w3 || !s3 || !w2 || !s2 || !hidden_bits ||
        load_input(argv[2], x) ||
        fp8_linear(x, INPUT, w1, s1, INTER, gate) ||
        fp8_linear(x, INPUT, w3, s3, INTER, up)) goto bad;
    for (size_t i = 0; i < INTER; i++) {
        float g = gate[i] > 10.0f ? 10.0f : gate[i];
        float u = up[i] > 10.0f ? 10.0f : (up[i] < -10.0f ? -10.0f : up[i]);
        gate[i] = g; up[i] = u;
        hidden_bits[i] = bf16_bits(silu(g) * u);
        hidden[i] = bf16_float(hidden_bits[i]);
    }
    if (fp8_linear(hidden, INTER, w2, s2, OUT_ROWS, out)) goto bad;
    for (size_t i = 0; i < OUT_ROWS; i++) out_bits[i] = bf16_bits(out[i]);
    if (write_exact(argv[9], gate, INTER * sizeof(float)) ||
        write_exact(argv[10], up, INTER * sizeof(float)) ||
        write_exact(argv[11], hidden_bits, INTER * sizeof(uint16_t)) ||
        write_exact(argv[12], out_bits, sizeof out_bits)) goto bad;
    free(x); free(gate); free(up); free(hidden); free(w1); free(s1); free(w3); free(s3); free(w2); free(s2); free(hidden_bits);
    return 0;
bad:
    free(x); free(gate); free(up); free(hidden); free(w1); free(s1); free(w3); free(s3); free(w2); free(s2); free(hidden_bits);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "routed") == 0) {
        int rc = routed(argc, argv);
        if (rc < 0) fprintf(stderr, "usage: %s routed input w1 s1 w3 s3 w2 s2 route out unused\n", argv[0]);
        return rc < 0 ? 2 : rc;
    }
    if (argc > 1 && strcmp(argv[1], "shared") == 0) {
        int rc = shared(argc, argv);
        if (rc < 0) fprintf(stderr, "usage: %s shared input w1 s1 w3 s3 w2 s2 gate up hidden out unused\n", argv[0]);
        return rc < 0 ? 2 : rc;
    }
    fprintf(stderr, "expected routed or shared mode\n");
    return 2;
}
