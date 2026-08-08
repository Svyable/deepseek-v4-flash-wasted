/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 */
/* Gate V1 — bit-level native quantization decode (docs/VALIDATION.md).
 *
 * E2M1 has 16 codes and E4M3 has 256, so "representative fixtures" would
 * be a needless approximation: this enumerates every code of both formats
 * and every UE8M0 scale byte. There is no sampling here and no tolerance
 * on the format tables — decode is exact or it is wrong.
 *
 * Expected values are derived from the format rules (sign, bias,
 * subnormal) rather than copied from the implementation's table, so a
 * typo in src/quant/ fails here instead of being confirmed by a mirror of
 * itself. The two disagree on purpose about *how* they compute; they must
 * agree on the result.
 *
 * WHAT THIS DOES NOT PROVE. Passing means these decoders conform to the
 * public OCP Microscaling definitions. It does not mean DeepSeek's
 * reference agrees, because that reference has never been read here
 * (docs/INVENTORY-0731.md). Nibble order and scale direction in
 * particular are conventions a spec cannot settle. V1 is only complete
 * when official fixtures confirm them; until then this is the cheap half
 * of the gate, and it is the half that catches the bugs that would
 * otherwise be blamed on the model.
 */

#include "../src/quant/fp4_e2m1.h"
#include "../src/quant/fp8_e4m3.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static void ck(int cond, const char *msg)
{
    if (!cond) {
        fails++;
        printf("  FAIL %s\n", msg);
    }
}

static void ckf(double got, double want, const char *msg)
{
    /* Exact: every value in these formats is a dyadic rational well
     * inside binary32's range, so there is nothing to round. */
    if (got != want) {
        fails++;
        printf("  FAIL %s (got %.17g want %.17g)\n", msg, got, want);
    }
}

/* ---------------------------------------------------------------- E2M1 -- */

/* The format rule, written out independently of the implementation's
 * lookup table: 1 sign, 2 exponent bits with bias 1, 1 mantissa bit.
 * E==0 is subnormal with no implicit leading one. */
static double e2m1_expected(unsigned code)
{
    unsigned sign = (code >> 3) & 1u;
    unsigned exp  = (code >> 1) & 3u;
    unsigned mant = code & 1u;
    double mag;

    if (exp == 0)
        mag = (double)mant * 0.5;                      /* M * 2^(1-1) * 2^-1 */
    else
        mag = (1.0 + 0.5 * (double)mant) * ldexp(1.0, (int)exp - 1);

    return sign ? -mag : mag;
}

static void test_e2m1(void)
{
    for (unsigned code = 0; code < 16; code++) {
        char msg[64];
        snprintf(msg, sizeof msg, "e2m1 code 0x%X", code);
        ckf(waste_e2m1_decode((uint8_t)code), e2m1_expected(code), msg);
    }

    /* Only the low nibble may be consulted; the high bits are another
     * element's data and reading them would silently corrupt every odd
     * column. */
    for (unsigned code = 0; code < 16; code++)
        ck(waste_e2m1_decode((uint8_t)(code | 0xF0)) ==
           waste_e2m1_decode((uint8_t)code),
           "e2m1 ignores the high nibble");

    /* Landmarks, spelled out so a reader can check them against the spec
     * without running the derivation above. */
    ckf(waste_e2m1_decode(0x0), 0.0, "e2m1 +0");
    ckf(waste_e2m1_decode(0x1), 0.5, "e2m1 min subnormal");
    ckf(waste_e2m1_decode(0x2), 1.0, "e2m1 one");
    ckf(waste_e2m1_decode(0x7), 6.0, "e2m1 max");
    ckf(waste_e2m1_decode(0xF), -6.0, "e2m1 min");

    /* Negative zero must keep its sign bit: it is not merely equal to
     * zero, and a decoder that returns +0 here has dropped the sign path. */
    ck(signbit(waste_e2m1_decode(0x8)), "e2m1 -0 keeps its sign");
    ck(!signbit(waste_e2m1_decode(0x0)), "e2m1 +0 has no sign");

    /* No code is NaN or infinite. */
    for (unsigned code = 0; code < 16; code++)
        ck(isfinite(waste_e2m1_decode((uint8_t)code)), "e2m1 is always finite");
}

/* --------------------------------------------------------------- UE8M0 -- */

static void test_ue8m0(void)
{
    for (unsigned e = 0; e < 255; e++) {
        char msg[64];
        double want = ldexp(1.0, (int)e - 127);
        snprintf(msg, sizeof msg, "ue8m0 exponent %u", e);
        ckf(waste_ue8m0_decode((uint8_t)e), want, msg);
        ck(!waste_ue8m0_is_nan((uint8_t)e), "ue8m0 below 0xFF is a number");
    }

    ck(waste_ue8m0_is_nan(0xFF), "ue8m0 0xFF is NaN");
    ck(isnan(waste_ue8m0_decode(0xFF)), "ue8m0 0xFF decodes to NaN");

    ckf(waste_ue8m0_decode(127), 1.0, "ue8m0 bias point is 1.0");
    ckf(waste_ue8m0_decode(128), 2.0, "ue8m0 one above bias is 2.0");
    ckf(waste_ue8m0_decode(126), 0.5, "ue8m0 one below bias is 0.5");

    /* Both ends stay finite and nonzero. 2^-127 is subnormal in binary32,
     * which is the end a hand-rolled exponent shift gets wrong: it
     * flushes to zero and every weight in that block silently vanishes. */
    ck(waste_ue8m0_decode(0) > 0.0f, "ue8m0 2^-127 does not flush to zero");
    ck(isfinite(waste_ue8m0_decode(254)), "ue8m0 2^127 is finite");
}

/* ---------------------------------------------------------------- E4M3 -- */

/* Independent derivation: 1 sign, 4 exponent bits with bias 7, 3 mantissa
 * bits, finite variant (only S.1111.111 is NaN). */
static double e4m3_expected(unsigned b, int *is_nan)
{
    unsigned sign = (b >> 7) & 1u;
    unsigned exp  = (b >> 3) & 0x0Fu;
    unsigned mant = b & 0x07u;
    double mag;

    *is_nan = ((b & 0x7Fu) == 0x7Fu);
    if (*is_nan)
        return 0.0;

    if (exp == 0)
        mag = ((double)mant / 8.0) * ldexp(1.0, 1 - 7);
    else
        mag = (1.0 + (double)mant / 8.0) * ldexp(1.0, (int)exp - 7);

    return sign ? -mag : mag;
}

static void test_e4m3(void)
{
    for (unsigned b = 0; b < 256; b++) {
        char msg[64];
        int want_nan;
        double want = e4m3_expected(b, &want_nan);
        float got = waste_e4m3_decode((uint8_t)b);

        snprintf(msg, sizeof msg, "e4m3 byte 0x%02X", b);
        if (want_nan) {
            ck(isnan(got), msg);
            ck(waste_e4m3_is_nan((uint8_t)b), "e4m3 NaN predicate agrees");
        } else {
            ckf(got, want, msg);
            ck(!waste_e4m3_is_nan((uint8_t)b), "e4m3 non-NaN predicate agrees");
            ck(isfinite(got), "e4m3 finite variant has no infinities");
        }
    }

    /* The landmark that separates e4m3fn from an IEEE-style E4M3: the top
     * exponent carries values. If someone "fixes" this into Inf/NaN the
     * maximum drops to 240 and every large trunk weight is wrong. */
    ckf(waste_e4m3_decode(0x7E), 448.0, "e4m3fn max is 448");
    ck(WASTE_E4M3_MAX == 448.0f, "advertised max matches");
    ckf(waste_e4m3_decode(0x78), 256.0, "e4m3 exponent 15 mantissa 0 is 256");
    ckf(waste_e4m3_decode(0x38), 1.0, "e4m3 one");
    ckf(waste_e4m3_decode(0x08), 0.015625, "e4m3 min normal 2^-6");
    ckf(waste_e4m3_decode(0x01), 0.001953125, "e4m3 min subnormal 2^-9");
    ckf(waste_e4m3_decode(0x00), 0.0, "e4m3 +0");
    ck(signbit(waste_e4m3_decode(0x80)), "e4m3 -0 keeps its sign");
    ck(isnan(waste_e4m3_decode(0x7F)) && isnan(waste_e4m3_decode(0xFF)),
       "e4m3 both NaN encodings");
}

/* ------------------------------------------------- FP4 packing + scales -- */

static uint8_t pack2(unsigned even_code, unsigned odd_code)
{
#if WASTE_FP4_LOW_NIBBLE_IS_EVEN
    return (uint8_t)((odd_code << 4) | even_code);
#else
    return (uint8_t)((even_code << 4) | odd_code);
#endif
}

static void test_fp4_layout(void)
{
    /* Two rows, 64 columns: two scale blocks per row, so the block
     * boundary at column 32 and the row stride are both exercised. */
    enum { ROWS = 2, COLS = 64 };
    uint8_t w[ROWS * (COLS / 2)];
    uint8_t s[ROWS * (COLS / 32)];
    float row[COLS];

    ck(waste_fp4_dims_ok(ROWS, COLS), "64 columns is a legal fp4 shape");
    ck(waste_fp4_weight_bytes(ROWS, COLS) == sizeof w, "weight plane size");
    ck(waste_fp4_scale_bytes(ROWS, COLS) == sizeof s, "scale plane size");

    /* Column c gets code (c % 8), which is a distinct positive value for
     * each residue, so a pair swap or a stride error changes the answer. */
    for (size_t r = 0; r < ROWS; r++)
        for (size_t c = 0; c < COLS; c += 2)
            w[r * (COLS / 2) + c / 2] = pack2((unsigned)(c % 8),
                                              (unsigned)((c + 1) % 8));

    /* Row 0: blocks scale by 1 and 2. Row 1: by 4 and 1/2. Four distinct
     * (row, block) scales means picking the wrong one is always visible. */
    s[0] = 127; s[1] = 128;
    s[2] = 129; s[3] = 126;

    for (size_t r = 0; r < ROWS; r++) {
        for (size_t c = 0; c < COLS; c++) {
            double base = e2m1_expected((unsigned)(c % 8));
            double scale = ldexp(1.0, (int)s[r * 2 + c / 32] - 127);
            char msg[64];
            snprintf(msg, sizeof msg, "fp4_at r%zu c%zu", r, c);
            ckf(waste_fp4_at(w, s, COLS, r, c), base * scale, msg);
        }
    }

    /* The block boundary itself: same code either side, different scale. */
    ck(waste_fp4_at(w, s, COLS, 0, 31) != waste_fp4_at(w, s, COLS, 0, 32) ||
       e2m1_expected(31 % 8) == 0.0,
       "column 31 and 32 use different scale blocks");

    /* decode_row must agree with element access everywhere. */
    for (size_t r = 0; r < ROWS; r++) {
        waste_fp4_decode_row(w, s, COLS, r, row);
        for (size_t c = 0; c < COLS; c++)
            ckf(row[c], waste_fp4_at(w, s, COLS, r, c), "decode_row == fp4_at");
    }

    /* Odd columns really do come from the other nibble: rewrite one byte
     * so its two halves differ and check both halves move independently. */
    w[0] = pack2(7u, 1u);                 /* even -> 6.0, odd -> 0.5 */
    ckf(waste_fp4_at(w, s, COLS, 0, 0), 6.0, "even column reads its nibble");
    ckf(waste_fp4_at(w, s, COLS, 0, 1), 0.5, "odd column reads the other");

    /* PIN THE BYTE LAYOUT WITH A LITERAL.
     *
     * Everything above packs through pack2(), which is compiled from the
     * same WASTE_FP4_LOW_NIBBLE_IS_EVEN as the decoder -- so flipping that
     * macro flips both halves and every check still passes. Mutation
     * testing caught exactly that: the file's highest-risk assumption was
     * the one thing its tests could not detect.
     *
     * So state the convention in raw hex, independent of the macro. Byte
     * 0x21 is high nibble 2, low nibble 1. Under the documented layout
     * (even column -> low nibble) column 0 is code 1 = 0.5 and column 1 is
     * code 2 = 1.0. If Gate 0 shows DeepSeek packs the other way, this
     * assertion is *supposed* to fail: changing the convention should cost
     * a deliberate edit here, not slip through as a silent re-mirroring. */
    w[0] = 0x21;
    s[0] = 127;                                       /* scale 1.0 */
    ckf(waste_fp4_at(w, s, COLS, 0, 0), 0.5,
        "byte 0x21: even column comes from the LOW nibble");
    ckf(waste_fp4_at(w, s, COLS, 0, 1), 1.0,
        "byte 0x21: odd column comes from the HIGH nibble");

    /* Shapes the format cannot store must be refused rather than read
     * past the end of the scale plane. */
    ck(!waste_fp4_dims_ok(1, 0), "zero columns refused");
    ck(!waste_fp4_dims_ok(0, 32), "zero rows refused");
    ck(!waste_fp4_dims_ok(1, 31), "non-multiple of the scale block refused");
    ck(!waste_fp4_dims_ok(1, 48), "48 columns is not a whole scale block");
    ck(waste_fp4_dims_ok(1, 4096), "a real expert K dimension is legal");
}

static void test_fp4_matvec(void)
{
    enum { ROWS = 3, COLS = 96 };
    uint8_t w[ROWS * (COLS / 2)];
    uint8_t s[ROWS * (COLS / 32)];
    float x[COLS], y[ROWS], dec[COLS];

    for (size_t i = 0; i < sizeof w; i++)
        w[i] = (uint8_t)((i * 37u + 11u) & 0xFFu);
    for (size_t i = 0; i < sizeof s; i++)
        s[i] = (uint8_t)(120u + (i % 9u));       /* 2^-7 .. 2^1 */
    for (size_t c = 0; c < COLS; c++)
        x[c] = (float)((int)c % 7 - 3) * 0.25f;

    waste_fp4_matvec(w, s, ROWS, COLS, x, y);

    /* Independent path: fully decode each row, then dot it. The matvec
     * hoists the scale per block, so agreeing here says that
     * factorization did not change the value. */
    for (size_t r = 0; r < ROWS; r++) {
        double acc = 0.0;
        char msg[64];
        waste_fp4_decode_row(w, s, COLS, r, dec);
        for (size_t c = 0; c < COLS; c++)
            acc += (double)dec[c] * (double)x[c];
        snprintf(msg, sizeof msg, "fp4 matvec row %zu", r);
        ckf(y[r], (float)acc, msg);
    }
}

/* ------------------------------------------------- FP8 block-scale grid -- */

static void test_fp8_layout(void)
{
    /* Deliberately ragged: 200x300 gives a 2x3 scale grid whose last row
     * block covers 72 rows and last column block 44 columns. A ceiling
     * error shows up as an out-of-bounds scale read or a wrong tile. */
    enum { ROWS = 200, COLS = 300 };

    ck(waste_fp8_scale_rows(ROWS) == 2, "200 rows -> 2 scale rows");
    ck(waste_fp8_scale_cols(COLS) == 3, "300 cols -> 3 scale cols");
    ck(waste_fp8_scale_count(ROWS, COLS) == 6, "2x3 scale grid");
    ck(waste_fp8_scale_rows(128) == 1, "an exact block is one scale row");
    ck(waste_fp8_scale_rows(129) == 2, "one past a block adds a scale row");
    ck(waste_fp8_scale_cols(1) == 1, "a single column still needs a scale");

    {
        static uint8_t w[ROWS * COLS];
        float s[6];
        static float row[COLS];

        for (size_t i = 0; i < sizeof w; i++)
            w[i] = 0x38;                       /* every weight decodes to 1.0 */
        for (size_t i = 0; i < 6; i++)
            s[i] = (float)(i + 1);             /* tile index is readable off */

        /* With all weights 1.0, each element equals its tile's scale, so
         * this checks tile selection directly at every boundary. */
        ckf(waste_fp8_at(w, s, COLS, 0, 0), 1.0, "tile (0,0)");
        ckf(waste_fp8_at(w, s, COLS, 0, 127), 1.0, "last column of tile 0");
        ckf(waste_fp8_at(w, s, COLS, 0, 128), 2.0, "first column of tile 1");
        ckf(waste_fp8_at(w, s, COLS, 0, 255), 2.0, "last column of tile 1");
        ckf(waste_fp8_at(w, s, COLS, 0, 256), 3.0, "ragged last column tile");
        ckf(waste_fp8_at(w, s, COLS, 127, 0), 1.0, "last row of tile row 0");
        ckf(waste_fp8_at(w, s, COLS, 128, 0), 4.0, "first row of tile row 1");
        ckf(waste_fp8_at(w, s, COLS, 199, 299), 6.0, "last element, last tile");

        waste_fp8_decode_row(w, s, COLS, 128, row);
        for (size_t c = 0; c < COLS; c++)
            ckf(row[c], waste_fp8_at(w, s, COLS, 128, c),
                "fp8 decode_row == fp8_at");
    }
}

static void test_fp8_matvec(void)
{
    enum { ROWS = 5, COLS = 300 };
    static uint8_t w[ROWS * COLS];
    float s[1 * 3];
    float x[COLS], y[ROWS];
    static float dec[COLS];

    for (size_t i = 0; i < sizeof w; i++) {
        uint8_t b = (uint8_t)((i * 53u + 7u) & 0xFFu);
        /* Keep NaN out of the accumulation; NaN handling is covered
         * exhaustively by the decode test, and one NaN here would mask
         * every other arithmetic error in the row. */
        if ((b & 0x7Fu) == 0x7Fu)
            b ^= 0x08u;
        w[i] = b;
    }
    for (size_t i = 0; i < 3; i++)
        s[i] = 0.5f * (float)(i + 1);
    for (size_t c = 0; c < COLS; c++)
        x[c] = (float)((int)c % 5 - 2) * 0.125f;

    waste_fp8_matvec(w, s, ROWS, COLS, x, y);

    for (size_t r = 0; r < ROWS; r++) {
        double acc = 0.0;
        char msg[64];
        waste_fp8_decode_row(w, s, COLS, r, dec);
        for (size_t c = 0; c < COLS; c++)
            acc += (double)dec[c] * (double)x[c];
        snprintf(msg, sizeof msg, "fp8 matvec row %zu", r);
        ckf(y[r], (float)acc, msg);
    }
}

/* ------------------------------------------------------ expert geometry -- */

static void test_expert_record_arithmetic(void)
{
    /* The shapes README §1 predicts for one routed expert. This is the
     * arithmetic the container design rests on, so it is checked against
     * the decoder's own size helpers rather than restated as a constant.
     *
     * Note this proves internal consistency, NOT the checkpoint: the
     * dimensions come from the handoff document, which has never been
     * reconciled with real 0731 headers. docs/EXPERIMENTS.md entry 2. */
    const size_t hidden = 4096, moe_inter = 2048;
    size_t w1 = waste_fp4_weight_bytes(moe_inter, hidden)
              + waste_fp4_scale_bytes(moe_inter, hidden);
    size_t w3 = w1;
    size_t w2 = waste_fp4_weight_bytes(hidden, moe_inter)
              + waste_fp4_scale_bytes(hidden, moe_inter);

    ck(waste_fp4_weight_bytes(moe_inter, hidden) == 4194304, "w1 packed bytes");
    ck(waste_fp4_scale_bytes(moe_inter, hidden) == 262144, "w1 scale bytes");
    ck(w1 + w3 + w2 == 13369344, "one expert record is 12.75 MiB of payload");
}

int main(void)
{
    test_e2m1();
    test_ue8m0();
    test_e4m3();
    test_fp4_layout();
    test_fp4_matvec();
    test_fp8_layout();
    test_fp8_matvec();
    test_expert_record_arithmetic();

    puts(fails ? "QUANT FAILED" : "PASS quantization decode (V1, format-conformance half)");
    return fails ? 1 : 0;
}
