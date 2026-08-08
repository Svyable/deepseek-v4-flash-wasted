/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 The deepseek-v4-flash-wasted authors.
 *
 * E2M1 4-bit float with UE8M0 per-K-block scales — the storage format
 * DeepSeek V4 Flash uses for routed experts (README §7).
 *
 * Scalar reference only. Correctness first: every later SIMD kernel is
 * checked against this one, so it is written to be obviously right rather
 * than fast (docs/VALIDATION.md §9).
 *
 * A routed-expert matrix is logically [rows, cols] but stored as
 * [rows, cols/2] packed nibbles plus a [rows, cols/32] UE8M0 scale plane:
 *
 *     value(r, c) = e2m1(nibble(r, c)) * 2^(scale[r][c/32] - 127)
 *
 * OFFICIAL 0731 STORAGE CONVENTIONS — SOURCE VERIFIED
 *
 * The public number formats are still checked independently and exhaustively
 * by tests/test_quant.c. The DeepSeek-specific layout choices that were open
 * in PR #3 are now settled by the pinned official release source:
 *
 *   deepseek-ai/DeepSeek-V4-Flash-0731
 *   9e165c30e2704aec5d9d593cce3eebd58bbef1cb
 *
 * inference/convert.py expands each packed byte as [low nibble, high nibble]
 * along K, so even logical column indices are the LOW nibble. inference/
 * kernel.py applies one E8M0 weight scale per 32 K values and multiplies the
 * corresponding GEMM partial by that scale. inference/model.py allocates FP4
 * weights as [out, in/2] and scales as [out, in/32] in float8_e8m0fnu.
 *
 * tests/fixtures/deepseek_v4/fp4_release_convention.json freezes the source
 * provenance and an implementation-independent literal case. Real checkpoint
 * header/tensor truth remains Gate A/V0; a real projection remains Gate C/V2.
 */
#ifndef WASTE_FP4_E2M1_H
#define WASTE_FP4_E2M1_H

#include <stddef.h>
#include <stdint.h>

/* Two E2M1 values share one byte. */
#define WASTE_FP4_PER_BYTE 2

/* One UE8M0 scale covers this many consecutive values along K (columns). */
#define WASTE_UE8M0_BLOCK 32

/* Verified against the pinned 0731 convert.py: low nibble is lower K index. */
#define WASTE_FP4_LOW_NIBBLE_IS_EVEN 1

/* Decode one 4-bit code. Only the low 4 bits of `code` are read.
 * E2M1 has no infinities and no NaN: all 16 codes are finite. */
float waste_e2m1_decode(uint8_t code);

/* Decode one UE8M0 scale byte to 2^(e-127).
 * 0xFF is NaN per the OCP MX E8M0 definition; 0x00..0xFE map to
 * 2^-127 .. 2^127, all representable in binary32 (2^-127 is subnormal). */
float waste_ue8m0_decode(uint8_t e);
int   waste_ue8m0_is_nan(uint8_t e);

/* True when a [rows, cols] matrix can be stored in this format at all:
 * columns must divide evenly into both nibble pairs and scale blocks.
 * Checked rather than assumed — a checkpoint is untrusted input, and a
 * ragged final block would otherwise read past the scale plane. */
int waste_fp4_dims_ok(size_t rows, size_t cols);

/* Bytes needed by each plane of a [rows, cols] matrix. */
size_t waste_fp4_weight_bytes(size_t rows, size_t cols);
size_t waste_fp4_scale_bytes(size_t rows, size_t cols);

/* One element. `w` is the packed plane, `s` the UE8M0 plane. */
float waste_fp4_at(const uint8_t *w, const uint8_t *s,
                   size_t cols, size_t r, size_t c);

/* One full row, dequantized to f32. `out` holds `cols` floats. */
void waste_fp4_decode_row(const uint8_t *w, const uint8_t *s,
                          size_t cols, size_t r, float *out);

/* y[r] = sum_c value(r, c) * x[c], for r in [0, rows).
 * Accumulates in double so the reference does not itself become a source
 * of error when a SIMD path is compared against it. */
void waste_fp4_matvec(const uint8_t *w, const uint8_t *s,
                      size_t rows, size_t cols,
                      const float *x, float *y);

#endif /* WASTE_FP4_E2M1_H */
