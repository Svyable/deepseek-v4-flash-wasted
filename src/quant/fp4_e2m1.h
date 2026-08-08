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
 * WHAT THIS FILE ASSUMES, AND WHY IT MATTERS
 *
 * The number formats below are public specifications (OCP Microscaling),
 * and `tests/test_quant.c` proves conformance to them exhaustively — all
 * 16 E2M1 codes and all 256 UE8M0 codes. That is a real proof, but it is a
 * proof about the *spec*, not about the checkpoint.
 *
 * Two choices cannot be settled from a spec and are not yet verified
 * against DeepSeek's reference, because huggingface.co was unreachable
 * (docs/INVENTORY-0731.md):
 *
 *   1. NIBBLE ORDER. This file puts even column indices in the LOW nibble.
 *      If the reference packs the other way, every decoded matrix is
 *      column-swapped in pairs — which produces plausible-looking garbage
 *      rather than an obvious failure. See WASTE_FP4_LOW_NIBBLE_IS_EVEN.
 *   2. SCALE DIRECTION. Here the stored UE8M0 exponent is a multiplier.
 *      A reference that stores the reciprocal would need `1/s`.
 *
 * Both are marked TBD-GATE0 in docs/TENSOR_MAP.md. Resolve them against
 * the official `inference/kernel.py` before trusting a decoded expert.
 */
#ifndef WASTE_FP4_E2M1_H
#define WASTE_FP4_E2M1_H

#include <stddef.h>
#include <stdint.h>

/* Two E2M1 values share one byte. */
#define WASTE_FP4_PER_BYTE 2

/* One UE8M0 scale covers this many consecutive values along K (columns). */
#define WASTE_UE8M0_BLOCK 32

/* Even column index -> low nibble. See the assumption note above. */
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
