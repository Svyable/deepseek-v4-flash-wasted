#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free preflight for the Gate C/V2 scalar quantized-linear seam.

This does NOT pass Gate C/V2: no real checkpoint projection is involved. It
checks source-derived arithmetic with closed-form values chosen so activation
quantization and both K-block scales have exact expected results.
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compiler():
    preferred = os.environ.get("CC")
    if preferred:
        return preferred
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    probe = r'''
#include "deepseek_v4_linear_ref.h"
#include "fp8_e4m3.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int bad;
static void ck(int cond, const char *msg)
{
    if (!cond) {
        bad++;
        fprintf(stderr, "FAIL %s\n", msg);
    }
}

int main(void)
{
    /* Official source rule: amax/448 rounded upward to a power of two.
     * For amax=1 this is 2^-8; normalized activation is exactly 256, E4M3
     * code 0x78. For amax=.75 it is 2^-9; normalized value is exactly 384. */
    float one[128], three_quarters[128];
    for (int i = 0; i < 128; i++) {
        one[i] = 1.0f;
        three_quarters[i] = 0.75f;
    }
    ck(waste_ds_v4_act_scale_ref(one, 128) == 0.00390625f,
       "amax=1 -> activation scale 2^-8");
    ck(waste_ds_v4_act_scale_ref(three_quarters, 128) == 0.001953125f,
       "amax=.75 -> activation scale 2^-9");
    ck(waste_ds_v4_e4m3_encode_ref(256.0f) == 0x78,
       "256 is exact E4M3FN code 0x78");
    ck(waste_ds_v4_e4m3_encode_ref(384.0f) == 0x7C,
       "384 is exact E4M3FN code 0x7C");
    ck(waste_ds_v4_e4m3_encode_ref(-256.0f) == 0xF8,
       "negative sign is retained by E4M3 encoder");

    enum { M = 1, K = 256, N = 2 };
    float x[M * K];
    uint8_t w[N * K];
    float scales[1 * 2];
    float y[M * N];

    /* Block 0 dequantizes exactly to 1.0. Block 1 dequantizes exactly to
     * .75. Every raw weight byte is E4M3 code 0x38 = 1.0. Weight scales are
     * 1 and 2 for the two K blocks, so each output is:
     *
     *   128 * 1.0 * 1.0 + 128 * .75 * 2.0 = 320
     *
     * 320 is exactly representable in BF16, so no tolerance is needed. */
    for (int i = 0; i < 128; i++) x[i] = 1.0f;
    for (int i = 128; i < K; i++) x[i] = 0.75f;
    for (int i = 0; i < N * K; i++) w[i] = 0x38;
    scales[0] = 1.0f;
    scales[1] = 2.0f;

    ck(waste_ds_v4_fp8_linear_ref(x, M, K, w, scales, N, y) == 0,
       "reference linear accepts a valid 256-wide projection");
    ck(y[0] == 320.0f && y[1] == 320.0f,
       "two scaled K blocks accumulate to the closed-form output 320");

    /* A non-128 K is invalid under the pinned official act_quant contract. */
    ck(waste_ds_v4_fp8_linear_ref(x, M, 129, w, scales, N, y) == -1,
       "non-128 activation K block is refused");

    /* BF16 tie-to-even smoke pins the final store as a deliberate operation,
     * not an accidental f32 output. */
    ck(waste_ds_v4_bf16_round_ref(1.0f) == 1.0f, "BF16 exact one");

    if (bad)
        return 1;
    puts("PASS source-derived Gate C scalar linear preflight");
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="v2-linear-") as td:
        src = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe.exe" if os.name == "nt" else "probe")
        with open(src, "w", encoding="utf-8") as f:
            f.write(probe)
        cmd = [
            cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-I", os.path.join(REPO, "src", "quant"),
            src,
            os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
            os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
            "-lm", "-o", exe,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            sys.stderr.write(built.stdout)
            sys.stderr.write(built.stderr)
            print("FAIL: could not compile Gate C scalar preflight")
            return 1
        ran = subprocess.run([exe], capture_output=True, text=True)
        if ran.returncode != 0:
            sys.stderr.write(ran.stdout)
            sys.stderr.write(ran.stderr)
            print("FAIL: Gate C scalar preflight")
            return 1
        sys.stdout.write(ran.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
