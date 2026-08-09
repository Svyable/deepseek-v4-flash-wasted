#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free preflight for DeepSeek quantized-linear scalar seams."""

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

    /* ---------------- resident FP8 ---------------- */
    enum { M = 1, K = 256, N = 2 };
    float x[M * K];
    uint8_t w[N * K];
    float scales[1 * 2];
    float y[M * N];

    for (int i = 0; i < 128; i++) x[i] = 1.0f;
    for (int i = 128; i < K; i++) x[i] = 0.75f;
    for (int i = 0; i < N * K; i++) w[i] = 0x38;  /* E4M3 1.0 */
    scales[0] = 1.0f;
    scales[1] = 2.0f;

    ck(waste_ds_v4_fp8_linear_ref(x, M, K, w, scales, N, y) == 0,
       "FP8 reference linear accepts valid projection");
    ck(y[0] == 320.0f && y[1] == 320.0f,
       "FP8 two scaled K128 blocks accumulate to 320");
    ck(waste_ds_v4_fp8_linear_ref(x, M, 129, w, scales, N, y) == -1,
       "FP8 non-K128 input is refused");

    /* ---------------- routed-expert FP4 ----------------
     * x=1 -> activation scale 2^-8 -> quantized activation 256.
     * FP4 code 0x2 = 1.0, packed low/high as byte 0x22.
     * Every K32 weight scale = E8M0 0x7f = 1.0.
     * Each sub-block contributes 32*256*(2^-8)*1 = 32, four -> 128. */
    float x4[128];
    uint8_t w4[64];
    uint8_t s4[4];
    float y4[1];
    for (int i = 0; i < 128; i++) x4[i] = 1.0f;
    for (int i = 0; i < 64; i++) w4[i] = 0x22;
    for (int i = 0; i < 4; i++) s4[i] = 0x7f;
    ck(waste_ds_v4_fp4_linear_ref(x4, 1, 128, w4, s4, 1, y4) == 0,
       "FP4 expert reference linear accepts valid K128 projection");
    ck(y4[0] == 128.0f,
       "FP8-activation x FP4-weight four K32 blocks accumulate to 128");
    ck(waste_ds_v4_fp4_linear_ref(x4, 1, 96, w4, s4, 1, y4) == -1,
       "FP4 expert linear refuses non-K128 input");

    ck(waste_ds_v4_bf16_round_ref(1.0f) == 1.0f, "BF16 exact one");

    if (bad)
        return 1;
    puts("PASS source-derived FP8 + FP4 scalar linear preflight");
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
            os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
            "-lm", "-o", exe,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            sys.stderr.write(built.stdout)
            sys.stderr.write(built.stderr)
            print("FAIL: could not compile quantized-linear scalar preflight")
            return 1
        ran = subprocess.run([exe], capture_output=True, text=True)
        if ran.returncode != 0:
            sys.stderr.write(ran.stdout)
            sys.stderr.write(ran.stderr)
            print("FAIL: quantized-linear scalar preflight")
            return 1
        sys.stdout.write(ran.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
