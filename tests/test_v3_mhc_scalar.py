#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free invariants for the DeepSeek V4 scalar mHC reference."""

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
#include "deepseek_v4_mhc_ref.h"
#include <math.h>
#include <stdio.h>

static int bad;
static void ck(int cond, const char *msg)
{
    if (!cond) { bad++; fprintf(stderr, "FAIL %s\n", msg); }
}

int main(void)
{
    float mixes[24] = {0}, scale[3] = {1,1,1}, base[24] = {0};
    float pre[4], post[4], comb[16];
    ck(waste_ds_v4_hc_split_sinkhorn(mixes, scale, base, 20, 1e-6f,
                                     pre, post, comb) == 0,
       "zero-logit Sinkhorn accepts valid inputs");
    for (int i = 0; i < 4; i++) {
        ck(fabsf(pre[i] - 0.500001f) < 2e-7f,
           "zero-logit pre is sigmoid(0)+eps");
        ck(fabsf(post[i] - 1.0f) < 2e-7f,
           "zero-logit post is 2*sigmoid(0)");
    }
    for (int j = 0; j < 4; j++) {
        for (int k = 1; k < 4; k++)
            ck(fabsf(comb[j*4+k] - comb[j*4]) < 2e-6f,
               "symmetric Sinkhorn row stays symmetric");
    }
    for (int k = 0; k < 4; k++) {
        float col = 0;
        for (int j = 0; j < 4; j++) col += comb[j*4+k];
        ck(fabsf(col - 1.0f) < 1e-4f, "Sinkhorn column approximately normalized");
    }

    /* Pin hc_post orientation with an intentionally non-symmetric matrix.
     * Output lane k must consume comb[source,k], not comb[k,source]. */
    float branch[1] = {10.0f};
    float residual[4] = {1,2,3,4};
    float post2[4] = {0.1f,0.2f,0.3f,0.4f};
    float comb2[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16,
    };
    float y[4];
    ck(waste_ds_v4_hc_post_ref(branch, residual, 1, post2, comb2, y) == 0,
       "hc_post accepts one-dimensional orientation fixture");
    /* k=0: 1*1 + 5*2 + 9*3 + 13*4 + .1*10 = 91 */
    ck(y[0] == 91.0f, "hc_post uses comb[source,output] for output 0");
    /* k=3: 4*1 + 8*2 + 12*3 + 16*4 + .4*10 = 124 */
    ck(y[3] == 124.0f, "hc_post uses comb[source,output] for output 3");

    if (bad) return 1;
    puts("PASS model-free mHC Sinkhorn invariants + hc_post orientation");
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="v3-mhc-scalar-") as td:
        src = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe.exe" if os.name == "nt" else "probe")
        with open(src, "w", encoding="utf-8") as f:
            f.write(probe)
        cmd = [
            cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-I", os.path.join(REPO, "src"),
            src, os.path.join(REPO, "src", "deepseek_v4_mhc_ref.c"),
            "-lm", "-o", exe,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            sys.stderr.write(built.stdout)
            sys.stderr.write(built.stderr)
            print("FAIL: could not compile model-free mHC invariant test")
            return 1
        ran = subprocess.run([exe], capture_output=True, text=True)
        sys.stdout.write(ran.stdout)
        sys.stderr.write(ran.stderr)
        return ran.returncode


if __name__ == "__main__":
    sys.exit(main())
