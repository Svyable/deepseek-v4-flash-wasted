#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free invariants for DeepSeek V4 router semantics."""

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
#include "deepseek_v4_router_ref.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int bad;
static void ck(int cond, const char *msg)
{
    if (!cond) { bad++; fprintf(stderr, "FAIL %s\n", msg); }
}

int main(void)
{
    float score[256], bias[256], selection[256];
    uint32_t ids[6];
    float weights[6];
    for (int i = 0; i < 256; i++) {
        score[i] = 1.0f + (float)i / 1000.0f;
        bias[i] = 0.0f;
    }

    /* Make experts 3,5,7,9,11,13 win selection by bias even though their
     * unbiased scores differ. Weights must still come from score[], not from
     * the huge biased selection values. */
    const uint32_t want[6] = {13,11,9,7,5,3};
    for (int j = 0; j < 6; j++)
        bias[want[j]] = 100.0f + (float)j;

    ck(waste_ds_v4_router_learned_ref(score, bias, 1.5f,
                                      ids, weights, selection) == 0,
       "learned router accepts valid fixture");
    for (int j = 0; j < 6; j++)
        ck(ids[j] == want[5-j], "learned selection follows correction bias order");

    float sum = 0.0f;
    for (int j = 0; j < 6; j++) sum += weights[j];
    ck(fabsf(sum - 1.5f) < 2e-6f, "learned weights normalize then scale to 1.5");
    ck(weights[0] < 0.3f,
       "learned weight does not use ~100-point correction-biased score");

    int64_t tid[6] = {200, 3, 17, 99, 5, 250};
    uint32_t hids[6];
    float hweights[6];
    ck(waste_ds_v4_router_hash_ref(score, tid, 1.5f, hids, hweights) == 0,
       "hash router accepts checkpoint IDs");
    for (int j = 0; j < 6; j++)
        ck(hids[j] == (uint32_t)tid[j], "hash router preserves tid2eid order");
    sum = 0.0f;
    for (int j = 0; j < 6; j++) sum += hweights[j];
    ck(fabsf(sum - 1.5f) < 2e-6f, "hash weights normalize then scale to 1.5");

    ck(waste_ds_v4_sqrt_softplus_ref(25.0f) == 5.0f,
       "softplus threshold path returns sqrt(x) above 20");
    ck(isfinite(waste_ds_v4_sqrt_softplus_ref(-100.0f)),
       "negative softplus path remains finite");

    if (bad) return 1;
    puts("PASS model-free router bias-selection + hash-ID semantics");
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="router-scalar-") as td:
        src = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe.exe" if os.name == "nt" else "probe")
        with open(src, "w", encoding="utf-8") as f:
            f.write(probe)
        cmd = [
            cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-I", os.path.join(REPO, "src"),
            src, os.path.join(REPO, "src", "deepseek_v4_router_ref.c"),
            "-lm", "-o", exe,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            sys.stderr.write(built.stdout)
            sys.stderr.write(built.stderr)
            print("FAIL: could not compile model-free router test")
            return 1
        ran = subprocess.run([exe], capture_output=True, text=True)
        sys.stdout.write(ran.stdout)
        sys.stderr.write(ran.stderr)
        return ran.returncode


if __name__ == "__main__":
    sys.exit(main())
