#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free Gate H test for the complete eight-group attention output path."""

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
        p = shutil.which(name)
        if p:
            return p
    return None


def main():
    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    probe = r'''
#include "deepseek_v4_attention_output_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GROUPS 8u
#define GROUP_IN 4096u
#define GROUP_RANK 1024u
#define FLAT_RANK 8192u
#define WO_A_ROWS 8192u

int main(void)
{
    float *heads = calloc(GROUPS * GROUP_IN, sizeof(float));
    uint8_t *wo_a = calloc(WO_A_ROWS * GROUP_IN, 1);
    uint8_t *wo_as = malloc(64u * 32u);
    uint8_t *wo_b = calloc(FLAT_RANK, 1); /* one output row */
    uint8_t *wo_bs = malloc(64u);          /* one K-scale row */
    float *lora = calloc(FLAT_RANK, sizeof(float));
    float out[1] = {0};
    if (!heads || !wo_a || !wo_as || !wo_b || !wo_bs || !lora)
        return 2;

    /* E4M3 1.0 = 0x38, E8M0 scale 1.0 = exponent byte 127. */
    for (size_t i = 0; i < 64u * 32u; i++) wo_as[i] = 127u;
    for (size_t i = 0; i < 64u; i++) wo_bs[i] = 127u;

    /* Each group contributes a distinct exact dyadic value through latent row 0.
     * wo_b row 0 then sums those eight latent coordinates. */
    float want = 0.0f;
    for (size_t g = 0; g < GROUPS; g++) {
        float x = (float)(g + 1u) / 8.0f; /* 0.125 .. 1.0, exactly E4M3 */
        heads[g * GROUP_IN] = x;
        size_t wr = g * GROUP_RANK;       /* first wo_a row of group g */
        wo_a[wr * GROUP_IN] = 0x38u;
        wo_b[g * GROUP_RANK] = 0x38u;
        want += x;
    }

    if (waste_ds_v4_attention_output_full_ref(
            heads, wo_a, wo_as, wo_b, wo_bs, 1, lora, out) != 0) {
        fprintf(stderr, "full output reference rejected valid construction\n");
        return 1;
    }
    if (fabsf(out[0] - want) > 1e-6f) {
        fprintf(stderr, "all-group output %.9g != expected %.9g\n", out[0], want);
        return 1;
    }
    for (size_t g = 0; g < GROUPS; g++) {
        float want_lora = (float)(g + 1u) / 8.0f;
        if (lora[g * GROUP_RANK] != want_lora) {
            fprintf(stderr, "group %zu latent %.9g != %.9g\n",
                    g, lora[g * GROUP_RANK], want_lora);
            return 1;
        }
    }

    /* Mutation: silently keep only group 0. This is exactly the incorrect
     * extrapolation from the old Gate-E one-group seam and must not pass. */
    for (size_t g = 1; g < GROUPS; g++)
        heads[g * GROUP_IN] = 0.0f;
    float mutated[1] = {0};
    if (waste_ds_v4_attention_output_full_ref(
            heads, wo_a, wo_as, wo_b, wo_bs, 1, NULL, mutated) != 0)
        return 1;
    if (mutated[0] == out[0]) {
        fprintf(stderr, "group-0-only mutation left full output unchanged\n");
        return 1;
    }

    printf("PASS Gate H full attention output: all 8 groups feed one wo_b quantized linear\n");
    printf("all-groups %.6f vs group0-only %.6f\n", out[0], mutated[0]);
    free(heads); free(wo_a); free(wo_as); free(wo_b); free(wo_bs); free(lora);
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="v6-attn-out-full-") as td:
        src = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe")
        with open(src, "w", encoding="utf-8") as f:
            f.write(probe)
        cmd = [
            cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I", os.path.join(REPO, "src"),
            src,
            os.path.join(REPO, "src", "deepseek_v4_attention_output_ref.c"),
            os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
            os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
            os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
            "-lm", "-o", exe,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode:
            sys.stderr.write(built.stdout + built.stderr)
            print("FAIL: could not compile full attention-output probe")
            return 1
        ran = subprocess.run([exe], capture_output=True, text=True)
        sys.stdout.write(ran.stdout)
        sys.stderr.write(ran.stderr)
        return ran.returncode


if __name__ == "__main__":
    sys.exit(main())
