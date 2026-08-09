#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen real layer-0 mHC Gate D fixture through scalar C."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v3_mhc_real")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    provenance_path = os.path.join(FIXTURE, "provenance.json")
    if not os.path.isfile(provenance_path):
        print("SKIP: real Gate D mHC fixture is not frozen yet")
        return 77

    with open(provenance_path, encoding="utf-8") as f:
        p = json.load(f)
    if p.get("revision") != "9e165c30e2704aec5d9d593cce3eebd58bbef1cb":
        print("FAIL: Gate D fixture revision is not the canonical 0731 release")
        return 1

    files = {
        "fn": os.path.join(FIXTURE, "hc_attn_fn.f32.bin"),
        "base": os.path.join(FIXTURE, "hc_attn_base.f32.bin"),
        "scale": os.path.join(FIXTURE, "hc_attn_scale.f32.bin"),
        "residual": os.path.join(FIXTURE, "residual.bf16.bin"),
        "branch": os.path.join(FIXTURE, "branch.bf16.bin"),
        "pre": os.path.join(FIXTURE, "expected-pre.bf16.bin"),
        "post": os.path.join(FIXTURE, "expected-post.bf16.bin"),
        "diag": os.path.join(FIXTURE, "expected-diagnostics.f32.bin"),
    }
    if not all(os.path.isfile(path) for path in files.values()):
        print("FAIL: incomplete Gate D fixture")
        return 1

    checks = {
        files["fn"]: p["parameters"]["fn"]["payload_sha256"],
        files["base"]: p["parameters"]["base"]["payload_sha256"],
        files["scale"]: p["parameters"]["scale"]["payload_sha256"],
        files["residual"]: p["inputs"]["residual"]["sha256"],
        files["branch"]: p["inputs"]["branch"]["sha256"],
        files["pre"]: p["expected"]["pre_sha256"],
        files["post"]: p["expected"]["post_sha256"],
        files["diag"]: p["diagnostics"]["sha256"],
    }
    for path, want in checks.items():
        got = sha256(path)
        if got != want:
            print(f"FAIL: {os.path.basename(path)} sha256 {got} != {want}")
            return 1

    expected_sizes = {
        "fn": 24 * 16384 * 4,
        "base": 24 * 4,
        "scale": 3 * 4,
        "residual": 4 * 4096 * 2,
        "branch": 4096 * 2,
        "pre": 4096 * 2,
        "post": 4 * 4096 * 2,
        "diag": (24 + 4 + 4 + 16) * 4,
    }
    for key, want in expected_sizes.items():
        got = os.path.getsize(files[key])
        if got != want:
            print(f"FAIL: {key} size {got} != {want}")
            return 1

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    probe = r'''
#include "deepseek_v4_mhc_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 4096u
#define FLAT (4u * DIM)
#define MIX 24u
#define DIAG 48u

static int read_exact(const char *path, void *dst, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(dst, 1, n, f);
    int extra = fgetc(f);
    fclose(f);
    return got == n && extra == EOF ? 0 : -1;
}

static float bf16_to_f32(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

static uint16_t bf16_round(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    uint32_t exp = u & 0x7f800000u;
    uint32_t frac = u & 0x007fffffu;
    if (exp == 0x7f800000u) {
        if (frac) u |= 0x00400000u;
        return (uint16_t)(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static int compare_bf16(const char *label, const float *got,
                        const uint16_t *want, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint16_t b = bf16_round(got[i]);
        if (b != want[i]) {
            fprintf(stderr, "%s[%zu]: got 0x%04x (%.9g), want 0x%04x\n",
                    label, i, b, got[i], want[i]);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 9) return 2;

    float *fn = malloc(MIX * FLAT * sizeof(float));
    float base[MIX], scale[3];
    uint16_t residual_b[FLAT], branch_b[DIM], want_pre[DIM], want_post[FLAT];
    float want_diag[DIAG];
    float *residual = malloc(FLAT * sizeof(float));
    float *branch = malloc(DIM * sizeof(float));
    float *pre_y = malloc(DIM * sizeof(float));
    float *post_y = malloc(FLAT * sizeof(float));
    if (!fn || !residual || !branch || !pre_y || !post_y) return 2;

    if (read_exact(argv[1], fn, MIX * FLAT * sizeof(float)) ||
        read_exact(argv[2], base, sizeof base) ||
        read_exact(argv[3], scale, sizeof scale) ||
        read_exact(argv[4], residual_b, sizeof residual_b) ||
        read_exact(argv[5], branch_b, sizeof branch_b) ||
        read_exact(argv[6], want_pre, sizeof want_pre) ||
        read_exact(argv[7], want_post, sizeof want_post) ||
        read_exact(argv[8], want_diag, sizeof want_diag)) {
        fprintf(stderr, "fixture read failed\n");
        return 2;
    }

    for (size_t i = 0; i < FLAT; i++) residual[i] = bf16_to_f32(residual_b[i]);
    for (size_t i = 0; i < DIM; i++) branch[i] = bf16_to_f32(branch_b[i]);

    float mixes[MIX], pre[4], post[4], comb[16];
    if (waste_ds_v4_hc_pre_ref(residual, DIM, fn, scale, base,
                               1e-6f, 20, 1e-6f,
                               pre_y, mixes, pre, post, comb) != 0) {
        fprintf(stderr, "hc_pre rejected fixture\n");
        return 1;
    }
    if (waste_ds_v4_hc_post_ref(branch, residual, DIM, post, comb, post_y) != 0) {
        fprintf(stderr, "hc_post rejected fixture\n");
        return 1;
    }

    if (compare_bf16("hc_pre", pre_y, want_pre, DIM) ||
        compare_bf16("hc_post", post_y, want_post, FLAT))
        return 1;

    /* Diagnostic values are f32 transcendental/Sinkhorn outputs. They localize
     * a mismatch without pretending libm has bit-identical sigmoid/exp across
     * platforms. The final BF16 state above remains exact. */
    float got_diag[DIAG];
    size_t q = 0;
    for (size_t i = 0; i < MIX; i++) got_diag[q++] = mixes[i];
    for (size_t i = 0; i < 4; i++) got_diag[q++] = pre[i];
    for (size_t i = 0; i < 4; i++) got_diag[q++] = post[i];
    for (size_t i = 0; i < 16; i++) got_diag[q++] = comb[i];

    float max_abs = 0.0f;
    size_t max_i = 0;
    for (size_t i = 0; i < DIAG; i++) {
        float e = fabsf(got_diag[i] - want_diag[i]);
        if (e > max_abs) { max_abs = e; max_i = i; }
    }
    if (max_abs > 2e-4f) {
        fprintf(stderr, "mHC diagnostic max_abs %.9g at %zu exceeds 2e-4\n",
                max_abs, max_i);
        return 1;
    }

    for (size_t k = 0; k < 4; k++) {
        float col = 0.0f;
        for (size_t j = 0; j < 4; j++) col += comb[j * 4 + k];
        if (fabsf(col - 1.0f) > 1e-4f) {
            fprintf(stderr, "comb column %zu sums to %.9g\n", k, col);
            return 1;
        }
    }

    printf("PASS real 0731 mHC: exact BF16 pre/post, diagnostic max_abs %.9g\n",
           max_abs);
    free(fn); free(residual); free(branch); free(pre_y); free(post_y);
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="v3-mhc-") as td:
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
            print("FAIL: could not compile Gate D mHC replay")
            return 1
        ran = subprocess.run(
            [exe, files["fn"], files["base"], files["scale"], files["residual"],
             files["branch"], files["pre"], files["post"], files["diag"]],
            capture_output=True, text=True)
        sys.stdout.write(ran.stdout)
        sys.stderr.write(ran.stderr)
        if ran.returncode != 0:
            print("FAIL: scalar C disagrees with frozen real mHC fixture")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
