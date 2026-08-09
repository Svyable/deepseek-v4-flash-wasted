#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen real-checkpoint Gate C projection through scalar C."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    REPO, "tests", "fixtures", "deepseek_v4", "v2_wq_a_real")


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
    if not os.path.exists(provenance_path):
        print("SKIP: real Gate C fixture is not frozen yet")
        return 77

    with open(provenance_path, encoding="utf-8") as f:
        p = json.load(f)
    if p.get("revision") != "9e165c30e2704aec5d9d593cce3eebd58bbef1cb":
        print("FAIL: Gate C fixture is not pinned to the canonical 0731 release")
        return 1

    weight = os.path.join(FIXTURE, "weight-rows0-8.e4m3.bin")
    scales = os.path.join(FIXTURE, "weight-scale-row0.e8m0.bin")
    x = os.path.join(FIXTURE, "input.bf16.bin")
    expected = os.path.join(FIXTURE, "expected.bf16.bin")
    required = [weight, scales, x, expected]
    if not all(os.path.isfile(path) for path in required):
        print("FAIL: incomplete Gate C fixture")
        return 1

    checks = {
        weight: p["weight"]["payload_sha256"],
        scales: p["weight_scale"]["payload_sha256"],
        x: p["input"]["sha256"],
        expected: p["expected"]["sha256"],
    }
    for path, want in checks.items():
        got = sha256(path)
        if got != want:
            print(f"FAIL: fixture hash mismatch for {os.path.basename(path)}: {got} != {want}")
            return 1

    if os.path.getsize(weight) != 8 * 4096:
        print("FAIL: real weight slice must be exactly 8x4096 E4M3 bytes")
        return 1
    if os.path.getsize(scales) != 32:
        print("FAIL: real weight-scale row must contain exactly 32 E8M0 bytes")
        return 1
    if os.path.getsize(x) != 4096 * 2 or os.path.getsize(expected) != 8 * 2:
        print("FAIL: input/expected fixture dimensions are inconsistent")
        return 1

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    probe = r'''
#include "deepseek_v4_linear_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 4096u
#define N 8u
#define KB (K / 128u)

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

static uint16_t f32_as_bf16_exact(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    if ((u & 0xffffu) != 0u) {
        fprintf(stderr, "reference returned non-BF16-representable f32: 0x%08x\n", u);
        exit(3);
    }
    return (uint16_t)(u >> 16);
}

static float e8m0(uint8_t code)
{
    if (code == 0xffu) return NAN;
    return ldexpf(1.0f, (int)code - 127);
}

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    uint8_t *w = malloc(N * K);
    uint8_t raw_s[KB];
    uint16_t raw_x[K];
    uint16_t want[N];
    float x[K];
    float scales[KB];
    float y[N];
    if (!w) return 2;

    if (read_exact(argv[1], w, N * K) ||
        read_exact(argv[2], raw_s, sizeof raw_s) ||
        read_exact(argv[3], raw_x, sizeof raw_x) ||
        read_exact(argv[4], want, sizeof want)) {
        free(w);
        fprintf(stderr, "could not read exact fixture files\n");
        return 2;
    }

    for (size_t i = 0; i < K; i++) x[i] = bf16_to_f32(raw_x[i]);
    for (size_t i = 0; i < KB; i++) {
        scales[i] = e8m0(raw_s[i]);
        if (!isfinite(scales[i])) {
            free(w);
            fprintf(stderr, "NaN E8M0 scale in fixture\n");
            return 2;
        }
    }

    if (waste_ds_v4_fp8_linear_ref(x, 1, K, w, scales, N, y) != 0) {
        free(w);
        fprintf(stderr, "scalar reference rejected real projection\n");
        return 1;
    }

    for (size_t i = 0; i < N; i++) {
        uint16_t got = f32_as_bf16_exact(y[i]);
        if (got != want[i]) {
            free(w);
            fprintf(stderr, "row %zu: got BF16 0x%04x want 0x%04x (%.9g)\n",
                    i, got, want[i], y[i]);
            return 1;
        }
    }
    free(w);
    puts("PASS real 0731 wq_a projection: scalar C == independent source oracle (exact BF16)");
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="v2-real-") as td:
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
            print("FAIL: could not compile real Gate C replay")
            return 1
        ran = subprocess.run(
            [exe, weight, scales, x, expected], capture_output=True, text=True)
        sys.stdout.write(ran.stdout)
        sys.stderr.write(ran.stderr)
        if ran.returncode != 0:
            print("FAIL: scalar C disagrees with frozen real Gate C fixture")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
