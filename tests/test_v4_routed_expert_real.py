#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay the frozen real layer-3 expert-2 primitive through scalar C."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    REPO, "tests", "fixtures", "deepseek_v4", "v4_routed_expert_real")


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
    pp = os.path.join(FIXTURE, "provenance.json")
    if not os.path.isfile(pp):
        print("SKIP: real routed-expert fixture is not frozen yet")
        return 77
    with open(pp, encoding="utf-8") as f:
        p = json.load(f)
    if p.get("revision") != "9e165c30e2704aec5d9d593cce3eebd58bbef1cb":
        print("FAIL: routed-expert fixture revision mismatch")
        return 1
    if p.get("layer") != 3 or p.get("expert") != 2:
        print("FAIL: routed-expert fixture target changed")
        return 1

    paths = {
        "w1": os.path.join(FIXTURE, "w1.fp4.bin"),
        "s1": os.path.join(FIXTURE, "w1-scale.e8m0.bin"),
        "w3": os.path.join(FIXTURE, "w3.fp4.bin"),
        "s3": os.path.join(FIXTURE, "w3-scale.e8m0.bin"),
        "w2": os.path.join(FIXTURE, "w2-rows0-8.fp4.bin"),
        "s2": os.path.join(FIXTURE, "w2-scale-rows0-8.e8m0.bin"),
        "x": os.path.join(FIXTURE, "input.bf16.bin"),
        "gate": os.path.join(FIXTURE, "gate-clamped.f32.bin"),
        "up": os.path.join(FIXTURE, "up-clamped.f32.bin"),
        "hidden": os.path.join(FIXTURE, "hidden-after-route.bf16.bin"),
        "out": os.path.join(FIXTURE, "expected-out8.bf16.bin"),
    }
    if not all(os.path.isfile(v) for v in paths.values()):
        print("FAIL: incomplete routed-expert fixture")
        return 1

    tensor_key = {
        "w1": "w1", "s1": "s1", "w3": "w3",
        "s3": "s3", "w2": "w2", "s2": "s2",
    }
    for key, tkey in tensor_key.items():
        got = sha256(paths[key])
        want = p["tensors"][tkey]["payload_sha256"]
        if got != want:
            print(f"FAIL: {key} sha256 {got} != {want}")
            return 1
    expected_hashes = {
        "x": p["input"]["sha256"],
        "gate": p["intermediate"]["gate_sha256"],
        "up": p["intermediate"]["up_sha256"],
        "hidden": p["intermediate"]["hidden_sha256"],
        "out": p["expected"]["sha256"],
    }
    for key, want in expected_hashes.items():
        got = sha256(paths[key])
        if got != want:
            print(f"FAIL: {key} sha256 {got} != {want}")
            return 1

    expected_sizes = {
        "w1": 2048 * 2048,
        "s1": 2048 * 128,
        "w3": 2048 * 2048,
        "s3": 2048 * 128,
        "w2": 8 * 1024,
        "s2": 8 * 64,
        "x": 4096 * 2,
        "gate": 2048 * 4,
        "up": 2048 * 4,
        "hidden": 2048 * 2,
        "out": 8 * 2,
    }
    for key, want in expected_sizes.items():
        got = os.path.getsize(paths[key])
        if got != want:
            print(f"FAIL: {key} size {got} != {want}")
            return 1

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    route_weight = float(p["route_weight"])
    probe = r'''
#include "deepseek_v4_expert_ref.h"
#include "quant/deepseek_v4_linear_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IN 4096u
#define MID 2048u
#define OUT 8u

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

static uint16_t f32_as_bf16(float x)
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

int main(int argc, char **argv)
{
    if (argc != 13) return 2;
    uint8_t *w1 = malloc(MID * (IN/2));
    uint8_t *s1 = malloc(MID * (IN/32));
    uint8_t *w3 = malloc(MID * (IN/2));
    uint8_t *s3 = malloc(MID * (IN/32));
    uint8_t *w2 = malloc(OUT * (MID/2));
    uint8_t *s2 = malloc(OUT * (MID/32));
    uint16_t xb[IN], want_hidden[MID], want_out[OUT];
    float want_gate[MID], want_up[MID];
    float x[IN], gate[MID], up[MID], hidden[MID], out[OUT];
    if (!w1 || !s1 || !w3 || !s3 || !w2 || !s2) return 2;

    if (read_exact(argv[1], w1, MID*(IN/2)) ||
        read_exact(argv[2], s1, MID*(IN/32)) ||
        read_exact(argv[3], w3, MID*(IN/2)) ||
        read_exact(argv[4], s3, MID*(IN/32)) ||
        read_exact(argv[5], w2, OUT*(MID/2)) ||
        read_exact(argv[6], s2, OUT*(MID/32)) ||
        read_exact(argv[7], xb, sizeof xb) ||
        read_exact(argv[8], want_gate, sizeof want_gate) ||
        read_exact(argv[9], want_up, sizeof want_up) ||
        read_exact(argv[10], want_hidden, sizeof want_hidden) ||
        read_exact(argv[11], want_out, sizeof want_out)) {
        fprintf(stderr, "expert fixture read failed\n");
        return 2;
    }
    float route_weight = strtof(argv[12], NULL);
    for (size_t i = 0; i < IN; i++) x[i] = bf16_to_f32(xb[i]);

    if (waste_ds_v4_routed_expert_ref(
            x, IN, MID, w1, s1, w3, s3, 10.0f, route_weight,
            w2, s2, OUT, gate, up, hidden, out) != 0) {
        fprintf(stderr, "routed expert rejected fixture\n");
        return 1;
    }

    float max_gate = 0.0f, max_up = 0.0f;
    for (size_t i = 0; i < MID; i++) {
        float eg = fabsf(gate[i] - want_gate[i]);
        float eu = fabsf(up[i] - want_up[i]);
        if (eg > max_gate) max_gate = eg;
        if (eu > max_up) max_up = eu;
        if (f32_as_bf16(hidden[i]) != want_hidden[i]) {
            fprintf(stderr, "hidden[%zu] BF16 mismatch got 0x%04x want 0x%04x\n",
                    i, f32_as_bf16(hidden[i]), want_hidden[i]);
            return 1;
        }
    }
    if (max_gate != 0.0f || max_up != 0.0f) {
        fprintf(stderr, "gate/up expected exact decoded BF16/clamp values; max %.9g %.9g\n",
                max_gate, max_up);
        return 1;
    }
    for (size_t i = 0; i < OUT; i++) {
        uint16_t got = f32_as_bf16(out[i]);
        if (got != want_out[i]) {
            fprintf(stderr, "out[%zu] BF16 mismatch got 0x%04x want 0x%04x (%.9g)\n",
                    i, got, want_out[i], out[i]);
            return 1;
        }
    }

    printf("PASS real routed expert 3/2: exact gate/up, hidden BF16, out8 BF16\n");
    free(w1); free(s1); free(w3); free(s3); free(w2); free(s2);
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="expert-real-") as td:
        src = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe.exe" if os.name == "nt" else "probe")
        with open(src, "w", encoding="utf-8") as f:
            f.write(probe)
        cmd = [
            cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-I", os.path.join(REPO, "src"),
            "-I", os.path.join(REPO, "src", "quant"),
            src,
            os.path.join(REPO, "src", "deepseek_v4_expert_ref.c"),
            os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
            os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
            os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
            "-lm", "-o", exe,
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            sys.stderr.write(built.stdout)
            sys.stderr.write(built.stderr)
            print("FAIL: could not compile real routed-expert replay")
            return 1
        ran = subprocess.run(
            [exe, paths["w1"], paths["s1"], paths["w3"], paths["s3"],
             paths["w2"], paths["s2"], paths["x"], paths["gate"], paths["up"],
             paths["hidden"], paths["out"], repr(route_weight)],
            capture_output=True, text=True)
        sys.stdout.write(ran.stdout)
        sys.stderr.write(ran.stderr)
        return ran.returncode


if __name__ == "__main__":
    sys.exit(main())
