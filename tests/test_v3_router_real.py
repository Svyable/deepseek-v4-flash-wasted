#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Replay frozen real learned/hash router fixtures through scalar C."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "tests", "fixtures", "deepseek_v4", "v3_router_real")


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
        print("SKIP: real router fixture is not frozen yet")
        return 77
    with open(pp, encoding="utf-8") as f:
        p = json.load(f)
    if p.get("revision") != "9e165c30e2704aec5d9d593cce3eebd58bbef1cb":
        print("FAIL: router fixture revision mismatch")
        return 1

    paths = {
        "lw": os.path.join(FIXTURE, "layer3-gate-weight.bf16.bin"),
        "lb": os.path.join(FIXTURE, "layer3-gate-bias.f32.bin"),
        "hw": os.path.join(FIXTURE, "layer0-gate-weight.bf16.bin"),
        "tid": os.path.join(FIXTURE, "layer0-tid2eid-token4242.i64.bin"),
        "x": os.path.join(FIXTURE, "input.bf16.bin"),
        "ld": os.path.join(FIXTURE, "learned-diagnostics.f32.bin"),
        "li": os.path.join(FIXTURE, "learned-ids.u32.bin"),
        "lwt": os.path.join(FIXTURE, "learned-weights.f32.bin"),
        "hd": os.path.join(FIXTURE, "hash-diagnostics.f32.bin"),
        "hi": os.path.join(FIXTURE, "hash-ids.u32.bin"),
        "hwt": os.path.join(FIXTURE, "hash-weights.f32.bin"),
    }
    if not all(os.path.isfile(v) for v in paths.values()):
        print("FAIL: incomplete real router fixture")
        return 1

    checks = {
        paths["lw"]: p["learned"]["weight"]["payload_sha256"],
        paths["lb"]: p["learned"]["bias"]["payload_sha256"],
        paths["hw"]: p["hash"]["weight"]["payload_sha256"],
        paths["tid"]: p["hash"]["tid2eid"]["payload_sha256"],
        paths["x"]: p["input"]["sha256"],
        paths["ld"]: p["learned"]["diagnostics_sha256"],
        paths["li"]: p["learned"]["ids_sha256"],
        paths["lwt"]: p["learned"]["weights_sha256"],
        paths["hd"]: p["hash"]["diagnostics_sha256"],
        paths["hi"]: p["hash"]["ids_sha256"],
        paths["hwt"]: p["hash"]["weights_sha256"],
    }
    for path, want in checks.items():
        got = sha256(path)
        if got != want:
            print(f"FAIL: {os.path.basename(path)} sha256 {got} != {want}")
            return 1

    cc = compiler()
    if not cc:
        print("SKIP: no C compiler available")
        return 77

    probe = r'''
#include "deepseek_v4_router_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 4096u
#define E 256u
#define K 6u

static int read_exact(const char *path, void *dst, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(dst, 1, n, f);
    int extra = fgetc(f);
    fclose(f);
    return got == n && extra == EOF ? 0 : -1;
}

int main(int argc, char **argv)
{
    if (argc != 12) return 2;
    uint16_t *lw = malloc(E * DIM * sizeof(uint16_t));
    uint16_t *hw = malloc(E * DIM * sizeof(uint16_t));
    uint16_t xb[DIM];
    float x[DIM], bias[E];
    int64_t tid[K];
    float want_ld[3*E], want_hd[2*E], want_lw[K], want_hw[K];
    uint32_t want_li[K], want_hi[K];
    if (!lw || !hw) return 2;

    if (read_exact(argv[1], lw, E*DIM*sizeof(uint16_t)) ||
        read_exact(argv[2], bias, sizeof bias) ||
        read_exact(argv[3], hw, E*DIM*sizeof(uint16_t)) ||
        read_exact(argv[4], tid, sizeof tid) ||
        read_exact(argv[5], xb, sizeof xb) ||
        read_exact(argv[6], want_ld, sizeof want_ld) ||
        read_exact(argv[7], want_li, sizeof want_li) ||
        read_exact(argv[8], want_lw, sizeof want_lw) ||
        read_exact(argv[9], want_hd, sizeof want_hd) ||
        read_exact(argv[10], want_hi, sizeof want_hi) ||
        read_exact(argv[11], want_hw, sizeof want_hw)) {
        fprintf(stderr, "router fixture read failed\n");
        return 2;
    }
    for (size_t i = 0; i < DIM; i++) x[i] = waste_ds_v4_bf16_to_f32(xb[i]);

    float lraw[E], lscore[E], lsel[E], lweights[K];
    uint32_t lids[K];
    if (waste_ds_v4_router_score_ref(x, DIM, lw, lraw, lscore) ||
        waste_ds_v4_router_learned_ref(lscore, bias, 1.5f,
                                       lids, lweights, lsel)) {
        fprintf(stderr, "learned router rejected real fixture\n");
        return 1;
    }

    for (size_t j = 0; j < K; j++) {
        if (lids[j] != want_li[j]) {
            fprintf(stderr, "learned id[%zu]=%u want %u\n", j, lids[j], want_li[j]);
            return 1;
        }
    }

    float hraw[E], hscore[E], hweights[K];
    uint32_t hids[K];
    if (waste_ds_v4_router_score_ref(x, DIM, hw, hraw, hscore) ||
        waste_ds_v4_router_hash_ref(hscore, tid, 1.5f, hids, hweights)) {
        fprintf(stderr, "hash router rejected real fixture\n");
        return 1;
    }
    for (size_t j = 0; j < K; j++) {
        if (hids[j] != want_hi[j] || hids[j] != (uint32_t)tid[j]) {
            fprintf(stderr, "hash id[%zu]=%u want %u/tid %lld\n",
                    j, hids[j], want_hi[j], (long long)tid[j]);
            return 1;
        }
    }

    float max_abs = 0.0f;
    size_t max_where = 0;
    for (size_t i = 0; i < E; i++) {
        float vals[5] = {lraw[i], lscore[i], lsel[i], hraw[i], hscore[i]};
        float wants[5] = {want_ld[i], want_ld[E+i], want_ld[2*E+i],
                          want_hd[i], want_hd[E+i]};
        for (size_t q = 0; q < 5; q++) {
            float err = fabsf(vals[q] - wants[q]);
            if (err > max_abs) { max_abs = err; max_where = q*E+i; }
        }
    }
    for (size_t j = 0; j < K; j++) {
        float e1 = fabsf(lweights[j] - want_lw[j]);
        float e2 = fabsf(hweights[j] - want_hw[j]);
        if (e1 > max_abs) { max_abs = e1; max_where = 5*E+j; }
        if (e2 > max_abs) { max_abs = e2; max_where = 5*E+K+j; }
    }
    if (max_abs > 2e-5f) {
        fprintf(stderr, "router diagnostic max_abs %.9g at %zu exceeds 2e-5\n",
                max_abs, max_where);
        return 1;
    }

    float lsum = 0.0f, hsum = 0.0f;
    for (size_t j = 0; j < K; j++) { lsum += lweights[j]; hsum += hweights[j]; }
    if (fabsf(lsum - 1.5f) > 2e-6f || fabsf(hsum - 1.5f) > 2e-6f) {
        fprintf(stderr, "router sums %.9g %.9g != 1.5\n", lsum, hsum);
        return 1;
    }

    printf("PASS real 0731 learned+hash router: exact IDs, diagnostic max_abs %.9g\n",
           max_abs);
    free(lw); free(hw);
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="router-real-") as td:
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
            print("FAIL: could not compile real router replay")
            return 1
        ran = subprocess.run(
            [exe, paths["lw"], paths["lb"], paths["hw"], paths["tid"], paths["x"],
             paths["ld"], paths["li"], paths["lwt"], paths["hd"], paths["hi"], paths["hwt"]],
            capture_output=True, text=True)
        sys.stdout.write(ran.stdout)
        sys.stderr.write(ran.stderr)
        return ran.returncode


if __name__ == "__main__":
    sys.exit(main())
