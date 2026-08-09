#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Model-free source-contract tests for the ratio-128 compressor."""

import ctypes
import math
import os
import shutil
import struct
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def bf16_bits(x):
    u = struct.unpack("<I", struct.pack("<f", f32(x)))[0]
    if (u & 0x7F800000) == 0x7F800000:
        if u & 0x007FFFFF:
            u |= 0x00400000
        return (u >> 16) & 0xFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16(x):
    return struct.unpack("<f", struct.pack("<I", bf16_bits(x) << 16))[0]


def compile_lib(td):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    out = os.path.join(td, "libcompress.so")
    cmd = [
        cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", os.path.join(REPO, "src"),
        "-I", os.path.join(REPO, "src", "quant"),
        os.path.join(REPO, "src", "deepseek_v4_compressor_ref.c"),
        os.path.join(REPO, "src", "deepseek_v4_attention_ref.c"),
        os.path.join(REPO, "src", "quant", "deepseek_v4_linear_ref.c"),
        os.path.join(REPO, "src", "quant", "fp4_e2m1.c"),
        os.path.join(REPO, "src", "quant", "fp8_e4m3.c"),
        "-lm", "-o", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise AssertionError("compressor scalar compile failed:\n" + p.stdout + p.stderr)
    return out


def arr_f(xs):
    return (ctypes.c_float * len(xs))(*[f32(x) for x in xs])


def pool_python(kv, score, ape, tokens, dim):
    out = []
    for d in range(dim):
        z = [f32(score[t*dim+d] + ape[t*dim+d]) for t in range(tokens)]
        m = max(z)
        e = [f32(math.exp(f32(v-m))) for v in z]
        den = f32(0.0)
        num = f32(0.0)
        for t in range(tokens):
            den = f32(den + e[t])
            num = f32(num + f32(kv[t*dim+d] * e[t]))
        out.append(f32(num / den))
    return out


def correction_dim(rotations, dim, base, original):
    return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))


def yarn_freq(pair, dim=64, original=65536, base=160000.0,
              factor=16.0, beta_fast=32.0, beta_slow=1.0):
    freq = 1.0 / (base ** ((2*pair)/dim))
    low = max(0.0, math.floor(correction_dim(beta_fast, dim, base, original)))
    high = min(dim-1.0, math.ceil(correction_dim(beta_slow, dim, base, original)))
    if low == high:
        high += 0.001
    ramp = min(1.0, max(0.0, (pair-low)/(high-low)))
    smooth = 1.0-ramp
    return freq/factor*(1.0-smooth) + freq*smooth


def main():
    with tempfile.TemporaryDirectory(prefix="v5-compressor-") as td:
        libpath = compile_lib(td)
        if not libpath:
            print("SKIP: no C compiler")
            return 77
        lib = ctypes.CDLL(libpath)

        pool = lib.waste_ds_v4_compressor_pool_ref
        pool.restype = ctypes.c_int
        pool.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_float)]
        tokens, dim = 3, 2
        kv = [1.0, 10.0, 2.0, 20.0, 4.0, 40.0]
        score = [0.0, 3.0, 1.0, 1.0, 2.0, -2.0]
        ape = [0.0, 0.5, 0.2, -0.1, -0.3, 0.7]
        expected = pool_python(kv, score, ape, tokens, dim)
        got = (ctypes.c_float * dim)()
        assert pool(arr_f(kv), arr_f(score), arr_f(ape), tokens, dim, got) == 0
        for i in range(dim):
            assert abs(got[i]-expected[i]) < 2e-6, (i, got[i], expected[i])

        # A wrong shared softmax based on dimension 0 must not reproduce dim 1.
        z0 = [score[t*dim] + ape[t*dim] for t in range(tokens)]
        m0 = max(z0)
        e0 = [math.exp(v-m0) for v in z0]
        wrong_dim1 = sum(kv[t*dim+1]*e0[t] for t in range(tokens))/sum(e0)
        assert abs(expected[1]-wrong_dim1) > 1.0

        rope = lib.waste_ds_v4_compressed_rope_ref
        rope.restype = ctypes.c_int
        rope.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_float,
                         ctypes.c_float, ctypes.c_float]
        x = [bf16(0.0)] * 512
        x[448] = bf16(1.0)       # rope pair 0  — ramp clamps to 0, extrapolated
        x[449] = bf16(-0.25)
        x[488] = bf16(0.5)       # rope pair 20 — ramp 0.5, the blended interior
        x[489] = bf16(0.75)
        x[510] = bf16(0.5)       # rope pair 31 — ramp clamps to 1, interpolated
        x[511] = bf16(0.75)
        pos0 = arr_f(x)
        assert rope(pos0, 512, 64, 0, 65536, 160000.0, 16.0, 32.0, 1.0) == 0
        assert [pos0[i] for i in range(512)] == x

        pos128 = arr_f(x)
        assert rope(pos128, 512, 64, 128, 65536, 160000.0, 16.0, 32.0, 1.0) == 0
        # Independently compute pair 0.
        freq0 = yarn_freq(0)
        a = 128.0 * freq0
        e0 = bf16(x[448]*math.cos(a) - x[449]*math.sin(a))
        e1 = bf16(x[448]*math.sin(a) + x[449]*math.cos(a))
        assert bf16_bits(pos128[448]) == bf16_bits(e0)
        assert bf16_bits(pos128[449]) == bf16_bits(e1)

        # Pair 0 cannot witness YaRN, and asserting that it does is worse than
        # not asserting: base**(-0/dim) == 1 for every base, and the ramp at
        # these betas spans low=15..high=25, so pair 0 is fully extrapolated
        # (smooth=1) and neither `base` nor `factor` reaches the angle. The
        # compressed and plain schemes are bit-identical there by construction.
        # This check therefore used to demand they differ at pair 0, which no
        # implementation can satisfy — and everything above it still passed
        # when yarn_freq was mutated to `return freq`, dropping factor and ramp
        # altogether. Pin the coincidence as the invariant it actually is:
        plain_a0 = 128.0 * 1.0
        same0 = bf16(x[448]*math.cos(plain_a0) - x[449]*math.sin(plain_a0))
        same1 = bf16(x[448]*math.sin(plain_a0) + x[449]*math.cos(plain_a0))
        assert (bf16_bits(pos128[448]), bf16_bits(pos128[449])) == \
               (bf16_bits(same0), bf16_bits(same1))

        # Pair 20 is the witness that discriminates. The ramp spans low=15 to
        # high=25, so pair 20 sits at ramp=0.5 — the interior, where the result
        # is a genuine blend of interpolated and extrapolated frequency rather
        # than a clamped endpoint. Pair 31 clamps to ramp=1 and its angles at
        # position 128 (7.3e-05 against 1.2e-03) both round to the same BF16
        # near 0.5, so it cannot see a dropped interpolation; pair 20's
        # 0.038 against 0.072 can.
        def rot(re, im, ang):
            return (bf16(re*math.cos(ang) - im*math.sin(ang)),
                    bf16(re*math.sin(ang) + im*math.cos(ang)))

        got20 = (bf16_bits(pos128[488]), bf16_bits(pos128[489]))
        want20 = rot(x[488], x[489], 128.0 * yarn_freq(20))
        assert got20 == (bf16_bits(want20[0]), bf16_bits(want20[1])), (got20, want20)

        # Two wrong frequencies for the same pair, each a one-line mistake:
        # ordinary base-10000 RoPE, and base-160000 with the YaRN blend
        # dropped. Both must produce a different answer than the contract.
        raw20 = 1.0 / (160000.0 ** ((2*20)/64))          # no interpolation
        plain20 = 1.0 / (10000.0 ** ((2*20)/64))         # wrong base entirely
        for wrong_freq, label in ((raw20, "un-blended"), (plain20, "base-10000")):
            w = rot(x[488], x[489], 128.0 * wrong_freq)
            assert got20 != (bf16_bits(w[0]), bf16_bits(w[1])), label

    print("PASS Gate E ratio-128 compressor scalar semantics: per-dim pooling + compressed YaRN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
