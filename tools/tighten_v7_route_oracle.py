#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary review helper: make V7 HC coefficient checks exact F32."""

from pathlib import Path

PATH = Path("tools/make_v7_layer4_route_fixture.py")


def main() -> int:
    s = PATH.read_text()

    dead = '''\ndef oracle_hc_pre_y_f32(x, pre):\n    """Independent HC-pre equation with the model's sequential F32 reduction."""\n    out = []\n    for d in range(DIM):\n        acc = chain.f32(0.0)\n        for j in range(HC):\n            acc = chain.f32(acc + chain.f32(pre[j] * x[j * DIM + d]))\n        out.append(acc)\n    return out\n\n'''
    if dead in s:
        s = s.replace(dead, "\n", 1)
    elif "def oracle_hc_pre_y_f32(" in s:
        raise SystemExit("dead HC-pre helper changed shape")

    old_attn = '''        param_delta = max(\n            max(abs(a - b) for a, b in zip(attn_post, o_post)),\n            max(abs(a - b) for a, b in zip(\n                [v for row in attn_comb for v in row],\n                [v for row in o_comb for v in row])))\n        if param_delta > 1e-5:\n            raise ValueError(f"layer4 HC-attn post/comb oracle delta {param_delta} > 1e-5")\n'''
    new_attn = '''        attn_c_bits = [chain.f32_bits(v) for v in attn_post] + [\n            chain.f32_bits(v) for row in attn_comb for v in row]\n        attn_o_bits = [chain.f32_bits(v) for v in o_post] + [\n            chain.f32_bits(v) for row in o_comb for v in row]\n        if attn_c_bits != attn_o_bits:\n            i = next(i for i, (a, b) in enumerate(zip(attn_c_bits, attn_o_bits)) if a != b)\n            raise ValueError(\n                f"layer4 HC-attn post/comb F32 mismatch at {i}: "\n                f"{attn_c_bits[i]:08x}!={attn_o_bits[i]:08x}")\n'''
    if old_attn in s:
        s = s.replace(old_attn, new_attn, 1)
    elif new_attn not in s:
        raise SystemExit("attention HC tolerance anchor drifted")

    old_ffn = '''        ffn_param_delta = max(\n            max(abs(a - b) for a, b in zip(ffn_post, fo_post)),\n            max(abs(a - b) for a, b in zip(\n                [v for row in ffn_comb for v in row],\n                [v for row in fo_comb for v in row])))\n        if ffn_param_delta > 1e-5:\n            raise ValueError(f"layer4 HC-FFN post/comb oracle delta {ffn_param_delta} > 1e-5")\n'''
    new_ffn = '''        ffn_c_bits = [chain.f32_bits(v) for v in ffn_post] + [\n            chain.f32_bits(v) for row in ffn_comb for v in row]\n        ffn_o_bits = [chain.f32_bits(v) for v in fo_post] + [\n            chain.f32_bits(v) for row in fo_comb for v in row]\n        if ffn_c_bits != ffn_o_bits:\n            i = next(i for i, (a, b) in enumerate(zip(ffn_c_bits, ffn_o_bits)) if a != b)\n            raise ValueError(\n                f"layer4 HC-FFN post/comb F32 mismatch at {i}: "\n                f"{ffn_c_bits[i]:08x}!={ffn_o_bits[i]:08x}")\n'''
    if old_ffn in s:
        s = s.replace(old_ffn, new_ffn, 1)
    elif new_ffn not in s:
        raise SystemExit("FFN HC tolerance anchor drifted")

    old_prov = '''        "hc_crosschecks": {\n            "attn_post_comb_max_abs": param_delta,\n            "ffn_post_comb_max_abs": ffn_param_delta,\n            "pre_boundaries": "independent Python and C exact at BF16",\n        },\n'''
    new_prov = '''        "hc_crosschecks": {\n            "attn_post_comb_exact_f32": True,\n            "ffn_post_comb_exact_f32": True,\n            "pre_boundaries": "independent Python and C exact at BF16",\n        },\n'''
    if old_prov in s:
        s = s.replace(old_prov, new_prov, 1)
    elif new_prov not in s:
        raise SystemExit("HC provenance tolerance anchor drifted")

    for stale in ("param_delta", "HC-attn post/comb oracle delta", "HC-FFN post/comb oracle delta"):
        if stale in s:
            raise SystemExit(f"stale HC tolerance marker remains: {stale}")

    PATH.write_text(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
