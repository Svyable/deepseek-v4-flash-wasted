#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic patcher for the Gate-H HC precision correction."""

from pathlib import Path

ROUTE = Path("tests/test_v6_ffn_route_real.py")
RUN = Path("tests/run.sh")


def patch_route() -> None:
    s = ROUTE.read_text()
    old_import = '''REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nsys.path.insert(0, os.path.join(REPO, "tests"))\nfrom test_v6_layer_composition_scalar import split  # noqa: E402\n'''
    new_import = '''REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nsys.path.insert(0, os.path.join(REPO, "tools"))\nsys.path.insert(0, os.path.join(REPO, "tests"))\nimport deepseek_v4_hc_oracle as hc_oracle  # noqa: E402\n'''
    if old_import in s:
        s = s.replace(old_import, new_import, 1)
    elif new_import not in s:
        raise SystemExit("Gate-H route import anchor drifted")

    old = '''def oracle_hc_params(x, fn, scale, base):\n    ss = sum(v * v for v in x)\n    r = 1.0 / math.sqrt(ss / len(x) + 1e-6)\n    mixes = [sum(fn[row * FLAT + c] * x[c] for c in range(FLAT)) * r\n             for row in range(MIX)]\n    return split(mixes, scale, base)\n\n\ndef oracle_hc_pre_y(x, pre):\n    return [sum(pre[j] * x[j * DIM + d] for j in range(HC)) for d in range(DIM)]\n\n\ndef oracle_hc_post(branch, residual, post, comb):\n    out = [0.0] * FLAT\n    for k in range(HC):\n        for d in range(DIM):\n            acc = f32(post[k] * branch[d])\n            for j in range(HC):\n                acc = f32(acc + f32(comb[j][k] * residual[j * DIM + d]))\n            out[k * DIM + d] = acc\n    return out\n'''
    new = '''def oracle_hc_params(x, fn, scale, base):\n    pre, post, comb, _ = hc_oracle.params(x, fn, scale, base, dim=DIM)\n    return pre, post, comb\n\n\ndef oracle_hc_pre_y(x, pre):\n    return hc_oracle.pre_y(x, pre, dim=DIM)\n\n\ndef oracle_hc_post(branch, residual, post, comb):\n    return hc_oracle.post_y(branch, residual, post, comb, dim=DIM)\n'''
    if old in s:
        s = s.replace(old, new, 1)
    elif new not in s:
        raise SystemExit("Gate-H route HC-oracle function anchor drifted")

    if "return split(mixes, scale, base)" in s:
        raise SystemExit("stale double-precision Gate-H Sinkhorn route oracle remains")
    ROUTE.write_text(s)


def patch_run() -> None:
    s = RUN.read_text()
    anchor = '''deepseek_replay "Gate H/V6 — model-free layer block wiring and five refusals" \\
                "test_v6_layer_composition_scalar.py"\n'''
    block = '''deepseek_replay "Gate H/V6 — bit-exact independent F32 HyperConnection oracle" \\
                "test_v6_hc_oracle_f32.py"\ndeepseek_replay "Gate H/V6 — corrected real layer-3 final HC precision boundary" \\
                "test_v6_layer3_hc_precision.py"\n'''
    if block not in s:
        if s.count(anchor) != 1:
            raise SystemExit(f"Gate-H tests/run anchor count={s.count(anchor)}")
        s = s.replace(anchor, block + anchor, 1)
    RUN.write_text(s)


def main() -> int:
    patch_route()
    patch_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
