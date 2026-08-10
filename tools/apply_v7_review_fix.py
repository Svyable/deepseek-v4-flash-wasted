#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic patcher for V7 maintainer-review fixes.

This file is removed by the validation workflow after the patch passes the
focused regressions and the complete frozen suites.
"""

from pathlib import Path


def patch_attention_helper():
    p = Path("tools/make_v6_attention_branch_fixture.py")
    s = p.read_text()
    anchor = "\n\ndef one_token_heads(x_bits, paths):\n"
    helper = '''


def one_row_online_attention(q_bits, kv_bits, sink):
    """One valid KV row through the real online-softmax operation order.

    Algebraically this looks like ``kv / (1 + exp(sink-score))``, but the
    source kernel first accumulates ``BF16(exp(0)) * kv`` into a +0 F32
    accumulator. That operation is observably different for a -0 KV lane:
    +0 + -0 becomes +0 under round-to-nearest. Keep the operational order;
    do not replace this with direct division.
    """
    if len(q_bits) != HEAD or len(kv_bits) != HEAD:
        raise ValueError("one-row attention requires one full 512-value head")
    qf = [qref.bf16_to_f32(b) for b in q_bits]
    kvf = [qref.bf16_to_f32(b) for b in kv_bits]
    scale = hist.f32(HEAD ** -0.5)
    dot = hist.f32(0.0)
    for a, b in zip(qf, kvf):
        dot = hist.f32(dot + hist.f32(a * b))
    score = hist.f32(dot * scale)

    best = score
    denom = hist.f32(0.0)
    acc = [hist.f32(0.0)] * HEAD
    e = hist.f32(math.exp(hist.f32(score - best)))
    denom = hist.f32(denom + e)
    e_bf16 = qref.bf16_to_f32(qref.bf16_bits(e))
    for k, v in enumerate(kvf):
        acc[k] = hist.f32(acc[k] + hist.f32(e_bf16 * v))

    denom = hist.f32(denom + hist.f32(math.exp(hist.f32(sink - best))))
    return [qref.bf16_bits(hist.f32(v / denom)) for v in acc]
'''
    if "def one_row_online_attention(" not in s:
        if s.count(anchor) != 1:
            raise SystemExit(f"one_token_heads anchor count={s.count(anchor)}")
        s = s.replace(anchor, helper + anchor, 1)

    old = '''    kv = hist.qat_nope(kv1)\n    kvf = [qref.bf16_to_f32(b) for b in kv]\n    scale = hist.f32(HEAD ** -0.5)\n\n    heads = []\n    q_all = []\n    for h in range(N_HEADS):\n        qn = hist.head_norm(qb[h*HEAD:(h+1)*HEAD])\n        q_all.extend(qn)\n        qf = [qref.bf16_to_f32(b) for b in qn]\n        dot = hist.f32(0.0)\n        for a, b in zip(qf, kvf):\n            dot = hist.f32(dot + hist.f32(a * b))\n        score = hist.f32(dot * scale)\n        # One valid KV row. Online softmax best=score, value weight BF16(exp(0))=1.\n        denom = hist.f32(1.0 + hist.f32(math.exp(hist.f32(sinks[h] - score))))\n        heads.extend(qref.bf16_bits(hist.f32(v / denom)) for v in kvf)\n'''
    new = '''    kv = hist.qat_nope(kv1)\n\n    heads = []\n    q_all = []\n    for h in range(N_HEADS):\n        qn = hist.head_norm(qb[h*HEAD:(h+1)*HEAD])\n        q_all.extend(qn)\n        heads.extend(one_row_online_attention(qn, kv, sinks[h]))\n'''
    if old in s:
        s = s.replace(old, new, 1)
    elif new not in s:
        raise SystemExit("one_token_heads implementation anchor drifted")
    p.write_text(s)


def patch_run_sh():
    p = Path("tests/run.sh")
    s = p.read_text()
    anchor = '''# V7 infrastructure: compare chained expected/runtime layer-boundary traces,\n# reject stale manifests/broken output->input chains, and stop at the first\n# differing layer/boundary rather than reporting a downstream symptom.\ndeepseek_replay "V7 — fail-closed first-divergence layer trace localizer" \\\n                "test_v7_localize.py"\n'''
    block = '''# V7 precision boundaries found by real chained-layer acquisition. These are\n# permanent because each caught an otherwise plausible, near-bit-exact oracle.\ndeepseek_replay "V7 — bit-exact independent F32 HyperConnection oracle" \\\n                "test_v7_hc_oracle_f32.py"\ndeepseek_replay "V7 — hc_pre sequential F32 reduction boundary" \\\n                "test_v7_hc_pre_f32_scalar.py"\ndeepseek_replay "V7 — sparse FP8 scalar differential and signed-zero cases" \\\n                "test_v7_sparse_fp8_oracle.py"\ndeepseek_replay "V7 — one-row online-softmax signed-zero operation order" \\\n                "test_v7_one_row_attention_signed_zero.py"\n\n'''
    if block not in s:
        if s.count(anchor) != 1:
            raise SystemExit(f"tests/run V7 localizer anchor count={s.count(anchor)}")
        s = s.replace(anchor, block + anchor, 1)
    p.write_text(s)


def main():
    patch_attention_helper()
    patch_run_sh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
