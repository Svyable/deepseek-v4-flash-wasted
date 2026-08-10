#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Pin one-row online-softmax signed-zero semantics independently.

The V7 layer-4 chain exposed a shortcut bug: direct ``kv / denom`` preserves a
negative-zero value lane, while the source online-softmax kernel first adds the
weighted value to a +0 F32 accumulator. Under round-to-nearest, +0 + -0 is +0.
This test makes that operation order load-bearing and also pins E4M3 0x80 decode
as negative zero.
"""

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import make_v2_real_fixture as qref  # noqa: E402
import make_v5_attention_history_real_fixture as hist  # noqa: E402
import make_v6_attention_branch_fixture as attn  # noqa: E402


def main():
    neg_zero = qref.e4m3_float(0x80)
    if neg_zero != 0.0 or math.copysign(1.0, neg_zero) >= 0.0:
        raise AssertionError("E4M3 0x80 must decode to negative zero")
    if qref.bf16_bits(neg_zero) != 0x8000:
        raise AssertionError("negative-zero E4M3 decode must survive BF16 cast")

    q = [0x0000] * attn.HEAD
    kv = [0x0000] * attn.HEAD
    kv[17] = 0x3f80      # +1.0, proves ordinary value arithmetic too.
    kv[333] = 0x8000     # -0.0, the exact V7 layer-4 failure lane.

    got = attn.one_row_online_attention(q, kv, 0.0)
    if len(got) != attn.HEAD:
        raise AssertionError("one-row attention output geometry drifted")
    if got[17] != 0x3f00:  # 1 / (1 + exp(0)) = 0.5
        raise AssertionError(f"ordinary one-row value drifted: {got[17]:04x} != 3f00")
    if got[333] != 0x0000:
        raise AssertionError(
            f"online accumulator must canonicalize -0 lane to +0: {got[333]:04x}")

    # Known-wrong mutation: algebraically collapse the online accumulation to
    # direct value/denominator division. It must visibly preserve -0 instead.
    kvf = [qref.bf16_to_f32(b) for b in kv]
    denom = hist.f32(2.0)
    direct = [qref.bf16_bits(hist.f32(v / denom)) for v in kvf]
    if direct[333] != 0x8000:
        raise AssertionError(
            "direct-division mutation no longer exposes signed-zero difference")
    if direct == got:
        raise AssertionError("direct-division mutation became invisible")

    print("PASS V7 one-row online-softmax preserves operational signed-zero semantics")
    print("runtime-order lane333=0x0000 direct-division mutation=0x8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
