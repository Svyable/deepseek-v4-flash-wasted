#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""One-shot reconciliation of V7 validation/roadmap claims with frozen evidence."""

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def patch_validation() -> None:
    path = Path("docs/VALIDATION.md")
    s = path.read_text()

    old = """**Status: Gates A/V0, B/V1, C/V2, README Gate D's mHC seam, Gate E/V5 attention-by-type, and Gate F/V4 are passed at their stated scalar/model-semantic evidence levels. Gate E includes ratio-0, shared grouped output, coherent ratio-128, and coherent ratio-4 CSA/indexer checkpoint evidence. Gate H/V6 is PARTIAL: layer-3 block wiring, the real HC composition, and the attention half composed for real (residual → hc_pre → real 64-head attention → hc_post) are passed; the FFN half still runs a stub, so the real router/MoE composed at the ffn_pre state is what remains. Gate G still owns converted-record/cache identity.**
"""
    new = """**Status: Gates A/V0, B/V1, C/V2, README Gate D's mHC seam, Gate E/V5 attention-by-type, Gate F/V4, and Gate H/V6 are passed at their stated scalar/model-semantic evidence levels. V7 multi-layer localization is also passed for one real consecutive layer-3 → layer-4 transition, with 18 exact typed boundaries across the ratio128-learned → ratio4-learned structural change. Gate I/V8 final hidden/norm/logits is the next numerical target. Gate G still owns converted-record/cache identity.**
"""
    s = replace_once(s, old, new, "validation top status")

    old = """Permanent no-network validation after registration:

```text
make check   63 passed, 0 failed, 13 skipped
make asan    62 passed, 0 failed, 14 skipped
```

**Evidence boundary:** Gate H is passed at the scalar/model-semantic level for one real layer-3 state transition. This is not yet a 43-layer forward, final-hidden/logit proof, greedy-generation proof, converted-container proof, cache/disk identity proof, or performance result. V7/I/K and Gate G remain separate gates.

### V7 — multi-layer localization

Checkpoint hidden states across multiple layers and stop at the first divergent layer.
"""
    new = """Permanent no-network validation after the V7 trace refinement:

```text
make check   74 passed, 0 failed, 13 skipped
make asan    73 passed, 0 failed, 14 skipped
```

**Evidence boundary:** Gate H is passed at the scalar/model-semantic level for one real layer-3 state transition. This is not yet a 43-layer forward, final-hidden/logit proof, greedy-generation proof, converted-container proof, cache/disk identity proof, or performance result. V7, I/K, and Gate G remain separate gates.

### V7 — multi-layer localization — **PASSED for a real consecutive two-layer scalar boundary trace**

The frozen trace in `tests/fixtures/deepseek_v4/v7_two_layer_real/` chains the corrected Gate-H layer-3 endpoint directly into a complete real layer-4 transition:

```text
layer 3  ratio128-learned  output SHA c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7
                              || byte-identical
layer 4  ratio4-learned     input  SHA c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7
layer 4  final              output SHA f32bec5a013bc5a0da98a6ac940dd4946fe0dad877714ffb6811c177837cf749
```

The complete layer-4 acquisition independently checks all six selected routed FP4 experts plus the shared FP8 expert before composition: **24,576 routed + 4,096 shared = 28,672 exact BF16 expert outputs**. The final HyperConnection transition is also checked at **16,384 exact F32 values** against the independent F32 oracle. Raw checkpoint expert payloads remain transient acquisition evidence and are not committed.

The V7 expected/runtime manifests compare **18 exact typed boundaries** — nine per layer:

```text
input -> attn_pre -> attention_branch -> after_attn -> ffn_pre
      -> router_ids -> router_weights -> moe_branch -> output
```

Trace integrity is checked before values: file geometry/SHA, consecutive layers, structural classes, and exact layer-output → next-layer-input chaining must all agree. The localizer then stops at the first differing boundary/index. Adversarial regressions independently mutate (1) the chained layer-3 output/layer-4 input, (2) layer-4 `attention_branch`, (3) layer-4 F32 `router_weights`, and (4) the terminal layer-4 output; each mutation must localize at that exact earliest seam.

Mechanical parent freshness is part of the gate: `tools/v7_parent_freshness.py` derives the current Gate-H endpoint from its provenance and requires all downstream V7 fixtures to consume that exact state. The explicitly superseded `0e65c4ec...` pre-F32-HyperConnection endpoint is not accepted.

**Evidence boundary:** V7 proves localization and exact scalar/model-semantic composition across these two consecutive real layers. It does **not** prove the remaining 41 transformer layers, final norm/head/logits, generation, converted-container/cache identity, RAM, or throughput. Gate I/V8 is next.
"""
    s = replace_once(s, old, new, "validation Gate H/V7 block")

    old = "| — | **V7** multi-layer localization | base-model bring-up |"
    new = "| — | **V7** multi-layer localization | **PASSED two-layer scalar/model-semantic** — real ratio128 layer 3 → ratio4 layer 4, 18 exact boundaries |"
    s = replace_once(s, old, new, "validation concordance V7 row")

    path.write_text(s)


def patch_roadmap() -> None:
    path = Path("ROADMAP.md")
    s = path.read_text()

    old = "| multi-layer localization/logits/generation | V7, I/V8, K/V9 | later base bring-up |"
    new = """| multi-layer localization | **V7** | **PASSED two-layer scalar/model-semantic** — real layer 3 ratio128 → layer 4 ratio4, 18 exact boundaries |
| final hidden/norm/logits | **I/V8** | **NEXT** |
| deterministic greedy generation | **K/V9** | after V8 logits |"""
    s = replace_once(s, old, new, "roadmap current snapshot")

    old = """**Immediate priority: V7 multi-layer localization, then Gate I/V8 logits.** Do not jump straight to generation: establish a reproducible way to carry the frozen layer-state contract across structurally different layers and stop at the first divergent layer. Gate K/V9 follows only after final hidden/norm/logits are pinned.
"""
    new = """**Immediate priority: Gate I/V8 final hidden/norm/logits.** V7 now provides the reproducible first-divergence contract across a real ratio128 → ratio4 layer transition, so downstream forward/logit work has a concrete localization mechanism. Do not jump straight to generation: Gate K/V9 follows only after final hidden/norm/logits are pinned.
"""
    s = replace_once(s, old, new, "roadmap immediate priority")

    old = """9. complete real layer-3 HC + attention + router + six-routed/shared-MoE transition — H/V6.

Next oracle seams:

1. multi-layer localization — V7;
2. final hidden/norm/logits — I/V8;
3. deterministic greedy generation — K/V9;
4. encoder/parser — J/V10.
"""
    new = """9. complete real layer-3 HC + attention + router + six-routed/shared-MoE transition — H/V6;
10. real consecutive layer-3 → layer-4 localization across ratio128 → ratio4, including the complete layer-4 MoE/final state and an 18-boundary trace — V7.

Next oracle seams:

1. final hidden/norm/logits — I/V8;
2. deterministic greedy generation — K/V9;
3. encoder/parser — J/V10.
"""
    s = replace_once(s, old, new, "roadmap oracle seam list")

    old = """Permanent suite floor after the Gate H precision correction:

```text
make check  65 passed, 0 failed, 13 skipped
make asan   64 passed, 0 failed, 14 skipped
```

Continue in this order:

1. multi-layer localization — **V7**;
2. final hidden/norm/head/logits — **I/V8**;
3. deterministic known-token generation — **K/V9**;
4. only then broaden to encoding/API and optimization gates.
"""
    new = """Permanent suite floor after V7 completion and trace refinement:

```text
make check  74 passed, 0 failed, 13 skipped
make asan   73 passed, 0 failed, 14 skipped
```

### V7 multi-layer localization — PASSED for a real consecutive two-layer scalar boundary trace

The corrected Gate-H layer-3 endpoint (`c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7`) is consumed byte-identically by a complete real layer-4 transition. Layer 4 crosses into the ratio4-learned structural class, selects experts `[237,90,185,120,197,239]`, independently validates **28,672 BF16 expert outputs**, and freezes final state SHA `f32bec5a013bc5a0da98a6ac940dd4946fe0dad877714ffb6811c177837cf749`.

The permanent V7 trace has nine boundaries per layer — `input`, `attn_pre`, `attention_branch`, `after_attn`, `ffn_pre`, `router_ids`, `router_weights`, `moe_branch`, `output` — for **18 exact boundaries total**. Mutations at the cross-layer chain, raw attention branch, F32 route weights, and terminal layer output are required to localize at the exact first divergent seam. Parent freshness is derived mechanically from current Gate-H provenance rather than a hard-coded historical hash.

This is deliberately **not** a claim about all 43 layers or final logits; it is the localization substrate for that work.

Continue in this order:

1. final hidden/norm/head/logits — **I/V8**;
2. deterministic known-token generation — **K/V9**;
3. only then broaden to encoding/API and optimization gates.
"""
    s = replace_once(s, old, new, "roadmap Phase 5 V7 block")

    path.write_text(s)


def main() -> int:
    patch_validation()
    patch_roadmap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
