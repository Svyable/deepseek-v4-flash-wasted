#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""Temporary deterministic docs patcher for the Gate-H HC precision correction."""

from pathlib import Path

OLD = "0e65c4ecb328d5067f2274e724f9f46e4a13218e7fdeb706f7d1a465c0ee4761"
NEW = "c3d175f8170b33f344a471739640f683c41fb8b9c2c69f1529f70b0479a1d8f7"
DIFFS = "1186, 10135, 10503, 10877, 11565, 11672, 11675, 13648, 13717"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    s = path.read_text()
    if old not in s:
        if new in s:
            return
        raise SystemExit(f"{label}: anchor not found")
    if s.count(old) != 1:
        raise SystemExit(f"{label}: anchor count={s.count(old)}")
    path.write_text(s.replace(old, new, 1))


def patch_roadmap() -> None:
    p = Path("ROADMAP.md")
    old = f'''Acquisition checked 28,672 expert BF16 outputs against an independent standalone oracle. The exact scalar-router F32 weights are recorded separately from the independent Python router (same IDs; max weight delta `1.49662139e-07`). Final state SHA-256: `{OLD}`.\n\nA real numerical boundary was found and pinned rather than tolerated: independent Sinkhorn `post`/`comb` are rounded to F32 before final `hc_post`; doing the Python computation at double precision changed exactly one of 16,384 final BF16 values (index 13,648).\n\nPermanent suite floor after Gate H registration:\n\n```text\nmake check  63 passed, 0 failed, 13 skipped\nmake asan   62 passed, 0 failed, 14 skipped\n```\n'''
    new = f'''Acquisition checked 28,672 expert BF16 outputs against an independent standalone oracle. The exact scalar-router F32 weights are recorded separately from the independent Python router (same IDs; max weight delta `1.49662139e-07`). The canonical final state SHA-256 is `{NEW}`.\n\nA later V7 review exposed that Gate H's original Python Sinkhorn still executed its internal reductions/normalizations in double precision and only rounded `post`/`comb` at the boundary. That is not model-equivalent. `tools/deepseek_v4_hc_oracle.py` now executes every HC operation in F32 and is bit-exact with `src/deepseek_v4_mhc_ref.c`. The correction changes 9 of 16,384 final BF16 values (indices {DIFFS}) while leaving `attn_pre`, `ffn_pre`, routing, all seven expert outputs, and the MoE branch unchanged. No checkpoint reacquisition was required.\n\nPermanent suite floor after the Gate H precision correction:\n\n```text\nmake check  65 passed, 0 failed, 13 skipped\nmake asan   64 passed, 0 failed, 14 skipped\n```\n'''
    replace_once(p, old, new, "ROADMAP Gate-H precision section")


def patch_validation() -> None:
    p = Path("docs/VALIDATION.md")
    old = f'''The final 16,384 BF16 values are exact. The frozen final-state SHA-256 is `{OLD}`; representative first values are `3f93 3e9a 3ed7 bcba 3e11 3e57 bd52 bf2d`.\n\nOne failed replay tightened the numerical contract rather than weakening it. The independent Python Sinkhorn initially carried `post`/`comb` in double precision into the final `hc_post`, while the model/runtime boundary is F32. That changed exactly one final BF16 value, index 13,648 (`378d` versus C `378e`). The fixture now explicitly rounds independent Sinkhorn `post`/`comb` to F32 before `hc_post`; no comparison tolerance was added. Provenance records that one boundary-sensitive value.\n'''
    new = f'''The final 16,384 BF16 values are exact. The canonical final-state SHA-256 is `{NEW}`; representative first values are `3f93 3e9a 3ed7 bcba 3e11 3e57 bd52 bf2d`.\n\nA V7 consecutive-layer review tightened the numerical contract again. Merely rounding Python Sinkhorn `post`/`comb` to F32 at the boundary was insufficient because its internal reductions and normalizations had still executed in Python double precision. The canonical independent oracle now performs every HyperConnection operation in F32, including RMS reduction, mix dot products, sigmoid/softmax, every Sinkhorn row/column reduction and normalization, and `hc_pre`/`hc_post` accumulation. It is bit-exact with the scalar C reference. The corrected final state differs from the earlier frozen Gate-H endpoint at exactly 9 BF16 indices ({DIFFS}); `attn_pre` and `ffn_pre` are unchanged, so routing and all expert/MoE acquisition evidence remain valid. The permanent regression explicitly refuses the old SHA `{OLD}`; no comparison tolerance was added.\n'''
    replace_once(p, old, new, "VALIDATION Gate-H precision paragraph")


def append_experiment() -> None:
    p = Path("docs/EXPERIMENTS.md")
    s = p.read_text()
    marker = "## 14. Gate-H final HC must execute Sinkhorn itself in F32, not only cast its outputs\n"
    if marker in s:
        return
    entry = f'''\n\n---\n\n{marker}\n**Status:** corrected and permanently replayed\n\nThe first Gate-H fix in experiment 12 was necessary but incomplete. It established that `post`/`comb` are F32 tensors at the final HC boundary and fixed one observed BF16 mismatch, but its independent Python producer still executed the RMS/mix/Sinkhorn arithmetic in Python double precision before casting those tensors to F32. A real consecutive-layer V7 trace exposed that this is not equivalent to the scalar/model dtype contract.\n\nA new independent oracle, `tools/deepseek_v4_hc_oracle.py`, now rounds every HyperConnection arithmetic operation in the model's F32 order and uses the platform `expf`/`sqrtf`. `tests/test_v6_hc_oracle_f32.py` compares 24 mixes, 4 pre coefficients, 4 post coefficients, 16 comb coefficients, `hc_pre` outputs, and `hc_post` outputs against `src/deepseek_v4_mhc_ref.c` at raw F32-bit equality. The real-data regression `tests/test_v6_layer3_hc_precision.py` then runs both paths on the frozen layer-3 checkpoint evidence.\n\nResult:\n\n```text\nold final SHA-256       {OLD}\ncorrected final SHA-256 {NEW}\nBF16 values changed     9 / 16,384\nchanged indices          {DIFFS}\nattn_pre                 unchanged\nffn_pre                  unchanged\nrouter/expert/MoE        unchanged\n```\n\nThe correction required no checkpoint or expert reacquisition: the stale boundary was after the already-frozen MoE branch. The old final SHA is now explicitly refused by the permanent test.\n\n**Lesson:** a later cast to the correct dtype does not retroactively make higher-precision reductions model-equivalent. For precision-sensitive reference paths, the operation sequence *and every intermediate dtype boundary* are part of the oracle contract.\n'''
    p.write_text(s.rstrip() + entry.rstrip() + "\n")


def main() -> int:
    patch_roadmap()
    patch_validation()
    append_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
