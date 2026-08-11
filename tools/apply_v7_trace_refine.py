#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""One-shot source patch for the V7 14 -> 18 boundary trace refinement."""

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def replace_n(text: str, old: str, new: str, count: int, label: str) -> str:
    got = text.count(old)
    if got != count:
        raise SystemExit(f"{label}: expected {count} matches, got {got}")
    return text.replace(old, new, count)


def patch_generator() -> None:
    path = Path("tools/make_v7_two_layer_trace.py")
    s = path.read_text()

    s = replace_once(
        s,
        '''BOUNDARIES = (\n    ("input", "BF16", [HC, DIM]),\n    ("attn_pre", "BF16", [DIM]),\n    ("after_attn", "BF16", [HC, DIM]),\n    ("ffn_pre", "BF16", [DIM]),\n    ("router_ids", "U32", [TOPK]),\n    ("moe_branch", "BF16", [DIM]),\n    ("output", "BF16", [HC, DIM]),\n)\n''',
        '''BOUNDARIES = (\n    ("input", "BF16", [HC, DIM]),\n    ("attn_pre", "BF16", [DIM]),\n    ("attention_branch", "BF16", [DIM]),\n    ("after_attn", "BF16", [HC, DIM]),\n    ("ffn_pre", "BF16", [DIM]),\n    ("router_ids", "U32", [TOPK]),\n    ("router_weights", "F32", [TOPK]),\n    ("moe_branch", "BF16", [DIM]),\n    ("output", "BF16", [HC, DIM]),\n)\n''',
        "BOUNDARIES",
    )

    write_u32 = '''def write_u32(path: str, values: list[int]) -> None:\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    with open(path, "wb") as f:\n        f.write(struct.pack(f"<{len(values)}I", *values))\n'''
    s = replace_once(
        s,
        write_u32,
        write_u32 + '''\n\ndef write_f32(path: str, values: list[float]) -> None:\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    with open(path, "wb") as f:\n        f.write(struct.pack(f"<{len(values)}f", *values))\n''',
        "write_f32 insertion",
    )

    s = replace_once(
        s,
        '''    l3_ids_path, l3_ids = checked_file(\n        L3MOE, l3_ids_meta, "layer3 router ids", TOPK, reader=read_u32)\n    l3_moe_meta = prov["l3_moe"]["outputs"]["moe_branch"]\n''',
        '''    l3_ids_path, l3_ids = checked_file(\n        L3MOE, l3_ids_meta, "layer3 router ids", TOPK, reader=read_u32)\n    l3_weights_meta = prov["l3_moe"]["outputs"]["weights"]\n    l3_weights_path, l3_weights = checked_file(\n        L3MOE, l3_weights_meta, "layer3 router weights", TOPK, reader=read_f32)\n    l3_moe_meta = prov["l3_moe"]["outputs"]["moe_branch"]\n''',
        "layer3 router weights load",
    )

    s = replace_once(
        s,
        '''    l4_ids = read_u32(l4_ids_path, TOPK)\n    l4_full_out = prov["l4_full"]["outputs"]\n''',
        '''    l4_ids = read_u32(l4_ids_path, TOPK)\n    l4_weights_path = os.path.join(L4ROUTE, l4_router["weights_file"])\n    if sha256_file(l4_weights_path) != l4_router["weights_sha256"]:\n        raise ValueError("layer4 router weights SHA drifted")\n    l4_weights = read_f32(l4_weights_path, TOPK)\n    l4_full_out = prov["l4_full"]["outputs"]\n''',
        "layer4 router weights load",
    )

    s = replace_once(
        s,
        '''            "router_ids": l3_ids, "moe_branch": l3_moe, "output": l3_output,\n''',
        '''            "router_ids": l3_ids, "router_weights": l3_weights,\n            "moe_branch": l3_moe, "output": l3_output,\n''',
        "layer3 frozen return",
    )
    s = replace_once(
        s,
        '''            "ffn_pre": l4_ffn_pre, "router_ids": l4_ids,\n            "moe_branch": l4_moe, "output": l4_output, "params": l4_params,\n''',
        '''            "ffn_pre": l4_ffn_pre, "router_ids": l4_ids,\n            "router_weights": l4_weights, "moe_branch": l4_moe,\n            "output": l4_output, "params": l4_params,\n''',
        "layer4 frozen return",
    )

    s = replace_n(
        s,
        '''        "input": frozen["input"], "attn_pre": attn_pre,\n        "after_attn": after_bits, "ffn_pre": ffn_pre,\n        "router_ids": frozen["router_ids"], "moe_branch": frozen["moe_branch"],\n        "output": final_bits,\n''',
        '''        "input": frozen["input"], "attn_pre": attn_pre,\n        "attention_branch": frozen["attn_branch"], "after_attn": after_bits,\n        "ffn_pre": ffn_pre, "router_ids": frozen["router_ids"],\n        "router_weights": frozen["router_weights"],\n        "moe_branch": frozen["moe_branch"], "output": final_bits,\n''',
        2,
        "independent/runtime trace returns",
    )

    s = replace_once(
        s,
        '''            ext = "u32.bin" if dtype == "U32" else "bf16.bin"\n            rel = os.path.join(f"layer{number}", f"{boundary}.{ext}")\n            path = os.path.join(root, rel)\n            if dtype == "U32":\n                write_u32(path, values[boundary])\n            else:\n                write_u16(path, values[boundary])\n''',
        '''            ext = {"U32": "u32.bin", "F32": "f32.bin"}.get(dtype, "bf16.bin")\n            rel = os.path.join(f"layer{number}", f"{boundary}.{ext}")\n            path = os.path.join(root, rel)\n            if dtype == "U32":\n                write_u32(path, values[boundary])\n            elif dtype == "F32":\n                write_f32(path, values[boundary])\n            else:\n                write_u16(path, values[boundary])\n''',
        "write_trace dtype dispatch",
    )

    path.write_text(s)


def patch_test() -> None:
    path = Path("tests/test_v7_two_layer_real.py")
    s = path.read_text()
    s = replace_once(
        s,
        'result.get("boundaries_compared") != 14',
        'result.get("boundaries_compared") != 18',
        "boundary count",
    )

    s = replace_once(
        s,
        '''def mutate_bf16(trace_root: str, manifest: dict, layer: int, boundary: str,\n                 index: int, xor_mask: int = 1) -> None:\n    item = next(x for x in manifest["layers"] if x["layer"] == layer)\n    b = next(x for x in item["boundaries"] if x["name"] == boundary)\n    if b["dtype"] != "BF16":\n        raise AssertionError("mutation helper is BF16-only")\n    path = os.path.join(trace_root, b["file"])\n    data = bytearray(open(path, "rb").read())\n    off = index * 2\n    old = struct.unpack_from("<H", data, off)[0]\n    struct.pack_into("<H", data, off, old ^ xor_mask)\n    with open(path, "wb") as f:\n        f.write(data)\n    b["sha256"] = sha(path)\n''',
        '''def mutate_boundary(trace_root: str, manifest: dict, layer: int, boundary: str,\n                    dtype: str, index: int, xor_mask: int = 1) -> None:\n    item = next(x for x in manifest["layers"] if x["layer"] == layer)\n    b = next(x for x in item["boundaries"] if x["name"] == boundary)\n    if b["dtype"] != dtype or dtype not in {"BF16", "F32"}:\n        raise AssertionError(f"mutation helper expected {dtype}, got {b['dtype']}")\n    width = 2 if dtype == "BF16" else 4\n    path = os.path.join(trace_root, b["file"])\n    data = bytearray(open(path, "rb").read())\n    off = index * width\n    if off + width > len(data):\n        raise AssertionError("mutation index out of range")\n    data[off] ^= xor_mask\n    with open(path, "wb") as f:\n        f.write(data)\n    b["sha256"] = sha(path)\n''',
        "generic mutation helper",
    )

    s = replace_once(s, 'mutate_bf16(root, manifest, 3, "output", 17)',
                     'mutate_boundary(root, manifest, 3, "output", "BF16", 17)',
                     "layer3 output mutation")
    s = replace_once(s, 'mutate_bf16(root, manifest, 4, "input", 17)',
                     'mutate_boundary(root, manifest, 4, "input", "BF16", 17)',
                     "layer4 input mutation")
    s = replace_once(s, 'mutate_bf16(root, manifest, 4, "output", 23)',
                     'mutate_boundary(root, manifest, 4, "output", "BF16", 23)',
                     "layer4 output mutation")

    anchor = '''        # Mutation 2 changes only the terminal layer-4 output and keeps the\n'''
    addition = '''        # Mutation 2 pins the raw attention branch as its own localization\n        # seam. Without this boundary an attention primitive drift would be\n        # reported only after HyperConnection post-composition.\n        with tempfile.TemporaryDirectory(prefix="v7-real-trace-attention-mutation-") as td:\n            root = os.path.join(td, "runtime")\n            shutil.copytree(os.path.join(FIX, "runtime"), root)\n            manifest = json.load(open(os.path.join(root, "trace.json"), encoding="utf-8"))\n            mutate_boundary(root, manifest, 4, "attention_branch", "BF16", 19)\n            mutated = loc.load_trace(write_manifest(root, manifest))\n            diff = loc.compare(expected, mutated)\n            if diff.get("status") != "diverged" or diff.get("layer") != 4 or \\\n               diff.get("boundary") != "attention_branch" or diff.get("index") != 19:\n                raise AssertionError(f"attention-branch mutation localized incorrectly: {diff}")\n\n        # Mutation 3 pins exact runtime route weights independently of IDs.\n        # A weight-only drift must not be deferred to the MoE branch boundary.\n        with tempfile.TemporaryDirectory(prefix="v7-real-trace-weight-mutation-") as td:\n            root = os.path.join(td, "runtime")\n            shutil.copytree(os.path.join(FIX, "runtime"), root)\n            manifest = json.load(open(os.path.join(root, "trace.json"), encoding="utf-8"))\n            mutate_boundary(root, manifest, 4, "router_weights", "F32", 2)\n            mutated = loc.load_trace(write_manifest(root, manifest))\n            diff = loc.compare(expected, mutated)\n            if diff.get("status") != "diverged" or diff.get("layer") != 4 or \\\n               diff.get("boundary") != "router_weights" or diff.get("index") != 2:\n                raise AssertionError(f"router-weight mutation localized incorrectly: {diff}")\n\n        # Mutation 4 changes only the terminal layer-4 output and keeps the\n'''
    s = replace_once(s, anchor, addition, "new mutation insertion")
    s = replace_once(
        s,
        "PASS V7 real two-layer trace: 14 boundaries exact, earliest-divergence mutations localized",
        "PASS V7 real two-layer trace: 18 boundaries exact, earliest-divergence mutations localized",
        "success text",
    )
    path.write_text(s)


def main() -> int:
    patch_generator()
    patch_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
