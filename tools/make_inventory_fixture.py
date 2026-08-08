#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The deepseek-v4-flash-wasted authors.
"""
make_inventory_fixture.py — a synthetic DeepSeek-V4-Flash-shaped checkpoint.

tools/inventory.py has to be exercised before anyone can reach the real
0731 weights, so this writes a checkpoint that is *structurally* real —
genuine safetensors headers, a genuine `model.safetensors.index.json`,
config-consistent shapes, FP4-packed expert matrices with UE8M0 K32 scales,
FP8 trunk weights with 128x128 scale blocks — and contains no weights at
all.

  tools/make_inventory_fixture.py /tmp/fixture
  tools/make_inventory_fixture.py /tmp/full --layers 43 --experts 256
  tools/make_inventory_fixture.py /tmp/withdata --with-data

**This is not DeepSeek-V4-Flash-0731 and its tensor names are not
authoritative.** They are the plausible HF-style names that README §1's
architecture implies, written without access to the checkpoint (see
LICENSES/README.md for why). A passing test here proves that
`inventory.py` classifies, measures and gates correctly given a checkpoint
of this shape — it does not prove the name table matches DeepSeek's. Gate 0
failing loudly on the real weights is the intended outcome if it does not.

By default the shards stop immediately after the header: every tensor's
`data_offsets` describe bytes that are not in the file. That is deliberate.
`inventory.py` promises to read headers only, and a fixture with no data
turns that promise into something the test suite can enforce — any attempt
to read a tensor hits EOF instead of quietly succeeding. Pass `--with-data`
for a fixture with real (zero-filled) payload bytes.
"""

import argparse
import json
import os
import struct
import sys

# The documented 0731 shape (README §1). Overridable so tests can run a
# small checkpoint and still exercise every code path.
DEFAULTS = {
    "layers": 43,
    "experts": 256,
    "hidden": 4096,
    "moe_inter": 2048,
    "top_k": 6,
    "shared": 1,
    "vocab": 129280,
    "hash_layers": 3,
    "q_lora": 1024,
    "heads": 64,
    "head_dim": 512,
    "indexer_heads": 64,
    "indexer_dim": 128,
    "hc_mult": 4,
    "dspark_layers": 1,
}

FP4_PER_BYTE = 2      # two E2M1 values per byte
FP4_SCALE_K = 32      # one UE8M0 scale per 32 values along K
FP8_BLOCK = 128       # 128x128 scaling blocks on trunk linears

DTYPE_BYTES = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4, "I32": 4}


class ShardWriter:
    """Accumulates tensors, then writes safetensors shards + an index."""

    def __init__(self, out_dir, shard_bytes):
        self.dir = out_dir
        self.shard_bytes = shard_bytes
        self.tensors = []      # (name, dtype, shape, nbytes)

    def add(self, name, dtype, shape):
        n = DTYPE_BYTES[dtype]
        for d in shape:
            n *= d
        self.tensors.append((name, dtype, list(shape), n))

    def _partition(self):
        shards, cur, cur_bytes = [], [], 0
        for t in self.tensors:
            if cur and cur_bytes + t[3] > self.shard_bytes:
                shards.append(cur)
                cur, cur_bytes = [], 0
            cur.append(t)
            cur_bytes += t[3]
        if cur:
            shards.append(cur)
        return shards

    def write(self, with_data):
        os.makedirs(self.dir, exist_ok=True)
        shards = self._partition()
        total = len(shards)
        weight_map, total_size = {}, 0

        for i, group in enumerate(shards, start=1):
            fn = f"model-{i:05d}-of-{total:05d}.safetensors"
            header, off = {}, 0
            for name, dtype, shape, nbytes in group:
                header[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [off, off + nbytes],
                }
                off += nbytes
                weight_map[name] = fn
                total_size += nbytes
            # safetensors requires the header be padded to 8-byte alignment
            blob = json.dumps(header, separators=(",", ":")).encode()
            pad = (-len(blob)) % 8
            blob += b" " * pad
            with open(os.path.join(self.dir, fn), "wb") as f:
                f.write(struct.pack("<Q", len(blob)))
                f.write(blob)
                if with_data:
                    f.write(b"\0" * off)

        index = {
            "metadata": {
                "total_size": total_size,
                "note": "SYNTHETIC FIXTURE - not DeepSeek-V4-Flash-0731",
            },
            "weight_map": weight_map,
        }
        with open(os.path.join(self.dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=1)
            f.write("\n")
        return total, total_size


def fp4_matrix(w, out_dim, in_dim, prefix):
    """A packed-FP4 weight plus its UE8M0 K32 scale, as README §7 describes.

    Logical [out, in]; stored [out, in/2] U8 with scale [out, in/32] U8.
    """
    w.add(f"{prefix}.weight", "U8", [out_dim, in_dim // FP4_PER_BYTE])
    w.add(f"{prefix}.weight_scale", "U8", [out_dim, in_dim // FP4_SCALE_K])


def fp8_matrix(w, out_dim, in_dim, prefix):
    """An FP8 E4M3 trunk linear plus its 128x128 F32 scale blocks."""
    w.add(f"{prefix}.weight", "F8_E4M3", [out_dim, in_dim])
    w.add(f"{prefix}.weight_scale_inv", "F32",
          [(out_dim + FP8_BLOCK - 1) // FP8_BLOCK,
           (in_dim + FP8_BLOCK - 1) // FP8_BLOCK])


def build_layer(w, cfg, layer, module="model"):
    p = f"{module}.layers.{layer}"
    h = cfg["hidden"]

    # norms
    w.add(f"{p}.input_layernorm.weight", "BF16", [h])
    w.add(f"{p}.post_attention_layernorm.weight", "BF16", [h])

    # manifold-constrained hyper-connections
    w.add(f"{p}.hyper_connections.alpha", "F32", [cfg["hc_mult"], h])
    w.add(f"{p}.hyper_connections.beta", "F32", [cfg["hc_mult"], h])

    # attention: LoRA'd Q, compressed KV, output projection
    qk = cfg["heads"] * cfg["head_dim"]
    fp8_matrix(w, cfg["q_lora"], h, f"{p}.self_attn.q_a_proj")
    fp8_matrix(w, qk, cfg["q_lora"], f"{p}.self_attn.q_b_proj")
    fp8_matrix(w, cfg["q_lora"], h, f"{p}.self_attn.kv_a_proj_with_mqa")
    fp8_matrix(w, qk, cfg["q_lora"], f"{p}.self_attn.kv_b_proj")
    fp8_matrix(w, h, qk, f"{p}.self_attn.o_proj")
    w.add(f"{p}.self_attn.q_a_layernorm.weight", "BF16", [cfg["q_lora"]])

    # compressed attention machinery: absent on the two ratio-0 layers
    if layer >= 2:
        fp8_matrix(w, h, h, f"{p}.self_attn.kv_compressor")
        idx = cfg["indexer_heads"] * cfg["indexer_dim"]
        fp8_matrix(w, idx, h, f"{p}.self_attn.indexer.wq")
        fp8_matrix(w, idx, h, f"{p}.self_attn.indexer.wk")

    # router. The first hash_layers use a stored token-id -> expert table
    # instead of a learned gate (README §11).
    if layer < cfg["hash_layers"]:
        w.add(f"{p}.mlp.gate.hash_table", "I32", [cfg["vocab"], cfg["top_k"]])
    w.add(f"{p}.mlp.gate.weight", "BF16", [cfg["experts"], h])
    w.add(f"{p}.mlp.gate.e_score_correction_bias", "F32", [cfg["experts"]])

    # shared expert (resident) — FP8 like the rest of the trunk
    for _ in range(cfg["shared"]):
        fp8_matrix(w, cfg["moe_inter"], h, f"{p}.mlp.shared_experts.w1")
        fp8_matrix(w, cfg["moe_inter"], h, f"{p}.mlp.shared_experts.w3")
        fp8_matrix(w, h, cfg["moe_inter"], f"{p}.mlp.shared_experts.w2")

    # routed experts (streamed) — native FP4
    for e in range(cfg["experts"]):
        base = f"{p}.mlp.experts.{e}"
        fp4_matrix(w, cfg["moe_inter"], h, f"{base}.w1")
        fp4_matrix(w, cfg["moe_inter"], h, f"{base}.w3")
        fp4_matrix(w, h, cfg["moe_inter"], f"{base}.w2")


def build(cfg, out_dir, shard_bytes, with_data):
    w = ShardWriter(out_dir, shard_bytes)
    h = cfg["hidden"]

    w.add("model.embed_tokens.weight", "BF16", [cfg["vocab"], h])
    for layer in range(cfg["layers"]):
        build_layer(w, cfg, layer)
    w.add("model.norm.weight", "BF16", [h])
    w.add("lm_head.weight", "BF16", [cfg["vocab"], h])

    # DSpark: an attached module past the end of the main stack. Named so
    # that it is identifiable *and* out of the main layer range, which is
    # what Gate 0's separability check looks for.
    for i in range(cfg["dspark_layers"]):
        layer = cfg["layers"] + i
        build_layer(w, cfg, layer, module="dspark")
        w.add(f"dspark.layers.{layer}.markov.weight", "BF16", [256, h])

    n_shards, total_size = w.write(with_data)

    config = {
        "_comment": "SYNTHETIC FIXTURE - not DeepSeek-V4-Flash-0731",
        "model_type": "deepseek_v4_flash",
        "num_hidden_layers": cfg["layers"],
        "hidden_size": h,
        "n_routed_experts": cfg["experts"],
        "n_shared_experts": cfg["shared"],
        "num_experts_per_tok": cfg["top_k"],
        "moe_intermediate_size": cfg["moe_inter"],
        "vocab_size": cfg["vocab"],
        "n_hash_layers": cfg["hash_layers"],
        "num_attention_heads": cfg["heads"],
        "q_lora_rank": cfg["q_lora"],
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return n_shards, total_size, len(w.tensors)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=int, default=val)
    ap.add_argument("--shard-bytes", type=int, default=4 << 30,
                    help="target bytes per shard (default 4 GiB)")
    ap.add_argument("--with-data", action="store_true",
                    help="write real zero-filled payload bytes; by default "
                         "shards end after the header")
    args = ap.parse_args(argv)

    cfg = {k: getattr(args, k) for k in DEFAULTS}
    if cfg["hidden"] % FP4_SCALE_K or cfg["moe_inter"] % FP4_SCALE_K:
        print(f"fixture: hidden and moe_inter must be multiples of "
              f"{FP4_SCALE_K}", file=sys.stderr)
        return 2

    n_shards, total, n_tensors = build(cfg, args.out_dir, args.shard_bytes,
                                       args.with_data)
    kind = "with data" if args.with_data else "headers only"
    print(f"  wrote {args.out_dir}: {n_tensors} tensors, {n_shards} shard(s), "
          f"{total / (1 << 30):.2f} GiB described ({kind})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
