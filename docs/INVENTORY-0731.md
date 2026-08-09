# INVENTORY-0731 — exact checkpoint totals

**Status: CHECKPOINT-VERIFIED. README Gate A / V0 PASSED against all 48 safetensors headers from `deepseek-ai/DeepSeek-V4-Flash-0731 @ 9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.**

The original bootstrap environment could not reach Hugging Face. PR #5 removed that blocker by adding a fail-closed Range-based header fetcher and running it on GitHub's Ubuntu runner. The first real inventory did exactly what the gate was designed to do: it exposed three narrow export-name assumptions (`layers.N` at the root, `ffn.gate`, and `head.weight`). Those were corrected against the checkpoint and the full run was repeated.

Validation run:

```text
GitHub Actions run: 31285866894
OS: Ubuntu 24.04.4 LTS
pinned revision: 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
48 / 48 shard headers fetched
make: PASS
make check: PASS
make asan: PASS
Gate A / V0: 6 PASS, 0 FAIL, 0 SKIP
```

The compact machine-readable evidence is committed at:

```text
reference/deepseek-v4-flash-0731.gate-a.json
```

The full per-tensor inventory was emitted and retained as the workflow artifact for the validating run; the repo keeps the compact summary rather than committing a tens-of-megabytes generated listing.

---

## Exact checkpoint totals

| quantity | checkpoint result |
|---|---:|
| safetensors shards | **48** |
| tensors | **72,317** |
| payload bytes | **166,878,536,440 B** |
| payload size | **155.417748 GiB** |
| main layers | **43** |
| routed experts / layer | **256** |
| shared experts / layer | **1** |
| routed experts / token / layer | **6** |
| bootstrap hash layers | **3** |

Byte buckets from safetensors `data_offsets`:

| bucket | tensors | exact bytes | GiB |
|---|---:|---:|---:|
| main routed experts | 66,048 | **147,169,738,752** | **137.062500** |
| shared experts | 258 | 1,082,196,480 | 1.007874 |
| attention / compressor / indexer | 784 | 5,400,422,016 | 5.029535 |
| mHC | 261 | 135,537,756 | 0.126229 |
| embedding + output head | 2 | 2,118,123,520 | 1.972656 |
| DSpark | 4,705 | 10,862,838,300 | 10.116807 |
| other resident tensors | 259 | 109,679,616 | 0.102147 |
| unclassified | **0** | **0** | **0** |

The base path excluding DSpark occupies approximately **145.300942 GiB of checkpoint payload**, of which **137.0625 GiB** is routed experts. This is storage accounting, not a runtime RAM floor: the point of the WASTE port is specifically not to resident-load that routed bank.

---

## Exact routed-expert result

The handoff estimate was correct down to the byte, but it is no longer an estimate.

For each routed expert the checkpoint contains:

```text
w1.weight  I8-packed FP4
w1.scale   F8_E8M0
w3.weight  I8-packed FP4
w3.scale   F8_E8M0
w2.weight  I8-packed FP4
w2.scale   F8_E8M0
```

Representative layer-0 expert shapes:

```text
layers.0.ffn.experts.0.w1.weight  I8        [2048, 2048]
layers.0.ffn.experts.0.w1.scale   F8_E8M0  [2048, 128]
layers.0.ffn.experts.0.w3.weight  I8        [2048, 2048]
layers.0.ffn.experts.0.w3.scale   F8_E8M0  [2048, 128]
layers.0.ffn.experts.0.w2.weight  I8        [4096, 1024]
layers.0.ffn.experts.0.w2.scale   F8_E8M0  [4096, 64]
```

Exact record arithmetic:

| quantity | result |
|---|---:|
| main routed expert records | **11,008** = 43 × 256 |
| checkpoint payload / expert | **13,369,344 B** = **12.75 MiB** |
| routed records touched / decode token | **258** = 43 × 6 |
| all-miss routed payload / decode token | **3,449,290,752 B** = **3.212402 GiB** |

These figures are payload only. A future WASTE container may add record headers and 4 KiB padding; cache hits will reduce actual read traffic.

This result strongly validates the core WASTE architecture choice: routed storage dominates the base checkpoint and access is top-k sparse.

---

## Exact dtype totals

| dtype | tensors | bytes |
|---|---:|---:|
| `I8` | 35,328 | 148,176,371,712 |
| `F8_E8M0` | 35,718 | 9,261,408,000 |
| `F8_E4M3` | 390 | 6,304,038,912 |
| `BF16` | 445 | 2,967,134,976 |
| `F32` | 433 | 150,966,520 |
| `I64` | 3 | 18,616,320 |

The checkpoint uses `I8` storage for packed FP4 expert payloads. Treating that byte plane as signed arithmetic would be wrong; it is packed-bit storage whose nibble interpretation is owned by Gate B / V1.

---

## Bootstrap routing is checkpoint-resident

Gate A confirms the first three layers carry exact token-ID-to-expert tables:

```text
layers.0.ffn.gate.tid2eid
layers.1.ffn.gate.tid2eid
layers.2.ffn.gate.tid2eid
```

Each is:

```text
dtype: I64
shape: [129280, 6]
```

This upgrades deterministic bootstrap routing and its prefetch opportunity from source/design evidence to **CHECKPOINT-VERIFIED**. For an input token ID, the six routed expert IDs in each of the first three layers can be known before any router matmul. The routing weights still follow the official model semantics; the table establishes IDs, not all floating-point routing arithmetic.

---

## Resident projection selected for Gate C / V2

Gate A identifies a compact representative quantized resident projection for the next numerical gate:

```text
layers.0.attn.wq_a.weight  F8_E4M3  [1024, 4096]
layers.0.attn.wq_a.scale   F8_E8M0  [8, 32]
```

Both tensors are in `model-00002-of-00048.safetensors`.

Gate C will freeze only a few real rows plus the corresponding scale row and compare a deterministic input against an independent pinned-source oracle and the scalar C `deepseek_v4_linear_ref` path. There is no reason to download the entire checkpoint to validate this projection.

---

## Gate A / V0 final checklist

| check | result |
|---|---|
| all main tensor names classified | **PASS — 72,317 tensors total** |
| no unexplained bytes | **PASS — 0 stray bytes** |
| dimensions agree with config | **PASS** |
| one `w1/w3/w2` + scales per routed expert | **PASS — 11,008 records** |
| first three layers expose token-ID → expert mapping | **PASS — layers 0,1,2** |
| DSpark separable from base stack | **PASS — 4,705 DSpark tensors, main layers 0..42** |

**Gate A / V0 is complete for the pinned 0731 release.**

If a future project change moves the model revision, this gate must be rerun; checkpoint-derived facts are revision-specific.
