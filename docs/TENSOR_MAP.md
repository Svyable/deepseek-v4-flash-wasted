# Tensor map — DeepSeek-V4-Flash-0731

**Status: DESIGN + SYNTHETIC-VERIFIED inventory machinery. Real checkpoint names/shapes/bytes remain BLOCKED until Gate 0 can read the pinned 0731 artifact.**

This document is the human-readable companion to `tools/inventory.py`. It exists to make every checkpoint tensor accountable: what it means, whether it is resident or streamed, which container object owns it, and which runtime component consumes it.

The final tensor map must be generated from the official checkpoint. Do not hand-maintain thousands of expert rows here.

## 1. Source of truth

Precedence:

1. official `DeepSeek-V4-Flash-0731 @ 9e165c3` checkpoint index and safetensors headers;
2. official 0731 inference/config code;
3. this map and `tools/inventory.py` rules.

If a real tensor name does not match the current classifier, **the classifier is wrong or incomplete**. Unknown main-model tensors are a Gate 0 failure.

## 2. Machine-readable inventory contract

The authoritative generated inventory should contain one row per checkpoint tensor with at least:

| Field | Meaning |
|---|---|
| `name` | exact checkpoint tensor name |
| `dtype` | safetensors dtype from the shard header |
| `shape` | stored tensor shape from the shard header |
| `stored_bytes` | exact `data_offsets[1] - data_offsets[0]` |
| `shard` | source safetensors shard |
| `layer` | normalized main/draft layer index where applicable |
| `module` | `main` or `dspark` |
| `subsystem` | normalized functional family |
| `expert` | routed expert index where applicable |
| `matrix` | normalized `w1`, `w2`, `w3` where applicable |
| `is_scale` | whether the row is quantization scale metadata |
| `scale_of` | weight tensor associated with the scale |
| `packing` | detectable values-per-byte/packing descriptor, never guessed |
| `placement` | initial `resident` or `streamed` candidate |

PR #1 implements most of this in `tools/inventory.py` using headers only.

Recommended generated artifact once the checkpoint is available:

```text
docs/inventory-0731.json
```

That JSON, not this Markdown table, should be used by scripts that need exact totals.

## 3. Current subsystem classifier

The inventory currently recognizes these functional buckets by inferred naming patterns:

| Subsystem | Initial placement | Purpose | Evidence state |
|---|---|---|---|
| `routed_expert` | streamed | sparse MoE expert matrices + associated scales | classifier SYNTHETIC-VERIFIED; real names TBD-GATE0 |
| `shared_expert` | resident | always-used expert path | TBD-GATE0 |
| `router_hash` | resident | early token/expert mapping if present | TBD-GATE0 |
| `router` | resident | learned expert scoring/correction | TBD-GATE0 |
| `csa_indexer` | resident | compressed sparse-attention selection/indexing | TBD-GATE0 |
| `compressor` | resident | compressed-attention projections/state transforms | TBD-GATE0 |
| `mhc` | resident | manifold-constrained Hyper-Connection parameters | TBD-GATE0 |
| `norm` | resident | normalization parameters | TBD-GATE0 |
| `attention` | resident | Q/KV/O and related attention projections | TBD-GATE0 |
| `embedding` | resident candidate | input embedding table | TBD-GATE0 |
| `lm_head` | resident candidate | output projection/head | TBD-GATE0 |
| `dspark_only` | separate optional module | speculative/draft-only parameters | TBD-GATE0 |

The placement column is a bootstrap policy, not a permanent storage law. Only routed experts are intentionally streamed in the first design.

## 4. Tensor-family mapping table

This is the implementation checklist. Populate the checkpoint-name and shape columns from the real inventory before binding C structs.

| Family | Exact checkpoint pattern | Exact stored shape/dtype | Container destination | Runtime consumer | Status |
|---|---|---|---|---|---|
| token embeddings | TBD-GATE0 | TBD-GATE0 | trunk | embedding lookup | BLOCKED |
| final norm | TBD-GATE0 | TBD-GATE0 | trunk | final model stage | BLOCKED |
| LM head | TBD-GATE0 | TBD-GATE0 | trunk or explicit on-disk-row strategy if later proven useful | logits | BLOCKED |
| per-layer norms | TBD-GATE0 | TBD-GATE0 | trunk | layer pre/post ops | BLOCKED |
| mHC params | TBD-GATE0 | TBD-GATE0 | trunk | `deepseek_v4/mhc` | BLOCKED |
| attention Q path | TBD-GATE0 | TBD-GATE0 | trunk | attention | BLOCKED |
| attention KV path | TBD-GATE0 | TBD-GATE0 | trunk | attention | BLOCKED |
| attention O path | TBD-GATE0 | TBD-GATE0 | trunk | attention | BLOCKED |
| compressors | TBD-GATE0 | TBD-GATE0 | trunk | compressed attention | BLOCKED |
| CSA indexers | TBD-GATE0 | TBD-GATE0 | trunk | sparse position selection | BLOCKED |
| learned routers | TBD-GATE0 | TBD-GATE0 | trunk | MoE router | BLOCKED |
| route correction/bias | TBD-GATE0 | TBD-GATE0 | trunk | MoE router | BLOCKED |
| hash/bootstrap route tables | TBD-GATE0 | TBD-GATE0 | trunk | early router/prefetch | BLOCKED |
| shared expert w1/w2/w3 | TBD-GATE0 | TBD-GATE0 | trunk | shared MoE path | BLOCKED |
| routed expert w1 | inferred aliases only | synthetic shape model only | expert bank record | streamed MoE | BLOCKED |
| routed expert w3 | inferred aliases only | synthetic shape model only | expert bank record | streamed MoE | BLOCKED |
| routed expert w2 | inferred aliases only | synthetic shape model only | expert bank record | streamed MoE | BLOCKED |
| routed expert scales | inferred suffixes only | synthetic K32 model only | same expert record | native FP4 apply | BLOCKED |
| resident FP8 scales | TBD-GATE0 | TBD-GATE0 | trunk beside owning tensor | native FP8 apply | BLOCKED |
| DSpark tensors | inferred namespace/layer signals only | TBD-GATE0 | separate optional DSpark files/section | phase-2 DSpark path | BLOCKED |

No C field should be declared solely because this table predicts a tensor. Bind only after the official inventory/reference establishes it.

## 5. Expert triplet invariant

The first proposed streamed unit is one complete routed expert.

Gate 0 must prove, for every main routed `(layer, expert)` pair:

- exactly one gate/`w1` matrix;
- exactly one up/`w3` matrix;
- exactly one down/`w2` matrix;
- all quantization scale tensors required to apply those matrices;
- no additional per-expert tensor required by the official forward path is omitted.

`tools/inventory.py` accepts both `w1/w2/w3` and `gate_proj/down_proj/up_proj` aliases for classification. That acceptance does **not** prove the official export uses either naming style.

If the real expert contains a fourth required tensor, the one-record format must change before conversion.

## 6. Scale ownership invariant

Scale tensors are attributed to the weight they scale. The generated map must make ownership unambiguous.

For each quantized weight:

```text
weight -> quantization format -> scale tensor(s) -> scale granularity/axis
```

A converter must not discover scale ownership from a heuristic once the real mapping has been established. The finalized mapping belongs in normalized config/manifest metadata or converter code covered by exact-name tests.

## 7. FP4 packing verification

Safetensors has no generic `FP4` dtype in the PR #1 inventory model. The current inventory therefore detects packing from stored dtype/shape/bytes rather than assuming `U8 == two FP4 values`.

Gate 0 must record:

- stored dtype;
- stored shape;
- logical matrix shape required by the official config/reference;
- values per stored byte;
- scale tensor shape;
- scale axis/granularity.

Only after those agree may `CONTAINER_V4.md` freeze the expert record payload.

## 8. DSpark separation

The base model must be loadable without executing DSpark.

Gate 0 must identify DSpark using checkpoint evidence, not only the current heuristic (`dspark`, `spark`, `mtp`, `speculat`, `draft`, or layer index past the main stack).

The finalized tensor map should assign every DSpark tensor to one of:

```text
dspark.shared_with_base     # only if truly the same stored tensor/reference
dspark.resident
dspark.streamed             # if the official path contains sparse draft experts
dspark.metadata
```

Do not duplicate base tensors into DSpark storage merely because the draft code references them.

## 9. Hash/bootstrap routing evidence

The README currently expects the first three layers to expose token-ID-to-expert mapping information. This is an architectural opportunity, not yet checkpoint-verified in this environment.

Gate 0/reference validation must answer:

1. Is the mapping stored as checkpoint tensors, generated from config, or encoded in reference code?
2. Is it static for a pinned model revision?
3. Does it map directly to all top-k expert IDs or to an intermediate bucket?
4. Is any score/weight still input-dependent after expert IDs are known?
5. Can the runtime prefetch selected records without performing model arithmetic first?

If the answer differs from the README assumption, update the README/architecture and remove the prefetch claim.

## 10. Gate 0 completion checklist

Before declaring this map checkpoint-verified:

- [ ] metadata downloaded from pinned revision;
- [ ] index-only inventory run and committed/logged;
- [ ] all real main-stack names classified;
- [ ] all shards/headers required for exact bytes available;
- [ ] strict inventory passes;
- [ ] no unexplained byte bucket;
- [ ] exact main-layer count established from config + tensor namespaces;
- [ ] exact expert count per MoE layer established;
- [ ] expert triplets/scales complete;
- [ ] native FP4 packing proven from storage/reference;
- [ ] resident FP8 families/scales mapped;
- [ ] mHC families mapped;
- [ ] attention/compressor/indexer families mapped;
- [ ] hash/bootstrap routing source mapped or assumption corrected;
- [ ] DSpark namespace and dependencies mapped;
- [ ] `docs/inventory-0731.json` generated;
- [ ] `docs/INVENTORY-0731.md` replaced/updated with measured totals;
- [ ] README estimates either replaced with checkpoint values or explicitly retained as historical estimates.

## 11. Change control

Once Gate 0 has passed, changes to a tensor mapping require one of:

- a new pinned DeepSeek checkpoint revision;
- a demonstrated inventory bug;
- a demonstrated official-reference interpretation bug.

Performance preferences are not a reason to relabel a tensor. Placement may evolve, semantics may not.