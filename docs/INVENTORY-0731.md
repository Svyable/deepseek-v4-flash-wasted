# INVENTORY-0731 — exact checkpoint totals

**Status: NOT YET MEASURED. This document contains no checkpoint-derived numbers. README Gate A / V0 is not yet complete.**

README §21 asks for this file to carry exact byte totals for `deepseek-ai/DeepSeek-V4-Flash-0731`. It cannot yet, and the reason is environmental rather than technical:

```text
huggingface.co:443 — the agent proxy answered 403 to CONNECT
                     (organization egress policy denial)
```

The checkpoint metadata — `config.json`, `model.safetensors.index.json`, and the shard headers — was unreachable from the machine that bootstrapped this repository. Per `/root/.ccr/README.md` a policy denial is reported, not retried or routed around.

`tools/inventory.py` is finished and tested. What is missing is the input.

Gate terminology follows `docs/VALIDATION.md` §4a: this document owns the checkpoint-inventory evidence for **README Gate A / V0**. “Gate 0” is historical shorthand and should not be used in new docs/PRs.

## What to run

From any environment that can reach Hugging Face:

```bash
python -m pip install 'huggingface_hub>=0.34'
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash-0731',
    revision='9e165c3',
    local_dir='reference/deepseek-v4-flash-0731',
    allow_patterns=['config.json', 'model.safetensors.index.json',
                    'tokenizer*', 'inference/*', 'encoding/*',
                    'LICENSE*', 'README.md'],
)
PY

# Names and counts, before a single weight byte is downloaded.
python3 tools/inventory.py reference/deepseek-v4-flash-0731

# Exact bytes, once the shards are present.
python3 tools/inventory.py reference/deepseek-v4-flash-0731 \
    --by-layer --strict --json docs/inventory-0731.json
```

The first run works on metadata alone and reports `mode: index-only`: it classifies every tensor name and evaluates the checks that do not need shapes, while byte totals stay explicitly unresolved rather than defaulting to zero. That is the run README §22 wants done before committing to a multi-hundred-GB download, because a mistaken tensor assumption is far cheaper to find there.

Then replace this file with the output, and update `README.md` §1 with the measured figures.

## Expect README Gate A / V0 to fail on the first real run

The classification table in `tools/inventory.py` (`RULES`) was written from the architecture description in README §1, **not** from the checkpoint. The tensor names are inferred. Unrecognised names deliberately fail Gate A/V0 rather than landing in a catch-all bucket, so the first real run is expected to report unclassified tensors.

That is the tool working. Extend `RULES` against the actual names, re-run, and record what the checkpoint said. README §5 is explicit about precedence:

> If the inventory disagrees with this README, update this README.
> The checkpoint wins.

The same applies to this file and to every estimate in README §1.

## The estimates this file must replace

From README §1, all first-order and **none of them measured**:

| quantity | estimate | status |
|---|---:|---|
| packed FP4 bytes per expert | 12,582,912 | unverified |
| UE8M0 K32 scale bytes per expert | 786,432 | unverified |
| bytes per expert record (pre-alignment) | 13,369,344 (12.75 MiB) | unverified |
| main routed expert records | 11,008 | unverified |
| routed experts per decode token | 258 | unverified |
| routed bytes per all-miss decode token | ≈3.21 GiB | unverified |

Nothing in this repository may cite these as measurements, and no RAM, disk-size or tokens/sec claim may be built on them.

## What has been verified

Only that the tool computes the right things given a checkpoint of the documented shape. `tests/test_inventory.py` builds a synthetic checkpoint from `tools/make_inventory_fixture.py` with the 0731 dimensions — 43 layers, 256 routed experts, hidden 4096, MoE intermediate 2048, FP4-packed expert matrices with UE8M0 K32 scales — and the tool reproduces the table above from it exactly: 11,008 records, 13,369,344 B each, 3.21 GiB per all-miss token.

**This confirms the arithmetic, not the checkpoint.** The fixture was generated from the same README §1 numbers it reproduces, so the agreement says the packing and scale math is self-consistent and that the per-record and per-token roll-ups are right. It says nothing whatsoever about what DeepSeek actually shipped. A synthetic fixture cannot corroborate its own source.

The fixture's shards deliberately stop after the safetensors header, which is how the header-only guarantee is enforced rather than asserted: every tensor's `data_offsets` point past end-of-file, so any code path that tried to read tensor data would hit EOF instead of quietly succeeding.

## README Gate A / V0 checks the real run must satisfy

From README §5, as implemented:

| check | what a failure means |
|---|---|
| all tensor names classified | `RULES` needs extending against real names |
| no bytes in an unexplained bucket | a subsystem is unaccounted for |
| dimensions agree with `config.json` | README §1's shape table is wrong |
| one w1/w2/w3 per routed expert, with scales | the expert record design in §6 needs revisiting |
| first three layers expose token-id → expert mapping | §11's deterministic-prefetch opportunity does not exist as described |
| DSpark tensors separable from the main stack | the 43-layer path cannot be brought up independently, contradicting §15 |

The last two are the ones with architectural consequences, so read their detail lines rather than just the pass/fail.

Passing this checklist supplies storage/tensor truth for Gate A/V0. It does **not** by itself complete Gate B/V1's official packing/scale arithmetic convention agreement; `NUMERICS.md` and `REFERENCE_ACCESS.md` own that distinction.