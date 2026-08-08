# Official reference access — unblock Gate V0/V1/V2 without over-downloading

**Status: BLOCKED in the original bootstrap environment by Hugging Face CONNECT 403. Highest-leverage external dependency.**

PR #3 narrowed the project blocker considerably. The public E2M1, UE8M0 and E4M3FN number formats are already implemented and exhaustively tested. What remains blocked is now mostly **DeepSeek-specific convention and checkpoint truth**:

- actual tensor names/shapes/storage;
- FP4 nibble ordering;
- scale direction/application;
- exact native scale tensor layout/dtype;
- official one-projection oracle output;
- model-specific arithmetic and encoding.

This document defines the smallest useful acquisition sequence. The goal is to unlock the most gates before committing to a multi-hundred-GB checkpoint download.

Do not bypass an organization/network policy. Run these steps from an authorized environment that can access the official repository.

---

## 1. Pin first, download second

Target repository:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
```

The handoff currently records `9e165c3` as the release revision seen during research. Before generating any durable fixture, resolve the full immutable commit SHA and record it.

Every fetched artifact or generated fixture should be attributable to:

```text
model repository
resolved commit SHA
artifact path
artifact SHA-256
retrieval/generator command
```

Do not generate golden fixtures from an unpinned moving branch.

---

## 2. Acquisition tier 0 — legal/provenance files

Before copying or adapting official source into this repository, retrieve the official license/notice material exactly as published.

PR #1 deliberately left:

```text
LICENSES/DEEPSEEK-MIT.txt.MISSING
```

rather than fabricating legal text from memory.

Resolve that marker before importing/adapting official DeepSeek source.

At minimum fetch whatever official repository paths provide:

```text
LICENSE / LICENSE-MODEL / NOTICE / README licensing section
```

Preserve the exact file and its source revision.

---

## 3. Acquisition tier 1 — tiny metadata/reference snapshot

This tier should be the first network operation because it can unlock names, semantics, encoding and convention review without weight shards.

Recommended allow-list:

```text
config.json
generation_config.json
model.safetensors.index.json
README.md
LICENSE*
inference/**
encoding/**
tokenizer*
*.model
*.json used by tokenizer/processor
```

Example with `huggingface_hub`:

```bash
python -m pip install 'huggingface_hub>=0.34'
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash-0731',
    revision='<FULL_RESOLVED_SHA>',
    local_dir='reference/deepseek-v4-flash-0731',
    allow_patterns=[
        'config.json',
        'generation_config.json',
        'model.safetensors.index.json',
        'README.md',
        'LICENSE*',
        'NOTICE*',
        'inference/*',
        'inference/**/*',
        'encoding/*',
        'encoding/**/*',
        'tokenizer*',
        '*.model',
    ],
)
PY
```

Do not change the pinned revision between metadata and later weight retrieval.

---

## 4. Immediate tier-1 questions to answer

Before downloading any tensor payloads, read the official reference and answer the conventions that PR #3 deliberately left unresolved.

### Q1 — FP4 nibble order

Find the exact code that extracts/expands packed E2M1 values.

Determine whether logical columns map as:

```text
even -> low nibble, odd -> high nibble
```

or the reverse.

Record:

- official path/function;
- resolved revision;
- exact shift/mask ordering;
- a small official-reference fixture using raw literal bytes.

Then reconcile the literal `0x21` assertion in `tests/test_quant.c`.

### Q2 — FP8 scale direction

Find the exact operation applied to the checkpoint tensor convention currently described as `weight_scale_inv`.

Determine whether reference arithmetic is equivalent to:

```text
e4m3(raw) * stored_scale
```

or

```text
e4m3(raw) / stored_scale
```

or another transformation.

Record the actual reference expression. Tensor naming is not enough.

### Q3 — exact scale storage

Confirm for relevant tensor families:

- scale tensor name;
- scale dtype;
- scale shape;
- block dimensions;
- row/column orientation;
- whether scales are stored transposed/reordered;
- whether expert FP4 and trunk FP8 use different scale conventions.

### Q4 — E4M3 variant

Confirm the target path actually uses finite `e4m3fn` semantics (max finite `448`, only `S.1111.111` NaN) for the checkpoint tensors being ported.

PR #3 has a correct public-format implementation of that variant, but the target tensor-family convention still belongs in the official-reference reconciliation.

### Q5 — expert matrix orientation/naming

Confirm real checkpoint names and logical orientation for routed-expert `w1/w2/w3` (or aliases). Do not assume the synthetic fixture's names prove the official export.

### Q6 — bootstrap/hash routing representation

Confirm whether the first three layers' deterministic routing data exists as checkpoint tensors, constants, code logic, or another representation.

This can materially change both `tools/inventory.py` rules and the proposed prefetch design.

---

## 5. Run index-only Gate V0 immediately

Once `config.json` and `model.safetensors.index.json` exist locally:

```bash
python3 tools/inventory.py reference/deepseek-v4-flash-0731
```

Expected mode:

```text
index-only
```

This run cannot produce exact tensor bytes/shapes without shard headers, but it can immediately expose:

- real tensor names;
- main-vs-DSpark namespace assumptions;
- unknown classifier rules;
- routed-expert naming patterns;
- scale partner naming;
- whether README/TENSOR_MAP assumptions need correction before weight acquisition.

**Unknown names should fail.** Do not weaken the classifier to make the first real run green.

Update `tools/inventory.py` rules against the official names and record corrections in `docs/INVENTORY-0731.md` / `docs/TENSOR_MAP.md`.

---

## 6. Header-only checkpoint truth is the next storage target

Full Gate V0 needs safetensors header metadata (dtype, shape, offsets) for all shards. The current local `HeaderOnlyIndex` reads only the 8-byte header length plus JSON header **after a shard file exists locally**, but obtaining an entire shard only to read its header is wasteful.

A high-leverage future tooling improvement is a remote/range header fetcher:

```text
GET first 8 bytes -> header length
GET next header_length bytes -> safetensors JSON header
```

against the official pinned shard URL, without downloading tensor payloads.

If the hosting path supports authenticated HTTP Range reliably, this could make Gate V0 checkpoint-header verification a tiny metadata operation rather than a full model download.

Until such tooling exists, use an authorized environment and the smallest supported retrieval mechanism that preserves official bytes and provenance. Do not reconstruct headers from config arithmetic.

---

## 7. Acquisition tier 2 — tiny real tensor/oracle sample

Before full conversion, acquire only enough real material to close V1 and V2 if the repository/hosting tooling permits selective shard retrieval.

Desired sample:

- one routed-expert FP4 matrix (or a small slice that includes raw packed bytes and scale bytes);
- its scale tensor;
- one representative FP8 trunk matrix/scale tile;
- deterministic input vector;
- official decoded values;
- official projection output.

Because safetensors shards may interleave many tensors, selective physical download may not be convenient. The important principle is to **close convention and one-projection gates before full conversion**, not necessarily to force a particular transport mechanism.

Generate frozen oracle fixtures per `docs/FIXTURES.md`.

---

## 8. Gate V1 closeout procedure

After official reference access:

1. cite official nibble extraction code;
2. cite official scale application code;
3. create literal official-derived convention fixtures;
4. run them against `src/quant/fp4_e2m1.*` and `src/quant/fp8_e4m3.*`;
5. if current assumptions disagree, let the tests fail;
6. change the implementation only after the official fixture exists;
7. update `docs/NUMERICS.md`, `docs/TENSOR_MAP.md`, `docs/VALIDATION.md`;
8. mark V1 passed only when public-format conformance **and** official convention agreement both hold.

Do not combine this with SIMD work.

---

## 9. Gate V2 minimal oracle

The next gate should remain one quantized linear projection.

Recommended fixture design:

```text
raw encoded weight bytes
raw scale bytes/values
input x
expected decoded weight subset
expected y
provenance
```

Choose dimensions that cross at least one relevant scale boundary.

The official generator and the C test must be separated as described in `docs/FIXTURES.md`; the C suite should replay the frozen fixture offline.

Once V2 passes, the native quantized arithmetic becomes usable as a proven model primitive rather than only a format decoder.

---

## 10. Acquisition tier 3 — full checkpoint

Only after metadata/schema/convention questions are understood should the project commit to full source-weight acquisition for conversion and end-to-end work.

Before starting:

- calculate source download size from official repository metadata;
- calculate destination/staging requirements;
- verify internal-NVMe target placement;
- verify resumable download behavior;
- pin revision;
- verify available disk space with margin;
- decide whether source shards can be reclaimed only after converter verification.

Then follow `docs/CONVERSION.md`.

---

## 11. What each acquisition tier unlocks

| Tier | Artifacts | Unlocks |
|---|---|---|
| 0 | license/notice | legal ability to copy/adapt official source with correct attribution |
| 1 | config/index/inference/encoding/tokenizer | real tensor names, convention review, encoder/oracle harness development, much of Gate V0 name mapping |
| header metadata | safetensors headers | exact dtype/shape/offset/byte totals; full Gate V0 storage truth |
| 2 | selected real tensor/oracle material | close V1, pass V2, establish projection tolerance |
| 3 | full checkpoint | converter, real container, model-layer/end-to-end gates |

The project should not treat “full checkpoint unavailable” as equivalent to “no useful progress possible.” PR #3 already demonstrated why.

---

## 12. Record the resolution

When access is restored, update all of the following in the same PR that first consumes the official artifacts:

- `docs/INVENTORY-0731.md` — remove/qualify the blocker and record commands/results;
- `docs/TENSOR_MAP.md` — real names/shapes/conventions;
- `docs/NUMERICS.md` — nibble/scale/reference agreement;
- `docs/VALIDATION.md` — V0/V1 status;
- `ROADMAP.md` — phase status;
- `UPSTREAM.md` / `LICENSES/` — official source provenance/license where applicable;
- `reference/README.md` or machine-readable provenance artifact — resolved revision/hashes;
- `docs/EXPERIMENTS.md` — any significant handoff assumption the official release disproves.

The first official-reference PR is expected to correct assumptions. A correction is progress, not a failure of the plan.