# Official reference access — README Gates A–C / V0–V2 without over-downloading

**Status: Tier 0 is resolved; pinned official source is accessible and the high-risk Gate B conventions are source-verified. The original bootstrap/current shell still cannot perform the real Hugging Face artifact fetch, so Gate A's 48-shard header run and Gate C's real projection fixture remain environment-dependent next steps.**

The repository no longer needs to treat “official reference access” as one binary blocker. Separate what is already established from what still requires an artifact-capable checkout:

- exact release SHA + official license — **resolved**;
- official `inference/` operation semantics — **source-verified**;
- FP4 nibble order / scale direction / scale geometry — **source-verified for Gate B/V1**;
- real exported tensor names/shapes/dtypes/bytes — **Gate A/V0 still open**;
- one real official quantized projection — **Gate C/V2 still open**;
- full checkpoint payload — deliberately deferred until cheaper gates justify it.

Canonical release:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

Do not bypass an organization/network policy. Run network steps from an authorized environment. Do not silently switch to a later model-card-only repository tip or a third-party conversion.

---

## 1. Pin first, download second — DONE

All durable fixtures/conversions should record:

```text
model repository
resolved commit SHA
artifact path
artifact SHA-256
retrieval/generator command
generator/port commit
```

The baseline above is the immutable 0731 release used by PR #5 source evidence. Moving it is an explicit project event.

---

## 2. Acquisition tier 0 — legal/provenance — DONE

PR #5 vendors the exact official release `LICENSE` as:

```text
LICENSES/DEEPSEEK-MIT.txt
```

and removes the old `DEEPSEEK-MIT.txt.MISSING` marker.

Recorded upstream facts:

```text
license: MIT
copyright: Copyright (c) 2023 DeepSeek
release SHA: 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

Before adapting a particular official source file, still record its source path/revision and preserve required attribution. The license blocker itself is closed.

---

## 3. Tier 1 — metadata/reference source

Useful small assets remain:

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
```

Where `huggingface_hub` is available, a normal metadata/source snapshot is fine as long as the full immutable SHA is used:

```bash
python -m pip install 'huggingface_hub>=0.34'
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash-0731',
    revision='9e165c30e2704aec5d9d593cce3eebd58bbef1cb',
    local_dir='reference/deepseek-v4-flash-0731',
    allow_patterns=[
        'config.json',
        'generation_config.json',
        'model.safetensors.index.json',
        'README.md',
        'LICENSE*',
        'NOTICE*',
        'inference/*', 'inference/**/*',
        'encoding/*', 'encoding/**/*',
        'tokenizer*', '*.model',
    ],
)
PY
```

PR #5's source review already records the release-level findings needed for the current quantization handoff in `OFFICIAL-0731-SOURCE.md`.

---

## 4. Gate B / V1 source questions — ANSWERED for the pinned release

### FP4 nibble order

Official converter behavior expands each packed byte along K as:

```text
low  = byte & 0x0f
high = (byte >> 4) & 0x0f
stack([low, high])
```

Therefore:

```text
lower/even logical K index -> low nibble
next/odd logical K index   -> high nibble
```

### Routed FP4 scale storage

Official model binding:

```text
logical weight: [out, in]
stored weight:  [out, in/2]  float4_e2m1fn_x2
scale:          [out, in/32] float8_e8m0fnu
```

### Scale direction

Official quantized GEMM multiplies block partials by activation scale × weight scale. The converter renames `weight_scale_inv` to `scale` without computing a reciprocal. The scale is a multiplier for the pinned operation.

### Resident FP8

The release uses finite `torch.float8_e4m3fn` with one weight scale per 128x128 block. The scale is again applied multiplicatively in the official GEMM.

### Independent fixture

PR #5 freezes:

```text
tests/fixtures/deepseek_v4/fp4_release_convention.json
```

with literal:

```text
packed byte 0x21
E8M0 scale 0x80 = 2.0
expected logical values [1.0, 2.0]
```

`tests/test_release_quant_fixture.py` compiles the actual scalar FP4 source against it. This removes the previous “nibble order and scale direction are unknown” blocker. A fresh full branch test run is still required before merge.

---

## 5. Gate A / V0 index-only inventory

Once `config.json` and `model.safetensors.index.json` are available locally:

```bash
python3 tools/inventory.py reference/deepseek-v4-flash-0731
```

Index-only mode can establish real names/shard assignment and immediately expose classifier mistakes. It cannot establish exact dtype/shape/bytes without headers.

Unknown main names remain failures. PR #5 has already added narrow source-verified bootstrap-route spellings:

```text
tid2eid   # official runtime field
tie2eid   # converter source spelling
```

Do not turn that into a generic router catch-all.

---

## 6. Gate A / V0 header-only checkpoint truth — TOOL IMPLEMENTED

PR #5 adds:

```text
tools/fetch_hf_headers.py
```

It implements the high-leverage Range strategy that this document previously described as future work.

For every indexed safetensors shard:

```text
GET bytes 0-7                  -> little-endian JSON-header length
GET bytes 8 .. 7+header_length -> exact safetensors JSON header
```

Safety properties:

- full immutable 40-hex revision required by default;
- `Accept-Encoding: identity`;
- HTTP **206 Partial Content required**;
- `Content-Range` validated;
- a server/proxy that returns HTTP 200 to a Range request is refused;
- header metadata/data offsets are validated;
- index↔header tensor agreement is checked;
- optional Hub token is never written to provenance.

The tool writes each shard as a native **header-only stub**:

```text
8-byte header length
original JSON header
(no tensor payload)
```

so the existing `HeaderOnlyIndex` is reused unchanged conceptually: the acquisition layer does transport/provenance, and `inventory.py` owns tensor accounting.

### Exact next commands

```bash
python3 tools/fetch_hf_headers.py \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --out reference/deepseek-v4-flash-0731

python3 tools/inventory.py reference/deepseek-v4-flash-0731 \
  --strict --by-layer --json docs/inventory-0731.json
```

The first real strict run is allowed to fail. Correct `RULES` only against actual official names, add regression cases, and rerun until every main tensor/byte is explained.

Gate A/V0 is passed only when all 48 headers and all exported main-model tensor families are reconciled. Source/config reasoning alone cannot substitute for that.

---

## 7. Tier 2 — tiny real tensor/oracle sample

After Gate A identifies exact resident tensor names/shards, acquire only enough payload/oracle material for Gate C/V2 before committing to full conversion.

Preferred first target: one representative resident quantized linear whose K dimension crosses a 128-wide activation/weight scale boundary.

Fixture should preserve:

```text
source tensor name
source shard/header identity
raw E4M3 weight bytes for bounded rows/tile
raw or decoded official weight scales
input activation
reference activation-quantized bytes/scales where practical
official expected output
reference source/device/dtype/kernel provenance
```

Physical selective retrieval may be inconvenient when many tensors share one shard. The principle is to keep the **oracle scope** tiny even if transport requires a larger shard download.

---

## 8. Gate C / V2 official-linear handoff

PR #5 now implements a source-derived scalar preflight in:

```text
src/quant/deepseek_v4_linear_ref.c
src/quant/deepseek_v4_linear_ref.h
tests/test_v2_linear_ref.py
```

Pinned source operation:

```text
for each activation row / K128 block:
  amax = max(max(abs(x)), 1e-4)
  scale = next_power_of_two(amax / 448)
  q = E4M3FN(clamp(x / scale, -448, 448))

for each FP8 K128 dot:
  accum += dot(qx, qw) * activation_scale * weight_scale

reference output -> BF16
```

The closed-form preflight intentionally chooses exact normalized values and expects `320`. It proves the scalar seam is executable; it is **not** a real V2 oracle.

To pass Gate C/V2:

1. select the real projection after Gate A name/header mapping;
2. generate the expected result from pinned official code without importing WASTE helpers;
3. freeze raw inputs/scales/expected output with provenance;
4. replay with `deepseek_v4_linear_ref`;
5. explain error and set a fixed tolerance;
6. verify known-wrong activation-scale, weight-scale, block-index and rounding mutations fail.

No SIMD or transformer integration before this gate.

---

## 9. Routing/source facts unlocked before real payloads

The pinned runtime uses a bootstrap token-to-expert table named:

```text
tid2eid
```

with logical shape:

```text
[vocab_size, num_experts_per_tok]
```

for `layer_id < n_hash_layers`. The converter also contains the spelling `tie2eid`.

Learned routing source semantics are recorded in `OFFICIAL-0731-SOURCE.md` and remain future Gate F/V4 implementation material. Source knowledge is useful for classifier/oracle design; exact exported tensor binding still belongs to Gate A.

---

## 10. Tier 3 — full checkpoint

Only after Gate A and Gate C have killed basic storage/numerical mistakes should the project commit to full source-weight acquisition/conversion.

Before starting:

- calculate official source download size;
- calculate output/staging free-space requirements;
- pin the same immutable revision;
- verify resumability;
- verify target NVMe/filesystem suitability;
- decide source-reclamation policy only after converter verification.

Then follow `CONVERSION.md`.

---

## 11. What each acquisition tier unlocks

| Tier | State | Artifacts | Unlocks |
|---|---|---|---|
| 0 | **DONE** | exact release license/provenance | legal/source provenance |
| official source/config | **DONE enough for current quant work** | pinned inference/config source | Gate B operation conventions; Gate C scalar design; routing design facts |
| index | **NEXT in artifact-capable checkout** | config + safetensors index | real tensor names/shard assignment; partial Gate A |
| header metadata | **NEXT** | all safetensors headers via Range stubs | exact dtype/shape/offset/byte totals; Gate A storage truth |
| selected payload/oracle | **NEXT after Gate A** | one real quantized projection + expected output | Gate C/V2 |
| full checkpoint | **DEFERRED** | all payloads | converter, real container, transformer/end-to-end gates |

The project has already demonstrated why “full checkpoint unavailable” does not mean “no useful work possible.”

---

## 12. Evidence update rule

When the first real header run lands, update in the same PR:

- `INVENTORY-0731.md` — exact commands/results and measured totals;
- `TENSOR_MAP.md` — exact exported names/shapes/layouts;
- `NUMERICS.md` — only if real storage contradicts the pinned source contract;
- `VALIDATION.md` — Gate A status and any downstream consequence;
- `ROADMAP.md` — Phase 1 status;
- `MEMORY_AND_IO.md` — checkpoint-derived storage facts only;
- `reference/` provenance metadata — resolved hashes;
- `EXPERIMENTS.md` — meaningful corrected assumptions.

When the real Gate C projection lands, update `NUMERICS.md` and `VALIDATION.md` with the fixture, error metrics/tolerance and mutation results.

Corrections are progress. The official artifact wins over the handoff, synthetic fixture, or source-level inference about how a tensor was ultimately exported.
