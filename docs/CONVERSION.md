# Conversion runbook — official DeepSeek 0731 to WASTE

**Status: DESIGN. `tools/inventory.py` exists; the DeepSeek converter does not. The bootstrap environment could not reach Hugging Face.**

This runbook defines the order of operations so the project never discovers a basic checkpoint mismatch after committing to a full conversion.

## 1. Non-negotiable input rule

The conversion source is the pinned official checkpoint:

```text
deepseek-ai/DeepSeek-V4-Flash-0731 @ 9e165c3
```

Do not use a GGUF, vLLM cache, transformed checkpoint, or third-party quantization as the converter source for the correctness baseline.

Third-party artifacts may be used later for behavioral smoke comparisons only.

## 2. License gate before source adaptation

PR #1 intentionally did not invent an MIT license file when the official repository was unreachable.

Before copying or adapting DeepSeek `inference/` or `encoding/` source into this repository:

1. fetch the exact official license/notice file from the pinned source;
2. place the appropriate copy/attribution under `LICENSES/`;
3. preserve required headers/notices for adapted files;
4. record source path and revision.

Checkpoint metadata inspection can proceed without copying source code into the repository, but legal/provenance material must be resolved before vendoring/adapting official code.

## 3. Stage A — metadata-only fetch

This is the first network operation because it is cheap enough to invalidate the plan before a weight download.

Example from `INVENTORY-0731.md`:

```bash
python -m pip install 'huggingface_hub>=0.34'
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash-0731',
    revision='9e165c3',
    local_dir='reference/deepseek-v4-flash-0731',
    allow_patterns=[
        'config.json',
        'model.safetensors.index.json',
        'generation_config.json',
        'tokenizer*',
        'inference/*',
        'encoding/*',
        'LICENSE*',
        'README.md',
    ],
)
PY
```

If organizational policy denies the request, report it and stop. Do not route around an explicit egress control.

## 4. Stage B — index-only inventory

Run before downloading shard data:

```bash
python3 tools/inventory.py reference/deepseek-v4-flash-0731
```

Expected mode:

```text
index-only
```

At this stage:

- names/counts can be classified;
- dtype/shape/stored-byte totals remain unresolved because the safetensors index does not contain them;
- checks that require headers should SKIP rather than use inferred zeros.

### Required action on first real run

Expect unknown names. Update `tools/inventory.py::RULES` against the actual checkpoint, add regression cases to `tests/test_inventory.py`, and re-run until all main-stack names are explained.

Do not change checkpoint names to fit the README.

## 5. Stage C — download planning

Before downloading full shards, compute and record:

- source-repository revision;
- file list and published sizes if available from the repository API/metadata;
- free space on staging volume;
- intended output volume free space;
- whether source and output can coexist;
- whether reclaiming source shards during conversion will eventually be implemented;
- expected filesystem support for large files;
- expected direct-I/O alignment constraints on the runtime/output filesystem.

Do not put a projected final container size in documentation as a measurement. Exact source sizes may be recorded from repository metadata; final container size exists only after converter dry-run/exact tensor accounting.

## 6. Stage D — full checkpoint availability

Use an official Hugging Face snapshot mechanism with the same pinned revision. The exact command may vary with `huggingface_hub` version and local storage policy.

Requirements:

- resumable download;
- no mutable `main` revision in production conversion scripts;
- verify that all files named by `model.safetensors.index.json` are present;
- preserve the unmodified checkpoint as the comparison source until converter validation passes.

The converter should never rewrite the source checkpoint in place.

## 7. Stage E — strict header inventory / Gate 0

With shards present:

```bash
python3 tools/inventory.py reference/deepseek-v4-flash-0731 \
  --by-layer --strict --json docs/inventory-0731.json
```

Before converter implementation proceeds, update:

- `docs/INVENTORY-0731.md` with exact totals;
- `docs/TENSOR_MAP.md` with exact family mappings;
- `README.md` estimates that the checkpoint disproves;
- `docs/MEMORY_AND_IO.md` only with checkpoint-derived storage facts, not throughput forecasts.

A strict inventory failure is a blocker, not a warning to suppress.

## 8. Stage F — converter dry run

The future `tools/convert_deepseek_v4.py` should support a no-payload/dry-run mode before it writes hundreds of gigabytes.

Dry-run output should include:

```text
source revision
recognized architecture/model family
main layer count
resident tensor count and exact input bytes
routed expert count and exact input bytes
DSpark exact input bytes
planned trunk output bytes
planned expert record bytes by layer
alignment/padding overhead
planned DSpark output bytes
peak temporary workspace estimate
minimum free output space
conversion format/version
```

All byte counts in this stage come from checkpoint headers + chosen deterministic storage transforms.

If the dry run cannot explain an output byte, the real run should not start.

## 9. Native-preserving baseline policy

The first converter exists to preserve the official model's semantics, not to minimize disk size.

Initial policy:

- retain official routed-expert FP4 representation/scale semantics as proven by Gate 0/reference code;
- retain official resident FP8 semantics where present;
- preserve small sensitive tensors at their official storage/semantic precision unless the reference requires a normalized representation;
- do not route through WASTE VQ3R/VQ4P merely because the upstream converter supports them;
- do not prune experts;
- do not substitute approximate experts;
- do not change router/attention precision for footprint.

Lossy transformations belong in `EXPERIMENTS.md` after end-to-end native parity exists.

## 10. Conversion phases

A robust converter should process in recoverable units.

### Phase 1 — normalized metadata

- parse config/index;
- validate Gate 0 assumptions;
- normalize model-family config;
- write temporary manifest/provenance data marked incomplete.

### Phase 2 — resident trunk

For each resident tensor:

1. locate the source shard/range;
2. read only the tensor required;
3. validate dtype/shape against inventory;
4. transform only as specified by the native-preserving format;
5. write aligned/output data;
6. record exact descriptor;
7. release source tensor memory before reading large unrelated tensors.

The converter should not require materializing the entire checkpoint in RAM.

### Phase 3 — routed experts by layer

Per layer:

1. resolve every expert and scale tensor;
2. validate completeness;
3. assemble one record per expert;
4. pad to declared alignment;
5. compute structural/checksum metadata;
6. write to a temporary bank;
7. verify the bank;
8. atomically rename it;
9. record layer completion.

A completed verified bank should be skipped on resume if its provenance/format inputs still match.

### Phase 4 — tokenizer/encoding/generation assets

Package/copy only what is allowed and necessary for a reproducible local run. Record source hashes/revision.

### Phase 5 — optional DSpark

Do not make DSpark conversion a prerequisite for a valid base container. Base conversion should complete and verify first.

### Phase 6 — final manifest

Write the valid/runnable final manifest atomically only after all mandatory base components pass verification.

## 11. Memory discipline during conversion

Inference is dependency-light; conversion may use Python/PyTorch/safetensors where appropriate.

Regardless of implementation language:

- never hold more expert layers/tensors than necessary;
- cap worker parallelism based on host RAM, not CPU count alone;
- expose `--jobs`/workspace estimates;
- prefer sequential shard access where practical;
- do not let multiple workers independently read the same huge source tensor;
- emit peak RSS and throughput during development runs.

A faster converter that OOMs midway through a multi-hour run is not an improvement.

## 12. Atomicity/recovery contract

Every resumable unit should follow:

```text
validate source -> write .tmp -> flush -> verify -> atomic rename -> mark complete
```

On startup:

- stale `.tmp` files are ignored/deleted only when safe;
- completed outputs are validated against current source/converter provenance;
- a changed format/revision invalidates incompatible partial output rather than silently reusing it;
- final runtime manifests never point at incomplete banks.

## 13. Verification levels

### Structural verification

- manifest schema/bounds;
- every declared file exists;
- exact file sizes;
- expert record headers and identities;
- offsets/alignment;
- optional checksums.

### Tensor round-trip verification

For selected tensors/experts:

```text
official checkpoint -> official reference operation
converted bytes      -> scalar C/reference operation
```

Compare under `VALIDATION.md`.

### Model-subset verification

Support conversion of selected layers/experts when possible so one real block can be validated before a complete model conversion.

A subset mode is a major gate: it can expose wrong matrix order, scale interpretation, transpose, or container layout without writing the full model.

### Full verification

After full conversion:

- all record checks pass;
- loader inventory matches converter manifest;
- final-logit and greedy-generation gates pass on real weights;
- cache-on/off parity passes.

Only then may the source checkpoint be considered disposable for this particular conversion environment.

## 14. Reclaim/delete policy

Do not implement destructive source-shard reclamation until:

- converter output is independently verified;
- restart/resume without source data is understood;
- a dry-run names exactly which shards become dead after each step;
- the user must opt in explicitly.

Default: keep the official checkpoint.

## 15. Converter CLI target

A future CLI might look like:

```bash
python3 tools/convert_deepseek_v4.py \
  --src reference/deepseek-v4-flash-0731 \
  --out ~/models/deepseek-v4-flash-0731.waste \
  --dry-run

python3 tools/convert_deepseek_v4.py \
  --src reference/deepseek-v4-flash-0731 \
  --out ~/models/deepseek-v4-flash-0731.waste \
  --jobs 2 \
  --verify
```

Do not treat these exact flags as frozen API; the requirements are the dry run, explicit source/output, controlled parallelism, resumability, and verification.

## 16. Conversion completion checklist

- [ ] official source revision pinned;
- [ ] official license/attribution resolved before source adaptation;
- [ ] Gate 0 strict inventory passes;
- [ ] tensor map updated;
- [ ] dry-run byte accounting complete;
- [ ] native FP4/FP8 scalar parity tests pass;
- [ ] format family/version cannot alias Kimi v0;
- [ ] subset conversion works;
- [ ] expert records round-trip against oracle;
- [ ] resume after interruption tested;
- [ ] final manifest is atomic;
- [ ] offline verifier passes whole output;
- [ ] real loader opens the container under a computed RAM budget;
- [ ] final-logit parity passes before any lossy post-conversion experiment.

The first successful full conversion is a correctness milestone, not automatically a performance result.