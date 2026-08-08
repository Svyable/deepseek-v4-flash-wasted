# reference/ — local official-artifact staging area

This directory documents how to stage official DeepSeek reference material locally without accidentally committing hundreds of gigabytes of checkpoint data or losing provenance.

The target is:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
```

Use `docs/REFERENCE_ACCESS.md` for the acquisition sequence and `docs/FIXTURES.md` for turning official outputs into small committed test fixtures.

The staged artifacts primarily feed **README Gate A / V0**, **Gate B / V1**, and **Gate C / V2**. `docs/VALIDATION.md` §4a is the canonical gate concordance.

## Intended local layout

```text
reference/
  README.md
  .gitignore
  deepseek-v4-flash-0731/       # local snapshot; ignored
    config.json
    generation_config.json
    model.safetensors.index.json
    inference/
    encoding/
    tokenizer...
    model-*.safetensors         # eventually; ignored
  provenance/                   # optional generated local reports
```

The local snapshot directory is intentionally ignored. Official source/checkpoint content should only be committed when a specific file is deliberately vendored under its license/provenance requirements.

## Do not commit raw model weights

Do not add:

- full `.safetensors` shards;
- converted `.waste` model directories;
- local download caches;
- temporary extracted tensor blobs;
- authentication tokens or Hugging Face cache metadata containing credentials.

Small oracle fixtures belong under `tests/fixtures/` (or another explicitly reviewed fixture path), not as anonymous slices of a local checkpoint.

## Pin the revision

Never stage reference material from an unrecorded moving branch and then generate golden fixtures from it.

Record at minimum:

```text
repository: deepseek-ai/DeepSeek-V4-Flash-0731
resolved revision: <full commit SHA>
retrieval date: <UTC>
retrieval command/tool version
```

For every artifact used to generate committed fixtures, record its path and SHA-256.

## Licensing

PR #1 intentionally left `LICENSES/DEEPSEEK-MIT.txt.MISSING` because the official repository was unreachable and legal text must not be reconstructed from memory.

Before copying/adapting official DeepSeek source into this repository:

1. retrieve the exact official license/notice material;
2. preserve it under `LICENSES/` as appropriate;
3. record the pinned source revision;
4. preserve required headers/notices in adapted files.

A local uncommitted snapshot may be used for inspection in an authorized environment, but committed adaptations must have resolved provenance/licensing.

## Expected first use

Once official access is available:

1. retrieve Tier 0/1 material from `docs/REFERENCE_ACCESS.md`;
2. run `tools/inventory.py reference/deepseek-v4-flash-0731` in index-only mode — **Gate A / V0**;
3. answer the FP4 nibble-order and scale-direction questions from official inference code — **Gate B / V1**;
4. generate small independent oracle fixtures and close Gate B/V1;
5. run one official quantized projection — **Gate C / V2**;
6. only then consider full checkpoint acquisition.

This directory is staging, not evidence by itself. Evidence becomes durable when its revision, hashes, generator, canonical gate, and expected outputs are recorded in the relevant docs/fixtures.