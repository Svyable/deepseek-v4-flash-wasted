# DeepSeek V4 Flash → WASTE: Link Pack

> Practical source pack for implementing, validating, serving, and eventually packaging **DeepSeek-V4-Flash-0731** on top of WASTE-style NVMe expert streaming.
>
> **Source-of-truth rule:** use the official DeepSeek checkpoint, config, inference code, encoder, tokenizer assets, and test cases as the correctness oracle. **Do not use a GGUF as the conversion source.** GGUF/vLLM/SGLang implementations are useful only as secondary behavioral references.
>
> Last curated: **2026-08-08**.

This file complements the implementation plan in [`README.md`](README.md). The README describes **what to build and in what order**; this file is the fast index for **where to look while building it**.

---

## 1. Source-of-truth hierarchy

When sources disagree, use this order:

1. **`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint contents** — tensor names, tensor shapes, dtypes, weight-index files, tokenizer assets, generation config.
2. **The 0731 `inference/` implementation** — exact forward semantics, quantization decoding, routing, attention, Hyper-Connections, DSpark integration.
3. **The 0731 `encoding/` implementation + tests** — exact message encoding and output parsing semantics.
4. **Official DeepSeek model card / technical report.**
5. **WASTE source and measured design notes** — storage, caching, memory planning, direct I/O, API separation, test methodology.
6. **NVIDIA NeMo DeepSeek-V4 implementation/docs** — valuable independent architecture/parity reference.
7. **vLLM / SGLang / llama.cpp / GGUF ports** — behavioral smoke tests and implementation ideas only.

Never change the model to make a third-party runtime agree. Differential tests should ultimately agree with the official DeepSeek reference.

For reproducible experiments, record the resolved Hugging Face revision and WASTE commit in every generated manifest/log.

---

## 2. Primary repositories

### Target repository

- **This project:** https://github.com/Svyable/deepseek-v4-flash-wasted
- **Implementation plan:** https://github.com/Svyable/deepseek-v4-flash-wasted/blob/main/README.md

### WASTE upstream

- **WASTE:** https://github.com/sqliteai/waste
- **Pinned WASTE baseline used by our README:** https://github.com/sqliteai/waste/commit/d9b919a791148b571e643d0af666bf19b4d733ab
- **Issues:** https://github.com/sqliteai/waste/issues
- **Forks:** https://github.com/sqliteai/waste/forks
- **Main-branch commits:** https://github.com/sqliteai/waste/commits/main

WASTE is the systems baseline: embeddable C inference, a hard RAM ceiling, aligned expert records, direct reads, a bounded expert cache, model conversion tooling, synthetic correctness tests, CLI, and a Python OpenAI-compatible server.

### Official DeepSeek checkpoints

- **DeepSeek-V4-Flash-0731 — primary target:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- **0731 file browser:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main
- **Original DeepSeek-V4-Flash:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- **Original Flash file browser:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main
- **DeepSeek-V4-Pro:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

Use **0731** for this project. The original Flash release is still useful for architecture history and cross-checking, while V4-Pro is an architecture-family reference rather than the sensible first WASTE target.

---

## 3. WASTE architecture: read these before editing the engine

### Design and project map

- **Top-level README:** https://github.com/sqliteai/waste/blob/main/README.md
- **Contributor/repository map:** https://github.com/sqliteai/waste/blob/main/CLAUDE.md
- **Engine internals:** https://github.com/sqliteai/waste/blob/main/docs/ENGINE.md
- **Container format:** https://github.com/sqliteai/waste/blob/main/docs/FORMAT.md
- **Correctness / feasibility gates:** https://github.com/sqliteai/waste/blob/main/docs/GATES.md
- **Performance / efficiency evidence:** https://github.com/sqliteai/waste/blob/main/docs/EFFICIENCY.md
- **Failed experiments / negative results:** https://github.com/sqliteai/waste/blob/main/docs/LEARNED.md
- **Current research directions:** https://github.com/sqliteai/waste/blob/main/docs/RESEARCH.md
- **K3-specific implementation notes:** https://github.com/sqliteai/waste/blob/main/docs/K3.md
- **Backend notes:** https://github.com/sqliteai/waste/blob/main/docs/BACKENDS.md

Read `LEARNED.md` before proposing an optimization. WASTE deliberately keeps failed ideas and negative measurements so they are not repeatedly rediscovered.

### Core C implementation

- **Public C API:** https://github.com/sqliteai/waste/blob/main/src/waste.h
- **High-level API implementation:** https://github.com/sqliteai/waste/blob/main/src/waste.c
- **Current model/forward path:** https://github.com/sqliteai/waste/blob/main/src/model.c
- **Model structures:** https://github.com/sqliteai/waste/blob/main/src/model.h
- **On-disk record structs:** https://github.com/sqliteai/waste/blob/main/src/waste_format.h
- **Expert cache:** https://github.com/sqliteai/waste/blob/main/src/ecache.c
- **Expert cache API/structures:** https://github.com/sqliteai/waste/blob/main/src/ecache.h
- **Memory planning:** https://github.com/sqliteai/waste/blob/main/src/memory.c
- **Platform abstraction / direct I/O helpers:** https://github.com/sqliteai/waste/blob/main/src/platform.h
- **Backend selection:** https://github.com/sqliteai/waste/blob/main/src/backend.c
- **SIMD helpers:** https://github.com/sqliteai/waste/blob/main/src/simd.h
- **All source files:** https://github.com/sqliteai/waste/tree/main/src

**Correction to earlier notes:** current WASTE does **not** have a `src/container.c`. Container/manifest loading is part of the model-loading path in `src/model.c`, while the frozen binary record layout is in `src/waste_format.h` and documented in `docs/FORMAT.md`.

### CLI, converter, and tests

- **CLI:** https://github.com/sqliteai/waste/tree/main/cli
- **Conversion tools:** https://github.com/sqliteai/waste/tree/main/tools
- **Current Kimi-family converter:** https://github.com/sqliteai/waste/blob/main/tools/convert.py
- **Memory-plan utility:** https://github.com/sqliteai/waste/blob/main/tools/memplan.py
- **Routing analysis:** https://github.com/sqliteai/waste/blob/main/tools/routing_stats.py
- **HF tracing utility:** https://github.com/sqliteai/waste/blob/main/tools/trace_hf.py
- **Tests:** https://github.com/sqliteai/waste/tree/main/tests
- **Main test runner:** https://github.com/sqliteai/waste/blob/main/tests/run.sh
- **Forward-pass tests:** https://github.com/sqliteai/waste/blob/main/tests/test_forward.c

### What we want to reuse from WASTE

The most important reusable concept is **placement**, not Kimi arithmetic:

- keep the always-used trunk resident;
- store routed MoE experts separately;
- make each independently requested expert record aligned and self-contained;
- read selected experts with explicit positional/direct I/O;
- use remaining RAM as a bounded cache under a hard process budget;
- overlap I/O with computation when correctness is unaffected;
- instrument cache hits, misses, bytes read, and timing;
- make the CLI and server clients of the same public C API;
- validate against a Python/original-model oracle layer by layer.

The DeepSeek port should reuse those systems pieces while implementing a new DeepSeek-V4 trunk rather than forcing the existing Kimi-specific KDA/MLA path to fit.

---

## 4. Existing WASTE OpenAI-compatible server

Do **not** build a second serving stack until this one has been evaluated for extension.

- **Serving design:** https://github.com/sqliteai/waste/blob/main/docs/SERVE.md
- **Server package:** https://github.com/sqliteai/waste/tree/main/serve
- **ctypes bridge to `libwaste`:** https://github.com/sqliteai/waste/blob/main/serve/engine.py
- **API request/response handling:** https://github.com/sqliteai/waste/blob/main/serve/api.py
- **HTTP server:** https://github.com/sqliteai/waste/blob/main/serve/server.py
- **Current K3 prompt renderer:** https://github.com/sqliteai/waste/blob/main/serve/xtml.py
- **Incremental response/tool-call parser:** https://github.com/sqliteai/waste/blob/main/serve/regions.py
- **Server tests:** https://github.com/sqliteai/waste/tree/main/tests/serve

WASTE already exposes an OpenAI-style local API using Python standard-library HTTP code plus `ctypes` calls into `libwaste`. The model arithmetic remains in C; Python owns protocol validation, prompt rendering, response parsing, and SSE framing.

That separation maps well to DeepSeek V4 because DeepSeek supplies **code-based conversation encoding/parsing rather than only a Jinja template**. The right approach is to add a DeepSeek encoder/parser adapter and differential tests, not to shoehorn V4 messages into K3 XTML.

---

## 5. DeepSeek-V4-Flash-0731: official source material

Start with the **0731 repository file browser** and only fall back to the original Flash repository when the file or explanation is genuinely shared.

### 0731 checkpoint and configuration

- **Model card:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- **All files:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main
- **Main config:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json
- **Generation config:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/generation_config.json
- **Safetensors index:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/model.safetensors.index.json
- **License:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE

The 0731 repository currently contains 48 safetensors shards plus the inference and encoding reference material. Treat the index and shard headers—not model-card parameter counts—as authoritative when inventorying bytes and tensor shapes.

### Official inference implementation

- **Inference directory:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference
- **Reference model:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py
- **Reference kernels:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/kernel.py
- **Reference converter:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/convert.py
- **Inference config:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/config.json

These are the most important implementation references after the weight index. When names in `config.json`, the model card, and inference code appear to differ, trace the actual checkpoint keys through the supplied converter/reference code before deciding what the WASTE manifest should contain.

### Official encoding / parsing

- **Encoding directory:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding
- **Encoding README:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/README.md

The release provides Python encoding/parsing code and test cases because the conversation format is not simply represented by a Jinja chat template. Port that behavior and differential-test it. Do not reverse-engineer prompts from example strings.

### Original Flash references

- **Original Flash model:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- **Original Flash files:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main
- **Original Flash encoding:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding
- **Original Flash inference:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
- **Original Flash config:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
- **Original Flash generation config:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/generation_config.json

Use these for comparison/history, not as a substitute for checking the 0731 files.

---

## 6. Architecture references

### Primary

- **DeepSeek-V4-Flash-0731 model card:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- **Official inference config:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/config.json
- **Official reference model:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py
- **DeepSeek V4 technical report:** https://arxiv.org/abs/2606.19348

### Independent architecture cross-checks

- **NVIDIA NeMo model coverage:** https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/deep-seek-v-4-flash
- **NVIDIA NeMo end-to-end DeepSeek V4 Flash recipe:** https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash
- **NVIDIA DeepSeek V4 article:** https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/
- **DeepSeek-V4-Pro model card:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

The invariant we should continuously re-check against the official config/reference is that the Flash backbone is a **43-layer all-MoE model** with **256 routed experts + one shared expert per block**, **top-6 routing**, `hc_mult=4` Hyper-Connections, and a hybrid SWA/CSA/HCA attention schedule. The first configured hash-routing layers use token-ID-driven expert selection rather than the normal score-based gate.

Do not translate those descriptions directly into C from a summary page. Use them to orient yourself, then implement from the official `inference/model.py`, `kernel.py`, and checkpoint shapes.

---

## 7. Quantization and checkpoint-format references

The first converter should preserve DeepSeek's released precision semantics. Requantization is a later experiment.

### DeepSeek reference

- **0731 inference model:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py
- **0731 kernels:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/kernel.py
- **0731 converter:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/convert.py
- **NVIDIA checkpoint-format notes:** https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash

The released model uses native low-precision formats, notably **packed FP4 routed experts with UE8M0-style scaling**, with predominantly FP8 treatment for non-expert/shared weights. Preserve those semantics in the first trustworthy converter.

### Safetensors / PyTorch

- **Safetensors repository:** https://github.com/huggingface/safetensors
- **Safetensors documentation:** https://huggingface.co/docs/safetensors/index
- **PyTorch serialization notes:** https://pytorch.org/docs/stable/notes/serialization.html

### WASTE format / converter references

- **WASTE format rationale:** https://github.com/sqliteai/waste/blob/main/docs/FORMAT.md
- **WASTE binary record structs:** https://github.com/sqliteai/waste/blob/main/src/waste_format.h
- **Kimi converter pattern:** https://github.com/sqliteai/waste/blob/main/tools/convert.py

A DeepSeek `.waste` container does **not** need to use WASTE's current VQ3R expert payload format. The reusable invariant is one aligned, independently readable expert record containing everything needed to execute that expert. A native-FP4 DeepSeek record format should be introduced and validated first.

---

## 8. Download and conversion

- **Hugging Face CLI:** https://huggingface.co/docs/huggingface_hub/guides/cli
- **`snapshot_download`:** https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download
- **Hugging Face download guide:** https://huggingface.co/docs/huggingface_hub/guides/download
- **Safetensors:** https://github.com/huggingface/safetensors

Recommended rules for automation:

1. Resolve and record an exact HF revision before conversion.
2. Download `config.json`, `generation_config.json`, tokenizer/encoding assets, the safetensors index, and shard metadata before downloading all weight bytes.
3. Run `tools/inventory.py` against headers/index first.
4. Support resumable conversion by layer/bank.
5. Never require all experts in RAM at once during conversion.
6. Write to a temporary path and atomically publish completed records/manifests.
7. Keep checksums/shape metadata sufficient to detect wrong-record reads.
8. Keep the source checkpoint until full differential validation passes.

---

## 9. Validation baselines

### Gold-standard baseline

- **Official inference code:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference
- **Official encoding/tests:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding
- **Official checkpoint:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

The validation harness should compare progressively:

- tokenizer / message encoding;
- FP4 and FP8 scalar decode kernels;
- individual linear operations;
- hash routing;
- learned MoE routing and weights;
- Hyper-Connection pre/post mixing;
- each attention variant;
- each transformer block;
- final hidden state;
- final logits;
- deterministic generation token sequence;
- response parsing / tool-call extraction;
- server-level OpenAI request/response behavior.

### Secondary behavioral baselines

Use these to smoke-test behavior or inspect how other projects solved compatibility problems. **Do not convert their quantized weights into WASTE.**

- **Unsloth DeepSeek-V4-Flash GGUF:** https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **vLLM:** https://github.com/vllm-project/vllm
- **SGLang:** https://github.com/sgl-project/sglang

GGUF may bake in a different quantizer, tensor transformation, prompt assumption, or runtime-specific workaround. It cannot establish converter fidelity.

---

## 10. Suggested synthetic-test strategy

Before converting the full checkpoint, follow WASTE's gate-first philosophy.

Useful WASTE references:

- **Gates:** https://github.com/sqliteai/waste/blob/main/docs/GATES.md
- **Synthetic tests:** https://github.com/sqliteai/waste/tree/main/tests
- **Container tests:** https://github.com/sqliteai/waste/blob/main/tests/test_container.c
- **Forward tests:** https://github.com/sqliteai/waste/blob/main/tests/test_forward.c

Build a tiny DeepSeek-V4-shaped fixture that preserves the **algorithmic structure** while shrinking dimensions:

- multiple HC streams;
- at least one hash-routed MoE layer;
- at least one learned-router MoE layer;
- one shared expert;
- top-k > 1;
- at least one SWA layer;
- one compressed-attention/indexer layer if practical;
- native FP4 expert record decode;
- native FP8 trunk path;
- a tiny tokenizer/encoding fixture;
- bounded cache with forced hit/miss cases;
- direct-I/O fallback tests.

The synthetic fixture should run in CI without downloading the real model.

---

## 11. Local API and UI

### Recommended topology

```text
Streamlit / desktop / CLI client
          |
          v
localhost OpenAI-compatible WASTE server
          |
          v
       libwaste
          |
          v
DeepSeek .waste container on fast NVMe
```

Keep the model/server process separate from a UI framework's script-rerun lifecycle. The UI should be replaceable without reopening a 100+ GB model.

### WASTE server

- **Server design:** https://github.com/sqliteai/waste/blob/main/docs/SERVE.md
- **Server package:** https://github.com/sqliteai/waste/tree/main/serve
- **ctypes bridge:** https://github.com/sqliteai/waste/blob/main/serve/engine.py
- **HTTP server:** https://github.com/sqliteai/waste/blob/main/serve/server.py
- **API layer:** https://github.com/sqliteai/waste/blob/main/serve/api.py

### Client/API references

- **OpenAI Python SDK:** https://github.com/openai/openai-python
- **OpenAI API reference:** https://platform.openai.com/docs/api-reference

### Streamlit option

- **Streamlit chat elements:** https://docs.streamlit.io/develop/api-reference/chat
- **Streamlit deployment concepts:** https://docs.streamlit.io/deploy
- **vLLM Streamlit integration example:** https://docs.vllm.ai/en/stable/deployment/frameworks/streamlit/

Streamlit is optional. The architectural requirement is that any UI talks to the local API rather than owning model lifetime itself.

---

## 12. macOS, Linux, and Windows systems references

### WASTE portability

- **WASTE README / platform notes:** https://github.com/sqliteai/waste/blob/main/README.md
- **Platform abstraction:** https://github.com/sqliteai/waste/blob/main/src/platform.h
- **Backend code:** https://github.com/sqliteai/waste/blob/main/src/backend.c
- **All source/backends:** https://github.com/sqliteai/waste/tree/main/src

### Linux

- **`open(2)` / `O_DIRECT`:** https://man7.org/linux/man-pages/man2/open.2.html
- **`pread(2)`:** https://man7.org/linux/man-pages/man2/pread.2.html

### macOS

- **Apple Accelerate:** https://developer.apple.com/documentation/accelerate
- **Apple Metal:** https://developer.apple.com/metal/
- **`fcntl` reference / `F_NOCACHE`:** https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html

### Windows

- **File buffering / unbuffered I/O:** https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering
- **`ReadFile`:** https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile
- **`_aligned_malloc`:** https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc
- **MinGW-w64:** https://www.mingw-w64.org/

Do not assume the platform's nominal sequential SSD benchmark predicts WASTE performance. Benchmark the actual pattern: aligned random-ish expert-sized positional reads at the concurrency the engine will issue, while bypassing or controlling the OS page cache as intended.

---

## 13. Performance and I/O measurement references

WASTE's experience is especially relevant here because performance intuition repeatedly failed until it was measured end-to-end.

- **WASTE efficiency measurements:** https://github.com/sqliteai/waste/blob/main/docs/EFFICIENCY.md
- **WASTE learned/negative results:** https://github.com/sqliteai/waste/blob/main/docs/LEARNED.md
- **WASTE feasibility gates:** https://github.com/sqliteai/waste/blob/main/docs/GATES.md
- **Expert cache:** https://github.com/sqliteai/waste/blob/main/src/ecache.c
- **Memory planning:** https://github.com/sqliteai/waste/blob/main/src/memory.c

For DeepSeek V4, measure at least:

- bytes per expert record;
- cold bytes requested per decode token;
- bytes actually read after cache hits;
- cache hit rate by layer and globally;
- next-token expert reuse;
- routing concentration / hot experts;
- I/O queue depth and achieved GB/s;
- expert decode/apply compute time;
- trunk compute time;
- attention time by attention type;
- Hyper-Connection time;
- prefill vs decode separately;
- memory floor vs cache budget;
- page-fault / swap behavior;
- direct-I/O vs buffered-I/O behavior;
- end-to-end tokens/sec.

Never optimize solely for cache hit rate: WASTE already demonstrated that a larger cache can increase hit rate while making the process dramatically slower if it pushes the machine into paging.

---

## 14. Agent reading order

Give a coding agent this sequence rather than a giant undirected context dump.

### Phase A — understand WASTE's invariants

1. https://github.com/sqliteai/waste/blob/main/CLAUDE.md
2. https://github.com/sqliteai/waste/blob/main/README.md
3. https://github.com/sqliteai/waste/blob/main/docs/FORMAT.md
4. https://github.com/sqliteai/waste/blob/main/docs/ENGINE.md
5. https://github.com/sqliteai/waste/blob/main/docs/GATES.md
6. https://github.com/sqliteai/waste/blob/main/docs/LEARNED.md
7. https://github.com/sqliteai/waste/blob/main/src/waste.h
8. https://github.com/sqliteai/waste/blob/main/src/model.c
9. https://github.com/sqliteai/waste/blob/main/src/ecache.c
10. https://github.com/sqliteai/waste/blob/main/tools/convert.py

### Phase B — understand the exact DeepSeek release

1. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json
2. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/model.safetensors.index.json
3. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference
4. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding
5. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/generation_config.json
6. tokenizer assets and all sidecar files in the 0731 repo
7. the technical report: https://arxiv.org/abs/2606.19348

### Phase C — build cheap kill-tests first

1. checkpoint inventory script;
2. scalar FP4 decode test against Python;
3. scalar FP8 test against Python;
4. tiny synthetic DeepSeek container;
5. hash-router parity;
6. learned-router parity;
7. Hyper-Connection parity;
8. attention parity;
9. one-block parity;
10. multi-block logits parity.

### Phase D — only then scale

1. resumable real-weight converter;
2. exact RAM/storage planner;
3. one/few real layers;
4. full base model conversion;
5. deterministic generation parity;
6. cache/read-ahead profiling;
7. CLI integration;
8. OpenAI server integration;
9. DeepSeek encoding/tool-call parity;
10. optional UI;
11. DSpark speculative decoding as a separate validated acceleration layer.

This sequence intentionally follows WASTE's synthetic-container + feasibility-gate culture. A 167 GB source checkpoint is too large to use as the inner development loop for basic parser, format, or arithmetic bugs.

---

## 15. Secondary implementations and community references

These can help answer questions like "how did another runtime map this tensor?" or "does this prompt produce the same kind of output?" They are not authoritative.

### Runtimes

- **vLLM:** https://github.com/vllm-project/vllm
- **SGLang:** https://github.com/sgl-project/sglang
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **NVIDIA NeMo AutoModel:** https://github.com/NVIDIA-NeMo/Automodel

### GGUF / converted checkpoints

- **Unsloth V4 Flash GGUF:** https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF
- **persadian V4 Flash GGUF:** https://huggingface.co/persadian/DeepSeek-V4-Flash-GGUF

### DSpark-related secondary checkpoint

- **0xSero/deepseek-v4-flash-0731-spark:** https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark

For DSpark implementation, prefer the module included in the official 0731 release and its official inference path. Use third-party extracted/convenience checkpoints only for inspection or cross-checking.

---

## 16. Fast decision table

| Question | First place to look |
|---|---|
| What tensor exists / what is its exact shape? | 0731 safetensors index + shard header |
| How should this tensor be interpreted? | 0731 `inference/model.py` / `kernel.py` |
| How should an OpenAI message become model input? | 0731 `encoding/` |
| How should model text become content/reasoning/tool calls? | 0731 `encoding/` parser/tests |
| How should experts be placed on disk? | WASTE `docs/FORMAT.md` + `waste_format.h` |
| How should RAM be budgeted? | WASTE `docs/ENGINE.md` + `src/memory.c` |
| How should expert caching work? | WASTE `src/ecache.c` + `docs/EFFICIENCY.md` |
| How should the C API be shaped? | WASTE `src/waste.h` |
| How should the CLI access capabilities? | WASTE `cli/` — through public API only |
| How should the local HTTP API work? | WASTE `docs/SERVE.md` + `serve/` |
| Is an optimization actually useful? | Measure it; read WASTE `LEARNED.md` first |
| Can a GGUF prove converter correctness? | **No** |
| Should DSpark block base-model bring-up? | **No** — base logits/generation first |

---

## 17. Non-negotiable porting rules

1. **Official 0731 checkpoint is the conversion source.**
2. **Official DeepSeek reference code is the arithmetic oracle.**
3. **Official encoding tests are the conversation-format oracle.**
4. **Preserve native FP4/FP8 semantics before experimenting with new quantization.**
5. **Reuse WASTE systems infrastructure; do not assume the Kimi trunk is reusable.**
6. **One expert request should map to one aligned disk record/read whenever practical.**
7. **Cache size stays inside a hard RAM budget; no accidental OS-cache dependency.**
8. **Synthetic CI tests come before full-model conversion.**
9. **Layer/intermediate/logit parity comes before speed work.**
10. **Base 43-layer generation comes before DSpark.**
11. **The OpenAI server should remain a client of the public native API.**
12. **Every benchmark records model revision, code commit, machine, storage device, budget, context, and cache settings.**

---

## 18. Core source list

The small set worth bookmarking immediately:

- https://github.com/Svyable/deepseek-v4-flash-wasted/blob/main/README.md
- https://github.com/sqliteai/waste
- https://github.com/sqliteai/waste/blob/main/docs/FORMAT.md
- https://github.com/sqliteai/waste/blob/main/docs/ENGINE.md
- https://github.com/sqliteai/waste/blob/main/docs/GATES.md
- https://github.com/sqliteai/waste/blob/main/docs/LEARNED.md
- https://github.com/sqliteai/waste/blob/main/src/waste.h
- https://github.com/sqliteai/waste/blob/main/src/model.c
- https://github.com/sqliteai/waste/blob/main/src/ecache.c
- https://github.com/sqliteai/waste/blob/main/tools/convert.py
- https://github.com/sqliteai/waste/blob/main/docs/SERVE.md
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/model.safetensors.index.json
- https://arxiv.org/abs/2606.19348
- https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash

If an implementation choice cannot be justified from those primary sources or from a measurement, treat it as a hypothesis and write a cheap test before committing the project to it.
