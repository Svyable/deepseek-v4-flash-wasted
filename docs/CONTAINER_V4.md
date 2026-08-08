# DeepSeek V4 container contract

**Status: DESIGN. No DeepSeek container has been written. Exact record fields remain `TBD-GATE0` until the official 0731 checkpoint/reference is available.**

This document defines the invariants a DeepSeek-specific WASTE container must satisfy. It intentionally does **not** redefine imported WASTE `docs/FORMAT.md`: that file documents Kimi format v0 and remains historical/upstream evidence.

## 1. Primary rule: never confuse model families

A DeepSeek container must be rejected by a Kimi-only reader and a Kimi container must be rejected by a DeepSeek-only path before tensor offsets are interpreted.

The manifest therefore needs explicit fields equivalent to:

```json
{
  "container_family": "waste",
  "format_version": "TBD-DISTINCT-FROM-KIMI-V0",
  "model_family": "deepseek_v4_flash",
  "model_revision": "9e165c3",
  "converter_revision": "<git sha>"
}
```

The exact format-version encoding is chosen in the implementation PR, but it must not alias the imported v0 semantics.

## 2. Design invariants inherited from WASTE

These are intended to remain true:

1. **One independently readable record per routed expert.** All bytes needed to apply that expert are co-located.
2. **Direct-I/O friendly.** Independently read records are aligned to the platform's required boundary; the baseline target is 4 KiB as in WASTE, subject to platform verification.
3. **Placement changes speed, never precision.** Cached and freshly read copies contain identical payload bytes.
4. **Hard bounds.** Manifest dimensions, offsets, lengths and record counts are validated before arithmetic or allocation.
5. **Resumable conversion.** Per-layer expert banks can be written/verified independently.
6. **Self-describing quantization.** Readers consume quantization descriptors from the manifest/record metadata rather than inferring formats from model names.
7. **Provenance.** Source model and converter revisions are embedded in the container.

## 3. Proposed directory layout

Exact names can change before the first format freeze:

```text
model.waste/
  manifest.json
  trunk.bin
  experts-L000.bin
  experts-L001.bin
  ...
  experts-L042.bin             # only if Gate 0 confirms 43 main MoE layers
  tokenizer/ or tokenizer assets
  encoding/ or normalized encoder assets
  generation.json
  provenance.json              # optional if not fully represented in manifest
  dspark/                      # optional phase-2 component
    manifest.json
    trunk.bin / banks as required by Gate 0
  usage.waste                  # optional runtime learned-hotlist/statistics
```

Do not create empty files merely to match this sketch. Gate 0 decides which components exist.

## 4. Manifest responsibilities

The manifest is the loader's untrusted input. It should contain enough information to validate the container without consulting Hugging Face.

Required categories:

### Identity/provenance

- container format version;
- model family/architecture identifier;
- source repository;
- resolved source revision;
- original/normalized config hash;
- converter repository revision;
- conversion timestamp/tool version if useful for diagnostics;
- whether optional DSpark assets are present.

### Model dimensions

Only checkpoint/reference-verified dimensions belong here. The converter reads them; the runtime does not hard-code the README table.

### Resident tensor index

For every resident tensor:

```text
canonical name
source name or source mapping id
format/dtype descriptor
offset
stored bytes
logical shape
stored shape
quantization scale location/granularity where applicable
alignment requirements
```

### Routed-expert bank index

For every main MoE layer:

```text
bank filename
expert count
record stride / offset table
record alignment
record format id
payload quantization descriptor
matrix shapes
scale shapes/granularity
optional checksum policy
```

Uniform stride is preferable for O(1) addressing but is not mandatory if the real checkpoint makes expert records nonuniform. Correctness and storage truth win over an assumed stride.

### Encoding/tokenizer

The container or adjacent install must identify the exact tokenizer/encoding assets needed to reproduce official token sequences. A model directory should not depend on an ambiguous latest remote template.

## 5. Resident trunk

The first correct converter preserves the official non-expert quantization semantics rather than dequantizing everything to f32 by default or requantizing it to WASTE's Kimi formats.

`trunk.bin` may contain a mix of:

- packed/quantized resident weights;
- scale tensors;
- small f16/f32 parameters;
- router/hash/indexer/compressor data;
- embeddings/head data according to the final memory/I/O design.

The exact formats and tensor families are populated from `TENSOR_MAP.md` after Gate 0.

### Loader rule

A tensor descriptor names its format. Code should switch on the descriptor, not on a substring of the tensor name.

## 6. Routed expert record

The payload is frozen only after Gate 0 proves the complete per-expert tensor set.

Conceptually:

```text
+------------------------------+ aligned record start
| record header                |
| matrix A packed payload      |
| matrix A scale payload       |
| matrix B packed payload      |
| matrix B scale payload       |
| matrix C packed payload      |
| matrix C scale payload       |
| optional verified extras     |
| padding                      |
+------------------------------+ aligned next record
```

Likely matrices correspond to normalized `w1`, `w3`, `w2`, but **matrix order and names are TBD-GATE0**.

The record must be sufficient for one expert apply after one cache miss. If official arithmetic needs an additional per-expert vector/bias/metadata tensor, include it in the record rather than adding another random read per expert.

## 7. Record header requirements

A minimal header should allow the read path to reject a misaddressed/corrupt record before interpreting payload offsets.

Candidate fields:

```text
magic / record format
header version
layer id
expert id
record byte length
quantization format id
matrix offsets/lengths or a fixed format definition
scale offsets/lengths
checksum (optional-at-runtime, always available for offline verification if adopted)
```

Rules:

- multi-byte endianness is explicit;
- structs have compile-time size checks if serialized directly;
- every offset is bounds-checked against the already-read record length;
- the header's `(layer, expert)` must match the requested identity;
- a short read is fatal for the current evaluation call;
- unknown record format/version is refused.

## 8. Quantization descriptors

The first format must describe the official native layouts exactly.

For an FP4 expert descriptor, Gate 0/reference work must settle:

- encoded element format (expected E2M1 from current handoff);
- storage dtype/packing order;
- logical versus stored dimensions;
- values per byte;
- scale encoding (expected UE8M0 from current handoff);
- scale block size/axis (currently expected K32);
- scale tensor ordering/strides;
- accumulation/output dtype semantics required for parity.

For resident FP8 tensors, settle equivalent fields including block scale shape/granularity.

Do not encode “DeepSeek FP4” as a single opaque assumption if the real checkpoint uses more than one layout.

## 9. Alignment and direct I/O

WASTE's imported format uses 4 KiB aligned, 4 KiB-multiple expert records. Retain that default if it is legal and efficient on the target filesystems.

The converter must calculate:

```text
payload_bytes
header_bytes
alignment_padding
record_bytes
```

and record the actual value. `record_bytes` is a storage fact; the runtime does not reconstruct it from README estimates.

Platform-specific unbuffered I/O may impose alignment on:

- file offset;
- read size;
- userspace buffer address.

`PLATFORM.md` owns the portability rules.

## 10. Checksums and verification

Imported WASTE v0 demonstrates a useful split:

- structural header validation on every read;
- optional payload checksum verification during inference;
- full offline container verification tool.

DeepSeek should preserve that shape unless measurement disproves it.

A converter should be able to verify each finished record immediately and resume without rewriting already validated banks.

The checksum algorithm/coverage is not frozen here. Reusing WASTE's CRC32 implementation is attractive because it already has accelerated/fallback paths, but the new format should document coverage explicitly.

## 11. Atomicity and resumability

A full model conversion is too expensive to make all-or-nothing.

For each output component:

1. write to a temporary filename;
2. flush/fsync as required by the recovery contract;
3. verify structural metadata/checksum;
4. atomically rename into place;
5. update progress/manifest only after the component is valid.

A partially converted directory must never make the runtime believe a missing layer bank is valid.

The final manifest should be written atomically after all mandatory base components pass conversion verification, or should contain an explicit incomplete state that the runtime refuses.

## 12. Conversion provenance

At minimum record:

```text
source model id
resolved source revision
source config hash
source index hash
converter commit
container format version
quantization policy (native-preserving baseline)
per-file sizes/hashes as appropriate
```

This allows two containers to be compared without guessing whether they came from the same checkpoint or converter.

## 13. DSpark storage boundary

DSpark is optional phase 2. Its presence must not change the interpretation of base-model records.

Preferred property:

```text
base container valid and runnable by itself
+ optional dspark/ assets
```

If the official checkpoint shares tensors between base and DSpark, the DSpark manifest should reference base tensors rather than duplicating them where practical.

The runtime must support opening a container with DSpark assets while explicitly disabling DSpark for validation.

## 14. Self-contained tokenizer/encoding

WASTE's Kimi container copies tokenizer assets so inference does not depend on a remote model repository. DeepSeek should aim for the same property, subject to the official encoder's license and runtime requirements.

Do not convert the official code-based encoding into a lossy hand-written template simply to make it easy to package. `API.md`/`VALIDATION.md` define encoder parity as part of correctness.

## 15. Parser hardening

The manifest and record headers are untrusted inputs even when the model came from a trusted publisher because files can be truncated or corrupted.

Tests must include:

- negative/overflowing dimensions represented in JSON forms the parser accepts;
- multiplication overflow for element/byte counts;
- offset + length overflow;
- out-of-file trunk tensors;
- bank smaller than declared record count;
- wrong layer/expert identity;
- unknown format ids;
- invalid alignment;
- truncated header/record;
- corrupted checksum;
- duplicate/conflicting tensor entries;
- DSpark metadata claiming base tensors that do not exist.

Use the imported fuzzing pattern rather than assuming generated manifests are always well-formed.

## 16. Format freeze gate

Do **not** call the DeepSeek format stable until:

- Gate 0 has exact tensor/scale mappings;
- at least one real expert record round-trips against the official reference;
- a synthetic complete container passes loader/fuzz/platform tests;
- a real subset conversion can reopen and reproduce oracle outputs;
- the memory planner uses manifest-derived rather than estimated sizes;
- resuming an interrupted conversion has been tested;
- the model-family/version discriminator prevents cross-family misreads.

Until then, format changes are expected and backward compatibility is not promised.