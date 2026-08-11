# DeepSeek V4 positional runtime I/O

This note records the storage boundary implemented by `src/deepseek_v4_runtime.{c,h}` after PR #22. It is intentionally narrower than a DeepSeek container format: the pinned evidence has not frozen filenames, per-record headers, record ordering, or alignment, so the runtime does not invent them.

## Goal

Turn a validated DeepSeek routed-record manifest into production-usable positional reads while preserving the WASTE invariant behind README **Gate G**:

> Placement may change timing, but it must not change expert identity, record bytes, or arithmetic.

The runtime therefore accepts an explicit byte offset for every `[layer][expert]` rather than assuming `expert * record_bytes`. A real family-open layer will eventually derive those offsets from pinned container evidence and hand them to this boundary.

## Public boundary

`waste_ds_v4_positional_bank` describes one layer bank:

- a `pread`-shaped `read_at` callback;
- caller-owned reader context;
- the bank's validated byte length;
- one explicit record offset per routed expert.

`waste_ds_v4_positional_source_init` requires exactly the manifest's layer count and exactly the manifest's routed-expert count for every layer. On success it deep-copies the offset index and bank descriptors. Mutating or freeing the caller's offset arrays after initialization cannot redirect a later cache miss.

Reader contexts remain caller-owned and must outlive both the positional source and any runtime using it.

`waste_ds_v4_runtime_init_positional` binds the validated source to the existing `waste_ecache` path. It refuses a source whose layer count, expert count, or record size does not match the manifest used by the runtime.

## Exact-read semantics

A storage callback follows `pread` semantics:

```text
>0   number of bytes copied; short positive progress is legal
 0   EOF / no more bytes
<0   read error
```

The positional source loops on short positive reads until the entire manifest-declared record has landed. It fails the cache miss on:

- EOF before the record is complete;
- zero progress;
- a negative reader result;
- a reader claiming more bytes than requested.

A partial record is never passed to `waste_ds_v4_manifest_bind_routed_record`.

`waste_ds_v4_fd_read_at` is the native-file adapter and reuses WASTE's cross-platform `waste_pread`, including signed 64-bit offsets on Windows. Source initialization therefore rejects record extents that cannot be represented in that signed-64-bit positional-I/O range.

## Bounds and identity checks

Before any storage read, initialization proves for every record:

```text
offset <= bank_bytes
record_bytes <= bank_bytes - offset
offset + record_bytes <= INT64_MAX
```

The subtraction form is deliberate: an untrusted near-`UINT64_MAX` offset cannot wrap into a plausible small endpoint.

No monotonicity, uniform stride, page alignment, record header, or expert-id ordering is required by this layer. Those properties belong to the eventual evidence-backed container format, not to generic runtime I/O.

## Model-free validation

`tests/test_deepseek_v4_runtime.c` covers the positional seam with synthetic data and fault injection:

- physical expert order is a permutation rather than expert-id order;
- the caller's offset table is mutated after initialization and cannot redirect the source;
- a non-page/non-record-divisor short-read chunk size is reassembled exactly;
- a second request for the same expert is a cache hit and performs no storage reads;
- wrong layer/expert ids are refused;
- wrong bank counts and wrong per-bank expert counts are refused;
- an offset whose record would run past the bank is refused before reading;
- zero-progress and over-reporting readers fail closed.

These checks are synthetic and do **not** promote Gate G. Gate G still needs the same real DeepSeek record observed on an actual storage miss and a cache hit, followed by identical arithmetic.

## Remaining family-open work

The next runtime layer may now be small and evidence-focused. It should:

1. parse the already-defined DeepSeek family manifest;
2. map/open the real resident trunk;
3. open each actual routed bank;
4. obtain bank byte lengths from the opened objects;
5. derive the explicit expert offset tables from pinned container metadata;
6. initialize `waste_ds_v4_positional_source`;
7. bind it through `waste_ds_v4_runtime_init_positional`;
8. keep `waste_ds_v4_manifest_step_refused` unconditional until the numerical gates close.

The open layer must not infer a filename, header, stride, order, or alignment rule merely because imported Kimi WASTE v0 used one.
