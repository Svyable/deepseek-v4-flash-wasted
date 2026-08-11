# DeepSeek V4 positional runtime I/O

This note records the storage boundary implemented by `src/deepseek_v4_runtime.{c,h}` after PRs #22–#23 and the explicit file-ownership layer in `src/deepseek_v4_file_runtime.{c,h}` built on top of it. It is intentionally narrower than a DeepSeek container format: the pinned evidence has not frozen filenames, per-record headers, record ordering, alignment, or direct-I/O policy, so the runtime does not invent them.

## Goal

Turn a validated DeepSeek routed-record manifest into production-usable positional reads while preserving the WASTE invariant behind README **Gate G**:

> Placement may change timing, but it must not change expert identity, record bytes, or arithmetic.

The runtime therefore accepts an explicit byte offset for every `[layer][expert]` rather than assuming `expert * record_bytes`. The file layer likewise accepts explicit caller-resolved paths rather than assuming `trunk.bin` or a per-layer filename pattern. A later evidence-backed family resolver can derive those paths and offsets from pinned container metadata and hand them to this boundary.

## Positional source boundary

`waste_ds_v4_positional_bank` describes one layer bank:

- a `pread`-shaped `read_at` callback;
- caller-owned reader context;
- the bank's validated byte length;
- one explicit record offset per routed expert.

`waste_ds_v4_positional_source_init` requires exactly the manifest's layer count and exactly the manifest's routed-expert count for every layer. On success it deep-copies the offset index and bank descriptors and freezes the routed six-plane record map. Mutating or freeing the caller's offset arrays after initialization cannot redirect a later cache miss.

Reader contexts remain caller-owned and must outlive both the positional source and any runtime using it.

`waste_ds_v4_runtime_init_positional` binds the validated source to the existing `waste_ecache` path. It refuses a source whose layer count, expert count, record size, or routed six-plane map does not match the manifest used by the runtime. Matching record byte length alone is deliberately insufficient: two individually valid maps can place the same planes in different byte order, and accepting the wrong one would produce plausible but incorrect tensor bindings.

`waste_ds_v4_runtime_manifest_validate` is the shared runtime-facing revalidation front door. Parsing already validates the manifest, but the resulting C struct is mutable; positional and file-resource bindings use the same function so post-parse mutation cannot be accepted by one path and refused by another.

## Explicit file ownership

`waste_ds_v4_file_runtime_open` closes the next production-risk seam without defining a family directory layout. Its caller supplies:

- an already-parsed `waste_ds_v4_manifest`, which is revalidated before I/O;
- one explicit resident-trunk path;
- one explicit bank path per manifest layer;
- one explicit record offset per routed expert;
- the cache budget/policy to bind after storage validation.

The open path then owns the native resources:

1. revalidate runtime-critical manifest identity/geometry and all obvious file/spec geometry before opening anything;
2. open the resident file and require its OS-reported size to equal `manifest->trunk_bytes` exactly;
3. allocate the final WASTE-aligned resident buffer once and fill it with exact positional reads — there is no whole-file temporary slurp followed by a second copy;
4. open every routed bank and query its actual byte length from the OS;
5. build `waste_ds_v4_positional_source` from those actual lengths plus the caller's explicit offsets;
6. bind that source to `waste_ecache` through `waste_ds_v4_runtime_init_positional`.

Paths and caller offset arrays are borrowed only during open. The source deep-copies placement metadata and the file runtime owns every bank descriptor plus the resident allocation after success.

Teardown is deliberately reverse-order:

```text
runtime/cache
  -> positional source
    -> bank file descriptors
      -> resident allocation
```

That ordering matters because the runtime's fetch user points at the positional source, and each positional bank's read context points into the owned descriptor array. A partial failure at any stage uses the same teardown path and leaves the output object zeroed. `waste_ds_v4_file_runtime_close` is safe again after it has zeroed a successfully opened object.

This layer currently uses ordinary read-only/random-access file opens. It does **not** silently enable WASTE's page-cache bypass/direct-I/O path because DeepSeek record alignment has not been frozen by evidence. Direct-versus-buffered parity belongs to Gate G once the real format proves the required offset/length alignment.

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

`waste_ds_v4_fd_read_at` is the native-file adapter and reuses WASTE's cross-platform `waste_pread`, including signed 64-bit offsets on Windows. Source initialization therefore rejects record extents that cannot be represented in that signed-64-bit positional-I/O range. The file-owned resident read uses the same exact-progress rules and refuses short/zero/error completion rather than exposing a partial trunk.

## Bounds and identity checks

Before any routed storage read, initialization proves for every record:

```text
offset <= bank_bytes
record_bytes <= bank_bytes - offset
offset + record_bytes <= INT64_MAX
```

The subtraction form is deliberate: an untrusted near-`UINT64_MAX` offset cannot wrap into a plausible small endpoint.

The file layer additionally requires the resident file's actual byte length to equal the validated manifest's declared trunk length. Routed bank files are bounded by their actual OS-reported sizes rather than a caller-supplied byte count.

No monotonicity, uniform stride, page alignment, record header, filename pattern, or expert-id ordering is required by these layers. Those properties belong to the eventual evidence-backed container format, not to generic runtime I/O.

## Model-free validation

`tests/test_deepseek_v4_runtime.c` covers the positional seam with synthetic data and fault injection:

- physical expert order is a permutation rather than expert-id order;
- the caller's offset table is mutated after initialization and cannot redirect the source;
- a valid, same-sized manifest with two routed planes reordered cannot reuse the source;
- a non-page/non-record-divisor short-read chunk size is reassembled exactly;
- a second request for the same expert is a cache hit and performs no storage reads;
- wrong layer/expert ids are refused;
- wrong bank counts and wrong per-bank expert counts are refused;
- an offset whose record would run past the bank is refused before reading;
- zero-progress and over-reporting readers fail closed.

`tests/test_deepseek_v4_file_runtime.c` separately exercises the ownership layer with actual local files:

- resident and routed bytes are read through owned native descriptors and then consumed by the existing runtime seams;
- resident FP8 arithmetic runs from the file-loaded aligned trunk;
- caller path/offset metadata is mutated after open without changing the opened identity;
- a resident file one byte short or one byte long is refused;
- a routed bank one byte short is refused from its actual file size;
- an explicit routed offset extending one byte past EOF is refused;
- invalid per-bank expert counts are refused before resource ownership begins;
- a missing bank after several previous bank opens forces full partial-open teardown;
- cleanup removes every owned handle and a second close after zeroing is harmless;
- the same output object can be reused after failed opens.

The dedicated file-runtime binary is built by the normal Makefile and is included in both `make check` and `make asan`; CI invokes the same target rather than hand-compiling a different test binary.

These checks use real OS files but synthetic DeepSeek bytes and therefore do **not** promote Gate G. Gate G still needs the same real DeepSeek record observed on an actual storage miss and cache hit, followed by identical arithmetic.

## Remaining family-open work

The generic runtime/resource substrate is now sufficient for a small evidence-focused family resolver. The next layer should:

1. read the real DeepSeek container metadata at the pinned revision;
2. derive the actual resident path, routed bank paths, bank topology, and explicit expert offset tables from that metadata;
3. parse/validate the existing family manifest and hand the resolved paths/offsets to `waste_ds_v4_file_runtime_open`;
4. add the first real disk-vs-cache byte/arithmetic identity test for Gate G;
5. only after real alignment evidence exists, compare buffered and page-cache-bypassing/direct-I/O reads without changing the authoritative bytes;
6. keep `waste_ds_v4_manifest_step_refused` unconditional until the numerical gates close.

The resolver must not infer a filename, header, stride, order, or alignment rule merely because imported Kimi WASTE v0 used one.
