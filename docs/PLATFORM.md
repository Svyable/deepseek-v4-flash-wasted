# Platform requirements — macOS, Linux and Windows

**Status: imported WASTE platform code is present and model-free-tested; DeepSeek-scale runtime behavior is not yet measured.**

This port inherits WASTE's goal of dependency-light C inference across macOS, Linux and Windows. Cross-platform compilation is necessary but not sufficient: a streamed model also depends on real unbuffered/direct I/O, aligned memory, large-file offsets and bounded memory behavior.

Gate terminology follows `docs/VALIDATION.md` §4a. Platform work primarily supports:

- **README Gate G** — placement/direct-I/O/cache mode must not change model meaning;
- **README Gate L** — real target-storage feasibility with the actual expert-record access pattern;
- **README Gate M** — cache recommendations measured under the real memory/storage environment.

G/L/M are systems/performance gates with no V-number.

## 1. Current baseline

- WASTE source imported at `d9b919a791148b571e643d0af666bf19b4d733ab`.
- Bootstrap test machine: Ubuntu 24.04, gcc 13.3.0, x86-64, 4 cores.
- Imported baseline: `31 passed, 0 failed, 12 skipped` before local changes.
- After PR #1 inventory test: `32 passed, 0 failed, 12 skipped`.
- After PR #3 native quantization tests: **34 passed, 0 failed, 12 skipped**.
- PR #3 ASan/UBSan run: **33 passed, 0 failed** with a clean warning-free rebuild.
- After the Gate A–F replays were given their own suite lines: **45 passed, 0 failed, 12 skipped**. Same tests, individually reported — `docs/EXPERIMENTS.md` entry 6.
- With PR #9's three Gate E/V5 replays named alongside them: **48 passed, 0 failed, 12 skipped**.
- The imported GitHub Actions matrix is parked under `.github/workflows-disabled/` because jobs failed before steps executed in this private repository. This is an account/Actions environment problem, not evidence about source portability.

Restore CI rather than shrinking its platform matrix once the account condition is resolved.

## 2. Common requirements

Every platform must provide:

- C11 compiler/toolchain;
- 64-bit file offsets;
- positional reads or a safe equivalent;
- thread primitives used by WASTE;
- aligned allocation suitable for unbuffered I/O;
- an implementation/fallback for page-cache bypass/direct I/O;
- a way to discover usable memory/cgroup/job/container limits where applicable;
- monotonic timing;
- atomic file replacement for converter outputs;
- filesystem support for very large model files.

Correctness may fall back to buffered I/O when a filesystem refuses bypass, but benchmark output must say that it did. Gate G requires the fallback/direct modes to expose the same record bytes/numerics; Gate L separately measures their performance characteristics.

## 3. Linux

### Direct I/O

The imported WASTE path uses Linux `O_DIRECT` behavior where available. Requirements commonly include alignment of:

- file offset;
- I/O length;
- userspace buffer.

Actual alignment can depend on filesystem/device. The DeepSeek converter/record layout should preserve at least the imported 4 KiB alignment target unless Gate A/container-format work proves another requirement, while the runtime must handle a platform refusal cleanly.

Do not infer direct-I/O success from `open()` flags alone. Report the runtime state and verify through the actual read path.

### Memory limits

Use the imported cgroup-aware memory capacity logic. A container on a high-RAM host may have a much smaller cgroup limit; sizing against host `MemTotal` can cause OOM kill.

The automatic DeepSeek budget must use the same principle and then apply DeepSeek-specific floor/cache sizing.

### Testing

Minimum Linux matrix eventually:

```text
x86-64 AVX2 baseline
x86-64 AVX-512 where real hardware is available
arm64/NEON
ASan/UBSan
container parser fuzzing
```

Do not label AVX-512 “tested” because it compiled on a CPU whose runtime dispatch selected AVX2.

## 4. macOS

### Page-cache bypass

Imported WASTE uses the macOS `fcntl`/`F_NOCACHE` style path rather than Linux `O_DIRECT` semantics. Preserve that abstraction in `platform.h`; DeepSeek model code should not contain OS-specific read calls.

### Apple silicon

NEON/vectorized CPU kernels are the first portable optimized target after scalar/official parity. Apple Accelerate/Metal are optional backends/experiments, not prerequisites for a correct base model.

Unified memory does not eliminate the hard-RAM-budget requirement. A too-large expert cache can still cause memory pressure/paging and destroy streaming performance.

### Internal versus external storage — Gate L

Upstream WASTE showed that interface/enclosure bandwidth can dominate a streamed model. Re-measure the actual DeepSeek expert-record pattern on the intended volume. “NVMe” describes the drive media, not necessarily the USB/Thunderbolt bridge performance.

## 5. Windows

### Build toolchain

Imported WASTE includes MinGW-w64 support and an upstream CI design that cross-builds on Linux and executes on a real Windows runner.

Maintain both checks:

- cross-compilation proves the source/toolchain link;
- execution on Windows proves `FILE_FLAG_NO_BUFFERING`, `ReadFile` offsets and aligned allocation behavior.

A successful cross-link is not a Windows runtime test.

### Unbuffered I/O

Windows `FILE_FLAG_NO_BUFFERING` has strict alignment requirements. The runtime should centralize these in the platform abstraction and use `_aligned_malloc`/appropriate matching free behavior where needed.

Tests should exercise real expert-like aligned offsets/read sizes on Windows, not only tiny sequential files.

### Large files

All offsets/lengths in container code must use explicitly 64-bit-safe types. Do not let `long` width vary the on-disk format or large-file addressing.

## 6. SIMD/backend policy

Order of implementation:

1. scalar correctness path;
2. official-reference parity for the relevant canonical gate;
3. architecture-neutral threaded baseline;
4. NEON/AVX2 optimized FP4/FP8 kernels;
5. AVX-512 only with actual runtime validation;
6. Metal/CUDA/other accelerators only after profiling and correctness gates.

Every optimized backend runs the same independent oracle fixtures and real final-logit tests.

The runtime should report which backend actually dispatched.

## 7. Storage qualification — README Gate L

Before a full DeepSeek performance run on a platform, record:

```text
OS/version
filesystem
physical storage model
connection/interface
encryption if relevant
container location
record size(s)
direct/bypass status
1-thread random record throughput
runtime reader-thread random record throughput
```

The benchmark working set should exceed RAM/page-cache effects enough to represent actual streaming.

Gate L results belong in `BENCHMARKS.md` and must not be presented as model tok/s by themselves.

## 8. Converter platform concerns

The converter may use Python/PyTorch and is not constrained to the dependency-free runtime policy.

Still require:

- atomic rename semantics on the output filesystem;
- sufficient free space checked before long writes;
- path handling for Windows/macOS/Linux;
- safe interruption/resume;
- no assumptions that source/output are on the same filesystem;
- no destructive reclaim by default.

When source and output are on separate devices, record both in conversion benchmarks.

## 9. Filesystem/direct-I/O fallback policy — Gate G first, Gate L second

Correctness hierarchy:

```text
direct/unbuffered path if supported
        |
        +-- success -> report enabled
        |
        +-- filesystem refuses -> explicit buffered fallback where safe
                                   + prominent benchmark/runtime warning
```

Do not fail model correctness solely because a developer's test filesystem does not support the performance path, unless the user explicitly required direct I/O.

Conversely, do not report buffered fallback performance as representative of the intended streaming configuration without saying so.

Before comparing performance, Gate G must show that direct and fallback read paths produce the same record/model semantics.

## 10. Platform test fixtures

Keep model-free fixtures small but structurally realistic:

- files larger than one alignment unit;
- multiple expert banks;
- record offsets past 4 GiB in a sparse-file test where feasible to catch 32-bit offset bugs without allocating huge disk space;
- aligned and deliberately misaligned buffers/sizes;
- short read/truncated bank;
- simultaneous reader threads;
- direct-I/O fallback behavior;
- manifest paths with spaces/non-ASCII where platform APIs support them.

PR #1's inventory fixture demonstrates the value of making a prohibited code path hit EOF by construction. Apply the same adversarial principle to I/O fixtures.

## 11. CI restoration plan

The imported workflow is preserved under `.github/workflows-disabled/ci.yml`.

Before restoring:

1. inspect repository/account Actions permissions and billing/minutes;
2. verify `actions/checkout@v4` is allowed;
3. widen the SPDX source glob so subdirectories such as `src/quant/` are included;
4. restore with the documented `git mv`;
5. trigger manually;
6. inspect actual logs rather than interpreting a pre-step failure as a code failure.

After DeepSeek kernels arrive, extend the matrix with **small model-free/official-derived replay fixtures**, not real 0731 weights.

## 12. Platform support definition

A platform is “supported” only when:

- it builds;
- model-free suite passes;
- native I/O wrapper tests pass or documented fallback works;
- scalar DeepSeek fixtures pass the relevant canonical gates;
- at least one optimized backend (or scalar if intentionally slow) reports correctly;
- a real/subset DeepSeek container opens and passes numerical validation where hardware access exists;
- Gate G placement identity holds for supported read/cache modes;
- memory planner respects platform/container limits;
- server/CLI basic operation works;
- Gate L/M performance claims, if made, are measured on that platform rather than inherited from another OS/device.

Do not reduce “supported” to “compiler emitted a binary.”