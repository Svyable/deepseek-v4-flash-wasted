# DSpark — phase-2 speculative decoding plan

**Status: DESIGN / DEFERRED. README Gate N. Base DeepSeek V4 logits and greedy generation must pass Gate I / V8 and Gate K / V9 before DSpark is implemented or enabled.**

The 0731 release track includes an attached speculative-decoding component referred to in the project handoff as DSpark. Its exact tensor namespace, dependencies and arithmetic remain subject to **Gate A / V0** checkpoint/reference validation before Gate N can begin.

Current README-level architecture notes include target main layers `40, 41, 42`, block size `5`, and Markov rank `256`. Treat these as **OFFICIAL-SPEC/HANDOFF, CHECKPOINT-UNVERIFIED in this repo** until the pinned checkpoint/reference is read.

Gate terminology follows `docs/VALIDATION.md` §4a:

- **Gate A / V0** maps and separates DSpark assets;
- **Gate I / V8** proves final base-model logits;
- **Gate K / V9** proves deterministic base generation;
- **Gate G** remains the placement-identity invariant if speculative execution changes cache/I/O behavior;
- **Gate N** is the final DSpark correctness + performance gate and intentionally has no V-number of its own.

## 1. Why DSpark is isolated

Speculative decoding adds at least three new failure dimensions:

- draft prediction arithmetic;
- acceptance/verification logic;
- performance accounting that can look faster while silently changing output.

Debugging those at the same time as the base attention/MoE port would make final-logit failures ambiguous. Therefore DSpark is not part of the minimum base-model loader/forward path.

## 2. Entry gate for README Gate N

Do not begin DSpark integration until all are true:

- **Gate A / V0** maps and separates DSpark tensors from the base stack;
- base converter can create a valid container without DSpark;
- **Gate I / V8** final-logit validation passes on real weights;
- **Gate K / V9** greedy generation passes on a deterministic prompt/token set;
- **Gate G** cache-on/cache-off and prefetch-off/on do not change base output;
- base performance/memory measurements exist so speculative speedup has a denominator.

## 3. Required loader modes

The container/runtime should support:

```text
A. base assets only
B. base + DSpark assets present, DSpark disabled
C. base + DSpark assets present, DSpark enabled
```

Modes A and B must be numerically equivalent for the base decode path. Merely having DSpark files installed must not alter scheduling, RAM budget, prompt encoding, or base logits unless the user enables it.

## 4. Gate A / V0 questions

The real checkpoint/reference must answer:

1. Which tensors are DSpark-only?
2. Which tensors are shared/referenced from the base model?
3. Are target-layer hidden states inputs to the draft module, and at exactly which point in the residual/mHC flow?
4. What state persists across tokens?
5. What is the meaning of the configured block size?
6. What does “Markov rank” parameterize in the released implementation?
7. How many speculative tokens can be proposed per step?
8. What logits/probabilities are used for verification?
9. How are EOS/stopping/context boundaries handled?
10. Does the official server/runtime expose knobs that must be preserved?

Do not infer answers from older MTP implementations if 0731 differs.

## 5. Container design

Preferred shape:

```text
model.waste/
  manifest.json
  trunk.bin
  experts-L*.bin
  ...base assets...
  dspark/
    manifest.json
    ...DSpark-only payloads...
```

The DSpark manifest should reference base tensors by stable canonical identity where they are genuinely shared rather than duplicating them.

Base validity must not depend on the `dspark/` directory.

## 6. Memory planning

DSpark must have a separate planner line:

```text
base RAM floor                  X
DSpark resident weights/state   Y
DSpark scratch                  Z
---------------------------------
RAM floor with DSpark         X+Y+Z
```

If enabling DSpark reduces available expert cache, the Gate N benchmark must include that cost. A speculative module can be compute-faster but end-to-end slower if its resident/scratch memory pushes the main expert cache into a worse Gate M region.

The user should be able to plan both modes before opening the full model.

## 7. Gate N correctness ladder

Speculative decoding is an acceleration technique. With deterministic sampling settings, enabling it must preserve the output distribution/accepted final tokens according to the official algorithm.

The D-levels below are **subchecks inside README Gate N**, not a fourth project-wide gate vocabulary.

### N-D1 — draft module primitive parity

Compare official and C for the smallest DSpark operation(s):

- inputs from target base layer(s);
- internal low-rank/Markov state if present;
- draft hidden/logit outputs.

### N-D2 — multi-token proposal parity

Given a fixed base state, compare:

- proposed token IDs;
- proposal probabilities/logits required by verification;
- updated DSpark state.

### N-D3 — acceptance decision parity

Construct fixtures that force:

- all proposals accepted;
- rejection at first proposal;
- rejection in the middle;
- rejection at last proposal;
- EOS inside proposal block;
- context boundary inside proposal block.

Acceptance/rejection indices and committed tokens must match official reference exactly.

### N-D4 — base fallback parity

A rejection must continue from the correct authoritative base-model state. There must be no hidden mutation from unaccepted draft tokens.

### N-D5 — end-to-end committed-token parity

For deterministic prompts, DSpark-on and official speculative reference should generate the same committed sequence. DSpark-off remains the diagnostic baseline.

These `N-D*` labels are local subchecks owned by Gate N; they are written with the `N-` prefix so they cannot be mistaken for README Gates A–N or global V-levels.

## 8. State rollback is load-bearing

A speculative implementation may temporarily evaluate states for tokens that are later rejected. Any stateful DeepSeek component must either:

- update into temporary/copy-on-write state until acceptance; or
- support exact rollback; or
- be recomputed safely from the last committed state.

This includes any attention/compression/indexer state and DSpark's own Markov/draft state.

Never commit state simply because a token was proposed.

## 9. Interaction with streamed experts — Gate G must remain true

Speculation can change expert I/O patterns substantially.

Measure:

- routed expert reads per committed token;
- expert reads for rejected speculative tokens;
- cache pollution from rejected tokens;
- whether draft evaluation can reuse base expert/cache work;
- useful versus wasted prefetch under speculation;
- total read GiB per **committed** token.

A higher raw evaluated-token rate is not the metric users experience.

Speculative scheduling/prefetch may change timing and traffic, but it must not cause cache placement to change authoritative model meaning. If it does, Gate G has failed before Gate N performance can be discussed.

## 10. Gate N performance metrics

Report together:

```text
base decode tok/s
DSpark committed tok/s
average proposed tokens/step
acceptance rate by proposal position
average committed tokens/verification step
base-model verification calls/committed token
expert read GiB/committed token
RAM floor delta
expert-cache delta
TTFT delta
```

A speedup without acceptance rate and memory/I/O cost is incomplete.

Record Gate N results in `BENCHMARKS.md`; include the base model's Gate I/V8 and Gate K/V9 status in the same entry.

## 11. API/CLI controls

DSpark should initially be explicit and easy to disable, for example conceptually:

```text
--speculative off|dspark
```

or a configuration field exposed through `waste.h` before the CLI/server.

The final public option is chosen only after the official serving semantics are understood. Do not invent request fields that look OpenAI-standard if they are project-specific.

For server use, a default may be considered only after correctness and stable positive end-to-end measurements on target hardware.

## 12. Benchmark sequence

Once N-D5 passes:

1. base versus DSpark with same RAM budget;
2. account for cache reduction from DSpark memory;
3. short and long generation lengths;
4. at least several prompt domains because acceptance can be workload-dependent;
5. context sweep;
6. thread/backend sweep;
7. I/O/prefetch metrics;
8. thermally stable repeated runs.

Add results to `BENCHMARKS.md` as README Gate N and surprising/negative findings to `EXPERIMENTS.md`.

## 13. Gate N pass / kill criteria

Gate N passes only if:

- committed output is provably equivalent to the official speculative algorithm;
- state rollback/commit semantics are exact;
- Gate G remains true;
- positive wall-clock benefit is measured using committed tokens under comparable RAM/cache accounting.

Keep DSpark optional or defer optimization if, on the target local hardware:

- committed output is not provably equivalent to the official algorithm;
- state rollback is fragile or requires duplicating an unacceptable fraction of context state;
- added resident memory causes a larger expert-cache/I/O regression than the speculative compute win;
- acceptance is too low for typical workloads to offset extra evaluation;
- complexity materially destabilizes the base path.

The project succeeds by running the full base model locally; DSpark is valuable only if it improves that result without weakening correctness.