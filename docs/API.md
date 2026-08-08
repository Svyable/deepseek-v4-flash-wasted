# API and serving contract

**Status: DESIGN. Imported WASTE server exists; DeepSeek encoding/model integration is NOT STARTED.**

WASTE already has a useful architecture: an embeddable C library, a CLI that is only a client of that library, and a stdlib Python server that reaches `libwaste` through `ctypes`. This port should preserve that separation.

Imported `docs/SERVE.md` describes Kimi XTML and remains upstream reference material. DeepSeek requires its own official encoding/parser semantics.

## 1. Architectural rule

> If the CLI or server needs an inference capability, expose it through the public C API first.

Do not create a private server-only model path.

The layers should remain:

```text
DeepSeek .waste container
        |
        v
libwaste public C API
        |
        +--> waste CLI
        |
        +--> Python ctypes bridge
                |
                +--> OpenAI-compatible localhost HTTP server
                        |
                        +--> optional UI/client
```

## 2. C library responsibilities

The native library owns model/runtime behavior:

- memory planning;
- model/container open/close;
- model-family validation;
- tokenizer primitives needed by the packaged DeepSeek encoder;
- evaluation/generation;
- context/session state;
- sampling primitives or generation configuration already provided by the library;
- routed expert cache and I/O;
- stats/telemetry;
- cancellation hooks where supported;
- detailed errors.

The library must not:

- call `exit()` for recoverable user/model errors;
- print protocol responses;
- own HTTP/SSE parsing;
- read server config files implicitly;
- silently choose a remote model/template;
- depend on Streamlit/Ollama/another host runtime.

## 3. DeepSeek model-family discovery

A caller should be able to inspect enough model metadata to know:

```text
model family / architecture
container format/version
source model revision
base model availability
DSpark assets present?
DSpark enabled?
context limit/current configured context
quantization/storage summary
```

Do not expose Kimi-specific fields as the generic model identity.

## 4. Memory planning API

Before open, hosts should be able to request a plan for:

- requested context length;
- thread count/CPU placement;
- optional DSpark enabled/disabled;
- explicit RAM budget or automatic resolution.

Plan output should include the rows described in `MEMORY_AND_IO.md`.

The server should use the same planner as the CLI. It must not allocate a second independent expert cache or infer memory limits itself.

## 5. Tokenization/encoding boundary

DeepSeek 0731 supplies code-based encoding behavior. Separate two concepts:

### Tokenizer primitive

Native/tokenizer layer performs exact text-to-token and token-to-text operations according to packaged tokenizer assets.

### Conversation encoder

Python server/host renders structured messages, tools and response constraints into the exact token sequence required by official DeepSeek encoding semantics.

This mirrors the useful design principle in imported WASTE `serve/`: protocol/chat logic belongs in Python; arithmetic/model state belongs in C.

Do **not** assume DeepSeek uses Kimi's markup/token split. Port what official `encoding/` does and test exact token sequences.

## 6. Response parser

If official DeepSeek output uses structured reasoning/tool regions, the server needs an incremental parser that can consume generated token IDs/pieces and emit OpenAI-compatible deltas.

Requirements:

- parser behavior is grounded in official encoding/reference tests;
- malformed/truncated output is handled as data, not a process crash;
- streaming and non-streaming use the same parser semantics;
- literal text that resembles structural markers must not be misparsed when token identity can distinguish it;
- committed token state is distinct from speculative DSpark tokens.

## 7. OpenAI-compatible HTTP scope

Use the imported server as the implementation base instead of replacing it with an unrelated serving stack.

The initial DeepSeek milestone should expose only endpoints/features that can be tested end-to-end. Typical progression:

1. health/model information;
2. non-streaming chat completions;
3. streaming chat completions;
4. tools/structured output only after official encoding/parser parity;
5. optional project-specific controls such as DSpark only through clearly documented extensions, not fake standard OpenAI fields.

Preserve compatibility where practical, but model correctness takes precedence over mimicking unsupported API features.

## 8. Model naming

The HTTP `model` field should resolve locally and deterministically. A recommended design is to expose one or more aliases plus the underlying container identity, for example:

```text
deepseek-v4-flash-0731
```

Do not make the server download a model because an arbitrary remote-style model name was requested.

A request for an unknown model should return a protocol error rather than silently use the only loaded model.

## 9. Generation settings

Map request fields onto the existing public generation API deliberately:

- max/new token limit;
- temperature;
- top-p/top-k if supported;
- stop conditions;
- seed if deterministic support exists;
- reasoning/thinking controls only if part of official DeepSeek behavior;
- tool choice/response format only when encoder semantics support them.

Fields with no faithful implementation should fail clearly or be documented as unsupported, not be ignored silently.

## 10. Streaming

The server should emit SSE deltas from committed model output.

Requirements:

- client disconnect/cancellation should stop generation promptly where the native API supports cancellation;
- stats are finalized even on cancellation where possible;
- DSpark proposals are never streamed before they are accepted/committed;
- parser state is incremental and identical to the non-streaming parse when concatenated;
- UTF-8/token boundary handling is tested.

## 11. Errors

Map native errors into useful HTTP responses without hiding the underlying detail.

Examples:

| Native condition | Server behavior |
|---|---|
| invalid/corrupt container | startup failure with exact file/tensor/record detail |
| RAM budget below floor | startup/request configuration error pointing to `plan` |
| context full | client-visible request error, not overwrite/out-of-bounds |
| expert read/checksum failure | terminate generation; never continue with bad bytes |
| unsupported model family/format | explicit incompatibility error |
| unsupported API feature | 4xx with a clear message |
| internal arithmetic invariant failure | 5xx + server log; never return fabricated completion |

## 12. Stats surface

Hosts should have access to the same underlying stats used by benchmarks:

- prompt/generated tokens;
- prefill/decode timings;
- expert hits/misses and bytes read;
- prefetch useful/wasted;
- direct-I/O state;
- RAM budget/cache bytes;
- backend/kernel selection;
- DSpark acceptance stats when enabled.

The public API does not need to expose every internal counter forever, but the server/CLI should not maintain conflicting duplicate definitions.

## 13. Security/local exposure

The first server target is localhost. Do not casually turn a dependency-free local model server into an unauthenticated LAN service.

Document/bind explicitly:

```text
127.0.0.1 by default
explicit --host/--listen for broader exposure
```

If remote exposure is later supported, consider authentication/reverse-proxy guidance separately. Model container parsing must remain hardened because local files are still untrusted inputs.

## 14. Optional UI topology

If a Streamlit or desktop UI is added, keep the model process separate:

```text
UI process
   -> localhost OpenAI-compatible server
      -> ctypes/libwaste
         -> container on target NVMe
```

Do not load the model inside Streamlit's script-rerun lifecycle. The server should own the long-lived model/cache/session resources.

## 15. API validation gates

Before calling serving complete:

- [ ] official DeepSeek encoder tests pass token-for-token;
- [ ] parser tests pass including malformed/truncated cases;
- [ ] direct C generation and server generation match for deterministic request;
- [ ] streaming concatenation equals non-streaming output;
- [ ] unknown model/unsupported fields fail clearly;
- [ ] context/RAM/read errors are propagated safely;
- [ ] client cancellation tested;
- [ ] server does not change RAM/cache plan relative to equivalent CLI run;
- [ ] DSpark disabled/enabled modes obey `DSPARK.md`;
- [ ] `VALIDATION.md` V8/V9 already pass on the underlying model.

API compatibility is the last layer over a correct model, not evidence that the model is correct.