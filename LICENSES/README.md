# Licenses and attribution

This tree combines code under more than one license. This directory collects
the governing texts so a reader does not have to guess which terms apply to
which files.

## WASTE — Apache License 2.0

- Text: [`WASTE-APACHE-2.0.txt`](WASTE-APACHE-2.0.txt)
- Notice: [`WASTE-NOTICE.txt`](WASTE-NOTICE.txt)
- Upstream: <https://github.com/sqliteai/waste>
- Imported commit: `d9b919a791148b571e643d0af666bf19b4d733ab`
- Copyright 2026 SQLite Cloud, Inc.

Both files are byte-identical copies of the `LICENSE` and `NOTICE` files that
came with the pinned import, which remain at the repository root. Apache-2.0
§4 requires that the license and the `NOTICE` contents travel with any
redistribution, so **do not delete the root copies** in favour of these.

Almost every file in `src/`, `cli/`, `serve/`, `tests/`, `tools/`, `docs/`, and
`examples/` originates from this import. `UPSTREAM.md` lists the files that do
not.

If you modify an imported file, Apache-2.0 §4(b) requires the modified file to
carry a prominent notice stating that you changed it. Add that notice as you
edit, not in a cleanup pass later.

## DeepSeek-V4-Flash-0731 — MIT

- Text: [`DEEPSEEK-MIT.txt`](DEEPSEEK-MIT.txt)
- Upstream: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- Release commit: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Upstream filename: `LICENSE`
- Upstream copyright line: `Copyright (c) 2023 DeepSeek`

The exact publisher license was retrieved from the pinned **Release
DeepSeek-V4-Flash-0731** commit and vendored verbatim. The old
`DEEPSEEK-MIT.txt.MISSING` marker has therefore been removed.

The release commit adds one `LICENSE` file and identifies the repository/model
as MIT-licensed. If a future pinned revision introduces additional model/code
license or notice files, vendor and record those separately rather than
assuming this file governs a materially different release.

### Provenance rule for DeepSeek-derived code and fixtures

Before copying or adapting any official `inference/` or `encoding/` source,
record at minimum:

```text
model repository
full immutable source commit SHA
source path
source/content hash where practical
this repository commit that imports/adapts it
```

Small official-oracle fixtures must also follow `docs/FIXTURES.md`: the
expected side must remain independent of the implementation/convention it is
meant to prove.

The canonical release baseline for current Gate A/V0, Gate B/V1 and Gate C/V2
work is:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
9e165c30e2704aec5d9d593cce3eebd58bbef1cb
```

A later model-card-only commit does not silently change the checkpoint/oracle
baseline. Moving the baseline requires an explicit project update and fixture
provenance change.
