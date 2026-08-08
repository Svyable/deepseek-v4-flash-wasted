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

## DeepSeek-V4-Flash-0731 — MIT (text not yet vendored)

- Upstream: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- Pinned revision: `9e165c3`
- Placeholder: [`DEEPSEEK-MIT.txt.MISSING`](DEEPSEEK-MIT.txt.MISSING)

**The DeepSeek license text is deliberately absent.** `huggingface.co` is
blocked by this environment's egress policy, so the authoritative `LICENSE`
file could not be retrieved. Writing an MIT template from memory would put a
fabricated legal document into the repository under a real party's name, with a
copyright line nobody verified — so the placeholder records the requirement
instead.

The repository `README.md` §0 states DeepSeek's weights and repository are
MIT-licensed. That is a claim to verify against the checkpoint, not a
substitute for the file.

### To resolve

From an environment that can reach Hugging Face:

```bash
python -m pip install 'huggingface_hub>=0.34'
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
for name in ('LICENSE', 'LICENSE-MODEL', 'LICENSE-CODE'):
    try:
        p = hf_hub_download(
            'deepseek-ai/DeepSeek-V4-Flash-0731',
            name,
            revision='9e165c3',
        )
    except Exception as exc:
        print(f'{name}: {exc}')
        continue
    shutil.copy(p, f'LICENSES/DEEPSEEK-{name}.txt')
    print(f'{name}: vendored')
PY
git rm LICENSES/DEEPSEEK-MIT.txt.MISSING
```

Check which of those filenames actually exist at the pinned revision — some
DeepSeek repositories split model and code licensing — and vendor each one you
find rather than assuming a single `LICENSE`.

This must be done before any DeepSeek reference code, tensor-name tables, or
converter logic derived from the official `inference/` sources is committed
here. Until then, nothing in this tree is derived from DeepSeek material.
