# Release packaging

RW-Cite uses separate publication targets for source code and benchmark data.

| Target | Contents |
|---|---|
| `github/` | Source, configurations, tests, and public documentation |
| `huggingface/` | Ten domain packs and the optional shared-corpus payload |

From a complete development checkout, build the source release with:

```bash
bash scripts/prepare_public_releases.sh --version 1.0 --code-only
```

Build the unified Hugging Face staging tree with:

```bash
bash scripts/prepare_public_releases.sh \
  --data-only --domains-only --data-tag YYYYMMDD --checksums
```

Add `--with-corpus --data-tier full` only when the optional corpus files are
available and their redistribution terms have been reviewed. Generated
payloads and archives are not part of the source release.
