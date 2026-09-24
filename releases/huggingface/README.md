---
pretty_name: RW-Cite
task_categories:
  - text-retrieval
  - feature-extraction
language:
  - en
license: other
---

# RW-Cite benchmark data

This staging area contains one Hugging Face dataset release with ten domain
packs under `domains/`. Each pack contains its graph, frozen splits, candidate
and ranking manifests, evaluation artifacts, checksums, and dataset card.

Build the domain payloads with:

```bash
bash scripts/prepare_public_releases.sh \
  --data-only --domains-only --data-tag YYYYMMDD --checksums
```

The optional `corpus/` payload is built separately and should be included only
after upstream licensing and redistribution constraints have been reviewed.
Upload and verification instructions are in `docs/DATASETS_HF_UPLOAD.md`.
