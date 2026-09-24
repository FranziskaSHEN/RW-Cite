# Optional corpus payload

This directory receives the optional metadata snapshot, topic supplements,
embeddings, and their manifest. Build it only when all required files are
available and redistribution is permitted:

```bash
bash scripts/prepare_public_releases.sh \
  --data-only --corpus-only --data-tag YYYYMMDD \
  --data-tier full --checksums
```
