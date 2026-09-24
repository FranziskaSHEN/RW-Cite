# Ten-domain benchmark payloads

This directory receives the ten domain packs defined in
`configs/paper_contract.yaml`:

```text
ewm, gw, driving, cosmo, radio, wsi, exo, sqc, fno, sce
```

Build and checksum the complete set with:

```bash
bash scripts/prepare_public_releases.sh \
  --data-only --domains-only --data-tag YYYYMMDD --checksums
```

Each generated pack must contain a graph checksum, frozen query manifests,
candidate identities, evaluation artifacts, and a dataset card free of local
paths or identifying information.
