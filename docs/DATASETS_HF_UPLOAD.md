# RW-Cite dataset release

RW-Cite publishes source code on GitHub and benchmark data through one Hugging
Face dataset repository. Keeping these products separate prevents large graphs,
model artifacts, and evaluation outputs from entering the source history.

The dataset repository contains the ten paper-aligned domain packs. An
optional `corpus/` subtree may additionally contain redistributable metadata
supplements and manifests for reconstructing the shared corpus.

## Dataset layout

```text
README.md
COLLECTION_MANIFEST.json
domains/
  ewm/
  gw/
  driving/
  cosmo/
  radio/
  wsi/
  exo/
  sqc/
  fno/
  sce/
corpus/                    Optional; governed by upstream licenses
```

Each domain pack should contain:

- the citation graph and graph checksum;
- frozen training, development, test, and frontier query manifests as
  applicable;
- full-gold citation IDs and citation-context provenance;
- candidate universes and ordered ranking windows;
- model configurations and checkpoint identifiers;
- per-query predictions and aggregate evaluation metrics;
- a manifest listing file sizes and SHA-256 checksums;
- a dataset card describing provenance, licensing, and intended use.

Generated logs, caches, temporary shards, paper source archives, API keys, and
machine-specific absolute paths must not be included.

## Build the staged dataset

From the repository root, build all ten domain packs:

```bash
bash scripts/prepare_public_releases.sh \
  --data-only --domains-only --data-tag YYYYMMDD --checksums
```

The output is written beneath `releases/huggingface/`. To include the optional
global corpus payload when all required source assets are present:

```bash
bash scripts/prepare_public_releases.sh \
  --data-only --with-corpus --data-tag YYYYMMDD \
  --data-tier full --checksums
```

Review every generated dataset card and manifest before upload. In particular,
confirm that the ten-domain counts match
[`../configs/paper_contract.yaml`](../configs/paper_contract.yaml).

## Validate the staging tree

Before publishing:

```bash
find releases/huggingface -type f -print0 | sort -z | xargs -0 sha256sum
```

Inspect the staging tree with the repository's release-hygiene tests and a
secret scanner appropriate to the publication environment. Also verify that:

1. the dataset card begins with valid YAML metadata;
2. every referenced file exists;
3. no training query appears in a test manifest;
4. graph, split, and candidate checksums agree with the evaluation artifacts;
5. aggregate metrics can be recomputed from saved per-query predictions;
6. upstream licenses permit redistribution of every included component.

## Upload with the current Hugging Face CLI

Install and authenticate with the current `hf` CLI:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf auth login
hf auth whoami
```

Set the destination once. During anonymous review, use the repository ID
provided by the anonymous hosting service; after acceptance, use the permanent
public namespace.

```bash
export RW_CITE_DATASET='<namespace>/RW-Cite'
```

Create the dataset repository if necessary, then upload the complete staged
tree in one commit:

```bash
hf repos create "$RW_CITE_DATASET" --type dataset --public --exist-ok

hf upload "$RW_CITE_DATASET" releases/huggingface . \
  --type dataset \
  --commit-message "Publish RW-Cite benchmark data"
```

For a reviewable pre-publication update, add `--create-pr` to `hf upload`.
Create an immutable release tag only after the uploaded manifests have been
verified:

```bash
hf repos tag create "$RW_CITE_DATASET" v1.0 \
  --type dataset \
  --message "RW-Cite benchmark release v1.0"
```

## Download and verify

Consumers can download the complete benchmark or selected domains:

```bash
hf download "$RW_CITE_DATASET" \
  --type dataset \
  --revision v1.0 \
  --local-dir datasets/rw-cite

hf download "$RW_CITE_DATASET" \
  --type dataset \
  --revision v1.0 \
  --include 'domains/ewm/**' \
  --local-dir datasets/rw-cite
```

Verify the local checkout against the Hub revision:

```bash
hf cache verify "$RW_CITE_DATASET" \
  --type dataset \
  --revision v1.0 \
  --local-dir datasets/rw-cite \
  --fail-on-missing-files
```

Base encoders are downloaded separately from their original publishers:

```bash
hf download BAAI/bge-large-en-v1.5 \
  --local-dir models/base/bge-large-en-v1.5

hf download allenai/scibert_scivocab_uncased \
  --local-dir models/base/scibert_scivocab_uncased
```

## Versioning

Paper results should cite an immutable dataset revision or tag. A new release
must receive a new tag whenever any graph, query list, candidate manifest,
evaluation label, or per-query prediction changes. Documentation-only changes
may update the default branch without altering an already published benchmark
tag.
