# RW-Cite

RW-Cite is a multi-domain benchmark and reference framework for cold-start
citation recommendation from a manuscript title and abstract. It provides an
automated pipeline for constructing domain-specific citation graphs from
published papers, frozen benchmark splits, reproducible comparison baselines,
and an evidence-aware ranking framework.

**Anonymous review repository:**
[https://anonymous.4open.science/r/RW-Cite/](https://anonymous.4open.science/r/RW-Cite/)

The reference framework first learns a structural Top-400 shortlist. It then
scores the same candidates with a citation-aware SciBERT cross-encoder and a
graph-attention network, combining both channels with the fixed score-and-rank
fusion defined in the paper. An optional downstream adapter generates
candidate-grounded citation sentences without changing the ranked references.

## Benchmark inventory

The public benchmark contains the ten domains reported in the paper. Counts
are domain entries and may overlap across independently constructed graphs.

| Domain | Retrieved papers | Graph nodes | Citation edges | Train queries | Test queries |
|---|---:|---:|---:|---:|---:|
| Embodied World Models (EWM) | 4,934 | 27,705 | 141,668 | 3,423 | 381 |
| Gravitational-wave physics (GW) | 5,568 | 29,027 | 130,359 | 2,528 | 281 |
| Autonomous driving | 4,385 | 26,952 | 109,473 | 2,946 | 328 |
| Physical cosmology | 4,993 | 36,575 | 132,656 | 2,598 | 289 |
| Radio astronomy | 7,172 | 24,165 | 65,214 | 1,460 | 163 |
| Computational pathology (WSI) | 4,972 | 19,547 | 50,660 | 1,665 | 185 |
| Exoplanet astronomy | 4,995 | 17,637 | 67,385 | 1,749 | 195 |
| Superconducting quantum computing (SQC) | 7,891 | 20,470 | 87,273 | 2,636 | 293 |
| Neural-operator turbulence (FNO) | 4,978 | 21,050 | 58,375 | 1,962 | 219 |
| Strongly correlated electrons (SCE) | 9,008 | 34,265 | 97,078 | 1,832 | 204 |
| **Total** | **58,896** | **257,393** | **940,141** | **22,799** | **2,538** |

The machine-readable publication contract is
[`configs/paper_contract.yaml`](configs/paper_contract.yaml), and the aggregate
values reported in the paper are recorded in
[`configs/paper_results.yaml`](configs/paper_results.yaml). Frozen per-query
artifacts in the dataset release remain the source of truth.

## Repository contents

```text
configs/                 Domain registry and paper-facing contracts
docs/                    Public construction and reproduction guides
experiments/ewm_top10/   External comparison experiments
rwcite/                  Installable Python package
scripts/                 Command-line launchers and release utilities
tests/                   Unit and contract tests
```

Large datasets, model checkpoints, logs, and generated outputs are not stored
in the source repository. They are distributed separately through the RW-Cite
dataset release.

## Installation

RW-Cite requires Python 3.10 or newer. Python 3.11 is recommended.

```bash
# Download and extract the source from the anonymous review repository above.
cd RW-Cite

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Install training and development dependencies only when needed:

```bash
python -m pip install -e ".[train]"
python -m pip install -e ".[dev]"
```

Run the public test suite with:

```bash
bash scripts/run_tests.sh
```

## Data and pretrained models

Place or link runtime assets beneath the checkout:

```text
datasets/arxiv-metadata-oai-snapshot.json
datasets/arxiv_topics/
datasets/topic_level_embeds/
models/base/bge-large-en-v1.5/
models/base/scibert_scivocab_uncased/
```

If these assets already exist in a shared data root, link them into the
checkout without copying multi-gigabyte files:

```bash
RWCITE_ASSET_SOURCE=/path/to/rwcite-assets \
  bash scripts/bootstrap_assets.sh
```

The benchmark graphs, splits, candidate manifests, and evaluation artifacts
are provided in the companion Hugging Face dataset. See
[`docs/DATASETS_HF_UPLOAD.md`](docs/DATASETS_HF_UPLOAD.md) for the public data
layout and current `hf` download commands.

## Constructing a domain benchmark

Domains are registered in [`configs/domains.yaml`](configs/domains.yaml), with
retrieval settings in `configs/config_<tag>.yaml`. To build a registered domain
graph:

```bash
bash scripts/run_domain_graph_build.sh \
  --domain ewm --cutover --seed-only
```

The pipeline performs multi-view topic retrieval, downloads and parses paper
sources, resolves citations from Introduction and Related Work-like sections,
adds resolvable cited-paper closure, cleans citation contexts, and writes the
domain graph. See [`docs/DOMAIN_GRAPH.md`](docs/DOMAIN_GRAPH.md) for the data
contract, outputs, and validation commands.

## Reference framework

The paper-facing framework uses the following fixed sequence:

1. admit and freeze temporally ordered query partitions;
2. remove outgoing edges from test and frontier sources in the scoring graph;
3. learn a structural Top-400 shortlist;
4. train the citation-aware SciBERT cross-encoder with random in-window
   negatives;
5. train the two-layer graph-attention scorer on the same candidate window;
6. combine both scores with fixed score-and-rank fusion.

The authoritative hyperparameters are in
[`configs/paper_contract.yaml`](configs/paper_contract.yaml). The final fusion
can be reproduced from compatible cross-encoder and GAT artifacts with:

```bash
DOMAIN=ewm bash scripts/run_score_rank_fusion.sh
```

Some artifact filenames retain historical identifiers for compatibility, but
public documentation uses the paper terminology throughout. See
[`docs/LAYOUT.md`](docs/LAYOUT.md) for the supported modules and artifact
locations.

## External comparisons

The comparison package includes BM25, pretrained BGE, unfine-tuned SciBERT, a
MasterSet-style SciBERT--NTX retriever, and an HLM-style pipeline whose trained
GTE retriever supplies the Analyzer--Decider Top-30 window.

```bash
export BENCHMARK_DATA_ROOT=/path/to/unpacked/benchmark
export EWM_GRAPH=/path/to/ewm_graph.gexf
export MODEL_ROOT=/path/to/models
export EVAL_PROTOCOL=development

bash experiments/ewm_top10/run_baselines.sh
bash experiments/ewm_top10/run_reference_baselines.sh
```

Full setup and API-based evaluation instructions are in
[`experiments/ewm_top10/README.md`](experiments/ewm_top10/README.md).

## Optional citation-sentence generation

The citation-sentence adapter receives a fixed ranked Top-10 and generates one
candidate-grounded sentence per reference. It cannot add, remove, or reorder
papers, and its generation metrics are separate from recommendation metrics.

```bash
DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh dump_pools
DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh build
DOMAIN=ewm NPROC=8 bash scripts/run_domain_rr_adapter_v6.sh train
DOMAIN=ewm NPROC=8 bash scripts/run_domain_rr_adapter_v6.sh eval
```

The `v6` suffix is retained only in implementation filenames for artifact
compatibility. See [`docs/RR_ADAPTER_V6.md`](docs/RR_ADAPTER_V6.md) for the
public method description.

## Documentation

- [Ten-domain benchmark contract](docs/DOMAIN_CONFIGS.md)
- [Domain-graph construction](docs/DOMAIN_GRAPH.md)
- [Source and artifact layout](docs/LAYOUT.md)
- [Citation-sentence adapter](docs/RR_ADAPTER_V6.md)
- [Hugging Face dataset layout](docs/DATASETS_HF_UPLOAD.md)
- [Comparison experiments](experiments/ewm_top10/README.md)

## Reproducibility

Every reported run should retain query lists, graph and candidate-manifest
checksums, model configuration, random seed, per-query rankings, and aggregate
metrics. Evaluation scripts verify query identity and candidate-window
compatibility before comparing methods. Dataset artifacts must be cited by an
immutable revision or release tag.

## License

Source code is released under the MIT License. Dataset components may inherit
additional terms from their upstream sources; consult the dataset card and
manifest before redistribution.

## Citation

Citation metadata will be added after the anonymous review period.
