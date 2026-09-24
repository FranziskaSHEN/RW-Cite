# Source and artifact layout

RW-Cite is distributed as one installable Python package with thin shell
launchers for long-running workflows.

```text
RW-Cite/
  configs/                 Domain registry and publication contracts
  docs/                    Public documentation
  experiments/ewm_top10/   External comparison experiments
  rwcite/
    adapter/               Citation-sentence generation
    cli/                   Command-line entry points
    gat/                   Graph-attention scoring and fusion
    graph/                 Retrieval, extraction, and graph construction
    latex/                 LaTeX and citation-context cleaning helpers
    ranker/                Candidate construction and cross-encoder ranking
    retrieve/              Metadata and dense retrieval
    runtime/               Paths, configuration, and shared services
  scripts/                 Reproducible shell launchers
  tests/                   Unit and contract tests
```

Runtime-only directories are present as empty mount points in the source
release:

```text
datasets/   Downloaded corpus and benchmark assets
models/     Pretrained and trained model artifacts
logs/       Generated logs and reports
outputs/    Comparison outputs when created locally
```

These files are distributed separately or generated locally and must not be
committed to the source repository.

## Installation and imports

```bash
python -m pip install -e .
export RWCITE_ROOT=/path/to/RW-Cite
```

After editable installation, import directly from `rwcite`; do not add custom
package directories to `PYTHONPATH`.

```python
from rwcite.runtime.config import load_domains_settings
from rwcite.graph.domain_retrieve import retrieve_and_write
```

## Public entry points

| Task | Entry point |
|---|---|
| Validate installation | `bash scripts/run_tests.sh` |
| Link shared assets | `bash scripts/bootstrap_assets.sh` |
| Refresh metadata | `bash scripts/run_metadata_fetch.sh` |
| Build a domain graph | `bash scripts/run_domain_graph_build.sh --domain <tag>` |
| Validate a graph | `python -m rwcite.cli.validate_graph` |
| Export the split-masked graph | `python -m rwcite.cli.export_nofuture_gexf` |
| Train the structural shortlist | `bash scripts/run_domain_train_pool_ranker.sh` |
| Train the citation-aware cross-encoder | `python -m rwcite.cli.train_ce_sent` |
| Train/evaluate graph attention | `scripts/run_gat_train.sh` and `scripts/run_gat_eval.sh` |
| Apply paper-defined fusion | `bash scripts/run_score_rank_fusion.sh` |
| Run external baselines | `experiments/ewm_top10/run_baselines.sh` |
| Run the HLM-style judge | `experiments/ewm_top10/run_hlm_judge.sh` |
| Build/train/evaluate the citation adapter | `scripts/run_domain_rr_adapter_v6.sh` |

The publication contract and aggregate paper values are maintained in
`configs/paper_contract.yaml` and `configs/paper_results.yaml`. Artifact names
that predate the publication terminology may remain for checkpoint
compatibility; their meaning is defined by these contracts.

## Artifact identity

Every derived ranking artifact should preserve:

- the graph checksum and split identifier;
- hashes of the training, development, and test query lists;
- the ordered candidate-manifest checksum;
- the model configuration and checkpoint identifier;
- the random seed and code revision;
- complete per-query rankings and aggregate metrics.

Fixed-window comparisons additionally require identical candidate IDs for
every query. Evaluation code should reject a comparison if the graph, split,
query list, or candidate-window identities do not match.

## Tests

```bash
python -m pip install -e ".[dev]"
bash scripts/run_tests.sh
```

The tests cover configuration contracts, citation-context cleaning, retrieval
gates, split generation, command-line parsing, comparison safeguards, and
release hygiene.
