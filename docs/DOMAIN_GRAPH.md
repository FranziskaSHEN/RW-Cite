# Domain-graph construction

This guide describes the automated graph-construction stage used by the
published RW-Cite benchmarks.

## Graph schema

Each domain graph is directed. A node represents a resolved paper and stores
its canonical identifier, title, abstract when available, publication date,
and source metadata. An edge `(u, v)` records that paper `u` cites paper `v`
in its Introduction or a Related Work-like section. Resolved edges retain the
local citation context and its cleaned representation.

The builder adds resolvable cited papers even when they were not retrieved as
domain seeds. This cited-paper closure preserves older foundational work and
prevents valid labels from disappearing solely because a cited paper falls
outside the initial retrieval set.

## Required assets

The default configuration expects:

```text
datasets/arxiv-metadata-oai-snapshot.json
datasets/arxiv_topics/
datasets/topic_level_embeds/
models/base/bge-large-en-v1.5/
```

Domain prototype queries and retrieval constraints are defined in
`configs/config_<tag>.yaml`; paths are registered in `configs/domains.yaml`.

## Build a registered domain

```bash
bash scripts/run_domain_graph_build.sh \
  --domain <tag> --cutover --seed-only
```

Important options:

| Option | Effect |
|---|---|
| `--domain <tag>` | Select a key from `configs/domains.yaml` |
| `--cutover` | Update the registered graph path after a successful build |
| `--seed-only` | Build from retrieved seeds plus cited-paper closure |
| `--expand` | Enable the optional budgeted expansion stage |
| `--skip-retrieval` | Reuse an existing retrieval manifest |
| `--skip-download` | Reuse previously downloaded paper sources |
| `--reset` | Rebuild the graph checkpoint for the selected domain |

The underlying Python entry point is:

```bash
python -m rwcite.cli.graph_pipeline --help
```

## Processing stages

1. **Domain retrieval.** Encode three-level topic text and domain prototype
   queries with BGE, then merge and deduplicate the selected results.
2. **Source acquisition.** Download paper sources incrementally while reusing
   existing archives and extracted sources.
3. **Section-aware extraction.** Parse citations from Introduction and Related
   Work-like sections and resolve bibliography entries to canonical IDs.
4. **Cited-paper closure.** Add resolved cited papers that were not part of the
   initial retrieval seeds.
5. **Context cleaning.** Remove layout commands, comments, labels, malformed
   prefixes, and other non-semantic fragments without changing graph topology.
6. **Validation.** Record node and edge counts, citation-context coverage,
   retrieval coverage, and a graph checksum.

## Expected artifacts

Paths depend on the selected domain configuration, but a completed build
contains the following logical artifacts:

| Artifact | Purpose |
|---|---|
| `description/test_graph_rr.gexf` | Complete extracted domain graph |
| `retrieval_nodes.json` | Deduplicated retrieval seeds |
| `failed_downloads.json` | Sources that could not be acquired |
| `description/rebuild_year10/` | Resume checkpoint and build summary |
| `data/splits/` | Frozen query partitions |
| `data/reference_recommend/` | Serialized ranking examples |
| `data/domain_embeddings.parquet` | Domain embedding cache |
| `data/retrieval_diag.json` | Retrieval and graph diagnostics |

Generated datasets, sources, checkpoints, and logs are runtime artifacts and
are intentionally excluded from the GitHub source release.

## Validation

Validate a completed graph before producing benchmark splits:

```bash
python -m rwcite.cli.validate_graph \
  --gexf /path/to/domain_graph.gexf
```

Useful diagnostics include the number of retained retrieval seeds, resolved
edge coverage, citation-context coverage, admitted-query coverage, duplicate
identifiers, and the graph checksum. The checksum must be carried into every
split and evaluation manifest derived from the graph.

## Temporal benchmark export

Query admission is computed on the complete extraction graph. The scoring
graph is then produced by removing outgoing edges from test and frontier
sources. Training and inference features are constructed from this split-masked
graph, while frozen citations from the complete graph remain evaluation labels.

The public dataset release includes both the graph identity and the frozen
split manifests required to audit this separation.
