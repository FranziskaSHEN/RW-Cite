# Ten-domain benchmark contract

RW-Cite constructs domain benchmarks automatically from a shared corpus of
3,144,903 paper metadata records. Each domain is independently retrieved,
parsed, resolved into a citation graph, and partitioned into frozen training
and test queries. No manual relevance labels are required for benchmark
construction: citations observed in Introduction and Related Work-like
sections provide the supervision and evaluation labels.

This document describes only the ten domains reported in the paper. Other
entries in `configs/domains.yaml` are extension examples and are not part of
the released benchmark suite.

## Published domains

| Tag | Domain | Retrieved papers | Graph nodes | Citation edges | Train queries | Test queries |
|---|---|---:|---:|---:|---:|---:|
| `ewm` | Embodied World Models | 4,934 | 27,705 | 141,668 | 3,423 | 381 |
| `gw` | Gravitational-wave physics | 5,568 | 29,027 | 130,359 | 2,528 | 281 |
| `driving` | Autonomous driving | 4,385 | 26,952 | 109,473 | 2,946 | 328 |
| `cosmo` | Physical cosmology | 4,993 | 36,575 | 132,656 | 2,598 | 289 |
| `radio` | Radio astronomy | 7,172 | 24,165 | 65,214 | 1,460 | 163 |
| `wsi` | Computational pathology | 4,972 | 19,547 | 50,660 | 1,665 | 185 |
| `exo` | Exoplanet astronomy | 4,995 | 17,637 | 67,385 | 1,749 | 195 |
| `sqc` | Superconducting quantum computing | 7,891 | 20,470 | 87,273 | 2,636 | 293 |
| `fno` | Neural-operator turbulence | 4,978 | 21,050 | 58,375 | 1,962 | 219 |
| `sce` | Strongly correlated electrons | 9,008 | 34,265 | 97,078 | 1,832 | 204 |
| **Total** |  | **58,896** | **257,393** | **940,141** | **22,799** | **2,538** |

Counts are domain entries rather than globally unique papers because the same
paper may occur in more than one independently constructed domain graph. The
authoritative machine-readable values are in
[`../configs/paper_contract.yaml`](../configs/paper_contract.yaml).

## Construction contract

Every published domain follows the same automated procedure:

1. encode three-level topic representations from the shared metadata corpus;
2. retrieve recent papers with domain-specific prototype queries;
3. download and parse paper sources;
4. retain citations from Introduction and Related Work-like sections;
5. resolve cited papers to canonical identifiers and add cited-paper closure;
6. clean citation-context text without changing graph topology;
7. admit eligible query papers and freeze temporal partitions;
8. export the complete extraction graph and its split-masked scoring graph;
9. serialize query, candidate, gold-label, and provenance manifests.

The complete extraction graph is used to freeze observed citation labels. The
split-masked graph removes outgoing edges from test and frontier sources before
model construction and scoring. Query outgoing edges therefore remain labels,
not inference features.

## Domain configuration

Each domain has two configuration layers:

- `configs/domains.yaml` records the public tag and artifact paths;
- `configs/config_<tag>.yaml` records prototype queries, retrieval limits,
  date filters, and source-download locations.

To validate the currently configured graph:

```bash
python -m rwcite.cli.validate_graph --gexf /path/to/domain_graph.gexf
```

To reconstruct a registered domain:

```bash
bash scripts/run_domain_graph_build.sh \
  --domain <tag> --cutover --seed-only
```

The release manifest records the graph checksum, query identifiers, candidate
manifest, and associated configuration. Results should only be compared when
these identifiers match the declared evaluation contract.

## Extending the benchmark

The construction pipeline is not limited to the ten released domains. A new
domain can be added by defining prototype queries and retrieval constraints,
registering its artifact paths, and running the same graph-construction and
validation commands. New domains should be reported separately until their
frozen manifests and graph statistics have been released.
