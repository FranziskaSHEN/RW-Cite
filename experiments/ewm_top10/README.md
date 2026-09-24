# EWM comparison experiments

This package reproduces the external baselines used to evaluate RW-Cite on a
shared EWM benchmark contract. It keeps corpus retrieval, controlled reranking,
and LLM-only reasoning as separate settings so that candidate availability is
not conflated with ordering quality.

## Included methods

- deterministic BM25, pretrained BGE, and unfine-tuned SciBERT controls;
- a MasterSet-style SciBERT bi-encoder trained with multi-positive NT-Xent;
- an HLM-style two-stage GTE retriever with graded supervision;
- a complete HLM-style pipeline in which the trained GTE retriever supplies
  the frozen Top-30 window consumed by the Analyzer--Decider; and
- an audited RW-Cite runner for the native structural Top-400 pipeline.

The HLM-style retriever preserves the source method's retrospective
co-citation supervision and is reported as a supplementary baseline. The
Analyzer--Decider cannot retrieve papers outside the Top-30 returned by the
HLM-style retriever.

## Evaluation contract

- All methods use the same query records and full-gold labels within a table.
- Candidate publication eligibility is applied before dense scoring.
- Same-month and unknown-date candidate evidence is excluded.
- Query outgoing citation edges are labels only and are not ranking inputs.
- Learned baselines use seeds 42, 43, and 44.
- Each Analyzer--Decider uses a fixed model identifier, prompt, reviewed
  training-only example, and temperature of zero in one complete evaluation.
- Per-query rankings are retained so every aggregate metric can be recomputed.

## Dense and retrieval baselines

Set the benchmark, graph, and model roots explicitly, then launch the trained
baselines:

```bash
export BENCHMARK_DATA_ROOT=/path/to/unpacked/benchmark
export EWM_GRAPH=/path/to/ewm_graph.gexf
export MODEL_ROOT=/path/to/models
export EVAL_PROTOCOL=development

bash experiments/ewm_top10/run_baselines.sh
bash experiments/ewm_top10/run_reference_baselines.sh
```

`run_baselines.sh` prepares the frozen data contract and neutral Top-400
window before training SciBERT-NTX and the HLM-style retriever. It then freezes
the HLM retriever's temporally admissible Top-30 for the Analyzer--Decider.
`run_reference_baselines.sh` adds BM25, pretrained BGE, and unfine-tuned
SciBERT under the same contract. Outputs are written beneath
`outputs/ewm_top10/` and are ignored by Git.

## HLM-style Analyzer--Decider

Run a smoke test before the complete evaluation:

```bash
export BASE_ROOT=/path/to/comparison/output
export EVAL_PROTOCOL=development
export HLM_MODEL=your-model-id
export HLM_RUN_LABEL=your-model-label
export HLM_ENDPOINT=http://host:port/v1
export HLM_API_KEY=your-token
export HLM_ONE_SHOT=/path/to/reviewed_training_example.txt
export HLM_COST_STATUS=not_provided

MODE=smoke bash experiments/ewm_top10/run_hlm_judge.sh
MODE=full bash experiments/ewm_top10/run_hlm_judge.sh
```

For endpoints whose reasoning mode suppresses structured response content, set
`HLM_DISABLE_THINKING=1`. API responses are cached to make interrupted runs
resumable without silently changing completed predictions. The launcher uses
the seed-42 HLM retriever by default and constructs its frozen Top-30 from the
existing retriever predictions if necessary. Set `HLM_RETRIEVER_SEED` only to
select another already trained seed deliberately.

## RW-Cite comparison runner

The native runner validates graph, split, checkpoint, fusion, and candidate
manifests before evaluation. Start it only with a reviewed configuration:

```bash
GPUS=4,5,6,7 bash scripts/launch_rwcite_comparison.sh \
  experiments/ewm_top10/configs/rw_cite.yaml \
  outputs/ewm_top10/rwcite_comparison
```

The output directory must be new and contained within the checkout. The runner
records provenance and refuses incompatible graph fingerprints or overlapping
query partitions.
