# Citation-sentence adapter

The optional citation-sentence adapter is downstream of recommendation. It
receives the reference framework's fixed ranked Top-10 and generates one
candidate-grounded sentence for each selected paper. It cannot add, remove, or
reorder references, so generation quality is evaluated separately from
ranking quality.

The `v6` suffix in implementation filenames is retained for compatibility
with frozen artifacts and is not a method name.

## Model and supervision

The adapter fine-tunes Qwen3-32B with quantized low-rank updates on the
attention projection layers. Training examples are derived only from training
queries. For each query, the score-and-rank fusion output is truncated to a
Top-50 pool and intersected with observed training citations. Prompts contain
the query title and abstract together with candidate metadata; targets are
cleaned citation-context sentences associated with those candidates.

| Setting | Value |
|---|---|
| Base model | Qwen3-32B |
| Updated modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA rank / scaling / dropout | 16 / 64 / 0.05 |
| Quantization / compute | 8-bit / bfloat16 |
| Maximum sequence length | 2,048 |
| Training epochs | 2 |
| Per-device batch / accumulation | 1 / 8 |
| Training candidate pool | Fixed fused Top-50 |
| Inference input | Fixed ranked Top-10 |
| Hidden reasoning | Disabled |

The authoritative configuration is
[`../configs/paper_contract.yaml`](../configs/paper_contract.yaml).

## Training and evaluation

Install the optional training dependencies and obtain the base model before
launching the adapter:

```bash
python -m pip install -e ".[train]"
python scripts/download_qwen3_32b.py
```

Run the four stages for a configured domain:

```bash
DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh dump_pools
DOMAIN=ewm bash scripts/run_domain_rr_adapter_v6.sh build
DOMAIN=ewm NPROC=8 bash scripts/run_domain_rr_adapter_v6.sh train
DOMAIN=ewm NPROC=8 bash scripts/run_domain_rr_adapter_v6.sh eval
```

Or run them sequentially:

```bash
DOMAIN=ewm NPROC=8 bash scripts/run_domain_rr_adapter_v6.sh all
```

Training and evaluation query IDs must be disjoint. Evaluation gold sentences
must never be included in prompts or training examples.

## Inference safeguards

The adapter generates a citation bundle in the ranker's fixed order. A
sentence that is inconsistent with its candidate title or nearly identical to
an earlier sentence in the same bundle is regenerated with a
candidate-specific prompt. The final artifact records the selected candidate
ID, generated sentence, validation outcome, and any regeneration decision.

## Metrics

The public evaluation reports:

- nonempty-output rate;
- candidate-ID consistency;
- title consistency;
- title-keyword alignment;
- pairwise distinctness;
- citation-context BERTScore F1.

These measurements assess output validity, grounding, diversity, and
reference-based semantic similarity. They do not constitute a ranking gain.
The aggregate values reported in the paper are recorded in
[`../configs/paper_results.yaml`](../configs/paper_results.yaml).
