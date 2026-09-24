from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-9)


def encode_batch(model, tokenizer, texts: list[str], max_length: int, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    hidden = model(**tokens).last_hidden_state
    return F.normalize(mean_pool(hidden, tokens["attention_mask"]), dim=-1)


@torch.no_grad()
def encode_corpus(model, tokenizer, texts: Iterable[str], max_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
    rows = list(texts)
    chunks = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        chunks.append(encode_batch(model, tokenizer, rows[start : start + batch_size], max_length, device).cpu())
    return torch.cat(chunks) if chunks else torch.empty((0, model.config.hidden_size))


def multi_positive_nt_xent(scores: torch.Tensor, positive_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = scores / temperature
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    weights = positive_mask.to(log_probs.dtype)
    counts = weights.sum(1).clamp_min(1.0)
    return -((log_probs * weights).sum(1) / counts).mean()


def neural_ndcg_loss(scores: torch.Tensor, relevance: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable NDCG using smooth pairwise ranks.

    This is kept local and explicit so the exact objective is recorded with the
    experiment. Relevance must be 2 (core), 1 (superficial), or 0 (non-cite).
    """
    pairwise = (scores.unsqueeze(0) - scores.unsqueeze(1)) / temperature
    soft_rank = 1.0 + torch.sigmoid(pairwise).sum(dim=1) - 0.5
    gains = torch.pow(2.0, relevance) - 1.0
    dcg = (gains / torch.log2(soft_rank + 1.0)).sum()
    ideal = torch.sort(relevance, descending=True).values
    positions = torch.arange(1, len(ideal) + 1, device=scores.device, dtype=scores.dtype)
    idcg = ((torch.pow(2.0, ideal) - 1.0) / torch.log2(positions + 1.0)).sum().clamp_min(1e-9)
    return 1.0 - dcg / idcg


def freeze_bottom_encoder_layers(model, count: int) -> None:
    base = getattr(model, "bert", None)
    if base is None:
        base = getattr(model, "roberta", None)
    if base is None:
        base = model
    layers = getattr(getattr(base, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError(f"cannot locate transformer layers on {type(model).__name__}")
    for layer in list(layers)[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
