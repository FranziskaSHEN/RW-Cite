"""Cross-encoder RR pool ranker (query × candidate joint scoring)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def rr_ce_params() -> dict[str, Any]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    def _f(name: str, default: float) -> float:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    # Opt-in only: explicit path or RR_USE_CE=1 (avoids broken CE shadowing MLP).
    explicit = (os.environ.get("RR_RANKER_CE_PATH") or "").strip()
    use = (os.environ.get("RR_USE_CE") or "").strip().lower() in ("1", "true", "yes")
    if explicit:
        path = explicit
    elif use:
        path = "models/base/scibert_scivocab_uncased"
    else:
        path = ""

    return {
        "path": path,
        "prefilter": _i("RR_CE_PREFILTER", 500),
        "batch_size": _i("RR_CE_BATCH", 32),
        "max_length": _i("RR_CE_MAX_LEN", 256),
        "full_u": (os.environ.get("RR_CE_FULL_U") or "").strip().lower()
        in ("1", "true", "yes"),
        # blend CE logit with -log(struct_rank+1); 0 = pure CE
        "struct_blend": _f("RR_CE_STRUCT_BLEND", 0.0),
    }


def _query_text(title: str, abstract: str) -> str:
    return f"{title or ''}\n{(abstract or '')}".strip()[:1200]


def _cand_text(c: dict[str, Any]) -> str:
    return f"{c.get('title') or ''}\n{c.get('abstract') or ''}".strip()[:1200]


class CERanker:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | None = None,
        max_length: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_length = int(max_length)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        max_length: int = 256,
    ) -> "CERanker":
        path = Path(path)
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        model.to(dev)
        return cls(model, tok, device=dev, max_length=max_length)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    @torch.inference_mode()
    def score_pairs(
        self,
        query: str,
        cand_texts: list[str],
        *,
        batch_size: int = 32,
    ) -> list[float]:
        if not cand_texts:
            return []
        out: list[float] = []
        q = (query or "").strip()
        bs = max(1, int(batch_size))
        for i in range(0, len(cand_texts), bs):
            batch = cand_texts[i : i + bs]
            enc = self.tokenizer(
                [q] * len(batch),
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.squeeze(-1)
            out.extend(float(x) for x in logits.detach().float().cpu().tolist())
        return out


_CE_CACHE: CERanker | None = None
_CE_PATH: str | None = None


def load_ce_ranker(path: str | None = None) -> CERanker | None:
    """Load CE from RR_RANKER_CE_PATH / default; None if missing."""
    global _CE_CACHE, _CE_PATH
    params = rr_ce_params()
    p = (path or params["path"] or "").strip()
    if not p:
        return None
    resolved = str(Path(p).resolve()) if Path(p).exists() else p
    # require config.json so empty dirs don't "load"
    cfg = Path(resolved) / "config.json"
    if not cfg.is_file():
        return None
    if _CE_CACHE is not None and _CE_PATH == resolved:
        return _CE_CACHE
    try:
        ce = CERanker.from_pretrained(
            resolved, max_length=int(params["max_length"])
        )
    except Exception:  # noqa: BLE001
        return None
    _CE_CACHE = ce
    _CE_PATH = resolved
    return ce


def rank_universe_ce(
    title: str,
    abstract: str,
    candidates: list[dict[str, Any]],
    ce: CERanker,
    *,
    n: int,
    batch_size: int = 32,
    struct_blend: float = 0.0,
    struct_ranks: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    import math

    q = _query_text(title, abstract)
    texts = [_cand_text(c) for c in candidates]
    scores = ce.score_pairs(q, texts, batch_size=batch_size)
    blend = float(struct_blend or 0.0)
    scored = []
    for c, sc in zip(candidates, scores):
        final = float(sc)
        if blend > 0.0 and struct_ranks is not None:
            r = struct_ranks.get(c["id"])
            if r is not None:
                final = final + blend * (-math.log1p(float(r)))
        scored.append(
            (
                final,
                {
                    "id": c["id"],
                    "title": c.get("title") or "",
                    "abstract": c.get("abstract") or "",
                    "source": c.get("source") or "ranked",
                    "score": final,
                    "ce_score": float(sc),
                },
            )
        )
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:n]]
