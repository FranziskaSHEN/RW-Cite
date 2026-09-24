"""Lazy BERTScore helper (AS-aligned L2 cite metric)."""

from __future__ import annotations

import os
import threading
from typing import Any


class BertScoreService:
    """Lazy singleton BERTScorer (bert_score, lang=en → RoBERTa-large)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scorer: Any = None

    def _get_scorer(self):
        if self._scorer is not None:
            return self._scorer
        with self._lock:
            if self._scorer is not None:
                return self._scorer
            from bert_score import BERTScorer

            # Prefer local HF cache (HF_HOME); optional override for offline/airgap.
            model_type = os.environ.get("RWCITE_BERTSCORE_MODEL", "").strip() or None
            kwargs: dict[str, Any] = {"lang": "en", "batch_size": 32, "device": None}
            if model_type:
                kwargs["model_type"] = model_type
            # Avoid re-hitting the hub when weights are already cached.
            if os.environ.get("RWCITE_BERTSCORE_LOCAL_ONLY", "1") == "1":
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self._scorer = BERTScorer(**kwargs)
            return self._scorer

    def score_many(
        self, candidates: list[str], references: list[str]
    ) -> list[dict[str, float]]:
        if len(candidates) != len(references):
            raise ValueError("candidates and references length mismatch")
        if not candidates:
            return []
        cands = [(c or "").strip() for c in candidates]
        refs = [(r or "").strip() for r in references]
        if any(not c or not r for c, r in zip(cands, refs)):
            raise ValueError("candidate and reference must be non-empty")
        scorer = self._get_scorer()
        p, r, f = scorer.score(cands, refs)
        return [
            {
                "precision": float(p[i].item()),
                "recall": float(r[i].item()),
                "f1": float(f[i].item()),
            }
            for i in range(len(cands))
        ]


_bertscore_service: BertScoreService | None = None


def get_bertscore_service() -> BertScoreService:
    global _bertscore_service
    if _bertscore_service is None:
        _bertscore_service = BertScoreService()
    return _bertscore_service
