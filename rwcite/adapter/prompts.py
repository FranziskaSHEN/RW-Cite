"""Map citation-generation and relevance-analysis samples to chat messages."""

from __future__ import annotations

from rwcite.ranker.reference_recommend import (
    build_analyze_user_content,
    build_cite_bundle_user_content,
    build_cite_one_user_content,
    build_rerank_with_analysis_user_content,
    format_analysis_block,
    format_gold_output,
)


def sample_to_messages(sample: dict) -> list[dict[str, str]]:
    task = (sample.get("task") or "").strip()
    title = sample.get("title") or ""
    abstract = sample.get("abstract") or ""

    if task == "cite_bundle":
        selected = list(sample.get("selected") or [])
        user = build_cite_bundle_user_content(title, abstract, selected)
        assistant = format_gold_output(
            [
                {
                    "id": s.get("id") or "",
                    "title": s.get("title") or "",
                    "sentence": s.get("sentence") or "",
                }
                for s in selected
            ]
        )
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    if task == "cite_one":
        paper = sample.get("paper") or (sample.get("selected") or [{}])[0]
        user = build_cite_one_user_content(title, abstract, paper)
        assistant = (paper.get("sentence") or "").strip()
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    if task == "analyze":
        paper = sample.get("paper") or {}
        user = build_analyze_user_content(title, abstract, paper)
        assistant = (sample.get("assistant") or "").strip()
        if not assistant:
            assistant = format_analysis_block(
                sample.get("relevance") or "none",
                sample.get("reason") or "",
            )
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    if task in ("rerank30", "rerank10"):
        k = int(sample.get("k") or (10 if task == "rerank10" else 30))
        cands = list(sample.get("candidates") or [])
        user = build_rerank_with_analysis_user_content(
            title, abstract, cands, k=k
        )
        ids = list(sample.get("assistant_ids") or sample.get("target_ids") or [])
        assistant = "\n".join(ids[:k])
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    raise ValueError(f"unsupported RR adapter task: {task!r}")
