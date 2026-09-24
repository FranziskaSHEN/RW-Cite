"""Load and merge base arXiv Topics corpus with local supplement files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

TOPICS_SUPPLEMENT = "datasets/arxiv_topics_supplement.jsonl"
EMBED_SUPPLEMENT = "datasets/topic_level_embeds/supplement_embeddings.parquet"
CORPUS_STATE = "datasets/corpus_state.json"


def _hf_load_dataset(*args, **kwargs):
    """Import HuggingFace datasets, bypassing the local datasets/ data directory."""
    # cwd often includes ./datasets/ (no __init__.py), which shadows the HF package.
    blocked = {"", os.getcwd()}
    root = os.environ.get("RWCITE_ROOT")
    if root:
        blocked.add(root)
    saved = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if p not in blocked]
        from datasets import load_dataset
    finally:
        sys.path[:] = saved
    return load_dataset(*args, **kwargs)


def load_corpus_state() -> dict:
    if not os.path.exists(CORPUS_STATE):
        return {}
    with open(CORPUS_STATE, encoding="utf-8") as f:
        return json.load(f)


def save_corpus_state(state: dict) -> None:
    os.makedirs(os.path.dirname(CORPUS_STATE) or ".", exist_ok=True)
    with open(CORPUS_STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_supplement_topics() -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not os.path.exists(TOPICS_SUPPLEMENT):
        return records
    with open(TOPICS_SUPPLEMENT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            records[entry["paper_id"]] = entry
    return records


def append_supplement_topics(entries: list[dict]) -> int:
    """Append topic records to supplement jsonl. Returns number appended."""
    if not entries:
        return 0
    os.makedirs(os.path.dirname(TOPICS_SUPPLEMENT) or ".", exist_ok=True)
    existing = load_supplement_topics()
    appended = 0
    with open(TOPICS_SUPPLEMENT, "a", encoding="utf-8") as f:
        for entry in entries:
            pid = entry["paper_id"]
            if pid in existing:
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing[pid] = entry
            appended += 1
    return appended


def load_hf_topics(cache_dir: str = "datasets/arxiv_topics"):
    return _hf_load_dataset("AliMaatouk/arXiv_Topics", cache_dir=cache_dir)


def build_id2topics(cache_dir: str = "datasets/arxiv_topics") -> dict[str, list]:
    """Merge HF Topics and local supplement into id -> [L1, L2, L3]."""
    concept_data = load_hf_topics(cache_dir=cache_dir)
    id2topics = {
        entry["paper_id"]: [entry["Level 1"], entry["Level 2"], entry["Level 3"]]
        for entry in concept_data["train"]
    }
    for pid, entry in load_supplement_topics().items():
        id2topics[pid] = [entry["Level 1"], entry["Level 2"], entry["Level 3"]]
    return id2topics


def build_paper_list(cache_dir: str = "datasets/arxiv_topics") -> list[str]:
    """Paper ids in retrieval order: HF corpus then sorted supplement ids."""
    concept_data = load_hf_topics(cache_dir=cache_dir)
    base_ids = list(concept_data["train"]["paper_id"])
    base_set = set(base_ids)
    supplement_ids = sorted(
        pid for pid in load_supplement_topics() if pid not in base_set
    )
    return base_ids + supplement_ids


def topic_text_for_level(topics: list[str]) -> str:
    return "".join(t + "," for t in topics)


def encode_topic_levels(
    model,
    tokenizer,
    paper_ids: list[str],
    id2topics: dict[str, list],
    batch_size: int = 256,
) -> np.ndarray:
    """BGE-encode L1+L2+L3 topic text and sum (same as retriever.py)."""
    import torch

    level_embs = []
    for level_idx in range(3):
        batch_embs: list[torch.Tensor] = []
        for i in tqdm(range(0, len(paper_ids), batch_size), desc=f"Level {level_idx + 1}"):
            batch_ids = paper_ids[i : i + batch_size]
            texts = [
                topic_text_for_level(id2topics[pid][level_idx]) for pid in batch_ids
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs.to("cuda"))
                emb = outputs.last_hidden_state[:, 0, :].cpu()
            batch_embs.append(emb)
        level_embs.append(torch.cat(batch_embs, dim=0))
    combined = level_embs[0] + level_embs[1] + level_embs[2]
    return combined.numpy()


def load_hf_embeddings(cache_dir: str = "datasets/topic_level_embeds") -> tuple[list[str], np.ndarray]:
    dataset = _hf_load_dataset("AliMaatouk/arXiv-Topics-Embeddings", cache_dir=cache_dir)
    table = dataset["train"]
    paper_ids = list(table["paper_id"])
    embeddings = np.stack(table["embedding"])
    return paper_ids, embeddings


def load_supplement_embeddings() -> tuple[list[str], np.ndarray]:
    if not os.path.exists(EMBED_SUPPLEMENT):
        return [], np.empty((0, 1024), dtype=np.float32)
    df = pd.read_parquet(EMBED_SUPPLEMENT)
    paper_ids = list(df["paper_id"])
    embeddings = np.stack(df["embedding"].tolist())
    return paper_ids, embeddings


def append_supplement_embeddings(paper_ids: list[str], embeddings: np.ndarray) -> int:
    """Append new rows to supplement embedding parquet."""
    if len(paper_ids) == 0:
        return 0
    os.makedirs(os.path.dirname(EMBED_SUPPLEMENT) or ".", exist_ok=True)
    new_df = pd.DataFrame({"paper_id": paper_ids, "embedding": list(embeddings)})
    if os.path.exists(EMBED_SUPPLEMENT):
        old_df = pd.read_parquet(EMBED_SUPPLEMENT)
        existing = set(old_df["paper_id"])
        new_df = new_df[~new_df["paper_id"].isin(existing)]
        if new_df.empty:
            return 0
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_parquet(EMBED_SUPPLEMENT, engine="pyarrow", compression="snappy")
    return len(new_df)


def load_merged_embeddings(
    use_hf_embeds: bool,
    cache_dir: str = "datasets/topic_level_embeds",
    local_parquet: str = "datasets/topic_level_embeds/arxiv_papers_embeds.parquet",
) -> tuple[list[str], np.ndarray]:
    """Load retrieval embeddings: HF/local base + supplement."""
    supplement_ids, supplement_embs = load_supplement_embeddings()
    supplement_map = dict(zip(supplement_ids, supplement_embs))

    if use_hf_embeds:
        base_ids, base_embs = load_hf_embeddings(cache_dir=cache_dir)
    elif os.path.exists(local_parquet):
        df = pd.read_parquet(local_parquet)
        base_ids = list(df["paper_id"])
        base_embs = np.stack(df["embedding"].tolist())
    else:
        raise FileNotFoundError(
            "No base embeddings found. Enable load_arxiv_embeds or generate local parquet first."
        )

    base_set = set(base_ids)
    extra_ids = sorted(pid for pid in supplement_map if pid not in base_set)
    if not extra_ids:
        return base_ids, base_embs

    extra_embs = np.stack([supplement_map[pid] for pid in extra_ids])
    merged_ids = base_ids + extra_ids
    merged_embs = np.vstack([base_embs, extra_embs])
    return merged_ids, merged_embs


def load_metadata_index(path: str = "datasets/arxiv-metadata-oai-snapshot.json") -> dict[str, dict]:
    papers: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            papers[entry["id"]] = entry
    return papers
