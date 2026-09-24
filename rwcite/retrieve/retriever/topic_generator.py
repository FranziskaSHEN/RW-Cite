"""Generate arXiv Topics-style L1/L2/L3 labels for new papers."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

TOPIC_PROMPT = """You classify arXiv papers into a three-level topic hierarchy.
Given a paper title and abstract, return exactly 3 topic labels for each level.

Level 1: broad academic domains (e.g., Computer Science, Physics, Mathematics, Engineering)
Level 2: intermediate subfields (e.g., Robotics, Machine Learning, Condensed Matter Physics)
Level 3: specific research topics (e.g., Robot Manipulation, Deep Learning Optimization)

Rules:
- Each level must contain exactly 3 distinct string labels.
- Labels should match the style of arXiv Topics dataset: short noun phrases, title case.
- Reflect the paper's primary and secondary fields (multi-label, cross-disciplinary allowed).

Respond with JSON only, no markdown:
{{"Level 1": ["...", "...", "..."], "Level 2": ["...", "...", "..."], "Level 3": ["...", "...", "..."]}}

Title: {title}

Abstract: {abstract}
"""

DEFAULT_LLM_BASE_URL = ""
DEFAULT_LLM_MODEL = "GLM-5.1"
DEFAULT_LLM_API_KEY_ENV = "RWCITE_LLM_API_KEY"


def get_llm_settings(config: dict | None = None) -> dict:
    cfg = (config or {}).get("corpus_update", {})
    return {
        "base_url": os.environ.get("RWCITE_LLM_BASE_URL", cfg.get("llm_base_url", DEFAULT_LLM_BASE_URL)),
        "model": os.environ.get("RWCITE_LLM_MODEL", cfg.get("llm_model", DEFAULT_LLM_MODEL)),
        "api_key_env": cfg.get("llm_api_key_env", DEFAULT_LLM_API_KEY_ENV),
    }


def make_llm_client(config: dict | None = None):
    """Create a client for a user-configured OpenAI-compatible endpoint."""
    from openai import OpenAI

    settings = get_llm_settings(config)
    if not settings["base_url"]:
        raise ValueError(
            "Set RWCITE_LLM_BASE_URL or corpus_update.llm_base_url before "
            "using the LLM topic backend."
        )
    api_key = (os.environ.get(settings["api_key_env"]) or os.environ.get("OPENAI_API_KEY") or "not-needed").strip()
    return OpenAI(base_url=settings["base_url"], api_key=api_key), settings["model"]


def _normalize_levels(raw: dict) -> dict:
    result = {}
    for level in ("Level 1", "Level 2", "Level 3"):
        values = raw.get(level, [])
        if isinstance(values, str):
            values = [values]
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if len(cleaned) < 3:
            while len(cleaned) < 3:
                cleaned.append(cleaned[-1] if cleaned else "General Research")
        result[level] = cleaned[:3]
    return result


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def generate_topics_llm(
    title: str,
    abstract: str,
    model: str | None = None,
    client=None,
    config: dict | None = None,
) -> dict:
    if client is None:
        client, default_model = make_llm_client(config)
        model = model or default_model
    prompt = TOPIC_PROMPT.format(title=title[:2000], abstract=abstract[:4000])
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(**kwargs)
    raw = _parse_json_response(response.choices[0].message.content or "{}")
    return _normalize_levels(raw)


# Backward-compatible alias
generate_topics_openai = generate_topics_llm


def generate_topics_heuristic(title: str, abstract: str, categories: str = "") -> dict:
    """Fallback when LLM is unavailable: map arXiv categories + keywords to L1/L2/L3."""
    cat_map = {
        "cs": ("Computer Science", "Artificial Intelligence", "Computer Science"),
        "math": ("Mathematics", "Mathematics", "Mathematical Analysis"),
        "physics": ("Physics", "Physics", "Theoretical Physics"),
        "astro-ph": ("Astronomy", "Astrophysics", "Astrophysics"),
        "cond-mat": ("Physics", "Condensed Matter Physics", "Materials Science"),
        "quant-ph": ("Physics", "Quantum Physics", "Quantum Mechanics"),
        "hep-ph": ("Physics", "Particle Physics", "High-Energy Physics"),
        "hep-th": ("Physics", "Theoretical Physics", "Quantum Field Theory"),
        "gr-qc": ("Physics", "General Relativity", "Theoretical Physics"),
        "stat": ("Statistics", "Statistics", "Statistical Inference"),
        "q-bio": ("Biology", "Computational Biology", "Biological Modeling"),
        "q-fin": ("Economics", "Finance", "Financial Modeling"),
        "eess": ("Engineering", "Signal Processing", "Electrical Engineering"),
    }
    sub_map = {
        "cs.RO": ("Robotics", "Robot Manipulation", "Autonomous Systems"),
        "cs.LG": ("Machine Learning", "Deep Learning Techniques", "Neural Network Optimization"),
        "cs.CV": ("Computer Vision", "Image Processing", "Visual Recognition"),
        "cs.CL": ("Natural Language Processing", "Language Model Evaluation", "Text Analysis"),
        "cs.AI": ("Artificial Intelligence", "AI Applications", "Intelligent Systems"),
    }
    primary = categories.split()[0] if categories else "cs.LG"
    prefix = primary.split(".")[0]
    l1, l2_default, l3_default = cat_map.get(prefix, ("Computer Science", "Computer Science", "Research Methods"))
    l2_triple = list(sub_map.get(primary, (l2_default, l3_default, l1)))
    text = (title + " " + abstract).lower()
    if "robot" in text:
        l2_triple[0] = "Robotics"
    if "grasp" in text or "manipulation" in text:
        l2_triple[2] = "Robot Manipulation"
    return {
        "Level 1": [l1, "Mathematics", "Engineering"] if prefix == "cs" else [l1, "Mathematics", "Physics"],
        "Level 2": l2_triple,
        "Level 3": [l2_triple[2], l2_triple[1], "Scientific Computing"],
    }


def _generate_one_paper(
    paper: dict,
    backend: str,
    model: str,
    client=None,
    config: dict | None = None,
) -> dict:
    pid = paper["id"]
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    categories = paper.get("categories", "")
    try:
        if backend in ("llm", "openai"):
            levels = generate_topics_llm(
                title, abstract, model=model, client=client, config=config
            )
        else:
            levels = generate_topics_heuristic(title, abstract, categories)
        return {"paper_id": pid, **levels}
    except Exception as exc:
        print(f"Topic generation failed for {pid}: {type(exc).__name__}: {exc}", flush=True)
        return {"paper_id": pid, **generate_topics_heuristic(title, abstract, categories)}


def generate_topics_batch(
    papers: list[dict],
    backend: str = "llm",
    model: str | None = None,
    sleep_sec: float = 0.0,
    config: dict | None = None,
    workers: int = 1,
) -> list[dict]:
    """Generate topic records for metadata dicts with id/title/abstract/categories."""
    if not papers:
        return []

    client = None
    resolved_model = model
    if backend in ("llm", "openai"):
        client, resolved_model = make_llm_client(config)
        if model:
            resolved_model = model

    if workers <= 1 or backend not in ("llm", "openai"):
        results: list[dict] = []
        for paper in papers:
            results.append(
                _generate_one_paper(paper, backend, resolved_model, client=client, config=config)
            )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        return results

    # Concurrent LLM requests: one client per thread (shared httpx client is not thread-safe).
    thread_local = threading.local()
    results: list[dict | None] = [None] * len(papers)

    def _task(index: int, paper: dict) -> tuple[int, dict]:
        if not hasattr(thread_local, "client"):
            thread_local.client, thread_local.model = make_llm_client(config)
        task_model = model or thread_local.model
        return index, _generate_one_paper(
            paper, backend, task_model, client=thread_local.client, config=config
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_task, i, paper) for i, paper in enumerate(papers)]
        done = 0
        for future in as_completed(futures):
            index, entry = future.result()
            results[index] = entry
            done += 1
            if done % 50 == 0 or done == len(papers):
                print(f"  topic workers: {done}/{len(papers)} papers done", flush=True)

    return [entry for entry in results if entry is not None]
