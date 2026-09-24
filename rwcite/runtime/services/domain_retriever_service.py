from __future__ import annotations

import glob
import os
import re
import threading
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

from rwcite.runtime.config import data_path, load_app_settings, load_domains_settings
from rwcite.runtime.deps import ensure_package_imports, data_cwd
from rwcite.runtime.services.domain_cache import DomainResourceCache
from rwcite.runtime.services.metadata_store import MetadataStore
from rwcite.runtime.services.survey_heuristics import is_survey_title


def _paper_recency(paper_id: str) -> int:
    m = re.match(r"^(\d{2})(\d{2})", paper_id)
    if not m:
        return 0
    yy, mm = int(m.group(1)), int(m.group(2))
    year = 1900 + yy if yy > 70 else 2000 + yy
    return year * 100 + mm


def _paper_year(paper_id: str, update_date: str = "") -> int | None:
    if update_date and len(update_date) >= 4 and update_date[:4].isdigit():
        return int(update_date[:4])
    m = re.match(r"^(\d{2})(\d{2})\.", paper_id)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy > 70 else 2000 + yy


class DomainRetrieverService:
    def __init__(self, cache: DomainResourceCache, metadata_store: MetadataStore | None = None):
        self.cache = cache
        self.metadata_store = metadata_store
        self._embedder = None
        self._tokenizer = None
        self._embedder_device = None  # torch.device
        self._embedder_lock = threading.Lock()
        self._id2topics: dict[str, list] | None = None
        self._domain_embeddings: dict[str, dict[str, np.ndarray]] = {}
        # Precomputed matrices for fast cosine: domain_id -> (ids, matrix L2-normalized)
        self._domain_matrices: dict[str, tuple[list[str], np.ndarray]] = {}
        self._on_demand_cache: dict[str, np.ndarray] = {}

    def _ensure_imports(self):
        ensure_package_imports()

    def _load_embedder(self, *, force_cpu: bool = False):
        """Load BGE encoder. Prefer CPU when sharing the process with 8-bit Llama.

        `.to(cuda)` after a bitsandbytes LLM is loaded often leaves meta tensors and
        breaks search with: Cannot copy out of meta tensor.
        Override: RWCITE_EMBEDDER_DEVICE=cpu|cuda|auto
        """
        if self._embedder is not None and not force_cpu:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        with self._embedder_lock:
            if self._embedder is not None and not force_cpu:
                return
            if force_cpu:
                self._embedder = None
                self._tokenizer = None
                self._embedder_device = None

            self._ensure_imports()
            embedder = data_path(load_domains_settings().shared.embedder)
            # Default CPU: same process hosts 8-bit Llama; CUDA BGE often ends up meta.
            prefer = (
                "cpu"
                if force_cpu
                else (os.environ.get("RWCITE_EMBEDDER_DEVICE") or "cpu")
                .strip()
                .lower()
            )
            load_kwargs: dict = {
                "local_files_only": True,
                "low_cpu_mem_usage": False,
            }
            with data_cwd():
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(embedder), **{k: v for k, v in load_kwargs.items() if k != "low_cpu_mem_usage"}
                    )
                    model = AutoModel.from_pretrained(str(embedder), **load_kwargs)
                except OSError:
                    self._tokenizer = AutoTokenizer.from_pretrained(str(embedder))
                    model = AutoModel.from_pretrained(
                        str(embedder), low_cpu_mem_usage=False
                    )

            device = torch.device("cpu")
            dtype = torch.float32
            if prefer in ("cuda", "auto") and torch.cuda.is_available():
                try:
                    free, _total = torch.cuda.mem_get_info(0)
                    # Need headroom alongside an optional 8-bit Llama on the same GPU.
                    if prefer == "cuda" or free > 3 * (1024**3):
                        cand = model.to(device="cuda", dtype=torch.float16)
                        p0 = next(cand.parameters())
                        if (not p0.is_meta) and p0.device.type == "cuda":
                            model = cand
                            device = torch.device("cuda")
                            dtype = torch.float16
                except Exception:
                    device = torch.device("cpu")
                    dtype = torch.float32

            if device.type == "cpu" and next(model.parameters()).device.type != "cpu":
                model = model.to(device="cpu", dtype=torch.float32)
            model.eval()
            self._embedder = model
            self._embedder_device = device
            # Keep dtype note for debugging; unused otherwise.
            _ = dtype

    @staticmethod
    def _split_topic_field(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(t).strip() for t in value if str(t).strip()]
        text = str(value).strip()
        if not text:
            return []
        # HF topics are usually lists; SQLite stores comma-separated strings
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(t).strip() for t in parsed if str(t).strip()]
            except (SyntaxError, ValueError):
                pass
        return [p.strip() for p in text.split(",") if p.strip()]

    def _topics_arrow_paths(self) -> list[Path]:
        cache_dir = data_path(load_domains_settings().shared.topics_cache)
        pattern = str(
            cache_dir / "AliMaatouk___ar_xiv_topics/default/*/*/ar_xiv_topics-train-*.arrow"
        )
        return [Path(p) for p in sorted(glob.glob(pattern))]

    def _load_id2topics_from_arrow(self) -> dict[str, list]:
        """Load topics from local HF arrow cache + supplement — no datasets.load_dataset."""
        import pyarrow as pa
        import pyarrow.ipc as pa_ipc

        id2topics: dict[str, list] = {}
        for shard in self._topics_arrow_paths():
            # memory_map lives on pyarrow, not pyarrow.ipc
            with pa.memory_map(str(shard), "r") as source:
                try:
                    table = pa_ipc.open_stream(source).read_all()
                except Exception:
                    table = pa_ipc.RecordBatchFileReader(source).read_all()
            cols = {name: table.column(name).to_pylist() for name in table.column_names}
            for i, pid in enumerate(cols["paper_id"]):
                id2topics[pid] = [
                    self._split_topic_field(cols["Level 1"][i]),
                    self._split_topic_field(cols["Level 2"][i]),
                    self._split_topic_field(cols["Level 3"][i]),
                ]

        supp = data_path("datasets/arxiv_topics_supplement.jsonl")
        if supp.exists():
            import json

            with open(supp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    pid = entry.get("paper_id")
                    if not pid:
                        continue
                    id2topics[pid] = [
                        self._split_topic_field(entry.get("Level 1")),
                        self._split_topic_field(entry.get("Level 2")),
                        self._split_topic_field(entry.get("Level 3")),
                    ]
        return id2topics

    def _load_id2topics_from_metadata(self, paper_ids: list[str]) -> dict[str, list]:
        if not self.metadata_store or not self.metadata_store.ready or not paper_ids:
            return {}
        meta = self.metadata_store.get_many(paper_ids)
        out: dict[str, list] = {}
        for pid, row in meta.items():
            l1 = self._split_topic_field(row.get("topics_l1"))
            l2 = self._split_topic_field(row.get("topics_l2"))
            l3 = self._split_topic_field(row.get("topics_l3"))
            if l1 or l2 or l3:
                out[pid] = [l1, l2, l3]
        return out

    def _load_id2topics(self):
        if self._id2topics is not None:
            return
        self._ensure_imports()
        # Prefer local arrow (avoids HuggingFace datasets + transformers dill bug)
        try:
            local = self._load_id2topics_from_arrow()
            if local:
                self._id2topics = local
                return
        except Exception:
            local = {}
        try:
            with data_cwd():
                from rwcite.retrieve.retriever.corpus_utils import build_id2topics

                topics_cache = str(data_path(load_domains_settings().shared.topics_cache))
                self._id2topics = build_id2topics(cache_dir=topics_cache)
        except Exception:
            self._id2topics = local or {}

    def _arrow_shard_paths(self) -> list[Path]:
        cache_dir = data_path(load_domains_settings().shared.embeddings_cache)
        pattern = str(
            cache_dir
            / "AliMaatouk___ar_xiv-topics-embeddings/default/*/*/ar_xiv-topics-embeddings-train-*.arrow"
        )
        return [Path(p) for p in sorted(glob.glob(pattern))]

    def _load_supplement_for_ids(self, needed: set[str], out: dict[str, np.ndarray]) -> set[str]:
        supp = data_path("datasets/topic_level_embeds/supplement_embeddings.parquet")
        if not supp.exists() or not needed:
            return needed
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        table = pq.read_table(supp, columns=["paper_id", "embedding"])
        mask = pc.is_in(table.column("paper_id"), value_set=pa.array(list(needed)))
        filtered = table.filter(mask)
        for pid, emb in zip(
            filtered.column("paper_id").to_pylist(),
            filtered.column("embedding").to_pylist(),
        ):
            out[pid] = np.asarray(emb, dtype=np.float32)
        return needed - set(out.keys())

    def _domain_cache_path(self, domain_id: str) -> Path:
        from rwcite.graph.domain_paths import domain_data_paths, working_dir_for_domain

        try:
            wd = working_dir_for_domain(domain_id, load_app_settings().data_root)
            rel = domain_data_paths(wd)["domain_embeddings"]
            path = data_path(rel)
        except (KeyError, OSError):
            shared = load_domains_settings().shared
            cache_dir = data_path(shared.domain_embeddings_cache)
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / f"{domain_id}_embeddings.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_domain_cache_file(self, domain_id: str) -> dict[str, np.ndarray] | None:
        path = self._domain_cache_path(domain_id)
        if not path.exists() or path.stat().st_size < 8:
            return None
        import pandas as pd

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f"WARN: ignore corrupt domain cache {path}: {exc}", flush=True)
            return None
        return {row.paper_id: np.asarray(row.embedding, dtype=np.float32) for row in df.itertuples()}

    def _save_domain_cache_file(self, domain_id: str, emb_map: dict[str, np.ndarray]) -> None:
        import fcntl
        import os
        import pandas as pd

        if not emb_map:
            return
        path = self._domain_cache_path(domain_id)
        lock_path = path.with_name(path.name + ".lock")
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        df = pd.DataFrame(
            {"paper_id": list(emb_map.keys()), "embedding": list(emb_map.values())}
        )
        df.to_parquet(tmp, compression="snappy")
        with open(lock_path, "a", encoding="utf-8") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                os.replace(tmp, path)
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

    def _load_hf_shards_for_ids(self, needed: set[str], out: dict[str, np.ndarray]) -> None:
        if not needed:
            return
        import pyarrow as pa
        import pyarrow.compute as pc

        needed_arr = pa.array(list(needed))
        for shard in self._arrow_shard_paths():
            if not needed:
                break
            with pa.memory_map(str(shard), "r") as source:
                reader = ipc.open_stream(source)
                table = reader.read_all()
            mask = pc.is_in(table.column("paper_id"), value_set=needed_arr)
            filtered = table.filter(mask)
            ids = filtered.column("paper_id").to_pylist()
            embs = filtered.column("embedding").to_pylist()
            for pid, emb in zip(ids, embs):
                out[pid] = np.asarray(emb, dtype=np.float32)
                needed.discard(pid)

    def _build_domain_matrix(self, domain_id: str, emb_map: dict[str, np.ndarray]) -> None:
        ids = list(emb_map.keys())
        if not ids:
            self._domain_matrices[domain_id] = ([], np.zeros((0, 0), dtype=np.float32))
            return
        mat = np.stack([emb_map[pid] for pid in ids]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms
        self._domain_matrices[domain_id] = (ids, mat)

    def _ensure_domain_embeddings(self, domain_id: str, candidate_ids: list[str]) -> dict[str, np.ndarray]:
        if domain_id in self._domain_embeddings:
            have = self._domain_embeddings[domain_id]
            if all(pid in have for pid in candidate_ids):
                return have
            # Cache incomplete vs current graph — fall through to fill gaps.

        found: dict[str, np.ndarray] = {}
        if domain_id in self._domain_embeddings:
            found.update(self._domain_embeddings[domain_id])
        else:
            cached = self._load_domain_cache_file(domain_id)
            if cached is not None:
                found.update(cached)

        n_before = len(found)
        needed = {pid for pid in candidate_ids if pid not in found} - set(
            self._on_demand_cache.keys()
        )
        needed = self._load_supplement_for_ids(needed, found)
        self._load_hf_shards_for_ids(needed, found)
        found.update(
            {k: self._on_demand_cache[k] for k in candidate_ids if k in self._on_demand_cache}
        )
        still_missing = [pid for pid in candidate_ids if pid not in found]
        if still_missing:
            self._load_id2topics()
            batch_size = 64
            for i in range(0, len(still_missing), batch_size):
                batch = still_missing[i : i + batch_size]
                embs = self._encode_topics_batch(batch)
                for pid, emb in zip(batch, embs):
                    self._on_demand_cache[pid] = emb
                    found[pid] = emb
        self._domain_embeddings[domain_id] = found
        self._build_domain_matrix(domain_id, found)
        # Persist only when coverage grew — 8-GPU workers must not rewrite the same parquet.
        if len(found) > n_before or not self._domain_cache_path(domain_id).exists():
            self._save_domain_cache_file(domain_id, found)
        return found

    def _encode_on_device(self, texts: list[str]):
        import torch

        assert self._tokenizer is not None and self._embedder is not None
        assert self._embedder_device is not None
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self._embedder_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._embedder(**inputs)
            hidden = outputs.last_hidden_state[:, 0, :]
            if getattr(hidden, "is_meta", False):
                raise NotImplementedError("embedder produced meta tensors")
            return hidden.float().cpu()

    def _encode_query(self, query: str) -> np.ndarray:
        self._load_embedder()
        try:
            vec = self._encode_on_device([query]).numpy().astype(np.float32)
        except NotImplementedError:
            # Broken CUDA/meta load (common after 8-bit LLM warmup) → CPU.
            self._load_embedder(force_cpu=True)
            vec = self._encode_on_device([query]).numpy().astype(np.float32)
        n = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(n, 1e-12)

    def _encode_topics_batch(self, paper_ids: list[str]) -> np.ndarray:
        from rwcite.retrieve.retriever.corpus_utils import topic_text_for_level

        self._load_embedder()
        self._load_id2topics()
        assert self._id2topics is not None
        level_embs = []
        for level_idx in range(3):
            texts = [
                topic_text_for_level(self._id2topics.get(pid, [[], [], []])[level_idx])
                for pid in paper_ids
            ]
            try:
                emb = self._encode_on_device(texts)
            except NotImplementedError:
                self._load_embedder(force_cpu=True)
                emb = self._encode_on_device(texts)
            level_embs.append(emb)
        combined = (level_embs[0] + level_embs[1] + level_embs[2]).numpy()
        return combined

    def _filter_by_topics(
        self, candidate_ids: list[str], l1: str | None, l2: str | None, l3: str | None
    ) -> list[str]:
        if not (l1 or l2 or l3):
            return candidate_ids
        # Prefer SQLite topics (same source as topic chips) — avoid loading full arrow.
        id2topics = self._load_id2topics_from_metadata(candidate_ids)
        if len(id2topics) < max(1, len(candidate_ids) // 10):
            self._load_id2topics()
            if self._id2topics:
                for pid in candidate_ids:
                    if pid not in id2topics and pid in self._id2topics:
                        id2topics[pid] = self._id2topics[pid]
        out = []
        for pid in candidate_ids:
            topics = id2topics.get(pid)
            if not topics:
                continue
            if l1 and l1 not in topics[0]:
                continue
            if l2 and l2 not in topics[1]:
                continue
            if l3 and l3 not in topics[2]:
                continue
            out.append(pid)
        return out

    def _filter_year_degree(
        self,
        candidate_ids: list[str],
        graph,
        year_from: int | None,
        year_to: int | None,
        min_in_degree: int | None,
        survey_filter: bool = False,
        meta_map: dict[str, dict] | None = None,
    ) -> list[str]:
        if (
            year_from is None
            and year_to is None
            and not min_in_degree
            and not survey_filter
        ):
            return candidate_ids
        need_meta = year_from is not None or year_to is not None or survey_filter
        if meta_map is None and need_meta and self.metadata_store and self.metadata_store.ready:
            # Prefer graph titles for survey; still load meta for year / title fallback.
            if year_from is not None or year_to is not None or graph is None:
                meta_map = self.metadata_store.get_many(candidate_ids)
            else:
                missing = [
                    pid
                    for pid in candidate_ids
                    if not ((graph.nodes[pid].get("title") if pid in graph else None) or "")
                ]
                meta_map = self.metadata_store.get_many(missing) if missing else {}
        meta_map = meta_map or {}
        out = []
        for pid in candidate_ids:
            if min_in_degree and graph is not None:
                if graph.in_degree(pid) < min_in_degree:
                    continue
            if year_from is not None or year_to is not None:
                y = _paper_year(pid, (meta_map.get(pid) or {}).get("update_date") or "")
                if y is None:
                    continue
                if year_from is not None and y < year_from:
                    continue
                if year_to is not None and y > year_to:
                    continue
            if survey_filter:
                title = ""
                if graph is not None and pid in graph:
                    title = graph.nodes[pid].get("title") or ""
                if not title:
                    title = (meta_map.get(pid) or {}).get("title") or ""
                if not is_survey_title(title):
                    continue
            out.append(pid)
        return out

    def search(
        self,
        scope: str = "domain",
        domain_id: str | None = None,
        query: str = "",
        offset: int = 0,
        limit: int = 20,
        l1: str | None = None,
        l2: str | None = None,
        l3: str | None = None,
        sort: str = "relevance",
        year_from: int | None = None,
        year_to: int | None = None,
        min_in_degree: int | None = None,
        survey_filter: bool = False,
    ) -> dict:
        if scope == "corpus":
            return self._search_corpus_text(
                query,
                offset,
                limit,
                sort,
                domain_id=domain_id,
                year_from=year_from,
                year_to=year_to,
                min_in_degree=min_in_degree,
                survey_filter=survey_filter,
            )
        if not domain_id:
            raise KeyError("domain is required for scope=domain")
        return self._search_domain_semantic(
            domain_id,
            query,
            offset,
            limit,
            l1,
            l2,
            l3,
            sort,
            year_from=year_from,
            year_to=year_to,
            min_in_degree=min_in_degree,
            survey_filter=survey_filter,
        )

    def _search_corpus_text(
        self,
        query: str,
        offset: int,
        limit: int,
        sort: str,
        domain_id: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_in_degree: int | None = None,
        survey_filter: bool = False,
    ) -> dict:
        if not self.metadata_store or not self.metadata_store.ready:
            return {
                "total": 0,
                "offset": offset,
                "limit": limit,
                "scope": "corpus",
                "mode": "text",
                "results": [],
                "year_histogram": [],
                "warning": "Metadata SQLite not available (Release uses JSONL snapshot only)",
            }

        graph = domain_cache = None
        if domain_id:
            try:
                domain_cache = self.cache.get(domain_id)
                graph = domain_cache.graph
            except KeyError:
                domain_id = None

        post_degree = bool(min_in_degree and graph is not None)
        if not query.strip():
            browse_sort = "recency" if sort in ("relevance", "recency", "influential") else sort
            if post_degree:
                # Over-fetch then apply in-degree; survey/year already in SQL.
                fetch_n = min(5000, max(limit * 50, 500))
                raw_ids, _ = self.metadata_store.browse(
                    0,
                    fetch_n,
                    sort=browse_sort,
                    year_from=year_from,
                    year_to=year_to,
                    survey_filter=survey_filter,
                )
                page_ids = [
                    pid
                    for pid in raw_ids
                    if pid in graph and graph.in_degree(pid) >= min_in_degree
                ]
                total = len(page_ids)
                page_ids = page_ids[offset : offset + limit]
            else:
                page_ids, total = self.metadata_store.browse(
                    offset,
                    limit,
                    sort=browse_sort,
                    year_from=year_from,
                    year_to=year_to,
                    survey_filter=survey_filter,
                )
            score_map: dict[str, float] = {}
        else:
            matched, total = self.metadata_store.search(
                query,
                allowed_ids=None,
                offset=0 if post_degree else offset,
                limit=min(5000, max(limit * 50, 500)) if post_degree else limit,
                sort=sort,
                year_from=year_from,
                year_to=year_to,
                survey_filter=survey_filter,
            )
            score_map = {pid: -rank for pid, rank in matched}
            page_ids = [pid for pid, _ in matched]
            if post_degree:
                page_ids = [
                    pid for pid in page_ids if pid in graph and graph.in_degree(pid) >= min_in_degree
                ]
                total = len(page_ids)
                page_ids = page_ids[offset : offset + limit]
            elif min_in_degree and not graph:
                # no graph — ignore degree filter; survey already applied in store
                pass

        results = self._format_corpus_results(
            page_ids, score_map, has_query=bool(query.strip()), graph=graph, domain_cache=domain_cache
        )
        hist = self.metadata_store.year_histogram(year_from=year_from, year_to=year_to)
        # Keep histogram compact for sidebar
        if len(hist) > 40:
            hist = hist[-40:]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "scope": "corpus",
            "mode": "text",
            "results": results,
            "year_histogram": hist,
        }

    def _format_corpus_results(
        self,
        page_ids: list[str],
        score_map: dict[str, float],
        has_query: bool,
        graph=None,
        domain_cache=None,
    ) -> list[dict]:
        meta_map = {}
        if self.metadata_store and self.metadata_store.ready:
            meta_map = self.metadata_store.get_many(page_ids)

        results = []
        for pid in page_ids:
            meta = meta_map.get(pid, {})
            in_domain = graph is not None and pid in graph
            abstract = meta.get("abstract") or ""
            title = meta.get("title") or ""
            journal_ref = (meta.get("journal_ref") or "").strip()
            results.append(
                {
                    "paper_id": pid,
                    "title": title,
                    "abstract": abstract[:500],
                    "authors": meta.get("authors") or "",
                    "categories": meta.get("categories") or "",
                    "topics_l1": meta.get("topics_l1") or "",
                    "topics_l2": meta.get("topics_l2") or "",
                    "topics_l3": meta.get("topics_l3") or "",
                    "update_date": meta.get("update_date") or "",
                    "journal_ref": journal_ref,
                    "arxiv_url": f"https://arxiv.org/abs/{pid}",
                    "score": score_map.get(pid) if has_query else None,
                    "in_degree": graph.in_degree(pid) if in_domain else None,
                    "in_domain": in_domain,
                    "is_retrieval_seed": (
                        pid in domain_cache.retrieval_seeds if domain_cache and in_domain else False
                    ),
                    "is_survey": is_survey_title(title),
                }
            )
        return results

    def _format_results(
        self, g, res, page_ids, score_map: dict, has_query: bool
    ) -> list[dict]:
        meta_map = {}
        if self.metadata_store and self.metadata_store.ready:
            meta_map = self.metadata_store.get_many(page_ids)

        results = []
        for pid in page_ids:
            attrs = dict(g.nodes[pid])
            meta = meta_map.get(pid, {})
            abstract = attrs.get("abstract") or meta.get("abstract") or ""
            title = attrs.get("title") or meta.get("title") or ""
            journal_ref = (meta.get("journal_ref") or "").strip()
            results.append(
                {
                    "paper_id": pid,
                    "title": title,
                    "abstract": abstract[:500],
                    "authors": meta.get("authors") or "",
                    "categories": meta.get("categories") or "",
                    "topics_l1": meta.get("topics_l1") or "",
                    "topics_l2": meta.get("topics_l2") or "",
                    "topics_l3": meta.get("topics_l3") or "",
                    "update_date": meta.get("update_date") or "",
                    "journal_ref": journal_ref,
                    "arxiv_url": f"https://arxiv.org/abs/{pid}",
                    "score": score_map.get(pid) if has_query else None,
                    "in_degree": g.in_degree(pid),
                    "in_domain": True,
                    "is_retrieval_seed": pid in res.retrieval_seeds,
                    "is_survey": is_survey_title(title),
                }
            )
        return results

    def _search_domain_semantic(
        self,
        domain_id: str,
        query: str = "",
        offset: int = 0,
        limit: int = 20,
        l1: str | None = None,
        l2: str | None = None,
        l3: str | None = None,
        sort: str = "relevance",
        year_from: int | None = None,
        year_to: int | None = None,
        min_in_degree: int | None = None,
        survey_filter: bool = False,
    ) -> dict:
        res = self.cache.get(domain_id)
        g = res.graph
        candidate_ids = self._filter_by_topics(list(res.node_ids), l1, l2, l3)
        candidate_ids = self._filter_year_degree(
            candidate_ids, g, year_from, year_to, min_in_degree, survey_filter=survey_filter
        )
        total = len(candidate_ids)

        sort_key = sort
        if sort_key == "influential":
            sort_key = "in_degree"

        if not query.strip():
            if sort_key == "in_degree":
                ranked = sorted(candidate_ids, key=lambda p: g.in_degree(p), reverse=True)
            elif sort_key == "recency":
                ranked = sorted(candidate_ids, key=_paper_recency, reverse=True)
            else:
                ranked = sorted(
                    candidate_ids,
                    key=lambda p: (g.in_degree(p), _paper_recency(p)),
                    reverse=True,
                )
            score_map: dict[str, float] = {}
        else:
            emb_map = self._ensure_domain_embeddings(domain_id, candidate_ids)
            if domain_id not in self._domain_matrices:
                self._build_domain_matrix(domain_id, emb_map)
            all_ids, mat = self._domain_matrices[domain_id]
            cand_set = set(candidate_ids)
            idxs = [i for i, pid in enumerate(all_ids) if pid in cand_set]
            if not idxs:
                return {
                    "total": 0,
                    "offset": offset,
                    "limit": limit,
                    "scope": "domain",
                    "mode": "semantic",
                    "results": [],
                    "year_histogram": [],
                }
            sub = mat[idxs]
            ids = [all_ids[i] for i in idxs]
            qvec = self._encode_query(query)
            sims = (sub @ qvec.T).ravel()
            order = np.argsort(-sims)
            if sort_key == "in_degree":
                order = np.array(
                    sorted(order, key=lambda i: (g.in_degree(ids[i]), float(sims[i])), reverse=True)
                )
            elif sort_key == "recency":
                order = np.array(
                    sorted(order, key=lambda i: (_paper_recency(ids[i]), float(sims[i])), reverse=True)
                )
            ranked = [ids[i] for i in order]
            score_map = {ids[i]: float(sims[i]) for i in order}

        page_ids = ranked[offset : offset + limit]
        results = self._format_results(
            g, res, page_ids, score_map, has_query=bool(query.strip())
        )
        hist_counts: dict[int, int] = {}
        meta_map = (
            self.metadata_store.get_many(candidate_ids)
            if self.metadata_store and self.metadata_store.ready and len(candidate_ids) <= 25000
            else {}
        )
        for pid in candidate_ids:
            y = _paper_year(pid, (meta_map.get(pid) or {}).get("update_date") or "")
            if y is not None:
                hist_counts[y] = hist_counts.get(y, 0) + 1
        hist = [{"year": y, "count": hist_counts[y]} for y in sorted(hist_counts)]
        if len(hist) > 40:
            hist = hist[-40:]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "scope": "domain",
            "mode": "semantic",
            "results": results,
            "year_histogram": hist,
        }

    def topic_vocab(self, domain_id: str) -> dict:
        res = self.cache.get(domain_id)
        node_ids = list(res.node_ids)

        # Fast path: SQLite metadata topics for domain nodes (no HF dataset load)
        id2topics = self._load_id2topics_from_metadata(node_ids)
        if len(id2topics) < max(1, len(node_ids) // 10):
            # Sparse metadata — fall back to local arrow / HF cache
            self._load_id2topics()
            assert self._id2topics is not None
            for pid in node_ids:
                if pid not in id2topics and pid in self._id2topics:
                    id2topics[pid] = self._id2topics[pid]

        l1, l2, l3 = set(), set(), set()
        counts_l1: dict[str, int] = {}
        counts_l2: dict[str, int] = {}
        for pid in node_ids:
            topics = id2topics.get(pid)
            if not topics:
                continue
            for t in topics[0]:
                l1.add(t)
                counts_l1[t] = counts_l1.get(t, 0) + 1
            for t in topics[1]:
                l2.add(t)
                counts_l2[t] = counts_l2.get(t, 0) + 1
            for t in topics[2]:
                l3.add(t)
        top_l1 = [t for t, _ in sorted(counts_l1.items(), key=lambda x: -x[1])[:24]]
        top_l2 = [t for t, _ in sorted(counts_l2.items(), key=lambda x: -x[1])[:24]]
        return {
            "l1": sorted(l1),
            "l2": sorted(l2),
            "l3": sorted(l3),
            "top_l1": top_l1,
            "top_l2": top_l2,
        }
