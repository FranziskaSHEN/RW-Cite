"""Stage the ten paper-domain packs for Hugging Face without duplicating data.

The merge input already contains complete domain packs.  This utility validates
that inventory, hard-links files when possible, removes internal release notes,
and regenerates publication-facing dataset cards and manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PAPER_DOMAINS = (
    "ewm",
    "gw",
    "driving",
    "cosmo",
    "radio",
    "wsi",
    "exo",
    "sqc",
    "fno",
    "sce",
)


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _ignore(path: str, names: list[str]) -> set[str]:
    base = Path(path)
    ignored: set[str] = set()
    for name in names:
        if name in {"README.md", "MANIFEST.json", "DATASETS_HF_UPLOAD.md"}:
            ignored.add(name)
        if base.name == "docs" and (
            name.startswith("RELEASE_") or name.startswith("ABLATION_")
        ):
            ignored.add(name)
    return ignored


def _find_source(source_root: Path, domain: str) -> Path:
    candidates = sorted(
        path
        for path in source_root.glob(f"RW-Cite-domain-{domain}-*")
        if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"missing prepacked domain: {domain}")
    return candidates[-1]


def _validate(source: Path, domain: str) -> dict:
    manifest_path = source / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("domain") != domain:
        raise ValueError(f"manifest domain mismatch in {source}")
    working = str(manifest.get("working_dir") or "").strip("/")
    required = (
        manifest.get("gexf"),
        f"{working}/data/splits/split_meta.json",
        manifest.get("struct_model"),
        manifest.get("ce_path"),
        f"{working}/ranker/gat_mvp/ckpt_e4.pt",
    )
    missing = [str(item) for item in required if not item or not (source / item).exists()]
    if missing:
        raise FileNotFoundError(f"{domain} pack is incomplete: {missing}")
    return manifest


def _file_manifest(stage: Path, checksums: bool) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name in {"README.md", "MANIFEST.json"}:
            continue
        stat = path.stat()
        row: dict[str, object] = {
            "path": path.relative_to(stage).as_posix(),
            "bytes": stat.st_size,
        }
        if checksums:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            row["sha256"] = digest.hexdigest()
        rows.append(row)
    return rows


def _dataset_card(domain: str, label: str, working: str, tag: str) -> str:
    return f"""---
pretty_name: RW-Cite — {label}
task_categories:
  - text-retrieval
language:
  - en
license: other
---

# RW-Cite: {label}

This data-only package contains the `{domain}` benchmark artifacts associated
with the RW-Cite paper. It includes the domain citation graph, frozen query
split, reference-recommendation records, learned structural Top-400 artifacts,
SciBERT cross-encoder checkpoint, graph-attention checkpoint and inputs, and
saved evaluation outputs.

The corresponding source code is distributed separately through the RW-Cite
GitHub repository. The package revision is `{tag}`. Its runtime root is
`{working}/` after extraction into an RW-Cite checkout.

## Installation

```bash
rsync -a configs/ /path/to/RW-Cite/configs/
rsync -a {working}/ /path/to/RW-Cite/{working}/
```

See `MANIFEST.json` for the exact file inventory, split metadata, model
identifiers, and graph path. Paper titles, abstracts, and citation-context text
originate from arXiv; downstream users remain responsible for observing the
applicable source-paper licenses and arXiv terms.
"""


def stage_domain(
    source_root: Path,
    output_root: Path,
    domain: str,
    requested_tag: str | None,
    checksums: bool,
) -> dict:
    source = _find_source(source_root, domain)
    source_manifest = _validate(source, domain)
    tag = (requested_tag or str(source_manifest.get("tag") or "")).lstrip("r")
    if not tag:
        tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    destination = output_root / f"RW-Cite-domain-{domain}-r{tag}"
    if destination.exists():
        raise FileExistsError(
            f"destination already exists; remove the generated payload before rebuilding: {destination}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{domain}-", dir=output_root))
    try:
        shutil.copytree(
            source,
            temporary,
            dirs_exist_ok=True,
            copy_function=_link_or_copy,
            ignore=_ignore,
        )
        label = str(source_manifest.get("label") or domain)
        working = str(source_manifest["working_dir"]).strip("/")
        files = _file_manifest(temporary, checksums)
        manifest = {
            "name": "RW-Cite-domain",
            "schema_version": 1,
            "domain": domain,
            "label": label,
            "tag": f"r{tag}",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "working_dir": f"{working}/",
            "gexf": source_manifest.get("gexf"),
            "struct_model": source_manifest.get("struct_model"),
            "ce_path": source_manifest.get("ce_path"),
            "split_meta": source_manifest.get("split_meta"),
            "method": {
                "candidate_window": 400,
                "cross_encoder": "allenai/scibert_scivocab_uncased",
                "graph_ranker": "two-layer graph-attention network",
                "fusion": "fixed score-and-rank fusion",
                "score_weight_gat": 0.4,
                "rank_fusion_offset": 20,
                "rank_fusion_weight": 0.4,
            },
            "embedder": source_manifest.get("embedder"),
            "files": files,
        }
        (temporary / "README.md").write_text(
            _dataset_card(domain, label, working, f"r{tag}"), encoding="utf-8"
        )
        (temporary / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "domain": domain,
        "label": manifest["label"],
        "path": destination.name,
        "files": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    rows = [
        stage_domain(source_root, output_root, domain, args.tag, args.checksums)
        for domain in PAPER_DOMAINS
    ]
    collection = {
        "name": "RW-Cite ten-domain benchmark collection",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domains": rows,
        "total_files": sum(int(row["files"]) for row in rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }
    (output_root / "COLLECTION_MANIFEST.json").write_text(
        json.dumps(collection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(f"{row['domain']}: {row['path']} ({row['files']} files)")


if __name__ == "__main__":
    main()
