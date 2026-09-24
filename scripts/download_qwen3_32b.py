#!/usr/bin/env python3
"""Download Qwen3-32B into models/base/Qwen3-32B (ModelScope default; HF optional)."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from rwcite.gat import protocol as P


def _expected_weight_files(out: Path) -> list[str]:
    idx = out / "model.safetensors.index.json"
    if not idx.is_file():
        return []
    data = json.loads(idx.read_text(encoding="utf-8"))
    return sorted(set((data.get("weight_map") or {}).values()))


def _missing_weights(out: Path) -> list[str]:
    expected = _expected_weight_files(out)
    if not expected:
        return []
    return [n for n in expected if not (out / n).is_file()]


def _assert_complete(out: Path) -> None:
    tok = out / "tokenizer_config.json"
    if not tok.is_file():
        raise SystemExit(f"ERROR: missing {tok}")
    missing = _missing_weights(out)
    incomplete = list(out.glob("*.incomplete"))
    if missing or incomplete:
        raise SystemExit(
            f"ERROR: incomplete download under {out}: "
            f"missing={missing[:8]}{'…' if len(missing) > 8 else ''} "
            f"n_missing={len(missing)} n_incomplete={len(incomplete)}"
        )


def _download_modelscope(
    repo: str,
    out: Path,
    *,
    max_workers: int,
) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        model_id=repo,
        local_dir=str(out),
        max_workers=int(max_workers),
    )


def _download_missing_one_by_one(repo: str, out: Path) -> None:
    """Fallback: pull only missing shards sequentially (more resilient)."""
    from modelscope.hub.file_download import model_file_download

    missing = _missing_weights(out)
    # also ensure tokenizer bits if somehow absent
    extras = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]
    for name in extras:
        if not (out / name).is_file():
            missing.append(name)
    # unique preserve order
    seen: set[str] = set()
    files = []
    for n in missing:
        if n not in seen:
            seen.add(n)
            files.append(n)
    for i, name in enumerate(files, 1):
        print(f"file {i}/{len(files)} {name}", flush=True)
        for attempt in range(1, 6):
            try:
                model_file_download(
                    model_id=repo,
                    file_path=name,
                    local_dir=str(out),
                )
                # modelscope may place under cache; ensure file exists at out
                if (out / name).is_file():
                    break
                raise RuntimeError(f"downloaded but missing at {out / name}")
            except Exception as e:
                print(f"  retry {attempt}/5: {e}", flush=True)
                time.sleep(min(60, 5 * attempt))
        else:
            raise RuntimeError(f"failed to download {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        choices=("modelscope", "hf"),
        default="modelscope",
        help="download backend (default: modelscope)",
    )
    ap.add_argument("--repo", default=None, help="ModelScope/HF id; default Qwen/Qwen3-32B")
    ap.add_argument("--out", default="models/base/Qwen3-32B")
    ap.add_argument("--min-free-gb", type=float, default=70.0)
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="parallel file workers for modelscope snapshot (use 1–2 on flaky nets)",
    )
    ap.add_argument(
        "--one-by-one",
        action="store_true",
        help="after first snapshot pass, finish missing files sequentially",
    )
    ap.add_argument(
        "--clean-incomplete",
        action="store_true",
        help="delete *.incomplete before each snapshot retry",
    )
    args = ap.parse_args(argv)

    root = P.root_path()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage(out.parent)
    free_gb = usage.free / (1024**3)
    if free_gb < float(args.min_free_gb):
        raise SystemExit(
            f"ERROR: free disk {free_gb:.1f}GB < {args.min_free_gb}GB at {out.parent}"
        )

    repo = args.repo or "Qwen/Qwen3-32B"
    last_err: Exception | None = None

    if args.source == "hf":
        from huggingface_hub import snapshot_download

        for attempt in range(1, int(args.retries) + 1):
            print(f"[attempt {attempt}/{args.retries}] hf {repo} → {out}", flush=True)
            try:
                snapshot_download(
                    repo_id=repo,
                    local_dir=str(out),
                    local_dir_use_symlinks=False,
                )
                _assert_complete(out)
                print("done", out, flush=True)
                return 0
            except Exception as e:
                last_err = e
                print(f"hf error: {e}", flush=True)
        raise SystemExit(f"ERROR: hf download failed: {last_err}")

    # ModelScope path
    for attempt in range(1, int(args.retries) + 1):
        if args.clean_incomplete:
            for p in out.glob("*.incomplete"):
                print(f"remove incomplete {p.name}", flush=True)
                p.unlink(missing_ok=True)
        print(
            f"[attempt {attempt}/{args.retries}] modelscope {repo} → {out} "
            f"workers={args.max_workers}",
            flush=True,
        )
        try:
            _download_modelscope(repo, out, max_workers=int(args.max_workers))
            _assert_complete(out)
            n = len(list(out.glob("model-*.safetensors")))
            print("done", out, f"weight_shards={n}", flush=True)
            return 0
        except SystemExit as e:
            last_err = e
            print(f"incomplete after snapshot: {e}", flush=True)
            if args.one_by_one or attempt >= 2:
                print("switching to one-by-one missing files…", flush=True)
                try:
                    _download_missing_one_by_one(repo, out)
                    _assert_complete(out)
                    n = len(list(out.glob("model-*.safetensors")))
                    print("done", out, f"weight_shards={n}", flush=True)
                    return 0
                except Exception as e2:
                    last_err = e2
                    print(f"one-by-one error: {e2}", flush=True)
        except Exception as e:
            last_err = e
            print(f"download error: {e}", flush=True)
            time.sleep(10)

    raise SystemExit(f"ERROR: download failed after {args.retries} attempts: {last_err}")


if __name__ == "__main__":
    raise SystemExit(main())
