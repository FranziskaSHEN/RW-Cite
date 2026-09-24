#!/usr/bin/env python3
"""Train RR v6 Cite QLoRA adapter (Qwen3-32B)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rwcite.adapter.paths import resolve_adapter_v6
from rwcite.adapter.train_qlora import _ALLOWED_V6, train_from_config
from rwcite.gat import protocol as P


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True, choices=("ewm", "sqc", "gw"))
    ap.add_argument("--config", default="configs/rr_adapter_v6.yaml")
    ap.add_argument("--fresh", action="store_true", default=True)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = P.root_path() if not args.root else Path(args.root)
    os.environ.setdefault("RWCITE_ROOT", str(root))
    paths = resolve_adapter_v6(args.domain, root)
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = root / cfg
    if not paths.train_jsonl.is_file():
        raise SystemExit(
            f"missing {paths.train_jsonl}; run build_rr_adapter_v6 first"
        )
    train_from_config(
        cfg,
        train_jsonl=paths.train_jsonl,
        output_dir=paths.model_dir,
        trainer_output_dir=paths.trainer_dir,
        fresh=bool(args.fresh),
        resume_from_checkpoint=args.resume_from_checkpoint,
        allowed_tasks=_ALLOWED_V6,
        wandb_project="rwcite-rr-adapter-v6",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
