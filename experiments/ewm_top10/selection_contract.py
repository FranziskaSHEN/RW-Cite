#!/usr/bin/env python3
"""Freeze and validate the development-selected comparison configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.ewm_top10.common import sha256


def _weights(model: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model / name
        if path.is_file():
            return path
    raise ValueError(f"model weights not found under {model}")


def _selected_settings(profile_path: Path) -> dict:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return {
        key: profile.get(key)
        for key in (
            "evidence_mode",
            "same_month_policy",
            "unknown_date_policy",
            "short_candidate",
            "max_sents",
            "max_length",
            "training_stage",
            "universe",
            "shortlist",
        )
    }


def create(args: argparse.Namespace) -> None:
    profile = Path(args.rw_profile).resolve()
    model = Path(args.rw_model).resolve()
    metrics = Path(args.dev_metrics).resolve()
    for path in (profile, metrics):
        if not path.is_file():
            raise ValueError(f"required development artifact is missing: {path}")
    manifest = {
        "selection_split": "development",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "rw_model": str(model),
        "rw_model_weights_sha256": sha256(_weights(model)),
        "rw_profile": str(profile),
        "rw_profile_sha256": sha256(profile),
        "selected_settings": _selected_settings(profile),
        "dev_metrics": str(metrics),
        "dev_metrics_sha256": sha256(metrics),
        "notes": args.notes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def validate(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    profile = Path(args.rw_profile).resolve()
    checks = {
        "selection_split": (manifest.get("selection_split"), "development"),
        "selected_settings": (manifest.get("selected_settings"), _selected_settings(profile)),
    }
    failures = [
        f"{key}: got {actual!r}, expected {expected!r}"
        for key, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise SystemExit("frozen selection mismatch:\n- " + "\n- ".join(failures))
    print("frozen development-selection contract validated")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--rw-model", required=True)
    create_parser.add_argument("--rw-profile", required=True)
    create_parser.add_argument("--dev-metrics", required=True)
    create_parser.add_argument("--out", required=True)
    create_parser.add_argument("--notes", default="")
    create_parser.set_defaults(function=create)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--rw-profile", required=True)
    validate_parser.set_defaults(function=validate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
