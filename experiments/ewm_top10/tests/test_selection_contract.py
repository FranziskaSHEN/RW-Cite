from __future__ import annotations

import argparse
import json
from unittest.mock import patch

from experiments.ewm_top10 import selection_contract


def _profile(path, *, created_at: str) -> None:
    path.write_text(
        json.dumps(
            {
                "created_at": created_at,
                "training_stage": "stage1",
                "evidence_mode": "temporal_recent",
                "same_month_policy": "drop",
                "unknown_date_policy": "drop",
                "short_candidate": True,
                "max_sents": 3,
                "max_length": 256,
                "universe": {"m": 400},
                "shortlist": {"window": 800, "score_m": 1000},
            }
        ),
        encoding="utf-8",
    )


def test_final_retrain_may_change_weights_but_not_selected_settings(tmp_path) -> None:
    dev_model = tmp_path / "dev_model"
    dev_model.mkdir()
    (dev_model / "model.safetensors").write_bytes(b"development weights")
    dev_profile = tmp_path / "dev_profile.json"
    final_profile = tmp_path / "final_profile.json"
    metrics = tmp_path / "metrics.json"
    manifest = tmp_path / "selection.json"
    _profile(dev_profile, created_at="development")
    _profile(final_profile, created_at="final retraining")
    metrics.write_text("{}", encoding="utf-8")
    create_args = argparse.Namespace(
        rw_profile=str(dev_profile),
        rw_model=str(dev_model),
        dev_metrics=str(metrics),
        out=str(manifest),
        notes="",
    )
    with patch("subprocess.check_output", return_value="commit\n"):
        selection_contract.create(create_args)
    selection_contract.validate(
        argparse.Namespace(manifest=str(manifest), rw_profile=str(final_profile))
    )


def test_final_profile_cannot_change_selected_window(tmp_path) -> None:
    profile = tmp_path / "profile.json"
    _profile(profile, created_at="dev")
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "selection_split": "development",
                "selected_settings": selection_contract._selected_settings(profile),
            }
        ),
        encoding="utf-8",
    )
    changed = json.loads(profile.read_text(encoding="utf-8"))
    changed["shortlist"]["window"] = 400
    profile.write_text(json.dumps(changed), encoding="utf-8")
    try:
        selection_contract.validate(
            argparse.Namespace(manifest=str(manifest), rw_profile=str(profile))
        )
    except SystemExit as error:
        assert "selected_settings" in str(error)
    else:
        raise AssertionError("changed test configuration was accepted")
