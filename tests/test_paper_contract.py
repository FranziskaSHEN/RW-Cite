from __future__ import annotations

from pathlib import Path

import yaml

from rwcite.gat.l0_recipe import DEFAULT_ALPHA, DEFAULT_RRF_K, DEFAULT_RRF_W
from rwcite.gat.model import GatMvpConfig
from rwcite.runtime.config import enabled_domains, load_app_settings, load_domains_settings


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_paper_contract_matches_runtime_defaults(repo_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("RWCITE_ROOT", str(repo_root))
    load_app_settings.cache_clear()
    load_domains_settings.cache_clear()

    contract = _yaml(repo_root / "configs" / "paper_contract.yaml")
    assert contract["corpus_records"] == 3_144_903
    assert set(contract["benchmark_domains"]) == set(enabled_domains())
    inventory = contract["benchmark_inventory"]
    assert set(inventory) == set(contract["benchmark_domains"])
    totals = [sum(row[i] for row in inventory.values()) for i in range(5)]
    assert totals == [58_896, 257_393, 940_141, 22_799, 2_538]
    assert list(contract["benchmark_totals"].values()) == totals

    framework = contract["reference_framework"]
    assert framework["structural_window"] == 400
    assert framework["graph_attention"]["layers"] == GatMvpConfig().n_layers == 2
    fusion = framework["score_and_rank_fusion"]
    assert fusion == {
        "gat_score_weight": DEFAULT_ALPHA,
        "reciprocal_rank_offset": DEFAULT_RRF_K,
        "reciprocal_rank_weight": DEFAULT_RRF_W,
    }


def test_paper_contract_matches_hlm_comparison(repo_root: Path) -> None:
    contract = _yaml(repo_root / "configs" / "paper_contract.yaml")
    hlm = _yaml(repo_root / "experiments" / "ewm_top10" / "configs" / "hlm_style.yaml")
    expected = contract["hlm_comparison"]
    assert hlm["retrieval_size"] == expected["retriever_window"] == 30
    assert hlm["output_size"] == expected["output_size"] == 10
    assert hlm["llm_temperature"] == expected["temperature"] == 0
    assert expected["evaluations"] == 1
    launcher = (repo_root / "experiments" / "ewm_top10" / "run_hlm_judge.sh").read_text(
        encoding="utf-8"
    )
    assert 'WINDOW="$BASE_ROOT/windows/hlm_retriever_seed${HLM_RETRIEVER_SEED}_top30.jsonl"' in launcher
    assert "--expected-method-prefix hlm_retriever" in launcher
    assert "--require-retrieval-method-prefix hlm_retriever" in launcher
    assert "--expected-window-size 30" in launcher
    assert '"${common_args[@]}" --replicate 1 --out "$output"' in launcher


def test_paper_contract_matches_adapter_config(repo_root: Path) -> None:
    contract = _yaml(repo_root / "configs" / "paper_contract.yaml")
    adapter = contract["citation_sentence_adapter"]
    cfg = _yaml(repo_root / "configs" / "rr_adapter_v6.yaml")
    train = cfg["training"]
    trainer = train["trainer_args"]
    qlora = train["qlora"]

    assert cfg["inference"]["base_model"].endswith(adapter["base_model"])
    assert trainer["num_train_epochs"] == adapter["epochs"]
    assert trainer["per_device_train_batch_size"] == adapter["per_device_batch_size"]
    assert trainer["gradient_accumulation_steps"] == adapter["gradient_accumulation_steps"]
    assert train["tokenizer"]["max_length"] == adapter["max_length"]
    assert qlora["rank"] == adapter["lora_rank"]
    assert qlora["lora_alpha"] == adapter["lora_alpha"]
    assert qlora["lora_dropout"] == adapter["lora_dropout"]
    assert qlora["target_modules"] == adapter["target_modules"]
    assert cfg["reference_recommend"]["n_pool"] == adapter["training_pool_size"]
    assert cfg["reference_recommend"]["k"] == adapter["inference_size"]

    trainer_source = (repo_root / "rwcite" / "adapter" / "train_qlora.py").read_text(
        encoding="utf-8"
    )
    assert "load_in_8bit=True" in trainer_source
    assert "bnb_8bit_compute_dtype=torch.bfloat16" in trainer_source
    assert "enable_thinking=False" in trainer_source


def test_cross_encoder_launcher_matches_paper_contract(repo_root: Path) -> None:
    contract = _yaml(repo_root / "configs" / "paper_contract.yaml")
    ce = contract["reference_framework"]["cross_encoder"]
    launcher = (repo_root / "scripts" / "run_ce_stage1.sh").read_text(encoding="utf-8")
    assert f'--epochs "${{EPOCHS:-{ce["epochs"]}}}"' in launcher
    assert f'--list-size "${{LIST_SIZE:-{ce["list_size"]}}}"' in launcher
    assert f'--max-length "${{MAX_LENGTH:-{ce["max_length"]}}}"' in launcher
    assert '--lr "${LR:-2e-5}"' in launcher
    assert f'--warmup-ratio "${{WARMUP_RATIO:-{ce["warmup_ratio"]}}}"' in launcher


def test_reported_results_are_internally_consistent(repo_root: Path) -> None:
    contract = _yaml(repo_root / "configs" / "paper_contract.yaml")
    results = _yaml(repo_root / "configs" / "paper_results.yaml")
    multidomain = results["multidomain"]
    assert set(multidomain) == set(contract["benchmark_domains"])
    assert all(row[2] > row[0] and row[2] > row[1] for row in multidomain.values())

    ewm = results["ewm_end_to_end"]["rwcite"]
    component = results["ewm_component_analysis"]
    assert ewm["hits_at_10"] == component["score_and_rank_fusion"]["hits_at_10"]
    assert round(ewm["hits_at_30"], 3) == component["score_and_rank_fusion"]["hits_at_30"]

    relevance = results["relevance_assessment"]
    assert relevance["queries"] * relevance["candidates_per_query"] == relevance["judgments"]
    assert abs(
        relevance["core_fraction"]
        + relevance["related_fraction"]
        + relevance["peripheral_fraction"]
        + relevance["none_fraction"]
        - 0.999
    ) < 1e-9  # rounded paper percentages
