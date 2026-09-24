from __future__ import annotations

from rwcite.graph.domain_paths import (
    DEFAULT_SPLIT_ID,
    domain_build_paths,
    domain_data_paths,
    domain_layout,
    domain_ranker_eval_paths,
    domain_ranker_paths,
    domain_ranker_pool_paths,
    domain_struct_ranker_paths,
)


def test_default_split_id():
    assert DEFAULT_SPLIT_ID == "test_admit_v1"


def test_domain_build_paths_under_retrieval_root():
    paths = domain_build_paths("gravitational_wave_retrieval/")
    assert paths["retrieval_nodes"] == "gravitational_wave_retrieval/retrieval_nodes.json"
    assert paths["failed_downloads"] == "gravitational_wave_retrieval/failed_downloads.json"
    assert paths["work_dir"] == "gravitational_wave_retrieval/description/rebuild_year10/"
    assert paths["graph_build_ckpt"] == (
        "gravitational_wave_retrieval/description/graph_build_ckpt/"
    )


def test_domain_data_paths():
    paths = domain_data_paths("gravitational_wave_retrieval/")
    assert paths["splits_dir"] == "gravitational_wave_retrieval/data/splits/"
    assert paths["jsonl_dir"] == "gravitational_wave_retrieval/data/reference_recommend/"
    assert paths["domain_embeddings"] == "gravitational_wave_retrieval/data/domain_embeddings.parquet"


def test_domain_ranker_paths_under_retrieval_root():
    paths = domain_ranker_paths(
        "gravitational_wave_retrieval/",
        base_tag="scibert_scivocab_uncased",
    )
    assert paths["citelink_model"] == "gravitational_wave_retrieval/ranker/citelink/model.npz"
    assert paths["ce_stage1"] == "gravitational_wave_retrieval/ranker/ce_sent/stage1/"
    assert paths["ce_c2s_hn"] == (
        "gravitational_wave_retrieval/ranker/ce_sent/c2s-hn-scibert_scivocab_uncased/"
    )


def test_domain_ranker_pool_and_eval_paths():
    pool = domain_ranker_pool_paths("gravitational_wave_retrieval/")
    assert pool["base_pairs"] == "gravitational_wave_retrieval/ranker/pool/ce_base_pairs.jsonl"
    struct = domain_struct_ranker_paths("gravitational_wave_retrieval/")
    assert struct["struct_model"] == "gravitational_wave_retrieval/ranker/struct/model.npz"
    assert struct["struct_feats"] == "gravitational_wave_retrieval/ranker/struct/feats/"
    ev = domain_ranker_eval_paths(
        "gravitational_wave_retrieval/",
        base_tag="scibert_scivocab_uncased",
    )
    assert ev["eval_dir"] == "gravitational_wave_retrieval/ranker/eval/"
    assert ev["c2s_report"].endswith("c2s_report_scibert_scivocab_uncased.json")


def test_domain_layout_merges_sections():
    layout = domain_layout(
        "superconducting_quantum_computing_retrieval/",
        base_tag="scibert_scivocab_uncased",
    )
    assert layout["retrieval_nodes"].startswith("superconducting_quantum_computing_retrieval/")
    assert layout["jsonl_dir"].startswith("superconducting_quantum_computing_retrieval/data/")
    assert layout["citelink_model"].startswith("superconducting_quantum_computing_retrieval/ranker/")
    assert layout["struct_model"].endswith("/ranker/struct/model.npz")
