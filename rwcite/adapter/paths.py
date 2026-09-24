"""Per-domain paths for the reported generator and archived adapter artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rwcite.adapter import ADAPTER_REL, ADAPTER_REL_V7
from rwcite.gat.domain_layout import GatDomainPaths, resolve_gat_domain
from rwcite.gat import protocol as P


@dataclass(frozen=True)
class AdapterV6Paths:
    domain: str
    lane: str
    root: Path
    adapter_dir: Path
    data_dir: Path
    model_dir: Path
    trainer_dir: Path
    pools_train: Path
    pools_test: Path
    train_jsonl: Path
    cite_eval_jsonl: Path
    meta_json: Path
    gat: GatDomainPaths

    @property
    def adapter_rr_rel(self) -> str:
        return f"{self.lane}/{ADAPTER_REL}/model"


@dataclass(frozen=True)
class AdapterV7Paths:
    domain: str
    lane: str
    root: Path
    adapter_dir: Path
    data_dir: Path
    model_dir: Path
    trainer_dir: Path
    pools_train: Path
    pools_test: Path
    train_jsonl: Path
    analyses_train: Path
    analyses_test: Path
    cite_eval_jsonl: Path
    meta_json: Path
    eval_json: Path
    gat: GatDomainPaths
    v6_pools_train: Path
    v6_pools_test: Path

    @property
    def adapter_rr_v7_rel(self) -> str:
        return f"{self.lane}/{ADAPTER_REL_V7}/model"


def resolve_adapter_v6(domain: str, root: Path | None = None) -> AdapterV6Paths:
    root = P.root_path() if root is None else Path(root)
    gat = resolve_gat_domain(domain, root)
    adapter_dir = root / gat.lane / ADAPTER_REL
    data_dir = adapter_dir / "data"
    return AdapterV6Paths(
        domain=gat.domain,
        lane=gat.lane,
        root=root,
        adapter_dir=adapter_dir,
        data_dir=data_dir,
        model_dir=adapter_dir / "model",
        trainer_dir=adapter_dir / "trainer_outputs",
        pools_train=data_dir / "pools_train_l0.jsonl",
        pools_test=data_dir / "pools_test_l0.jsonl",
        train_jsonl=data_dir / "train.jsonl",
        cite_eval_jsonl=data_dir / "cite_eval.jsonl",
        meta_json=data_dir / "meta.json",
        gat=gat,
    )


def resolve_adapter_v7(domain: str, root: Path | None = None) -> AdapterV7Paths:
    root = P.root_path() if root is None else Path(root)
    gat = resolve_gat_domain(domain, root)
    adapter_dir = root / gat.lane / ADAPTER_REL_V7
    data_dir = adapter_dir / "data"
    v6 = root / gat.lane / ADAPTER_REL / "data"
    return AdapterV7Paths(
        domain=gat.domain,
        lane=gat.lane,
        root=root,
        adapter_dir=adapter_dir,
        data_dir=data_dir,
        model_dir=adapter_dir / "model",
        trainer_dir=adapter_dir / "trainer_outputs",
        pools_train=data_dir / "pools_train_l0.jsonl",
        pools_test=data_dir / "pools_test_l0.jsonl",
        train_jsonl=data_dir / "train.jsonl",
        analyses_train=data_dir / "analyses_train.jsonl",
        analyses_test=data_dir / "analyses_test.jsonl",
        cite_eval_jsonl=data_dir / "cite_eval.jsonl",
        meta_json=data_dir / "meta.json",
        eval_json=data_dir / "eval_v7.json",
        gat=gat,
        v6_pools_train=v6 / "pools_train_l0.jsonl",
        v6_pools_test=v6 / "pools_test_l0.jsonl",
    )


def ensure_adapter_dirs(paths: AdapterV6Paths) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.trainer_dir.mkdir(parents=True, exist_ok=True)


def ensure_adapter_v7_dirs(paths: AdapterV7Paths) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.trainer_dir.mkdir(parents=True, exist_ok=True)


def resolve_v7_pool(paths: AdapterV7Paths, *, split: str) -> Path:
    """Prefer v7 data pool; fall back to v6 dump."""
    local = paths.pools_train if split == "train" else paths.pools_test
    if local.is_file() or local.is_symlink():
        try:
            if local.resolve().is_file():
                return local
        except OSError:
            pass
    upstream = paths.v6_pools_train if split == "train" else paths.v6_pools_test
    if upstream.is_file():
        return upstream
    raise FileNotFoundError(
        f"missing {split} L0 pool at {local} and {upstream}; "
        "run dump_rr_l0_pools first"
    )
