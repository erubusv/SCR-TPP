from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.schema import NormalizedRule, true_rules_from_config


DEFAULT_CONFIG_ROOT = Path("data/paper_suite/configs/hetero_source_2000_adjusted")
DEFAULT_DATASET_ROOT = Path("data/paper_suite/datasets/hetero_source_2000_adjusted")
DEFAULT_SEEDS = (111, 222, 333)


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    seed: int
    config_path: Path
    dataset_path: Path

    def load_dataset(self) -> dict[str, Any]:
        with self.dataset_path.open("rb") as f:
            return pickle.load(f)

    def true_rules(self) -> list[NormalizedRule]:
        return true_rules_from_config(self.config_path)


def iter_synthetic_cases(
    *,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    only: set[str] | None = None,
) -> list[SyntheticCase]:
    config_root = Path(config_root)
    dataset_root = Path(dataset_root)
    cases: list[SyntheticCase] = []
    for cfg_path in sorted(config_root.glob("*.yaml")):
        name = cfg_path.stem
        if only is not None and name not in only:
            continue
        for seed in seeds:
            data_path = dataset_root / f"seed_{int(seed)}" / f"{name}.pkl"
            if not data_path.exists():
                raise FileNotFoundError(f"missing synthetic dataset: {data_path}")
            cases.append(SyntheticCase(name=name, seed=int(seed), config_path=cfg_path, dataset_path=data_path))
    return cases


def merged_sequences(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return list(dataset.get("train", [])) + list(dataset.get("val", [])) + list(dataset.get("test", []))


def write_all_as_train_dataset(case: SyntheticCase, output_path: str | Path) -> Path:
    """Write a synthetic benchmark pickle where every sequence is train data.

    Synthetic rule-recovery experiments do not use validation/test metrics. This
    adapter removes ambiguity for external baselines whose loaders only read a
    `train` split.
    """

    dataset = case.load_dataset()
    sequences = merged_sequences(dataset)
    metadata = dict(dataset.get("metadata", {}))
    metadata["benchmark_split_strategy"] = "all_sequences_as_train"
    metadata["source_dataset_path"] = str(case.dataset_path)
    adapted = {
        "train": sequences,
        "val": [],
        "test": [],
        "metadata": metadata,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(adapted, f)
    return output_path
