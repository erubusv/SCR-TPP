from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


OFFICIAL_ROOT = Path("external/official_baselines/Logic_Point_Processes_ICLR")
_ORIGINAL_TELLER_GET_FEATURE: Any | None = None


def _template_cache_key(template: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(int(x) for x in template.get("body_predicate_idx", [])),
        tuple(int(x) for x in template.get("body_predicate_sign", [])),
        tuple(int(x) for x in template.get("head_predicate_sign", [])),
        tuple((int(a), int(b)) for a, b in template.get("temporal_relation_idx", [])),
        tuple(str(x) for x in template.get("temporal_relation_type", [])),
    )


def _cached_get_feature(
    self: Any,
    cur_time: float,
    head_predicate_idx: int,
    history: dict[str, Any],
    template: dict[str, Any],
) -> Any:
    original = _ORIGINAL_TELLER_GET_FEATURE
    if original is None:
        raise RuntimeError("official TELLER get_feature was not captured before cache installation")
    max_entries = int(getattr(self, "_codex_feature_cache_entries", 0))
    if max_entries <= 0:
        return original(self, cur_time, head_predicate_idx, history, template)
    cache = getattr(self, "_codex_feature_cache", None)
    if cache is None:
        cache = OrderedDict()
        self._codex_feature_cache = cache
        self._codex_feature_cache_hits = 0
        self._codex_feature_cache_misses = 0
    key = (
        id(history),
        float(cur_time),
        int(head_predicate_idx),
        float(getattr(self, "time_window", 0.0)),
        float(getattr(self, "Time_tolerance", 0.0)),
        float(getattr(self, "decay_rate", 0.0)),
        bool(getattr(self, "use_decay", True)),
        _template_cache_key(template),
    )
    if key in cache:
        cache.move_to_end(key)
        self._codex_feature_cache_hits = int(getattr(self, "_codex_feature_cache_hits", 0)) + 1
        return cache[key]
    value = original(self, cur_time, head_predicate_idx, history, template)
    cache[key] = value
    self._codex_feature_cache_misses = int(getattr(self, "_codex_feature_cache_misses", 0)) + 1
    if len(cache) > max_entries:
        cache.popitem(last=False)
    return value


def _install_official_equivalent_acceleration(model_cls: Any) -> None:
    global _ORIGINAL_TELLER_GET_FEATURE
    if getattr(model_cls, "_codex_official_equivalent_cache_installed", False):
        return
    _ORIGINAL_TELLER_GET_FEATURE = model_cls.get_feature
    model_cls.get_feature = _cached_get_feature
    model_cls._codex_official_equivalent_cache_installed = True


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _load_dataset(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _target_from_config(config: dict[str, Any]) -> int:
    for rule in config.get("rules", []):
        if "target" in rule:
            return int(rule["target"])
    return int(config["num_event_types"]) - 1


def _notation(num_types: int) -> list[str]:
    names = []
    for idx in range(int(num_types)):
        if idx < 26:
            names.append(chr(ord("A") + idx))
        else:
            names.append(f"X{idx}")
    return names


def _sequences_for_scope(dataset: dict[str, Any], split_scope: str) -> list[dict[str, Any]]:
    split_scope = str(split_scope)
    has_splits = any(key in dataset for key in ("train", "val", "test"))
    if split_scope == "train":
        sequences = list(dataset.get("train", []))
    elif split_scope == "train_val":
        sequences = list(dataset.get("train", [])) + list(dataset.get("val", []))
    elif split_scope == "all":
        sequences = list(dataset.get("train", [])) + list(dataset.get("val", [])) + list(dataset.get("test", []))
    else:
        raise ValueError(f"unsupported split_scope: {split_scope!r}")
    if has_splits and not sequences:
        raise ValueError(f"split_scope={split_scope!r} selected no sequences from an explicitly split dataset")
    if not sequences:
        sequences = list(dataset.get("sequences", []))
    if not sequences:
        raise ValueError("no sequences found; expected split keys or an unsplit `sequences` field")
    return sequences


def _as_teller_dataset(
    sequences: list[dict[str, Any]],
    *,
    num_event_types: int,
) -> dict[int, dict[int, dict[str, list[float] | list[int]]]]:
    """Convert point-event sequences to the official TELLER transition format.

    TELLER stores every predicate as a sequence of state transitions. Our
    synthetic events are point events, so each occurrence is represented as a
    positive transition. This is the least-invasive adapter: it leaves the
    official feature, likelihood, rule-generation, and pruning code unchanged.
    """

    out: dict[int, dict[int, dict[str, list[float] | list[int]]]] = {}
    for sample_id, seq in enumerate(sequences):
        sample = {idx: {"time": [0.0], "state": [0]} for idx in range(int(num_event_types))}
        for t, event in zip(seq.get("time", []), seq.get("event", [])):
            event = int(event)
            if event < 0 or event >= int(num_event_types):
                continue
            sample[event]["time"].append(float(t))
            sample[event]["state"].append(1)
        out[int(sample_id)] = sample
    return out


def _extract_rules(model: Any, *, target: int) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule_id, rule in model.logic_template.get(int(target), {}).items():
        weight = float(model.model_parameter[int(target)][int(rule_id)]["weight"].detach().cpu().numpy()[0])
        if abs(weight) <= 0.0:
            continue
        rules.append(
            {
                "sources": [int(src) for src in rule["body_predicate_idx"]],
                "sign": "excitation" if weight >= 0.0 else "inhibition",
                "target": int(target),
                "score": abs(weight),
                "temporal_relations": [
                    {
                        "pair": [int(a), int(b)],
                        "relation": str(rel),
                    }
                    for (a, b), rel in zip(rule["temporal_relation_idx"], rule["temporal_relation_type"])
                ],
                "raw": model.get_rule_str(rule, int(target)),
            }
        )
    return rules


def _master_params(model: Any, *, target: int) -> dict[str, Any]:
    params = model.model_parameter[int(target)]
    base = None
    if "base" in params:
        base = float(params["base"].detach().cpu().numpy()[0])
    weights = []
    for rule_id in sorted(model.logic_template.get(int(target), {}).keys()):
        weights.append(float(params[int(rule_id)]["weight"].detach().cpu().numpy()[0]))
    return {"base": base, "weights": weights}


def run_official_teller(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    seed: int,
    algorithm: str = "REFS",
    time_limit: float = 600.0,
    worker_num: int = 4,
    num_epoch: int = 5,
    batch_size: int = 64,
    max_rule_body_length: int | None = None,
    max_num_rule: int | None = None,
    split_scope: str = "train",
    exclude_target_from_sources: bool = False,
    feature_cache_entries: int = 200000,
) -> dict[str, Any]:
    started = time.perf_counter()
    official_root = OFFICIAL_ROOT.resolve()
    if not official_root.exists():
        raise FileNotFoundError(f"missing official TELLER clone: {official_root}")
    sys.path.insert(0, str(official_root))
    from logic_learning import Logic_Learning_Model  # type: ignore

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    _install_official_equivalent_acceleration(Logic_Learning_Model)

    config = _load_yaml(config_path)
    dataset = _load_dataset(dataset_path)
    sequences = _sequences_for_scope(dataset, split_scope)
    num_event_types = int(config["num_event_types"])
    target = _target_from_config(config)
    if bool(exclude_target_from_sources):
        body = [idx for idx in range(num_event_types) if idx != int(target)]
    else:
        body = list(range(num_event_types))
    training_dataset = _as_teller_dataset(sequences, num_event_types=num_event_types)

    model = Logic_Learning_Model([int(target)])
    model.num_predicate = int(num_event_types)
    model.predicate_set = list(range(num_event_types))
    model.body_pred_set = body
    model.predicate_notation = _notation(num_event_types)
    model.instant_pred_set = list(range(num_event_types))
    # TELLER uses a finite history window in its synthetic runner.
    model.time_window = float(config.get("time_window", config.get("baseline_time_window", 10.0)))
    model.time_limit = float(time_limit)
    model.worker_num = int(worker_num)
    model.batch_size = min(max(int(batch_size), 1), max(int(len(training_dataset)), 1))
    model.num_epoch = int(num_epoch)
    model.num_epoch_final = max(int(num_epoch), 8)
    model.max_rule_body_length = min(int(max_rule_body_length) if max_rule_body_length is not None else 3, len(body))
    model.max_num_rule = int(max_num_rule) if max_num_rule is not None else max(20, len(config.get("rules", [])) * 3)
    model.use_exp_kernel = True
    model.use_2_bases = False
    model.reverse_head_sign = True
    model.print_time = False
    model.weight_lr = 0.0001
    model.base_lr = 0.00005
    model.init_base = 0.01
    model.init_weight = 0.1
    model.gain_threshold = 0.02
    model.weight_threshold = 0.05
    model.strict_weight_threshold = 0.1
    model.init_params()
    model._codex_feature_cache_entries = int(feature_cache_entries)
    model._codex_feature_cache = OrderedDict()
    model._codex_feature_cache_hits = 0
    model._codex_feature_cache_misses = 0

    log_buffer = io.StringIO()
    terminated_by_time_limit = False
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        try:
            if str(algorithm).upper() == "RAFS":
                model.RAFS(int(target), training_dataset, {}, tag=f"benchmark_{Path(config_path).stem}_{seed}", init_params=False)
            elif str(algorithm).upper() == "REFS":
                model.REFS(int(target), training_dataset, {}, tag=f"benchmark_{Path(config_path).stem}_{seed}", init_params=False)
            else:
                model.Brute(int(target), training_dataset)
        except SystemExit as exc:
            terminated_by_time_limit = "exceeding maxinum time" in str(exc)
            if not terminated_by_time_limit:
                raise

    rules = _extract_rules(model, target=int(target))
    feature_cache = getattr(model, "_codex_feature_cache", {})
    feature_cache_size = int(len(feature_cache))
    feature_cache_hits = int(getattr(model, "_codex_feature_cache_hits", 0))
    feature_cache_misses = int(getattr(model, "_codex_feature_cache_misses", 0))
    model._codex_feature_cache = OrderedDict()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(Path(output_path).with_suffix(".official_teller.pkl"))
    with Path(checkpoint_path).open("wb") as f:
        pickle.dump(model, f)
    payload = {
        "model": "TELLER",
        "target": int(target),
        "predicted_rules": rules,
        "metadata": {
            "implementation_type": "official_equivalent_accelerated_teller",
            "acceleration_note": (
                "Official TELLER source is executed unchanged except for an exact LRU "
                "cache around get_feature. Cache misses call the official implementation, "
                "so selected rules and fitted weights are intended to match uncached "
                "official execution under the same seed and hyperparameters."
            ),
            "source_url": "https://github.com/FengMingquan-sjtu/Logic_Point_Processes_ICLR",
            "source_commit": "4160b27bb4606e87129bd36985fbba2ccdb9c925",
            "algorithm": str(algorithm).upper(),
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "seed": int(seed),
            "num_sequences": int(len(sequences)),
            "batch_size": int(model.batch_size),
            "max_rule_body_length": int(model.max_rule_body_length),
            "max_num_rule": int(model.max_num_rule),
            "split_scope": str(split_scope),
            "include_target_history": not bool(exclude_target_from_sources),
            "feature_cache_entries": int(feature_cache_entries),
            "feature_cache_size": feature_cache_size,
            "feature_cache_hits": feature_cache_hits,
            "feature_cache_misses": feature_cache_misses,
            "time_limit": float(time_limit),
            "terminated_by_time_limit": bool(terminated_by_time_limit),
            "teller_master_params": _master_params(model, target=int(target)),
            "official_model_pickle_path": checkpoint_path,
            "runtime_sec": float(time.perf_counter() - started),
            "stdout_tail": log_buffer.getvalue()[-8000:],
        },
    }
    output_path = Path(output_path)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Official TELLER runner with a point-event data adapter.")
    parser.add_argument("--model", default="TELLER")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithm", default="REFS", choices=["REFS", "RAFS", "Brute"])
    parser.add_argument("--time_limit", type=float, default=600.0)
    parser.add_argument("--worker_num", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_rule_body_length", type=int, default=None)
    parser.add_argument("--max_num_rule", type=int, default=None)
    parser.add_argument("--split_scope", choices=["train", "train_val", "all"], default="train")
    parser.add_argument("--exclude_target_from_sources", action="store_true")
    parser.add_argument("--feature_cache_entries", type=int, default=200000)
    args = parser.parse_args()
    payload = run_official_teller(
        config_path=args.config,
        dataset_path=args.data,
        output_path=args.output,
        seed=int(args.seed),
        algorithm=str(args.algorithm),
        time_limit=float(args.time_limit),
        worker_num=int(args.worker_num),
        num_epoch=int(args.num_epoch),
        batch_size=int(args.batch_size),
        max_rule_body_length=args.max_rule_body_length,
        max_num_rule=args.max_num_rule,
        split_scope=str(args.split_scope),
        exclude_target_from_sources=bool(args.exclude_target_from_sources),
        feature_cache_entries=int(args.feature_cache_entries),
    )
    print(json.dumps({"model": "TELLER", "rules": len(payload["predicted_rules"])}, indent=2))


if __name__ == "__main__":
    main()
