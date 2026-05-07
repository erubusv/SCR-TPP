from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from scipy.optimize import linear_sum_assignment
import yaml


SIGN_ALIASES = {
    "+": "excitation",
    "pos": "excitation",
    "positive": "excitation",
    "exc": "excitation",
    "excitation": "excitation",
    "promote": "excitation",
    "promotes": "excitation",
    "-": "inhibition",
    "neg": "inhibition",
    "negative": "inhibition",
    "inh": "inhibition",
    "inhibition": "inhibition",
    "inhibit": "inhibition",
    "inhibits": "inhibition",
}


def normalize_sign(sign: str) -> str:
    key = str(sign).strip().lower()
    if key not in SIGN_ALIASES:
        raise ValueError(f"unknown rule sign: {sign!r}")
    return SIGN_ALIASES[key]


def _source_name_to_index(token: str) -> int:
    token = str(token).strip()
    if not token:
        raise ValueError("empty source token")
    if token.upper().startswith("X") and token[1:].isdigit():
        return int(token[1:])
    if token.isdigit():
        return int(token)
    if len(token) == 1 and token.isalpha():
        return ord(token.upper()) - ord("A")
    raise ValueError(f"cannot parse source token: {token!r}")


def _target_name_to_index(token: str | None, fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)
    if token is None:
        raise ValueError("target is required")
    token = str(token).strip()
    if token.upper().startswith("T") and token[1:].isdigit():
        return int(token[1:])
    if token.isdigit():
        return int(token)
    if len(token) == 1 and token.isalpha() and token.upper() != "T":
        return ord(token.upper()) - ord("A")
    raise ValueError(f"cannot infer numeric target from {token!r}; pass target explicitly")


@dataclass(frozen=True, order=True)
class NormalizedRule:
    """Canonical signed source-set rule used across all benchmark models."""

    sources: tuple[int, ...]
    sign: str
    target: int
    temporal_relations: tuple[str, ...] = field(default_factory=tuple, compare=False)
    raw: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(sorted({int(src) for src in self.sources})))
        object.__setattr__(self, "sign", normalize_sign(self.sign))
        object.__setattr__(self, "target", int(self.target))
        object.__setattr__(
            self,
            "temporal_relations",
            tuple(str(rel) for rel in self.temporal_relations),
        )
        if not self.sources:
            raise ValueError("rule sources cannot be empty")

    @property
    def key(self) -> tuple[tuple[int, ...], str, int]:
        return (self.sources, self.sign, self.target)

    def to_dict(self, *, include_relations: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sources": list(self.sources),
            "sign": self.sign,
            "target": int(self.target),
        }
        if include_relations and self.temporal_relations:
            out["temporal_relations"] = list(self.temporal_relations)
        if self.raw is not None:
            out["raw"] = str(self.raw)
        return out

    @staticmethod
    def from_obj(obj: Any, *, default_target: int | None = None) -> "NormalizedRule":
        if isinstance(obj, NormalizedRule):
            return obj
        if isinstance(obj, str):
            return parse_rule_string(obj, default_target=default_target)
        if not isinstance(obj, dict):
            raise TypeError(f"cannot parse rule object of type {type(obj)!r}")
        if "sources" in obj:
            sources = tuple(int(src) for src in obj["sources"])
        elif "source_set" in obj:
            sources = tuple(int(src) for src in obj["source_set"])
        elif "body" in obj:
            sources = tuple(_source_name_to_index(src) for src in obj["body"])
        elif "condition" in obj and isinstance(obj["condition"], dict):
            sources = tuple(int(src) for src in obj["condition"].keys())
        else:
            raise ValueError(f"rule object has no sources/source_set/body: {obj!r}")
        sign = obj.get("sign")
        if sign is None:
            w_pos = float(obj.get("W_pos", 0.0))
            w_neg = float(obj.get("W_neg", 0.0))
            if w_pos > 0.0 and w_neg == 0.0:
                sign = "excitation"
            elif w_neg > 0.0 and w_pos == 0.0:
                sign = "inhibition"
            else:
                raise ValueError(f"cannot infer sign from rule object: {obj!r}")
        target = int(obj.get("target", default_target if default_target is not None else -1))
        if target < 0:
            raise ValueError(f"rule object has no target and no default target: {obj!r}")
        temporal_relations = tuple(obj.get("temporal_relations", obj.get("relations", ())) or ())
        return NormalizedRule(
            sources=sources,
            sign=str(sign),
            target=target,
            temporal_relations=temporal_relations,
            raw=json.dumps(obj, sort_keys=True),
        )


RULE_TEXT_RE = re.compile(
    r"^\s*(?P<body>.+?)\s*->\s*(?P<target>[^:]+?)(?:\s*:\s*(?P<sign>.+))?\s*$"
)


def parse_rule_string(text: str, *, default_target: int | None = None) -> NormalizedRule:
    """Parse strings such as ``A and C -> T : inhibition``.

    Pass ``default_target`` when the string uses a symbolic target such as ``T``.
    """

    raw = str(text)
    match = RULE_TEXT_RE.match(raw)
    if not match:
        raise ValueError(f"cannot parse rule string: {text!r}")
    body = match.group("body")
    target_token = match.group("target").strip()
    sign = match.group("sign") or "excitation"
    body = re.sub(r"\bAND\b", "and", body, flags=re.IGNORECASE)
    parts = [part.strip() for part in body.split(" and ") if part.strip()]
    sources = tuple(_source_name_to_index(part) for part in parts)
    target = _target_name_to_index(target_token, default_target)
    return NormalizedRule(sources=sources, sign=sign, target=target, raw=raw)


def normalize_rules(
    rules: Iterable[Any],
    *,
    default_target: int | None = None,
) -> list[NormalizedRule]:
    unique = {}
    for rule in rules:
        parsed = NormalizedRule.from_obj(rule, default_target=default_target)
        unique[parsed.key] = parsed
    return [unique[key] for key in sorted(unique)]


def source_jaccard(a: NormalizedRule, b: NormalizedRule) -> float:
    left = set(a.sources)
    right = set(b.sources)
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def _rule_pair_score(
    true_rule: NormalizedRule,
    pred_rule: NormalizedRule,
    *,
    require_same_sign: bool,
) -> float:
    if int(true_rule.target) != int(pred_rule.target):
        return 0.0
    if require_same_sign and str(true_rule.sign) != str(pred_rule.sign):
        return 0.0
    return source_jaccard(true_rule, pred_rule)


def _source_jaccard_match(
    *,
    true_rules: list[NormalizedRule],
    pred_rules: list[NormalizedRule],
    require_same_sign: bool,
) -> dict[str, Any]:
    if not true_rules or not pred_rules:
        score_sum = 0.0
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "score_sum": score_sum,
            "pairs": [],
        }

    scores = [
        [
            _rule_pair_score(true_rule, pred_rule, require_same_sign=require_same_sign)
            for pred_rule in pred_rules
        ]
        for true_rule in true_rules
    ]
    row_ind, col_ind = linear_sum_assignment([[-value for value in row] for row in scores])

    pairs = []
    score_sum = 0.0
    for true_idx, pred_idx in sorted(zip(row_ind, col_ind), key=lambda item: (int(item[0]), int(item[1]))):
        score = float(scores[int(true_idx)][int(pred_idx)])
        if score <= 0.0:
            continue
        score_sum += score
        true_rule = true_rules[int(true_idx)]
        pred_rule = pred_rules[int(pred_idx)]
        pairs.append(
            {
                "true_index": int(true_idx),
                "predicted_index": int(pred_idx),
                "jaccard": float(score),
                "true_rule": true_rule.to_dict(),
                "predicted_rule": pred_rule.to_dict(),
            }
        )

    recall = float(score_sum) / float(max(len(true_rules), 1))
    precision = float(score_sum) / float(max(len(pred_rules), 1))
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "score_sum": float(score_sum),
        "pairs": pairs,
    }


def load_json_or_yaml(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def true_rules_from_config(config_path: str | Path) -> list[NormalizedRule]:
    cfg = load_json_or_yaml(config_path)
    rules = []
    for rule in cfg.get("rules", []):
        rules.append(NormalizedRule.from_obj(rule, default_target=rule.get("target")))
    return normalize_rules(rules)


def match_rule_sets(
    predicted: Iterable[Any],
    truth: Iterable[Any],
    *,
    default_target: int | None = None,
) -> dict[str, Any]:
    pred_rules = normalize_rules(predicted, default_target=default_target)
    true_rules = normalize_rules(truth, default_target=default_target)
    pred_by_key = {rule.key: rule for rule in pred_rules}
    true_by_key = {rule.key: rule for rule in true_rules}
    matched_keys = sorted(set(pred_by_key) & set(true_by_key))
    missing_keys = sorted(set(true_by_key) - set(pred_by_key))
    extra_keys = sorted(set(pred_by_key) - set(true_by_key))
    recall = len(matched_keys) / max(len(true_by_key), 1)
    precision = len(matched_keys) / max(len(pred_by_key), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    source_jaccard_match = _source_jaccard_match(
        true_rules=true_rules,
        pred_rules=pred_rules,
        require_same_sign=False,
    )
    signed_source_jaccard_match = _source_jaccard_match(
        true_rules=true_rules,
        pred_rules=pred_rules,
        require_same_sign=True,
    )
    return {
        "predicted_rules": [rule.to_dict() for rule in pred_rules],
        "true_rules": [rule.to_dict() for rule in true_rules],
        "matched": [true_by_key[key].to_dict() for key in matched_keys],
        "missing": [true_by_key[key].to_dict() for key in missing_keys],
        "extra": [pred_by_key[key].to_dict() for key in extra_keys],
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "source_jaccard_recall": float(source_jaccard_match["recall"]),
        "source_jaccard_precision": float(source_jaccard_match["precision"]),
        "source_jaccard_f1": float(source_jaccard_match["f1"]),
        "signed_source_jaccard_recall": float(signed_source_jaccard_match["recall"]),
        "signed_source_jaccard_precision": float(signed_source_jaccard_match["precision"]),
        "signed_source_jaccard_f1": float(signed_source_jaccard_match["f1"]),
        "source_jaccard_pairs": source_jaccard_match["pairs"],
        "signed_source_jaccard_pairs": signed_source_jaccard_match["pairs"],
    }


@dataclass
class SyntheticRuleRecoveryResult:
    model: str
    dataset: str
    seed: int
    predicted_rules: list[dict[str, Any]]
    matched: list[dict[str, Any]]
    missing: list[dict[str, Any]]
    extra: list[dict[str, Any]]
    recall: float
    precision: float
    f1: float
    runtime_sec: float | None
    source_jaccard_recall: float = 0.0
    source_jaccard_precision: float = 0.0
    source_jaccard_f1: float = 0.0
    signed_source_jaccard_recall: float = 0.0
    signed_source_jaccard_precision: float = 0.0
    signed_source_jaccard_f1: float = 0.0
    source_jaccard_pairs: list[dict[str, Any]] = field(default_factory=list)
    signed_source_jaccard_pairs: list[dict[str, Any]] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        if not self.dataset:
            raise ValueError("dataset is required")
        if int(self.seed) <= 0:
            raise ValueError("seed must be positive")
        for field_name in (
            "recall",
            "precision",
            "f1",
            "source_jaccard_recall",
            "source_jaccard_precision",
            "source_jaccard_f1",
            "signed_source_jaccard_recall",
            "signed_source_jaccard_precision",
            "signed_source_jaccard_f1",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")
        for rule_list_name in ("predicted_rules", "matched", "missing", "extra"):
            for rule in getattr(self, rule_list_name):
                NormalizedRule.from_obj(rule)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "model": self.model,
            "dataset": self.dataset,
            "seed": int(self.seed),
            "predicted_rules": self.predicted_rules,
            "matched": self.matched,
            "missing": self.missing,
            "extra": self.extra,
            "recall": float(self.recall),
            "precision": float(self.precision),
            "f1": float(self.f1),
            "source_jaccard_recall": float(self.source_jaccard_recall),
            "source_jaccard_precision": float(self.source_jaccard_precision),
            "source_jaccard_f1": float(self.source_jaccard_f1),
            "signed_source_jaccard_recall": float(self.signed_source_jaccard_recall),
            "signed_source_jaccard_precision": float(self.signed_source_jaccard_precision),
            "signed_source_jaccard_f1": float(self.signed_source_jaccard_f1),
            "source_jaccard_pairs": list(self.source_jaccard_pairs),
            "signed_source_jaccard_pairs": list(self.signed_source_jaccard_pairs),
            "runtime_sec": None if self.runtime_sec is None else float(self.runtime_sec),
            "model_metadata": dict(self.model_metadata),
        }


@dataclass
class RealWorldPredictionResult:
    model: str
    dataset: str
    target: str
    nll: float | None
    time_mae: float | None
    time_rmse: float | None = None
    type_acc: float | None = None
    topk: dict[str, float] = field(default_factory=dict)
    runtime_sec: float | None = None
    learned_rules: list[dict[str, Any]] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model is required")
        if not self.dataset:
            raise ValueError("dataset is required")
        if not self.target:
            raise ValueError("target is required")
        for field_name in ("nll", "time_mae", "time_rmse", "type_acc"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite or null")
        if self.type_acc is not None and not (0.0 <= float(self.type_acc) <= 1.0):
            raise ValueError("type_acc must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "model": self.model,
            "dataset": self.dataset,
            "target": self.target,
            "nll": self.nll,
            "time_mae": self.time_mae,
            "time_rmse": self.time_rmse,
            "type_acc": self.type_acc,
            "topk": dict(self.topk),
            "runtime_sec": self.runtime_sec,
            "learned_rules": list(self.learned_rules),
            "model_metadata": dict(self.model_metadata),
        }
