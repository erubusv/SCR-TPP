from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


RELATION_NAMES = ("before", "equal", "after", "none")


MODEL_INFO: dict[str, dict[str, Any]] = {
    "CLNN": {
        "implementation_type": "restricted_paper_based_reimplementation_weighted_clock_logic_tpp",
        "source_url": "https://openreview.net/forum?id=YfUICnZMwk7",
        "official_code_available": False,
        "signed": True,
        "paper_components": [
            "first/last event masking and clock signals",
            "POP/SOP smooth truth degrees, Eqs. (2)-(3)",
            "weighted conjunction/disjunction with literal complements, Eqs. (4)-(6)",
            "architecture-cell softmax relaxation, Eq. (7)",
            "wCL-informed log-link intensity, Eq. (8)",
            "TPP maximum likelihood with numerical integral, Eqs. (9)-(12)",
        ],
    },
    "NSTPP": {
        "implementation_type": "restricted_paper_based_reimplementation_neural_symbolic_tpp",
        "source_url": "https://proceedings.mlr.press/v235/yang24ag.html",
        "official_code_available": False,
        "signed": False,
        "paper_components": [
            "Horn rule with temporal relations, Eq. (1)",
            "additive NS-TPP intensity, Eq. (2)",
            "fixed predicate embeddings and learnable rule embeddings, Eqs. (3)-(5)",
            "static and temporal neural-symbolic features, Eqs. (6)-(9)",
            "soft-min robust feature construction, Eq. (10)",
            "single-rule sequential covering and final refinement, Eq. (11)",
        ],
    },
    "CLUSTER": {
        "implementation_type": "restricted_paper_based_reimplementation_latent_causal_rule_em",
        "source_url": "https://proceedings.mlr.press/v238/kuang24a.html",
        "official_code_available": True,
        "signed": False,
        "paper_components": [
            "Horn rule and Boolean temporal feature, Eqs. (2)-(3)",
            "mixture-of-components rule intensity, Eq. (4)",
            "complete-data likelihood and EM posterior, Eqs. (5)-(9)",
            "continuous rule relaxation and dummy predicates, Eqs. (10)-(11)",
            "Laplace Boolean feature and temporal soft relation approximation, Eqs. (12)-(14)",
            "differentiable relaxed top-K subset sampling, Alg. 1 and Eq. (15)",
            "closed-form mixture-prior update, Eq. (16)",
        ],
    },
}


@dataclass(frozen=True)
class Design:
    source_ids: tuple[int, ...]
    target: int
    time_horizon: float
    event_facts: torch.Tensor
    grid_facts: torch.Tensor
    event_clocks: torch.Tensor
    grid_clocks: torch.Tensor
    event_last_times: torch.Tensor
    grid_last_times: torch.Tensor
    event_times: torch.Tensor
    grid_times: torch.Tensor
    event_grid_positions: torch.Tensor
    event_valid: torch.Tensor
    grid_valid: torch.Tensor
    grid_weights: torch.Tensor
    event_sequence_ids: torch.Tensor
    grid_sequence_ids: torch.Tensor
    sequence_count: int


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _load_dataset(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


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
    if not sequences and isinstance(dataset, dict):
        sequences = list(dataset.get("sequences", []))
    if not sequences:
        raise ValueError("no sequences found; expected split keys or an unsplit `sequences` field")
    return sequences


def _target_from_config(config: dict[str, Any]) -> int:
    for rule in config.get("rules", []):
        if "target" in rule:
            return int(rule["target"])
    return int(config["num_event_types"]) - 1


def _source_ids(config: dict[str, Any], target: int, *, include_target_history: bool) -> tuple[int, ...]:
    if bool(include_target_history):
        return tuple(range(int(config["num_event_types"])))
    return tuple(idx for idx in range(int(config["num_event_types"])) if int(idx) != int(target))


def _sequence_horizon(seq: dict[str, Any], fallback_horizon: float, *, horizon_mode: str) -> float:
    if str(horizon_mode) == "config":
        return float(fallback_horizon)
    for key in ("time_horizon", "t_stop", "stop_time", "window", "duration"):
        if key in seq:
            return max(float(seq[key]), 1e-6)
    times = seq.get("time", [])
    if times:
        return max(float(max(times)), 1e-6)
    return float(fallback_horizon)


def _events_by_source(seq: dict[str, Any], source_ids: tuple[int, ...]) -> dict[int, np.ndarray]:
    out = {int(src): [] for src in source_ids}
    for t, event in zip(seq.get("time", []), seq.get("event", [])):
        event = int(event)
        if event in out:
            out[event].append(float(t))
    return {src: np.asarray(vals, dtype=np.float64) for src, vals in out.items()}


def _clock_row(
    *,
    by_source: dict[int, np.ndarray],
    source_ids: tuple[int, ...],
    t: float,
    time_horizon: float,
    masking: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    no_event_clock = 1.5 * float(time_horizon)
    facts: list[float] = []
    clocks: list[float] = []
    last_times: list[float] = []
    valid: list[float] = []
    for src in source_ids:
        times = by_source[int(src)]
        pos = int(np.searchsorted(times, float(t), side="left")) - 1
        if pos < 0:
            facts.append(0.0)
            clocks.append(no_event_clock)
            last_times.append(0.0)
            valid.append(0.0)
            continue
        use_pos = 0 if str(masking) == "first" else pos
        src_time = float(times[use_pos])
        facts.append(1.0)
        clocks.append(max(float(t) - src_time, 0.0))
        last_times.append(src_time)
        valid.append(1.0)
    return facts, clocks, last_times, valid


def _build_design(
    *,
    config: dict[str, Any],
    dataset: dict[str, Any],
    grid_size: int,
    masking: str,
    split_scope: str,
    include_target_history: bool,
    horizon_mode: str,
    device: str,
) -> Design:
    target = _target_from_config(config)
    source_ids = _source_ids(config, target, include_target_history=include_target_history)
    time_horizon = float(config.get("time_horizon", config.get("time_window", 120.0)))
    sequences = _sequences_for_scope(dataset, split_scope)

    event_facts: list[list[float]] = []
    grid_facts: list[list[float]] = []
    event_clocks: list[list[float]] = []
    grid_clocks: list[list[float]] = []
    event_last_times: list[list[float]] = []
    grid_last_times: list[list[float]] = []
    event_times: list[float] = []
    grid_times: list[float] = []
    event_grid_positions: list[int] = []
    event_valid: list[list[float]] = []
    grid_valid: list[list[float]] = []
    grid_weights: list[float] = []
    event_sequence_ids: list[int] = []
    grid_sequence_ids: list[int] = []

    for seq_idx, seq in enumerate(sequences):
        by_source = _events_by_source(seq, source_ids)
        seq_horizon = _sequence_horizon(seq, time_horizon, horizon_mode=horizon_mode)
        dt = seq_horizon / max(int(grid_size), 1)
        for t, event in zip(seq.get("time", []), seq.get("event", [])):
            if int(event) != int(target):
                continue
            facts, clocks, last_times, valid = _clock_row(
                by_source=by_source,
                source_ids=source_ids,
                t=float(t),
                time_horizon=seq_horizon,
                masking=masking,
            )
            event_facts.append(facts)
            event_clocks.append(clocks)
            event_last_times.append(last_times)
            event_times.append(float(t))
            event_grid_positions.append(int(np.searchsorted((np.arange(int(grid_size)) + 0.5) * dt, float(t), side="right")) - 1)
            event_valid.append(valid)
            event_sequence_ids.append(int(seq_idx))
        for idx in range(int(grid_size)):
            t = (idx + 0.5) * dt
            facts, clocks, last_times, valid = _clock_row(
                by_source=by_source,
                source_ids=source_ids,
                t=float(t),
                time_horizon=seq_horizon,
                masking=masking,
            )
            grid_facts.append(facts)
            grid_clocks.append(clocks)
            grid_last_times.append(last_times)
            grid_times.append(float(t))
            grid_valid.append(valid)
            grid_weights.append(float(dt))
            grid_sequence_ids.append(int(seq_idx))

    p = len(source_ids)

    def tensor2(rows: list[list[float]]) -> torch.Tensor:
        if not rows:
            return torch.zeros((0, p), dtype=torch.float32, device=device)
        return torch.tensor(rows, dtype=torch.float32, device=device)

    return Design(
        source_ids=source_ids,
        target=target,
        time_horizon=time_horizon,
        event_facts=tensor2(event_facts),
        grid_facts=tensor2(grid_facts),
        event_clocks=tensor2(event_clocks),
        grid_clocks=tensor2(grid_clocks),
        event_last_times=tensor2(event_last_times),
        grid_last_times=tensor2(grid_last_times),
        event_times=torch.tensor(event_times, dtype=torch.float32, device=device),
        grid_times=torch.tensor(grid_times, dtype=torch.float32, device=device),
        event_grid_positions=torch.tensor(event_grid_positions, dtype=torch.long, device=device),
        event_valid=tensor2(event_valid),
        grid_valid=tensor2(grid_valid),
        grid_weights=torch.tensor(grid_weights, dtype=torch.float32, device=device),
        event_sequence_ids=torch.tensor(event_sequence_ids, dtype=torch.long, device=device),
        grid_sequence_ids=torch.tensor(grid_sequence_ids, dtype=torch.long, device=device),
        sequence_count=int(len(sequences)),
    )


def _softmin(values: torch.Tensor, *, rho: float, dim: int = -1) -> torch.Tensor:
    return -torch.logsumexp(-float(rho) * values, dim=dim) / float(rho) + math.log(values.shape[dim]) / float(rho)


def _weighted_softmin(values: torch.Tensor, *, temperature: float, dim: int = -1) -> torch.Tensor:
    weights = torch.softmax(-values / max(float(temperature), 1e-8), dim=dim)
    return torch.sum(weights * values, dim=dim)


def _weighted_softmax(values: torch.Tensor, *, temperature: float, dim: int = -1) -> torch.Tensor:
    weights = torch.softmax(values / max(float(temperature), 1e-8), dim=dim)
    return torch.sum(weights * values, dim=dim)


def _relation_facts(times: torch.Tensor, valid: torch.Tensor, *, delta: float) -> torch.Tensor:
    # Output shape: Q x P x P x 4 for before/equal/after/none.
    diff = times.unsqueeze(2) - times.unsqueeze(1)
    valid_pair = (valid.unsqueeze(2) * valid.unsqueeze(1)).clamp(0.0, 1.0)
    before = ((diff < -float(delta)).float() * valid_pair)
    equal = ((torch.abs(diff) <= float(delta)).float() * valid_pair)
    after = ((diff > float(delta)).float() * valid_pair)
    none = torch.ones_like(before)
    return torch.stack([before, equal, after, none], dim=-1)


def _relation_string(left_source: int, right_source: int, relation: str, *, threshold: float | None = None) -> str:
    if threshold is None:
        return f"{int(left_source)} {relation} {int(right_source)}"
    return f"{int(left_source)} {relation} {int(right_source)}; threshold={float(threshold):.6g}"


def _nll_loglink(event_eta: torch.Tensor, grid_eta: torch.Tensor, grid_weights: torch.Tensor) -> torch.Tensor:
    return -torch.sum(event_eta) + torch.sum(grid_weights * torch.exp(torch.clamp(grid_eta, -30.0, 30.0)))


def _nll_additive(event_lambda: torch.Tensor, grid_lambda: torch.Tensor, grid_weights: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -torch.sum(torch.log(torch.clamp(event_lambda, min=eps))) + torch.sum(grid_weights * grid_lambda)


class CLNNModel(torch.nn.Module):
    def __init__(self, num_sources: int, *, num_formulas: int, alpha: float) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_formulas = int(num_formulas)
        self.alpha = float(alpha)
        self.pairs = [(left, right) for left in range(self.num_sources) for right in range(left + 1, self.num_sources)]
        pair_count = max(len(self.pairs), 1)
        self.u_sop_raw = torch.nn.Parameter(torch.empty(num_formulas, num_sources).uniform_(0.0, 1.0))
        self.u_pop_raw = torch.nn.Parameter(torch.empty(num_formulas, pair_count).uniform_(0.0, 1.0))
        self.w_sop_raw = torch.nn.Parameter(torch.empty(num_formulas, 2 * num_sources).uniform_(0.0, 1.0))
        self.w_pop_select_raw = torch.nn.Parameter(torch.empty(num_formulas, pair_count).uniform_(0.0, 1.0))
        self.w_pop_dir_raw = torch.nn.Parameter(torch.empty(num_formulas, pair_count, 2).uniform_(0.0, 1.0))
        self.beta_sop_raw = torch.nn.Parameter(torch.zeros(num_formulas))
        self.beta_pop_literal_raw = torch.nn.Parameter(torch.zeros(num_formulas, pair_count))
        self.beta_pop_select_raw = torch.nn.Parameter(torch.zeros(num_formulas))
        self.beta_formula_and_raw = torch.nn.Parameter(torch.zeros(num_formulas))
        self.beta_formula_or_raw = torch.nn.Parameter(torch.zeros(num_formulas))
        self.arch_raw = torch.nn.Parameter(torch.empty(num_formulas, 2).uniform_(0.0, 1.0))
        self.w_phi = torch.nn.Parameter(torch.empty(num_formulas).uniform_(0.0, 1.0))
        self.rho = torch.nn.Parameter(torch.empty(()).uniform_(0.0, 1.0))

    def _node_beta(self, beta_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weight_sum = torch.sum(weights, dim=-1)
        lower = self.alpha + (1.0 - self.alpha) * weight_sum
        upper = 1.0 - self.alpha + self.alpha * weight_sum
        if abs(float(self.alpha) - 0.5) < 1e-8:
            return lower
        beta_unit = torch.sigmoid(beta_raw)
        beta_unit = beta_unit.unsqueeze(1).expand_as(lower) if beta_unit.dim() == 1 else beta_unit.unsqueeze(1).expand_as(lower)
        return lower + beta_unit * torch.clamp(upper - lower, min=0.0)

    def _and_node(self, values: torch.Tensor, weights: torch.Tensor, beta_raw: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(weights)
        beta = self._node_beta(beta_raw, weights)
        out = beta - torch.sum(weights * (1.0 - values), dim=-1)
        return torch.clamp(out, 0.0, 1.0)

    def _or_node(self, values: torch.Tensor, weights: torch.Tensor, beta_raw: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(weights)
        beta = self._node_beta(beta_raw, weights)
        out = 1.0 - beta + torch.sum(weights * values, dim=-1)
        return torch.clamp(out, 0.0, 1.0)

    def formula_truth(self, clocks: torch.Tensor) -> torch.Tensor:
        q = clocks.shape[0]
        if q == 0:
            return torch.zeros((0, self.num_formulas), dtype=clocks.dtype, device=clocks.device)
        u_sop = F.softplus(self.u_sop_raw)
        sop = torch.sigmoid(u_sop.unsqueeze(1) - clocks.unsqueeze(0))
        sop_literals = torch.cat([sop, 1.0 - sop], dim=-1)
        sop_select = self._and_node(
            sop_literals,
            self.w_sop_raw.unsqueeze(1).expand(-1, q, -1),
            self.beta_sop_raw,
        )

        poc_values = []
        if self.pairs:
            u_pop = self.u_pop_raw
            for pair_idx, (left, right) in enumerate(self.pairs):
                diff = clocks[:, left] - clocks[:, right]
                threshold = u_pop[:, pair_idx].unsqueeze(1)
                p_left_before_right = torch.sigmoid(diff.unsqueeze(0) - threshold)
                p_right_before_left = torch.sigmoid((-diff).unsqueeze(0) - threshold)
                pair_vals = torch.stack([p_left_before_right, p_right_before_left], dim=-1)
                dir_w = self.w_pop_dir_raw[:, pair_idx, :].unsqueeze(1).expand(-1, q, -1)
                poc_values.append(self._and_node(pair_vals, dir_w, self.beta_pop_literal_raw[:, pair_idx]))
            poc = torch.stack(poc_values, dim=-1)
        else:
            poc = torch.ones((self.num_formulas, q, 1), dtype=clocks.dtype, device=clocks.device)

        pop_select = self._and_node(
            poc,
            self.w_pop_select_raw.unsqueeze(1).expand(-1, q, -1),
            self.beta_pop_select_raw,
        )
        formula_inputs = torch.stack([pop_select, sop_select], dim=-1)
        conj = self._and_node(formula_inputs, torch.ones_like(formula_inputs), self.beta_formula_and_raw)
        disj = self._or_node(formula_inputs, torch.ones_like(formula_inputs), self.beta_formula_or_raw)
        arch = torch.softmax(self.arch_raw, dim=-1).unsqueeze(1)
        truth = arch[..., 0] * conj + arch[..., 1] * disj
        return truth.transpose(0, 1)

    def forward(self, event_clocks: torch.Tensor, grid_clocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        event_truth = self.formula_truth(event_clocks)
        grid_truth = self.formula_truth(grid_clocks)
        event_eta = self.rho + event_truth @ self.w_phi
        grid_eta = self.rho + grid_truth @ self.w_phi
        return event_eta, grid_eta

    def extract_rules(
        self,
        source_ids: tuple[int, ...],
        target: int,
        *,
        max_rules: int,
        max_order: int,
    ) -> list[dict[str, Any]]:
        with torch.no_grad():
            formula_rank = torch.argsort(torch.abs(self.w_phi), descending=True).cpu().tolist()
            w_sop = F.softplus(self.w_sop_raw).cpu().numpy()
            w_pair = F.softplus(self.w_pop_select_raw).cpu().numpy()
            w_dir = F.softplus(self.w_pop_dir_raw).cpu().numpy()
            u_sop = F.softplus(self.u_sop_raw).cpu().numpy()
            u_pop = self.u_pop_raw.cpu().numpy()
            arch = torch.softmax(self.arch_raw, dim=-1).cpu().numpy()
            out: list[dict[str, Any]] = []
            occupied: set[tuple[tuple[int, ...], str]] = set()
            for formula_idx in formula_rank:
                if len(out) >= int(max_rules):
                    break
                sop_components: list[tuple[float, tuple[int, ...], str | None, dict[str, Any]]] = []
                poc_components: list[tuple[float, tuple[int, ...], str | None, dict[str, Any]]] = []
                for src_pos, src in enumerate(source_ids):
                    pos_score = float(w_sop[formula_idx, src_pos])
                    neg_score = float(w_sop[formula_idx, src_pos + self.num_sources])
                    is_negated = bool(neg_score > pos_score)
                    score = neg_score if is_negated else pos_score
                    sop_components.append(
                        (
                            score,
                            (int(src),),
                            None,
                            {
                                "kind": "SOP",
                                "source": int(src),
                                "threshold": float(u_sop[formula_idx, src_pos]),
                                "negated": is_negated,
                                "weight": score,
                            },
                        )
                    )
                for pair_idx, (left, right) in enumerate(self.pairs):
                    direction = int(np.argmax(w_dir[formula_idx, pair_idx]))
                    select_score = float(w_pair[formula_idx, pair_idx])
                    pair_score = float(select_score * w_dir[formula_idx, pair_idx, direction])
                    left_src = int(source_ids[int(left)])
                    right_src = int(source_ids[int(right)])
                    rel_left = left_src if direction == 0 else right_src
                    rel_right = right_src if direction == 0 else left_src
                    relation = _relation_string(
                        rel_left,
                        rel_right,
                        "before",
                        threshold=float(u_pop[formula_idx, pair_idx]),
                    )
                    poc_components.append(
                        (
                            pair_score,
                            (left_src, right_src),
                            relation,
                            {
                                "kind": "POP",
                                "left_source": rel_left,
                                "right_source": rel_right,
                                "relation": "before",
                                "threshold": float(u_pop[formula_idx, pair_idx]),
                                "negated": False,
                                "weight": pair_score,
                            },
                        )
                    )
                sop_components.sort(key=lambda item: (-float(item[0]), tuple(item[1]), str(item[2])))
                poc_components.sort(key=lambda item: (-float(item[0]), tuple(item[1]), str(item[2])))
                selected_sources: set[int] = set()
                selected_relations: list[str] = []
                selected_components: list[dict[str, Any]] = []
                # CLNN discretizes by retaining the top-k strongest POCs and
                # top-k strongest SOPs, then the dominant POP direction inside
                # each POC.  No relative threshold is part of the paper rule
                # extraction.
                candidates = poc_components[: max(int(max_order), 1)] + sop_components[: max(int(max_order), 1)]
                candidates.sort(key=lambda item: (-float(item[0]), tuple(item[1]), str(item[2])))
                for score, component_sources, relation, component_meta in candidates:
                    if len(selected_sources | set(component_sources)) > max(int(max_order), 1):
                        continue
                    selected_sources.update(int(src) for src in component_sources)
                    selected_components.append(component_meta)
                    if relation is not None:
                        selected_relations.append(str(relation))
                    if len(selected_sources) >= max(int(max_order), 1):
                        break
                sources = tuple(sorted(selected_sources))
                if not sources:
                    continue
                sign = "excitation" if float(self.w_phi[formula_idx]) >= 0.0 else "inhibition"
                key = (sources, sign)
                if key in occupied:
                    continue
                occupied.add(key)
                architecture = "and" if int(np.argmax(arch[formula_idx])) == 0 else "or"
                out.append(
                    {
                        "sources": list(sources),
                        "sign": sign,
                        "target": int(target),
                        "temporal_relations": selected_relations,
                        "score": float(abs(float(self.w_phi[formula_idx]))),
                        "formula_structure": architecture,
                        "dominant_components": selected_components,
                        "raw": f"CLNN formula={formula_idx} extracted_by_topk_strongest_POC_SOP",
                    }
                )
            return out


class NSTPPModel(torch.nn.Module):
    def __init__(self, num_sources: int, *, num_rules: int, max_rule_length: int, tau: float, softmin_rho: float) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_rules = int(num_rules)
        self.max_rule_length = int(max_rule_length)
        self.tau = float(tau)
        self.softmin_rho = float(softmin_rho)
        emb_dim = int(num_sources) + 1
        self.register_buffer("predicate_embeddings", torch.eye(emb_dim))
        self.predicate_embeddings[0].zero_()
        self.register_buffer("relation_embeddings", torch.eye(4, emb_dim)[:4])
        self.rule_content = torch.nn.Parameter(torch.randn(num_rules, max_rule_length, emb_dim) * 0.05)
        pair_count = max(max_rule_length * (max_rule_length - 1) // 2, 1)
        self.rule_relations = torch.nn.Parameter(torch.randn(num_rules, pair_count, emb_dim) * 0.05)
        self.gamma_raw = torch.nn.Parameter(torch.zeros(num_rules))
        self.base_raw = torch.nn.Parameter(torch.tensor(-4.0))

    def _slot_probs(self) -> torch.Tensor:
        logits = torch.matmul(self.rule_content, self.predicate_embeddings.T) / max(self.tau, 1e-4)
        return torch.softmax(logits, dim=-1)

    def _relation_probs(self) -> torch.Tensor:
        logits = torch.matmul(self.rule_relations, self.relation_embeddings.T) / max(self.tau, 1e-4)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _sample_indices(probs: torch.Tensor, *, sample_gumbel: bool) -> torch.Tensor:
        if bool(sample_gumbel):
            uniform = torch.rand_like(probs).clamp(1e-8, 1.0 - 1e-8)
            logits = torch.log(torch.clamp(probs, min=1e-8)) - torch.log(-torch.log(uniform))
            return torch.argmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)

    def features(
        self,
        facts: torch.Tensor,
        last_times: torch.Tensor,
        valid: torch.Tensor,
        *,
        delta: float,
        sample_gumbel: bool,
    ) -> torch.Tensor:
        q = facts.shape[0]
        if q == 0:
            return torch.zeros((0, self.num_rules), dtype=facts.dtype, device=facts.device)
        facts_aug = torch.cat([torch.ones((q, 1), dtype=facts.dtype, device=facts.device), facts], dim=1)
        probs = self._slot_probs()
        slot_idx = self._sample_indices(probs, sample_gumbel=sample_gumbel)
        selected_prob = torch.gather(probs, dim=-1, index=slot_idx.unsqueeze(-1)).squeeze(-1)
        flat_idx = slot_idx.reshape(-1)
        selected_fact = facts_aug[:, flat_idx].reshape(q, self.num_rules, self.max_rule_length)
        selected_prob = selected_prob.unsqueeze(0).expand(q, -1, -1)
        static_terms = torch.stack([selected_fact, selected_prob], dim=-1).reshape(q, self.num_rules, -1)
        static_feature = _softmin(static_terms, rho=self.softmin_rho, dim=-1)

        relation_terms: list[torch.Tensor] = []
        rel_probs = self._relation_probs()
        rel_facts_real = _relation_facts(last_times, valid, delta=delta)
        rel_idx = self._sample_indices(rel_probs, sample_gumbel=sample_gumbel)
        pair_idx = 0
        for i in range(self.max_rule_length):
            for j in range(i + 1, self.max_rule_length):
                left_idx = slot_idx[:, i]
                right_idx = slot_idx[:, j]
                chosen_rel = rel_idx[:, pair_idx]
                chosen_rel_prob = torch.gather(rel_probs[:, pair_idx, :], dim=-1, index=chosen_rel.unsqueeze(-1)).squeeze(-1)
                pair_fact = torch.zeros((q, self.num_rules), dtype=facts.dtype, device=facts.device)
                for rule_idx in range(self.num_rules):
                    li = int(left_idx[rule_idx].detach().cpu())
                    ri = int(right_idx[rule_idx].detach().cpu())
                    rel = int(chosen_rel[rule_idx].detach().cpu())
                    if li == 0 or ri == 0 or li == ri:
                        pair_fact[:, rule_idx] = 1.0 if rel == 3 else 0.0
                    else:
                        pair_fact[:, rule_idx] = rel_facts_real[:, li - 1, ri - 1, rel]
                relation_terms.append(torch.minimum(pair_fact, chosen_rel_prob.unsqueeze(0).expand(q, -1)))
                pair_idx += 1
        if relation_terms:
            temporal_feature = _softmin(torch.stack(relation_terms, dim=-1), rho=self.softmin_rho, dim=-1)
        else:
            temporal_feature = torch.ones_like(static_feature)
        return static_feature * temporal_feature

    def forward(
        self,
        event_facts: torch.Tensor,
        grid_facts: torch.Tensor,
        event_last_times: torch.Tensor,
        grid_last_times: torch.Tensor,
        event_valid: torch.Tensor,
        grid_valid: torch.Tensor,
        *,
        delta: float,
        active_rule_limit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        event_phi = self.features(event_facts, event_last_times, event_valid, delta=delta, sample_gumbel=self.training)
        grid_phi = self.features(grid_facts, grid_last_times, grid_valid, delta=delta, sample_gumbel=self.training)
        gamma = F.softplus(self.gamma_raw)
        base = F.softplus(self.base_raw)
        if active_rule_limit is not None:
            limit = max(0, min(int(active_rule_limit), self.num_rules))
            event_lambda = base + event_phi[:, :limit] @ gamma[:limit]
            grid_lambda = base + grid_phi[:, :limit] @ gamma[:limit]
        else:
            event_lambda = base + event_phi @ gamma
            grid_lambda = base + grid_phi @ gamma
        return event_lambda, grid_lambda, event_phi, grid_phi

    def extract_rules(
        self,
        source_ids: tuple[int, ...],
        target: int,
        *,
        max_rules: int,
        gamma_threshold: float,
        slot_threshold: float,
    ) -> list[dict[str, Any]]:
        with torch.no_grad():
            probs = self._slot_probs().cpu().numpy()
            rel_probs = self._relation_probs().cpu().numpy()
            gammas = F.softplus(self.gamma_raw).cpu().numpy()
            rank = np.argsort(-gammas)
            rel_names = ["before", "equal", "after", "none"]
            out = []
            occupied: set[tuple[int, ...]] = set()
            for rule_idx in rank:
                if len(out) >= int(max_rules):
                    break
                if float(gammas[rule_idx]) <= float(gamma_threshold):
                    continue
                selected = []
                for slot in range(self.max_rule_length):
                    idx = int(np.argmax(probs[rule_idx, slot]))
                    confidence = float(np.max(probs[rule_idx, slot]))
                    null_confidence = float(probs[rule_idx, slot, 0])
                    if idx > 0 and confidence >= max(float(slot_threshold), null_confidence):
                        selected.append(int(source_ids[idx - 1]))
                sources = tuple(sorted(set(selected)))
                if not sources or sources in occupied:
                    continue
                occupied.add(sources)
                rels: list[str] = []
                pair_idx = 0
                slot_sources = [
                    (
                        int(np.argmax(probs[rule_idx, slot])) - 1
                        if int(np.argmax(probs[rule_idx, slot])) > 0
                        and float(np.max(probs[rule_idx, slot])) >= max(float(slot_threshold), float(probs[rule_idx, slot, 0]))
                        else -1
                    )
                    for slot in range(self.max_rule_length)
                ]
                for left_slot in range(self.max_rule_length):
                    for right_slot in range(left_slot + 1, self.max_rule_length):
                        left_pos = int(slot_sources[left_slot])
                        right_pos = int(slot_sources[right_slot])
                        rel_name = rel_names[int(np.argmax(rel_probs[rule_idx, pair_idx]))]
                        pair_idx += 1
                        if left_pos < 0 or right_pos < 0 or left_pos == right_pos:
                            continue
                        rels.append(
                            _relation_string(
                                int(source_ids[left_pos]),
                                int(source_ids[right_pos]),
                                rel_name,
                            )
                        )
                out.append(
                    {
                        "sources": list(sources),
                        "sign": "excitation",
                        "target": int(target),
                        "temporal_relations": rels,
                        "score": float(gammas[rule_idx]),
                        "raw": f"NSTPP rule={rule_idx} argmax_predicate_slots",
                    }
                )
            return out


class CLUSTERModel(torch.nn.Module):
    def __init__(
        self,
        num_sources: int,
        *,
        num_rules: int,
        max_rule_length: int,
        dummy_count: int,
        tau: float,
        laplace_scale: float,
        softmin_rho: float,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_rules = int(num_rules)
        self.max_rule_length = int(max_rule_length)
        self.dummy_count = int(dummy_count)
        self.tau = float(tau)
        self.laplace_scale = float(laplace_scale)
        self.softmin_rho = float(softmin_rho)
        self.rule_weights_raw = torch.nn.Parameter(torch.randn(num_rules, num_sources + dummy_count) * 0.05)
        pair_count = max(max_rule_length * (max_rule_length - 1) // 2, 1)
        self.relation_logits = torch.nn.Parameter(torch.zeros(num_rules, pair_count, 4))
        self.gamma_raw = torch.nn.Parameter(torch.zeros(num_rules))
        self.base_raw = torch.nn.Parameter(torch.tensor(-4.0))
        self.pi_logits = torch.nn.Parameter(torch.zeros(num_rules + 1), requires_grad=False)

    def relaxed_topk_slots(self, *, sample_gumbel: bool | None = None) -> torch.Tensor:
        logits = F.softplus(self.rule_weights_raw)
        remaining = torch.log(torch.clamp(logits, min=1e-8))
        if bool(self.training if sample_gumbel is None else sample_gumbel):
            uniform = torch.rand_like(remaining).clamp(1e-8, 1.0 - 1e-8)
            remaining = remaining - torch.log(-torch.log(uniform))
        slots: list[torch.Tensor] = []
        khot = torch.zeros_like(remaining)
        for _ in range(self.max_rule_length):
            probs = torch.softmax(remaining / max(self.tau, 1e-4), dim=-1)
            slots.append(probs)
            khot = khot + probs
            remaining = remaining + torch.log(torch.clamp(1.0 - probs, min=1e-8))
        return torch.stack(slots, dim=1)

    def relaxed_topk(self, *, sample_gumbel: bool | None = None) -> torch.Tensor:
        return torch.sum(self.relaxed_topk_slots(sample_gumbel=sample_gumbel), dim=1)

    def features(
        self,
        facts: torch.Tensor,
        last_times: torch.Tensor,
        valid: torch.Tensor,
        *,
        delta: float,
        use_temporal: bool = True,
        slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = facts.shape[0]
        if q == 0:
            return torch.zeros((0, self.num_rules), dtype=facts.dtype, device=facts.device)
        dummy = torch.ones((q, self.dummy_count), dtype=facts.dtype, device=facts.device)
        facts_aug = torch.cat([facts, dummy], dim=1)
        if slots is None:
            slots = self.relaxed_topk_slots()
        a = torch.sum(slots, dim=1)
        coverage = torch.einsum("qn,rn->qr", facts_aug, a)
        static_k = torch.exp(-torch.abs(coverage - float(self.max_rule_length)) / max(self.laplace_scale, 1e-6))
        if not bool(use_temporal):
            return static_k

        source_slot_probs = slots[:, :, : self.num_sources]
        rel_facts_full = _relation_facts(last_times, valid, delta=delta)
        pair_terms: list[torch.Tensor] = []
        rel_probs = torch.softmax(self.relation_logits, dim=-1)
        pair_idx = 0
        for left_slot in range(self.max_rule_length):
            for right_slot in range(left_slot + 1, self.max_rule_length):
                pair_rel = torch.einsum(
                    "ra,rb,qabk->qrk",
                    source_slot_probs[:, left_slot, :],
                    source_slot_probs[:, right_slot, :],
                    rel_facts_full,
                )
                left_source_mass = torch.sum(source_slot_probs[:, left_slot, :], dim=-1)
                right_source_mass = torch.sum(source_slot_probs[:, right_slot, :], dim=-1)
                alpha = rel_probs[:, min(pair_idx, rel_probs.shape[1] - 1), :]
                rel_before = pair_rel[..., 0] * alpha[:, 0].unsqueeze(0)
                rel_equal = pair_rel[..., 1] * alpha[:, 1].unsqueeze(0)
                rel_after = pair_rel[..., 2] * alpha[:, 2].unsqueeze(0)
                rel_none = alpha[:, 3].unsqueeze(0) * (
                    1.0
                    - pair_rel[..., 0] * alpha[:, 0].unsqueeze(0)
                    - pair_rel[..., 1] * alpha[:, 1].unsqueeze(0)
                    - pair_rel[..., 2] * alpha[:, 2].unsqueeze(0)
                )
                dummy_pair = (1.0 - (left_source_mass * right_source_mass)).unsqueeze(0).clamp(0.0, 1.0)
                rel_none = torch.maximum(rel_none, dummy_pair * alpha[:, 3].unsqueeze(0))
                rel_values = torch.stack([rel_before, rel_equal, rel_after, rel_none], dim=-1)
                pair_terms.append(_weighted_softmax(rel_values, temperature=self.softmin_rho, dim=-1))
                pair_idx += 1
        if not pair_terms:
            pair_rel = torch.einsum(
                "ra,rb,qabk->qrk",
                source_slot_probs[:, 0, :],
                source_slot_probs[:, 0, :],
                rel_facts_full,
            )
            alpha = rel_probs[:, min(pair_idx, rel_probs.shape[1] - 1), :]
            source_mass = torch.sum(source_slot_probs[:, 0, :], dim=-1)
            rel_values = torch.stack(
                [
                    pair_rel[..., 0] * alpha[:, 0].unsqueeze(0),
                    pair_rel[..., 1] * alpha[:, 1].unsqueeze(0),
                    pair_rel[..., 2] * alpha[:, 2].unsqueeze(0),
                    torch.maximum(
                        pair_rel[..., 3] * alpha[:, 3].unsqueeze(0),
                        (1.0 - source_mass * source_mass).unsqueeze(0) * alpha[:, 3].unsqueeze(0),
                    ),
                ],
                dim=-1,
            )
            pair_terms.append(_weighted_softmax(rel_values, temperature=self.softmin_rho, dim=-1))
        temporal = _weighted_softmin(torch.stack(pair_terms, dim=-1), temperature=self.softmin_rho, dim=-1)
        return static_k * temporal

    def component_intensities(
        self,
        event_facts: torch.Tensor,
        grid_facts: torch.Tensor,
        event_last_times: torch.Tensor,
        grid_last_times: torch.Tensor,
        event_valid: torch.Tensor,
        grid_valid: torch.Tensor,
        *,
        delta: float,
        use_temporal: bool = True,
        sample_gumbel: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self.relaxed_topk_slots(sample_gumbel=self.training if sample_gumbel is None else sample_gumbel)
        event_phi = self.features(event_facts, event_last_times, event_valid, delta=delta, use_temporal=use_temporal, slots=slots)
        grid_phi = self.features(grid_facts, grid_last_times, grid_valid, delta=delta, use_temporal=use_temporal, slots=slots)
        base = F.softplus(self.base_raw)
        gamma = F.softplus(self.gamma_raw)
        event_components = torch.cat([base.expand(event_phi.shape[0], 1), event_phi * gamma.unsqueeze(0)], dim=1)
        grid_components = torch.cat([base.expand(grid_phi.shape[0], 1), grid_phi * gamma.unsqueeze(0)], dim=1)
        return event_components, grid_components, event_phi, grid_phi

    def marginal_lambdas(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        event_components, grid_components, event_phi, grid_phi = self.component_intensities(*args, **kwargs)
        pi = torch.softmax(self.pi_logits, dim=-1)
        return event_components @ pi, grid_components @ pi, event_phi, grid_phi

    def update_pi_closed_form(
        self,
        event_components: torch.Tensor,
        grid_components: torch.Tensor,
        grid_weights: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            pi = torch.softmax(self.pi_logits, dim=-1)
            numer = event_components * pi.unsqueeze(0)
            q = numer / torch.clamp(torch.sum(numer, dim=1, keepdim=True), min=1e-8)
            q_sum = torch.sum(q, dim=0)
            integrated = torch.sum(grid_components * grid_weights.unsqueeze(1), dim=0)
            positive = q_sum > 1e-8
            if not bool(torch.any(positive)):
                new_pi = torch.ones_like(q_sum) / float(q_sum.numel())
            else:
                low = -float(torch.min(integrated[positive]).detach().cpu()) + 1e-8
                high = 1.0
                def mass(c_value: float) -> float:
                    return float(torch.sum(q_sum[positive] / torch.clamp(integrated[positive] + c_value, min=1e-8)).detach().cpu())
                while mass(high) > 1.0:
                    high *= 2.0
                for _ in range(64):
                    mid = 0.5 * (low + high)
                    if mass(mid) > 1.0:
                        low = mid
                    else:
                        high = mid
                new_pi = torch.zeros_like(q_sum)
                new_pi[positive] = q_sum[positive] / torch.clamp(integrated[positive] + high, min=1e-8)
            new_pi = torch.clamp(new_pi, min=1e-8)
            new_pi = new_pi / torch.sum(new_pi)
            self.pi_logits.copy_(torch.log(new_pi))

    def extract_rules(
        self,
        source_ids: tuple[int, ...],
        target: int,
        *,
        max_rules: int,
        gamma_threshold: float,
        slot_threshold: float,
    ) -> list[dict[str, Any]]:
        with torch.no_grad():
            slot_probs = self.relaxed_topk_slots(sample_gumbel=False).cpu().numpy()
            khot_probs = np.sum(slot_probs, axis=1)
            gammas = F.softplus(self.gamma_raw).cpu().numpy()
            pi = torch.softmax(self.pi_logits, dim=-1).cpu().numpy()
            rule_scores = pi[1:] * gammas
            rel_probs = torch.softmax(self.relation_logits, dim=-1).cpu().numpy()
            rank = np.argsort(-rule_scores)
            rel_names = ["before", "equal", "after", "none"]
            out = []
            occupied: set[tuple[int, ...]] = set()
            for rule_idx in rank:
                if len(out) >= int(max_rules):
                    break
                if float(rule_scores[rule_idx]) <= float(gamma_threshold):
                    continue
                # The CLUSTER relaxation represents a rule by a relaxed k-hot
                # subset vector.  Reading per-slot argmaxes is not equivalent:
                # the same source can win several exchangeable slots and then
                # collapse to a false singleton after deduplication.  Extract
                # the hard subset from the aggregated relaxed k-hot vector.
                hard_positions = np.argsort(-khot_probs[rule_idx])[: self.max_rule_length]
                selected_positions = [
                    int(idx)
                    for idx in hard_positions
                    if int(idx) < self.num_sources and float(khot_probs[rule_idx, int(idx)]) >= float(slot_threshold)
                ]
                sources = tuple(sorted({int(source_ids[int(idx)]) for idx in selected_positions}))
                if not sources or sources in occupied:
                    continue
                occupied.add(sources)
                rels: list[str] = []
                pair_idx = 0
                slot_sources = [
                    int(source_ids[int(idx)])
                    if int(idx) < self.num_sources and float(khot_probs[rule_idx, int(idx)]) >= float(slot_threshold)
                    else None
                    for idx in hard_positions
                ]
                for left_pos in range(self.max_rule_length):
                    for right_pos in range(left_pos + 1, self.max_rule_length):
                        if pair_idx >= rel_probs.shape[1]:
                            continue
                        left_src = slot_sources[left_pos]
                        right_src = slot_sources[right_pos]
                        if left_src is not None and right_src is not None and int(left_src) != int(right_src):
                            rels.append(
                                _relation_string(
                                    int(left_src),
                                    int(right_src),
                                    rel_names[int(np.argmax(rel_probs[rule_idx, pair_idx]))],
                                )
                            )
                        pair_idx += 1
                out.append(
                    {
                        "sources": list(sources),
                        "sign": "excitation",
                        "target": int(target),
                        "temporal_relations": rels,
                        "score": float(rule_scores[rule_idx]),
                        "raw_gamma": float(gammas[rule_idx]),
                        "mixture_weight": float(pi[int(rule_idx) + 1]),
                        "raw": f"CLUSTER rule={rule_idx} relaxed_topK_argmax",
                    }
                )
            return out


def _fit_clnn(design: Design, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], CLNNModel]:
    if abs(float(args.alpha) - 0.5) > 1e-8:
        raise ValueError("CLNN restricted baseline fixes alpha=0.5; non-default alpha would be an extended variant.")
    model = CLNNModel(len(design.source_ids), num_formulas=args.num_formulas, alpha=args.alpha).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    for _ in range(int(args.epochs)):
        opt.zero_grad(set_to_none=True)
        event_eta, grid_eta = model(design.event_clocks, design.grid_clocks)
        loss = _nll_loglink(event_eta, grid_eta, design.grid_weights)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    rules = model.extract_rules(
        design.source_ids,
        design.target,
        max_rules=args.max_rules,
        max_order=args.max_rule_length,
    )
    return rules, {"loss_start": losses[0] if losses else None, "loss_end": losses[-1] if losses else None}, model


def _mask_nstpp_stage_gradients(model: NSTPPModel, train_rule_idx: int) -> None:
    idx = int(train_rule_idx)
    if model.rule_content.grad is not None:
        mask = torch.zeros_like(model.rule_content.grad)
        mask[idx].fill_(1.0)
        model.rule_content.grad.mul_(mask)
    if model.rule_relations.grad is not None:
        mask = torch.zeros_like(model.rule_relations.grad)
        mask[idx].fill_(1.0)
        model.rule_relations.grad.mul_(mask)
    if model.gamma_raw.grad is not None:
        mask = torch.zeros_like(model.gamma_raw.grad)
        mask[idx] = 1.0
        model.gamma_raw.grad.mul_(mask)


def _fit_nstpp(design: Design, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], NSTPPModel]:
    model = NSTPPModel(
        len(design.source_ids),
        num_rules=args.max_rules,
        max_rule_length=args.max_rule_length,
        tau=args.tau,
        softmin_rho=args.softmin_rho,
    ).to(args.device)
    losses = []
    stage_base_raw_values: list[float] = []
    # The paper learns rules through sequential covering. Covering is a
    # sequence-level state here: after a rule explains any target event in a
    # sequence, that sequence is removed from the next single-rule stage.
    sequence_active = torch.ones((design.sequence_count,), dtype=torch.float32, device=args.device)
    for rule_limit in range(1, int(args.max_rules) + 1):
        train_rule_idx = int(rule_limit) - 1
        best_loss = float("inf")
        best_rule_content = None
        best_rule_relations = None
        best_gamma_raw = None
        best_stage_base = None
        stage_epochs = max(int(args.epochs) // max(int(args.max_rules), 1), 1)
        for _restart in range(max(int(getattr(args, "search_restarts", 1)), 1)):
            with torch.no_grad():
                model.rule_content[train_rule_idx].normal_(0.0, 0.05)
                model.rule_relations[train_rule_idx].normal_(0.0, 0.05)
                model.gamma_raw[train_rule_idx].zero_()
            stage_base_raw = torch.nn.Parameter(model.base_raw.detach().clone())
            stage_opt = torch.optim.Adam(
                [model.rule_content, model.rule_relations, model.gamma_raw, stage_base_raw],
                lr=args.lr,
            )
            model.train()
            for _ in range(stage_epochs):
                stage_opt.zero_grad(set_to_none=True)
                event_weights = sequence_active[design.event_sequence_ids]
                grid_weights = design.grid_weights * sequence_active[design.grid_sequence_ids]
                event_phi = model.features(
                    design.event_facts,
                    design.event_last_times,
                    design.event_valid,
                    delta=args.delta,
                    sample_gumbel=True,
                )
                grid_phi = model.features(
                    design.grid_facts,
                    design.grid_last_times,
                    design.grid_valid,
                    delta=args.delta,
                    sample_gumbel=True,
                )
                gamma = F.softplus(model.gamma_raw)
                base = F.softplus(stage_base_raw)
                event_lambda = base + event_phi[:, train_rule_idx] * gamma[train_rule_idx]
                grid_lambda = base + grid_phi[:, train_rule_idx] * gamma[train_rule_idx]
                weighted_event = -torch.sum(event_weights * torch.log(torch.clamp(event_lambda, min=1e-8)))
                loss = weighted_event + torch.sum(grid_weights * grid_lambda)
                loss.backward()
                _mask_nstpp_stage_gradients(model, train_rule_idx)
                stage_opt.step()
                losses.append(float(loss.detach().cpu()))
            final_loss = float(loss.detach().cpu())
            if final_loss < best_loss:
                best_loss = final_loss
                best_rule_content = model.rule_content[train_rule_idx].detach().clone()
                best_rule_relations = model.rule_relations[train_rule_idx].detach().clone()
                best_gamma_raw = model.gamma_raw[train_rule_idx].detach().clone()
                best_stage_base = stage_base_raw.detach().clone()
        with torch.no_grad():
            if best_rule_content is not None:
                model.rule_content[train_rule_idx].copy_(best_rule_content)
                model.rule_relations[train_rule_idx].copy_(best_rule_relations)
                model.gamma_raw[train_rule_idx].copy_(best_gamma_raw)
        stage_base_raw = torch.nn.Parameter(best_stage_base if best_stage_base is not None else model.base_raw.detach().clone())
        stage_base_raw_values.append(float(stage_base_raw.detach().cpu()))
        with torch.no_grad():
            model.eval()
            _, _, event_phi, _ = model(
                design.event_facts,
                design.grid_facts,
                design.event_last_times,
                design.grid_last_times,
                design.event_valid,
                design.grid_valid,
                delta=args.delta,
            )
            covered = event_phi[:, train_rule_idx] >= float(args.cover_threshold)
            covered_sequence = torch.zeros_like(sequence_active)
            if covered.numel() > 0:
                covered_sequence.scatter_reduce_(
                    0,
                    design.event_sequence_ids,
                    covered.float(),
                    reduce="amax",
                    include_self=True,
                )
            sequence_active = sequence_active * (1.0 - covered_sequence)
            if float(torch.sum(sequence_active).detach().cpu()) <= 0.0:
                break
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(int(args.refine_epochs)):
        opt.zero_grad(set_to_none=True)
        event_lambda, grid_lambda, _, _ = model(
            design.event_facts,
            design.grid_facts,
            design.event_last_times,
            design.grid_last_times,
            design.event_valid,
            design.grid_valid,
            delta=args.delta,
            active_rule_limit=None,
        )
        loss = _nll_additive(event_lambda, grid_lambda, design.grid_weights)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    rules = model.extract_rules(
        design.source_ids,
        design.target,
        max_rules=args.max_rules,
        gamma_threshold=args.gamma_threshold,
        slot_threshold=float(getattr(args, "slot_threshold", 0.0)),
    )
    return (
        rules,
        {
            "loss_start": losses[0] if losses else None,
            "loss_end": losses[-1] if losses else None,
            "stage_local_base_raw_values": stage_base_raw_values,
            "stage_base_policy": "stage-local base for sequential covering; global base is learned only in final refinement",
        },
        model,
    )


def _per_event_component_integrals(design: Design, grid_components: torch.Tensor) -> torch.Tensor:
    integrals = torch.zeros(
        (int(design.event_times.shape[0]), int(grid_components.shape[1])),
        dtype=grid_components.dtype,
        device=grid_components.device,
    )
    if int(design.sequence_count) <= 0 or int(design.grid_weights.numel()) == 0:
        return integrals
    grid_per_sequence = int(design.grid_weights.numel()) // int(design.sequence_count)
    if grid_per_sequence <= 0 or grid_per_sequence * int(design.sequence_count) != int(design.grid_weights.numel()):
        # Fallback for non-rectangular grids. The benchmark adapters currently
        # use rectangular midpoint grids, so this path is not expected in the
        # paper synthetic run.
        for seq_id in torch.unique(design.grid_sequence_ids).detach().cpu().tolist():
            grid_mask = design.grid_sequence_ids == int(seq_id)
            event_mask = design.event_sequence_ids == int(seq_id)
            if not bool(torch.any(grid_mask)) or not bool(torch.any(event_mask)):
                continue
            grid_idx = torch.nonzero(grid_mask, as_tuple=False).squeeze(1)
            event_idx = torch.nonzero(event_mask, as_tuple=False).squeeze(1)
            order = torch.argsort(design.grid_times[grid_idx])
            grid_idx = grid_idx[order]
            grid_times = design.grid_times[grid_idx]
            cumulative = torch.cumsum(grid_components[grid_idx] * design.grid_weights[grid_idx].unsqueeze(1), dim=0)
            positions = torch.searchsorted(grid_times, design.event_times[event_idx], right=True) - 1
            valid = positions >= 0
            if bool(torch.any(valid)):
                integrals[event_idx[valid]] = cumulative[positions[valid]]
        return integrals
    weighted = (grid_components * design.grid_weights.unsqueeze(1)).reshape(
        int(design.sequence_count),
        grid_per_sequence,
        int(grid_components.shape[1]),
    )
    cumulative = torch.cumsum(weighted, dim=1)
    positions = design.event_grid_positions
    valid = positions >= 0
    if bool(torch.any(valid)):
        integrals[valid] = cumulative[design.event_sequence_ids[valid], positions[valid]]
    return integrals


def _cluster_log_p_components(
    model: CLUSTERModel,
    design: Design,
    args: argparse.Namespace,
    *,
    use_temporal: bool,
    sample_gumbel: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    event_components, grid_components, event_phi, grid_phi = model.component_intensities(
        design.event_facts,
        design.grid_facts,
        design.event_last_times,
        design.grid_last_times,
        design.event_valid,
        design.grid_valid,
        delta=args.delta,
        use_temporal=use_temporal,
        sample_gumbel=sample_gumbel,
    )
    pi = torch.softmax(model.pi_logits, dim=-1)
    integrals = _per_event_component_integrals(design, grid_components)
    log_p = (
        torch.log(torch.clamp(pi, min=1e-8)).unsqueeze(0)
        + torch.log(torch.clamp(event_components, min=1e-8))
        - integrals
    )
    return log_p, event_components, grid_components, event_phi


def _fit_cluster(design: Design, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], CLUSTERModel]:
    model = CLUSTERModel(
        len(design.source_ids),
        num_rules=args.max_rules,
        max_rule_length=args.max_rule_length,
        dummy_count=max(int(args.dummy_count), int(args.max_rule_length)),
        tau=args.tau,
        laplace_scale=args.laplace_scale,
        softmin_rho=args.softmin_rho,
    ).to(args.device)
    content_params = [model.rule_weights_raw, model.gamma_raw, model.base_raw]
    relation_params = [model.rule_weights_raw, model.relation_logits, model.gamma_raw, model.base_raw]
    opt_content = torch.optim.AdamW(content_params, lr=args.lr, weight_decay=args.weight_decay)
    opt_relation = torch.optim.AdamW(
        relation_params,
        lr=float(getattr(args, "relation_lr", 0.0035)),
        weight_decay=args.weight_decay,
    )
    losses = []
    for _ in range(int(args.em_iters)):
        model.tau = float(args.tau)
        with torch.no_grad():
            log_p, _, _, _ = _cluster_log_p_components(
                model,
                design,
                args,
                use_temporal=False,
                sample_gumbel=False,
            )
            q_content = torch.softmax(log_p, dim=1)
        for _inner in range(int(args.epochs)):
            if int(args.epochs) > 1:
                frac = float(_inner) / float(max(int(args.epochs) - 1, 1))
                model.tau = max(float(getattr(args, "tau_min", 0.1)), float(args.tau) * (float(getattr(args, "tau_min", 0.1)) / float(args.tau)) ** frac)
            opt_content.zero_grad(set_to_none=True)
            log_p_train, _, _, _ = _cluster_log_p_components(
                model,
                design,
                args,
                use_temporal=False,
                sample_gumbel=True,
            )
            loss = -torch.sum(q_content.detach() * log_p_train)
            loss.backward()
            opt_content.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            model.tau = float(getattr(args, "tau_min", 0.1))
            log_p, _, _, _ = _cluster_log_p_components(
                model,
                design,
                args,
                use_temporal=True,
                sample_gumbel=False,
            )
            q_relation = torch.softmax(log_p, dim=1)
        for _inner in range(int(getattr(args, "relation_epochs", 20))):
            model.tau = float(getattr(args, "tau_min", 0.1))
            opt_relation.zero_grad(set_to_none=True)
            log_p_train, _, _, _ = _cluster_log_p_components(
                model,
                design,
                args,
                use_temporal=True,
                sample_gumbel=True,
            )
            loss = -torch.sum(q_relation.detach() * log_p_train)
            loss.backward()
            opt_relation.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            log_p, _, _, _ = _cluster_log_p_components(
                model,
                design,
                args,
                use_temporal=True,
                sample_gumbel=False,
            )
            q = torch.softmax(log_p, dim=1)
            pi = torch.clamp(torch.mean(q, dim=0), min=1e-5)
            pi = pi / torch.sum(pi)
            model.pi_logits.copy_(torch.log(pi))
    rules = model.extract_rules(
        design.source_ids,
        design.target,
        max_rules=args.max_rules,
        gamma_threshold=args.gamma_threshold,
        slot_threshold=float(getattr(args, "slot_threshold", 0.0)),
    )
    return rules, {"loss_start": losses[0] if losses else None, "loss_end": losses[-1] if losses else None}, model


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    config = _load_yaml(args.config)
    dataset = _load_dataset(args.data)
    design = _build_design(
        config=config,
        dataset=dataset,
        grid_size=args.grid_size,
        masking=args.masking,
        split_scope=str(getattr(args, "split_scope", "train")),
        include_target_history=not bool(getattr(args, "exclude_target_from_sources", False)),
        horizon_mode=str(getattr(args, "horizon_mode", "config")),
        device=args.device,
    )
    if args.model == "CLNN":
        predicted_rules, fit_meta, model = _fit_clnn(design, args)
        init_args = {
            "num_sources": int(len(design.source_ids)),
            "num_formulas": int(args.num_formulas),
            "alpha": float(args.alpha),
        }
    elif args.model == "NSTPP":
        predicted_rules, fit_meta, model = _fit_nstpp(design, args)
        init_args = {
            "num_sources": int(len(design.source_ids)),
            "num_rules": int(args.max_rules),
            "max_rule_length": int(args.max_rule_length),
            "tau": float(args.tau),
            "softmin_rho": float(args.softmin_rho),
        }
    elif args.model == "CLUSTER":
        predicted_rules, fit_meta, model = _fit_cluster(design, args)
        init_args = {
            "num_sources": int(len(design.source_ids)),
            "num_rules": int(args.max_rules),
            "max_rule_length": int(args.max_rule_length),
            "dummy_count": max(int(args.dummy_count), int(args.max_rule_length)),
            "tau": float(model.tau),
            "laplace_scale": float(args.laplace_scale),
            "softmin_rho": float(args.softmin_rho),
        }
    else:
        raise ValueError(f"unsupported model: {args.model}")

    info = MODEL_INFO[str(args.model)]
    optimizer_name = "Adam" if str(args.model) == "NSTPP" else "AdamW"
    checkpoint_path = str(Path(args.output).with_suffix(".pt"))
    torch.save(
        {
            "model": str(args.model),
            "state_dict": model.state_dict(),
            "init_args": init_args,
            "source_ids": tuple(int(src) for src in design.source_ids),
            "target": int(design.target),
            "grid_size": int(args.grid_size),
            "masking": str(args.masking),
            "split_scope": str(getattr(args, "split_scope", "train")),
            "include_target_history": not bool(getattr(args, "exclude_target_from_sources", False)),
            "horizon_mode": str(getattr(args, "horizon_mode", "config")),
            "delta": float(args.delta),
            "use_temporal": True,
            "intensity_family": "loglink" if str(args.model) == "CLNN" else "additive",
        },
        checkpoint_path,
    )
    return {
        "model": str(args.model),
        "target": int(design.target),
        "predicted_rules": predicted_rules,
        "metadata": {
            "implementation_type": str(info["implementation_type"]),
            "source_url": str(info["source_url"]),
            "paper_components": list(info["paper_components"]),
            "official_code_available": bool(info.get("official_code_available", False)),
            "exactness_note": (
                "This is a paper-based reimplementation of the stated model equations "
                "and learning structure. Choices not specified in the paper, including "
                "initialization, optimizer schedule, quadrature grid, convergence "
                "thresholds, and discrete rule extraction from relaxations, are "
                "recorded here."
            ),
            "under_specified_choices": {
                "optimizer": optimizer_name,
                "numerical_integral": "uniform midpoint quadrature",
                "masking": str(args.masking),
                "split_scope": str(getattr(args, "split_scope", "train")),
                "include_target_history": not bool(getattr(args, "exclude_target_from_sources", False)),
                "horizon_mode": str(getattr(args, "horizon_mode", "config")),
                "grid_size": int(args.grid_size),
                "clnn_alpha": float(args.alpha) if str(args.model) == "CLNN" else None,
                "clnn_alpha_policy": "fixed at the paper/default alpha=0.5" if str(args.model) == "CLNN" else None,
                "epochs": int(args.epochs),
                "refine_epochs": int(args.refine_epochs),
                "learning_rate": float(args.lr),
                "relation_learning_rate": float(getattr(args, "relation_lr", 0.0035)),
                "weight_decay": 0.0 if str(args.model) == "NSTPP" else float(args.weight_decay),
                "tau": float(args.tau),
                "tau_min": float(getattr(args, "tau_min", 0.1)),
                "delta": float(args.delta),
                "softmin_rho": float(args.softmin_rho),
                "gamma_threshold": float(args.gamma_threshold),
                "slot_threshold": float(getattr(args, "slot_threshold", 0.0)),
                "extraction_note": "Discrete rules are read by paper-stated argmax/top-k rules where specified.",
                "cover_threshold": float(args.cover_threshold),
                "relation_epochs": int(getattr(args, "relation_epochs", 20)),
                "search_restarts": int(getattr(args, "search_restarts", 1)),
            },
            "config_path": str(args.config),
            "dataset_path": str(args.data),
            "paper_intensity_checkpoint_path": checkpoint_path,
            "seed": int(args.seed),
            "num_sources": int(len(design.source_ids)),
            "num_sequences": int(design.sequence_count),
            "num_target_events": int(design.event_facts.shape[0]),
            "num_grid_points": int(design.grid_facts.shape[0]),
            "max_rules": int(args.max_rules),
            "max_rule_length": int(args.max_rule_length),
            "supports_inhibition": bool(info["signed"]),
            "runtime_sec": float(time.perf_counter() - started),
            **fit_meta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned local rule baseline reimplementations.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_INFO))
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid_size", type=int, default=24)
    parser.add_argument("--masking", choices=["first", "last"], default="last")
    parser.add_argument("--split_scope", choices=["train", "train_val", "all"], default="train")
    parser.add_argument("--exclude_target_from_sources", action="store_true")
    parser.add_argument("--horizon_mode", choices=["sequence_max", "config"], default="config")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--refine_epochs", type=int, default=60)
    parser.add_argument("--em_iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--relation_lr", type=float, default=0.0035)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_formulas", type=int, default=8)
    parser.add_argument("--max_rules", type=int, default=8)
    parser.add_argument("--max_rule_length", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--softmin_rho", type=float, default=20.0)
    parser.add_argument("--dummy_count", type=int, default=3)
    parser.add_argument("--laplace_scale", type=float, default=0.5)
    parser.add_argument("--gamma_threshold", type=float, default=1e-3)
    parser.add_argument("--slot_threshold", type=float, default=0.0)
    parser.add_argument("--search_restarts", type=int, default=1)
    parser.add_argument("--cover_threshold", type=float, default=0.5)
    parser.add_argument("--relation_epochs", type=int, default=20)
    args = parser.parse_args()
    payload = run_baseline(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output), "rules": len(payload["predicted_rules"]), "metadata": payload["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
