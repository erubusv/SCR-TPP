from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import pickle
import re
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2
import torch
import yaml

from conjunctive_rule_initializer import (
    SourceBasisCache,
    auto_grid_step,
    build_midpoint_grid,
    build_seq_event_arrays,
    collect_target_events,
    estimate_source_kernels,
    feasible_subset_list,
    format_rule,
    gt_rules_from_config,
    normalize_piecewise_score,
    summarize_results,
)
from data.synthetic import (
    create_rules_from_config,
    generate_canonical_loglink_data,
)
import rule_dependent_kernel_active_set as rd
from runtime_resources import configure_runtime_resources, default_cpu_threads


ROOT = Path(__file__).resolve().parents[3]
PAPER_SUITE = ROOT / "data" / "paper_suite"
PAPER_CONFIGS = PAPER_SUITE / "configs"
FINAL_SYNTHETIC_CONFIGS = PAPER_CONFIGS / "hetero_source_2000_adjusted"
RESULTS_PATH = PAPER_SUITE / "results" / "paper_benchmark_results_real_world.json"
FINAL_KERNEL_SMOOTHNESS_RIDGE = 1e-3
FINAL_SUPPORT_BETA_THRESHOLD = 1e-4
FINAL_REPORT_KERNEL_DF = True
FINAL_MAX_RULE_ORDER = 3
FINAL_SIEVE_STEPS = 20
FINAL_SIEVE_BATCH_SIZE = 16
FINAL_EXACT_BATCH_SIZE = 8
_FINAL_SCORE_SUBSETS = None
_FINAL_SCORE_SOURCE_IDS: tuple[int, ...] | None = None
_FINAL_SCORE_NUM_KNOTS = 7
_FINAL_SUPPORT_IDENTITY_CACHE: dict[rd.SupportKey, float] = {}


def _progress(msg: str) -> None:
    print(f"[paper-runner] {msg}", flush=True)


def _set_final_support_score_context(
    *,
    subsets,
    source_ids: tuple[int, ...],
    num_knots: int = 7,
) -> None:
    global _FINAL_SCORE_SUBSETS, _FINAL_SCORE_SOURCE_IDS, _FINAL_SCORE_NUM_KNOTS, _FINAL_SUPPORT_IDENTITY_CACHE
    _FINAL_SCORE_SUBSETS = subsets
    _FINAL_SCORE_SOURCE_IDS = tuple(int(src) for src in source_ids)
    _FINAL_SCORE_NUM_KNOTS = max(int(num_knots), 2)
    _FINAL_SUPPORT_IDENTITY_CACHE = {}


def _log_choose(n: int, k: int) -> float:
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return float("inf")
    if k == 0 or k == n:
        return 0.0
    return float(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _support_identity_code(
    *,
    active_rules: list[rd.ActiveRule],
    subsets,
) -> float:
    rules = rd.sort_unique_sign_exclusive_rules(list(active_rules))
    if not rules:
        return 0.0
    order_sign_universe: dict[tuple[int, str], int] = {}
    for subset in subsets:
        order = int(len(subset))
        if order <= 0:
            continue
        for sign in ("exc", "inh"):
            key = (int(order), str(sign))
            order_sign_universe[key] = int(order_sign_universe.get(key, 0)) + 1

    selected_counts: dict[tuple[int, str], int] = {}
    for rule in rules:
        key = (int(len(subsets[int(rule.idx)])), str(rule.sign))
        selected_counts[key] = int(selected_counts.get(key, 0)) + 1

    code = 0.0
    for key, count in selected_counts.items():
        code += _log_choose(int(order_sign_universe.get(key, 0)), int(count))
    return float(code)


def _support_identity_mdl_penalty(
    *,
    active_rules: list[rd.ActiveRule],
    subsets,
) -> float:
    rules = rd.sort_unique_sign_exclusive_rules(list(active_rules))
    key = rd.support_key_from_rules(rules)
    cached = _FINAL_SUPPORT_IDENTITY_CACHE.get(key)
    if cached is not None:
        return float(cached)
    code_length = float(_support_identity_code(active_rules=rules, subsets=subsets))
    _FINAL_SUPPORT_IDENTITY_CACHE[key] = float(code_length)
    return float(code_length)


def _support_identity_bic_scale_penalty(
    *,
    active_rules: list[rd.ActiveRule],
    subsets,
) -> float:
    # Convert an MDL code length in nats to the -2 log-likelihood scale.
    return 2.0 * _support_identity_mdl_penalty(
        active_rules=active_rules,
        subsets=subsets,
    )


def _kernel_shape_identity_bic_scale_penalty(
    *,
    active_rules: list[rd.ActiveRule],
    subsets,
    num_knots: int,
) -> float:
    # One finite-knot dictionary choice per selected rule-source kernel.
    kernel_blocks = rd.kernel_group_count(list(active_rules), subsets)
    return 2.0 * float(kernel_blocks) * math.log(float(max(int(num_knots), 2)))


def _support_base_mdl_score(
    result: rd.SupportEvalResult,
    *,
    signed_universe_size: int,
) -> float:
    _ = int(signed_universe_size)
    if _FINAL_SCORE_SUBSETS is None or _FINAL_SCORE_SOURCE_IDS is None:
        raise RuntimeError("Block-MDL score requires initialized support-score context")
    return (
        float(result.bic)
        + _support_identity_bic_scale_penalty(
            active_rules=list(result.active_rules),
            subsets=_FINAL_SCORE_SUBSETS,
        )
        + _kernel_shape_identity_bic_scale_penalty(
            active_rules=list(result.active_rules),
            subsets=_FINAL_SCORE_SUBSETS,
            num_knots=int(_FINAL_SCORE_NUM_KNOTS),
        )
    )


def _final_support_score(
    result: rd.SupportEvalResult,
    *,
    signed_universe_size: int,
) -> float:
    return _support_base_mdl_score(result, signed_universe_size=int(signed_universe_size))


def _final_support_better(
    cand: rd.SupportEvalResult,
    ref: rd.SupportEvalResult,
    *,
    signed_universe_size: int,
) -> bool:
    ref_score = _final_support_score(ref, signed_universe_size=int(signed_universe_size))
    cand_score = _final_support_score(cand, signed_universe_size=int(signed_universe_size))
    if float(cand_score) + 1e-8 < float(ref_score):
        return True
    if abs(float(cand_score) - float(ref_score)) <= 1e-8:
        return len(cand.active_rules) < len(ref.active_rules)
    return False


def _conditional_amplitude_threshold(
    *,
    num_sequences: int,
    signed_universe_size: int,
) -> tuple[float, float]:
    """One-sided finite-family amplitude threshold on the -2logL scale.

    Support and kernel-shape complexity is already charged by J(S).  The final
    admissibility condition therefore tests the selected fitted activation's
    nonnegative signed amplitude only.  Under the one-sided beta>=0 boundary,
    the null tail is the 0.5*chi^2_1 mixture tail, so P(mixture>x)<=alpha is
    enforced by chi^2_1 survival probability 2*alpha.
    """

    n_eff = rd.bic_sample_size(int(num_sequences))
    move_count = max(int(signed_universe_size), 1)
    alpha = float(n_eff) ** -2 / float(move_count)
    chi_tail = min(max(2.0 * float(alpha), 1e-300), 1.0 - 1e-12)
    return float(chi2.isf(chi_tail, 1.0)), 1.0


def _conditional_necessity_stats(
    *,
    current_result: rd.SupportEvalResult,
    without_result: rd.SupportEvalResult,
    dropped_rule: rd.ActiveRule,
    subsets,
    num_sequences: int,
    signed_universe_size: int,
) -> dict:
    n_eff = rd.bic_sample_size(int(num_sequences))
    move_count = max(int(signed_universe_size), 1)
    delta_n = float(n_eff) ** -2
    alpha_channel = float(delta_n) / float(move_count)
    evidence = (
        _final_support_score(without_result, signed_universe_size=int(signed_universe_size))
        - _final_support_score(current_result, signed_universe_size=int(signed_universe_size))
    )
    amplitude_threshold, df_delta = _conditional_amplitude_threshold(
        num_sequences=int(num_sequences),
        signed_universe_size=int(signed_universe_size),
    )
    amplitude_margin = float(evidence) - float(amplitude_threshold)
    block_mdl_improves = float(evidence) > 0.0
    amplitude_pass = bool(block_mdl_improves and float(amplitude_margin) > 0.0)
    certified = bool(amplitude_pass)
    return {
        "rule_idx": int(dropped_rule.idx),
        "rule_sign": str(dropped_rule.sign),
        "rule_order": int(len(subsets[int(dropped_rule.idx)])),
        "evidence": float(evidence),
        "alpha_channel": float(alpha_channel),
        "amplitude_df": float(df_delta),
        "amplitude_threshold": float(amplitude_threshold),
        "amplitude_margin": float(amplitude_margin),
        "amplitude_pass": bool(amplitude_pass),
        "certified": bool(certified),
        "certificate_margin": float(amplitude_margin),
    }


def _amplitude_admissible_addition(
    *,
    base_result: rd.SupportEvalResult,
    candidate_result: rd.SupportEvalResult,
    num_sequences: int,
    signed_universe_size: int,
) -> bool:
    evidence = (
        _final_support_score(base_result, signed_universe_size=int(signed_universe_size))
        - _final_support_score(candidate_result, signed_universe_size=int(signed_universe_size))
    )
    threshold, _df = _conditional_amplitude_threshold(
        num_sequences=int(num_sequences),
        signed_universe_size=int(signed_universe_size),
    )
    return bool(float(evidence) > float(threshold))


def _conditional_necessity_certificate(
    *,
    current_result: rd.SupportEvalResult,
    subsets,
    basis_cache: SourceBasisCache,
    template_rule_heights,
    grid_weights: np.ndarray,
    device: torch.device,
    torch_basis_cache: rd.TorchBasisCache | None,
    num_sequences: int,
    penalty_scale: float,
    kernel_smoothness_ridge: float,
    signed_universe_size: int,
) -> tuple[rd.SupportEvalResult, dict[str, int], list[dict]]:
    stats = {
        "passes": 0,
        "drop_evals": 0,
        "rejects": 0,
        "kept_by_amplitude": 0,
    }
    rows: list[dict] = []
    best = current_result
    while True:
        stats["passes"] += 1
        current = rd.sort_unique_sign_exclusive_rules(list(best.active_rules))
        if not current:
            break
        failing: list[tuple[float, float, int, str, rd.ActiveRule, dict]] = []
        for drop_pos, rule in enumerate(current):
            trial = current[:drop_pos] + current[drop_pos + 1 :]
            without_result = rd.evaluate_support_exact(
                active_rules=trial,
                subsets=subsets,
                basis_cache=basis_cache,
                base_rule_heights=best.rule_heights,
                template_rule_heights=template_rule_heights,
                grid_weights_train=grid_weights,
                grid_weights_val=grid_weights,
                device=device,
                torch_basis_cache=torch_basis_cache,
                opt_steps=60,
                lr=0.05,
                penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
                penalty_scale=float(penalty_scale),
                num_val_sequences=int(num_sequences),
                kernel_smoothness_ridge=float(kernel_smoothness_ridge),
                support_cache=None,
                warm_start_result=best,
            )
            stats["drop_evals"] += 1
            row = _conditional_necessity_stats(
                current_result=best,
                without_result=without_result,
                dropped_rule=rule,
                subsets=subsets,
                num_sequences=int(num_sequences),
                signed_universe_size=int(signed_universe_size),
            )
            row["pass"] = int(stats["passes"])
            rows.append(dict(row))
            if bool(row["amplitude_pass"]):
                stats["kept_by_amplitude"] += 1
            if not bool(row["certified"]):
                failing.append(
                    (
                        float(row["certificate_margin"]),
                        float(row["evidence"]),
                        int(rule.idx),
                        str(rule.sign),
                        rule,
                        row,
                    )
                )
        if not failing:
            break

        _margin, _evidence, _idx, _sign, dropped, dropped_row = sorted(failing)[0]
        next_rules = [
            rule
            for rule in current
            if not (int(rule.idx) == int(dropped.idx) and str(rule.sign) == str(dropped.sign))
        ]
        best = rd.evaluate_support_exact(
            active_rules=next_rules,
            subsets=subsets,
            basis_cache=basis_cache,
            base_rule_heights=best.rule_heights,
            template_rule_heights=template_rule_heights,
            grid_weights_train=grid_weights,
            grid_weights_val=grid_weights,
            device=device,
            torch_basis_cache=torch_basis_cache,
            opt_steps=60,
            lr=0.05,
            penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
            penalty_scale=float(penalty_scale),
            num_val_sequences=int(num_sequences),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=None,
            warm_start_result=best,
        )
        stats["rejects"] += 1
        _progress(
            "conditional_necessity_certificate:rejected "
            f"idx={int(dropped.idx)} sign={dropped.sign} "
            f"evidence={float(dropped_row['evidence']):.3f} "
            f"amp={float(dropped_row['amplitude_margin']):.3f} "
            f"rules={len(best.active_rules)}"
        )
    return best, stats, rows


def _load_rule_discovery_dataset(data_path: str | Path) -> tuple[list[dict], list[dict], dict]:
    """Use every synthetic sequence for support discovery.

    The paper synthetic benchmark evaluates true-rule recovery, not predictive
    generalization, so holding out validation/test sequences only weakens the
    discovery problem.  Return the same full sequence list as both fitting and
    scoring data so the existing exact-refit code evaluates one all-data
    objective without changing the mathematical criterion.
    """

    with open(Path(str(data_path)), "rb") as f:
        dataset = pickle.load(f)
    full = (
        list(dataset.get("train", []))
        + list(dataset.get("val", []))
        + list(dataset.get("test", []))
    )
    if not full:
        raise ValueError(f"dataset has no sequences: {data_path}")
    return full, list(full), dataset["metadata"]



def _build_full_likelihood_selection_context(prepared: dict) -> dict:
    full, _scoring, metadata = _load_rule_discovery_dataset(prepared["data_path"])
    num_types = int(metadata["num_types"])
    target = int(prepared["target"])
    source_ids = tuple(sorted(k for k in range(num_types) if k != target))
    time_horizon = float(prepared["time_horizon"])
    grid_step = float(prepared["grid_step"])
    full_arrays = build_seq_event_arrays(full, num_types)
    event_seq, event_times = collect_target_events(full, target=target)
    grid_seq, grid_times, grid_weights = build_midpoint_grid(
        full,
        time_horizon=float(time_horizon),
        step=float(grid_step),
    )
    basis_cache = SourceBasisCache(
        source_ids=source_ids,
        knots=np.asarray(prepared["knots"], dtype=np.float64),
        train_arrays=full_arrays,
        val_arrays=full_arrays,
        train_event_seq_ids=event_seq,
        train_event_times=event_times,
        train_grid_seq_ids=grid_seq,
        train_grid_times=grid_times,
        val_event_seq_ids=event_seq,
        val_event_times=event_times,
        val_grid_seq_ids=grid_seq,
        val_grid_times=grid_times,
    )
    device = prepared["device"]
    return {
        "basis_cache": basis_cache,
        "torch_basis_cache": rd.TorchBasisCache(basis_cache, device),
        "grid_weights": np.asarray(grid_weights, dtype=np.float64),
        "num_sequences": int(len(full)),
    }


class _SieveSupportEvaluator:
    """Fast support scorer using a deterministic knot-hat kernel sieve.

    The evaluator keeps the canonical log-link and strict AND activation, but
    replaces per-candidate raw kernel refits with coefficient updates on
    precomputed source/basis witness responses.
    """

    def __init__(
        self,
        *,
        prepared: dict,
        basis_cache: SourceBasisCache,
        torch_basis_cache: rd.TorchBasisCache,
        grid_weights: np.ndarray,
        source_ids: tuple[int, ...],
        num_sequences: int,
        signed_universe_size: int,
        steps: int = 20,
        batch_size: int = 16,
    ) -> None:
        self.prepared = prepared
        self.subsets = prepared["subsets"]
        self.template_rule_heights = prepared["template_rule_heights"]
        self.basis_cache = basis_cache
        self.torch_basis_cache = torch_basis_cache
        self.grid_weights_np = np.asarray(grid_weights, dtype=np.float64)
        self.device = prepared["device"]
        self.source_ids = tuple(int(src) for src in source_ids)
        self.num_sequences = int(num_sequences)
        self.signed_universe_size = int(signed_universe_size)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.num_knots = int(len(basis_cache.knots))
        self.max_rule_order = max(int(len(subset)) for subset in self.subsets)
        self.grid_weights = torch.as_tensor(self.grid_weights_np, dtype=torch.float32, device=self.device)

        first_src = int(self.source_ids[0])
        first_tr_ev, first_tr_gr, _first_va_ev, _first_va_gr = self.torch_basis_cache.arrays(first_src)
        self.num_events = int(first_tr_ev.num_queries)
        self.grid_len = int(first_tr_gr.num_queries)
        self._basis_response: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._source_event_scope: dict[int, torch.Tensor] = {}
        self._source_grid_scope: dict[int, torch.Tensor] = {}
        self._source_set_scope_cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._pricing_atom_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._pricing_atom_cache_bytes = 0
        self._pricing_atom_cache_max_bytes = self._infer_pricing_atom_cache_budget()
        self._support_cache: dict[rd.SupportKey, rd.SupportEvalResult] = {}
        self._atom_indices_by_order = {
            order: torch.as_tensor(
                np.asarray(list(itertools.product(range(self.num_knots), repeat=order)), dtype=np.int64),
                dtype=torch.int64,
                device=self.device,
            )
            for order in range(1, int(self.max_rule_order) + 1)
        }
        self._precompute_basis_responses()

    def _infer_pricing_atom_cache_budget(self) -> int:
        # Keep real-world scoring memory-bounded.
        return 0

    def _precompute_basis_responses(self) -> None:
        eye = torch.eye(self.num_knots, dtype=torch.float32, device=self.device)
        for src in self.source_ids:
            combined, sizes = self.torch_basis_cache.combined_arrays(int(src), ("tr_ev", "tr_gr"))
            n_ev, n_gr = sizes
            event_chunks: list[torch.Tensor] = []
            grid_chunks: list[torch.Tensor] = []
            # Chunk over basis rows to avoid a large response tensor.
            basis_chunk = 1 if self.device.type == "cuda" else max(1, int(self.num_knots))
            for start in range(0, int(self.num_knots), int(basis_chunk)):
                h_chunk = eye[start: start + int(basis_chunk)]
                z_chunk = rd.witness_response_torch_batched(combined, h_chunk)
                event_chunks.append(z_chunk[:, : int(n_ev)].contiguous())
                grid_chunks.append(z_chunk[:, int(n_ev): int(n_ev) + int(n_gr)].contiguous())
                del z_chunk, h_chunk
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            event_basis = torch.cat(event_chunks, dim=0).contiguous()
            grid_basis = torch.cat(grid_chunks, dim=0).contiguous()
            self._basis_response[int(src)] = (event_basis, grid_basis)
            self._source_event_scope[int(src)] = torch.amax(event_basis, dim=0) > 1e-8
            self._source_grid_scope[int(src)] = torch.amax(grid_basis, dim=0) > 1e-8
            del combined, event_chunks, grid_chunks
            # Exact refits rebuild selected rule-source arrays on demand.
            if hasattr(self.torch_basis_cache, "_combo_cache"):
                self.torch_basis_cache._combo_cache.clear()
            if hasattr(self.torch_basis_cache, "_cache"):
                self.torch_basis_cache._cache.clear()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _scope_masks_for_sources(self, srcs: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(sorted(int(src) for src in srcs))
        cached = self._source_set_scope_cache.get(key)
        if cached is not None:
            return cached
        event_scope = torch.ones((self.num_events,), dtype=torch.bool, device=self.device)
        grid_scope = torch.ones((self.grid_len,), dtype=torch.bool, device=self.device)
        for src in key:
            event_scope = event_scope & self._source_event_scope[int(src)]
            grid_scope = grid_scope & self._source_grid_scope[int(src)]
        cached = (event_scope, grid_scope)
        self._source_set_scope_cache[key] = cached
        return cached

    def _pricing_atom_products(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(idx)
        cached = self._pricing_atom_cache.get(idx)
        if cached is not None:
            return cached

        srcs = tuple(int(src) for src in self.subsets[int(idx)])
        event_mats = [self._basis_response[int(src)][0] for src in srcs]
        grid_mats = [self._basis_response[int(src)][1] for src in srcs]
        event_scope, grid_scope = self._scope_masks_for_sources(srcs)
        atom_indices = self._atom_indices_by_order[int(len(srcs))]
        atom_chunk_size = 32

        def product_chunk(mats: list[torch.Tensor], atom_chunk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            scope_len = int(mask.sum().item())
            if scope_len == 0:
                return torch.empty((int(atom_chunk.shape[0]), 0), dtype=torch.float32, device=self.device)
            out = torch.ones((int(atom_chunk.shape[0]), scope_len), dtype=torch.float32, device=self.device)
            for source_pos, mat in enumerate(mats):
                out = out * mat[atom_chunk[:, int(source_pos)]][:, mask]
            return out

        event_sum_chunks: list[torch.Tensor] = []
        grid_atom_chunks: list[torch.Tensor] = []
        for start in range(0, int(atom_indices.shape[0]), int(atom_chunk_size)):
            atom_chunk = atom_indices[start: start + int(atom_chunk_size)]
            event_atoms = product_chunk(event_mats, atom_chunk, event_scope)
            grid_atoms = product_chunk(grid_mats, atom_chunk, grid_scope)
            event_sum = torch.sum(event_atoms, dim=1) if int(event_atoms.shape[1]) else torch.zeros(
                (int(atom_chunk.shape[0]),), dtype=torch.float32, device=self.device
            )
            event_sum_chunks.append(event_sum.contiguous())
            grid_atom_chunks.append(grid_atoms.contiguous())

        event_sums = torch.cat(event_sum_chunks, dim=0).contiguous()
        grid_atom_matrix = torch.cat(grid_atom_chunks, dim=0).contiguous()
        entry_bytes = int(event_sums.numel() * event_sums.element_size())
        entry_bytes += int(grid_atom_matrix.numel() * grid_atom_matrix.element_size())
        if (
            int(self._pricing_atom_cache_max_bytes) > 0
            and int(self._pricing_atom_cache_bytes) + int(entry_bytes) <= int(self._pricing_atom_cache_max_bytes)
        ):
            cached = (event_sums, grid_atom_matrix, grid_scope)
            self._pricing_atom_cache[idx] = cached
            self._pricing_atom_cache_bytes += int(entry_bytes)
            return cached
        return event_sums, grid_atom_matrix, grid_scope

    def _pricing_atom_statistics(
        self,
        idx: int,
        *,
        fisher_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return atom event sums, weighted grid sums, and Fisher diagonals.

        The synthetic runner materializes every finite-cone atom over the full
        scoped grid.  On MIMIC this can exceed GPU memory.  This real-world
        path computes the exact same sufficient statistics by streaming over
        scoped grid indices, so the pricing bound and candidate ordering are
        unchanged.
        """
        srcs = tuple(int(src) for src in self.subsets[int(idx)])
        event_mats = [self._basis_response[int(src)][0] for src in srcs]
        grid_mats = [self._basis_response[int(src)][1] for src in srcs]
        event_scope, grid_scope = self._scope_masks_for_sources(srcs)
        atom_indices = self._atom_indices_by_order[int(len(srcs))]
        atom_chunk_size = 8 if self.device.type == "cuda" else 32
        query_chunk_size = 262144 if self.device.type == "cuda" else 1048576
        event_idx = torch.nonzero(event_scope, as_tuple=False).squeeze(1)
        grid_idx = torch.nonzero(grid_scope, as_tuple=False).squeeze(1)

        def product_indices(
            mats: list[torch.Tensor],
            atom_chunk: torch.Tensor,
            query_idx: torch.Tensor,
        ) -> torch.Tensor:
            if int(query_idx.numel()) == 0:
                return torch.empty((int(atom_chunk.shape[0]), 0), dtype=torch.float32, device=self.device)
            out = torch.ones((int(atom_chunk.shape[0]), int(query_idx.numel())), dtype=torch.float32, device=self.device)
            for source_pos, mat in enumerate(mats):
                out = out * mat[atom_chunk[:, int(source_pos)]][:, query_idx]
            return out

        event_sum_chunks: list[torch.Tensor] = []
        grid_sum_chunks: list[torch.Tensor] = []
        info_chunks: list[torch.Tensor] = []
        for start in range(0, int(atom_indices.shape[0]), int(atom_chunk_size)):
            atom_chunk = atom_indices[start: start + int(atom_chunk_size)]
            event_atoms = product_indices(event_mats, atom_chunk, event_idx)
            event_sums = (
                torch.sum(event_atoms, dim=1)
                if int(event_atoms.shape[1])
                else torch.zeros((int(atom_chunk.shape[0]),), dtype=torch.float32, device=self.device)
            )
            grid_sums = torch.zeros((int(atom_chunk.shape[0]),), dtype=torch.float32, device=self.device)
            infos = torch.zeros((int(atom_chunk.shape[0]),), dtype=torch.float32, device=self.device)
            for q_start in range(0, int(grid_idx.numel()), int(query_chunk_size)):
                q_idx = grid_idx[q_start: q_start + int(query_chunk_size)]
                grid_atoms = product_indices(grid_mats, atom_chunk, q_idx)
                weight = fisher_weight[q_idx].reshape(1, -1)
                grid_sums = grid_sums + torch.sum(grid_atoms * weight, dim=1)
                infos = infos + torch.sum(grid_atoms * grid_atoms * weight, dim=1)
                del grid_atoms, weight, q_idx
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            event_sum_chunks.append(event_sums.contiguous())
            grid_sum_chunks.append(grid_sums.contiguous())
            info_chunks.append(infos.contiguous())
            del atom_chunk, event_atoms, event_sums, grid_sums, infos
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return (
            torch.cat(event_sum_chunks, dim=0).contiguous(),
            torch.cat(grid_sum_chunks, dim=0).contiguous(),
            torch.cat(info_chunks, dim=0).contiguous(),
            grid_scope,
        )

    @staticmethod
    def _normalized_rules(rules: list[rd.ActiveRule]) -> list[rd.ActiveRule]:
        return rd.sort_unique_sign_exclusive_rules(list(rules))

    @classmethod
    def _support_key(cls, rules: list[rd.ActiveRule]) -> rd.SupportKey:
        return rd.support_key_from_rules(cls._normalized_rules(rules))

    def _initial_height(self, rule: rd.ActiveRule, src: int) -> np.ndarray:
        sign_map = self.template_rule_heights.get(str(rule.sign), {})
        h = sign_map.get((int(rule.idx), int(src)))
        if h is None:
            h = self.prepared["rule_heights"][(int(rule.idx), int(src))]
        return rd.ensure_feasible_score_heights_numpy(np.asarray(h, dtype=np.float64)).astype(np.float32, copy=False)

    def _feature_from_heights(
        self,
        *,
        idx: int,
        heights_by_src: dict[int, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        event_feature = torch.ones((self.num_events,), dtype=torch.float32, device=self.device)
        grid_feature = torch.ones((self.grid_len,), dtype=torch.float32, device=self.device)
        for src in self.subsets[int(idx)]:
            heights_np = np.asarray(heights_by_src[int(src)], dtype=np.float32)
            heights = torch.as_tensor(heights_np, dtype=torch.float32, device=self.device).reshape(1, -1)
            event_basis, grid_basis = self._basis_response[int(src)]
            event_feature = event_feature * (heights @ event_basis).reshape(-1)
            grid_feature = grid_feature * (heights @ grid_basis).reshape(-1)
        return event_feature, grid_feature

    def _result_rule_feature(
        self,
        result: rd.SupportEvalResult,
        rule: rd.ActiveRule,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heights_by_src: dict[int, np.ndarray] = {}
        for src in self.subsets[int(rule.idx)]:
            h = result.rule_heights.get((int(rule.idx), int(src)))
            if h is None:
                h = self._initial_height(rule, int(src))
            heights_by_src[int(src)] = np.asarray(h, dtype=np.float32)
        return self._feature_from_heights(idx=int(rule.idx), heights_by_src=heights_by_src)

    @staticmethod
    def _rule_signed_beta(result: rd.SupportEvalResult, rule: rd.ActiveRule) -> float:
        if str(rule.sign) == "exc":
            return float(result.exc_params.get(int(rule.idx), 0.0))
        return -float(result.inh_params.get(int(rule.idx), 0.0))

    @staticmethod
    def _rule_direction_sign(rule: rd.ActiveRule) -> float:
        return 1.0 if str(rule.sign) == "exc" else -1.0

    def score_pricing_blocks(
        self,
        *,
        current_result: rd.SupportEvalResult,
        candidate_rules: list[rd.ActiveRule],
        penalty_delta,
    ) -> list[tuple[float, rd.ActiveRule]]:
        """Scoped finite-cone score bound for inactive signed rules.

        For a candidate strict-AND rule, only the event/grid locations where all
        sources can witness the rule are scored.  The kernel search direction is
        the finite nonnegative cone spanned by products of source knot-basis
        responses.  We use a diagonal cone upper bound on the one-step BIC
        improvement, so candidates whose bound cannot beat the stated MDL
        increment are safely ignored by this pricing pass.
        """
        current_rules = self._normalized_rules(list(current_result.active_rules))
        eta_grid = torch.zeros((self.grid_len,), dtype=torch.float32, device=self.device)
        for rule in current_rules:
            _event_feature, grid_feature = self._result_rule_feature(current_result, rule)
            signed_beta = self._rule_signed_beta(current_result, rule)
            eta_grid = eta_grid + float(signed_beta) * grid_feature

        lambda_grid = float(current_result.mu) * torch.exp(torch.clamp(eta_grid, min=-40.0, max=40.0))
        fisher_weight = lambda_grid * self.grid_weights

        priced: list[tuple[float, rd.ActiveRule]] = []
        rules_by_idx: dict[int, dict[str, rd.ActiveRule]] = {}
        for rule in candidate_rules:
            rules_by_idx.setdefault(int(rule.idx), {})[str(rule.sign)] = rule

        with torch.no_grad():
            for idx, sign_rules in sorted(rules_by_idx.items(), key=lambda item: int(item[0])):
                if not tuple(int(src) for src in self.subsets[int(idx)]):
                    continue
                event_sums_all, grid_sum, info, grid_scope = self._pricing_atom_statistics(
                    int(idx),
                    fisher_weight=fisher_weight,
                )
                if int(grid_scope.sum().item()) == 0:
                    continue
                upper_gain_by_sign = {"exc": 0.0, "inh": 0.0}
                signed_score = event_sums_all - grid_sum
                safe_info = torch.clamp(info, min=1e-8)
                exc_gains = torch.where(
                    (signed_score > 0.0) & (info > 1e-8),
                    signed_score * signed_score / safe_info,
                    torch.zeros_like(signed_score),
                )
                inh_gains = torch.where(
                    (signed_score < 0.0) & (info > 1e-8),
                    signed_score * signed_score / safe_info,
                    torch.zeros_like(signed_score),
                )
                if bool(torch.any(torch.isfinite(exc_gains)).item()):
                    upper_gain_by_sign["exc"] += float(
                        torch.sum(torch.nan_to_num(exc_gains, nan=0.0, posinf=0.0, neginf=0.0)).item()
                    )
                if bool(torch.any(torch.isfinite(inh_gains)).item()):
                    upper_gain_by_sign["inh"] += float(
                        torch.sum(torch.nan_to_num(inh_gains, nan=0.0, posinf=0.0, neginf=0.0)).item()
                    )
                for sign, rule in sign_rules.items():
                    required = float(penalty_delta(rule))
                    margin = float(upper_gain_by_sign[str(sign)] - required)
                    if margin > 1e-8:
                        priced.append((margin, rule))
        return sorted(priced, key=lambda item: (-float(item[0]), int(item[1].idx), str(item[1].sign)))

    def _constant_result(self) -> rd.SupportEvalResult:
        exposure = float(np.sum(self.grid_weights_np))
        mu = max(float(self.num_events) / max(exposure, 1e-8), 1e-8)
        nll = -(math.log(mu) * float(self.num_events)) + mu * exposure
        bic = 2.0 * nll + math.log(float(rd.bic_sample_size(int(self.num_sequences))))
        return rd.SupportEvalResult(
            bic=float(bic),
            mu=float(mu),
            exc_params={},
            inh_params={},
            rule_heights={},
            arrays_out=None,
            active_rules=[],
        )

    def evaluate_many(self, rule_lists: list[list[rd.ActiveRule]]) -> dict[rd.SupportKey, rd.SupportEvalResult]:
        unique: dict[rd.SupportKey, list[rd.ActiveRule]] = {}
        for rules in rule_lists:
            normalized = self._normalized_rules(list(rules))
            unique[self._support_key(normalized)] = normalized
        out: dict[rd.SupportKey, rd.SupportEvalResult] = {}
        pending_by_size: dict[int, list[list[rd.ActiveRule]]] = {}
        for key, rules in unique.items():
            cached = self._support_cache.get(key)
            if cached is not None:
                out[key] = cached
            else:
                pending_by_size.setdefault(int(len(rules)), []).append(rules)
        for rule_count, group in sorted(pending_by_size.items(), key=lambda item: int(item[0])):
            if int(rule_count) == 0:
                result = self._constant_result()
                for rules in group:
                    key = self._support_key(rules)
                    self._support_cache[key] = result
                    out[key] = result
                continue
            for start in range(0, len(group), self.batch_size):
                chunk = group[start: start + self.batch_size]
                chunk_results = self._evaluate_same_size(chunk)
                for key, result in chunk_results.items():
                    self._support_cache[key] = result
                    out[key] = result
        return out

    def _evaluate_same_size(self, active_rule_lists: list[list[rd.ActiveRule]]) -> dict[rd.SupportKey, rd.SupportEvalResult]:
        batch_count = int(len(active_rule_lists))
        rule_count = int(len(active_rule_lists[0]))
        if any(int(len(rules)) != rule_count for rules in active_rule_lists):
            raise ValueError("sieve batch expects equal-size supports")

        sign_np = np.zeros((batch_count, rule_count), dtype=np.float32)
        entry_support: list[int] = []
        entry_rule_row: list[int] = []
        entry_rule_idx: list[int] = []
        entry_source: list[int] = []
        entry_init_heights: list[np.ndarray] = []
        for batch_row, rules in enumerate(active_rule_lists):
            for rule_row, rule in enumerate(rules):
                sign_np[int(batch_row), int(rule_row)] = 1.0 if str(rule.sign) == "exc" else -1.0
                for src in self.subsets[int(rule.idx)]:
                    entry_support.append(int(batch_row))
                    entry_rule_row.append(int(rule_row))
                    entry_rule_idx.append(int(rule.idx))
                    entry_source.append(int(src))
                    entry_init_heights.append(self._initial_height(rule, int(src)))

        raw = torch.nn.Parameter(
            torch.as_tensor(np.stack(entry_init_heights, axis=0), dtype=torch.float32, device=self.device)
        )
        sign_tensor = torch.as_tensor(sign_np, dtype=torch.float32, device=self.device)
        beta_current = torch.full((batch_count, rule_count), 0.1, dtype=torch.float32, device=self.device)
        entry_support_tensor = torch.as_tensor(entry_support, dtype=torch.int64, device=self.device)
        entry_rule_flat_tensor = torch.as_tensor(
            [
                int(batch_row) * int(rule_count) + int(rule_row)
                for batch_row, rule_row in zip(entry_support, entry_rule_row)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        source_to_entries: dict[int, list[int]] = {}
        for entry_id, src in enumerate(entry_source):
            source_to_entries.setdefault(int(src), []).append(int(entry_id))
        source_to_entry_tensors = {
            int(src): torch.as_tensor(entry_ids, dtype=torch.int64, device=self.device)
            for src, entry_ids in source_to_entries.items()
        }
        opt = torch.optim.Adam([raw], lr=0.05)
        best_bic = np.full((batch_count,), np.inf, dtype=np.float64)
        best_raw = raw.detach().clone()
        best_beta = beta_current.detach().clone()

        def normalize_rows(x: torch.Tensor) -> torch.Tensor:
            y = torch.clamp(x, min=0.0)
            peak = torch.amax(y, dim=1, keepdim=True)
            return y / torch.clamp(peak, min=1e-8)

        def build_features() -> tuple[torch.Tensor, torch.Tensor]:
            normalized = normalize_rows(raw)
            flat_rule_count = int(batch_count) * int(rule_count)
            event_flat = torch.ones((flat_rule_count, self.num_events), dtype=torch.float32, device=self.device)
            grid_flat = torch.ones((flat_rule_count, self.grid_len), dtype=torch.float32, device=self.device)
            for src, entry_tensor in source_to_entry_tensors.items():
                heights = normalized[entry_tensor]
                event_basis, grid_basis = self._basis_response[int(src)]
                z_event = heights @ event_basis
                z_grid = heights @ grid_basis
                rows = entry_rule_flat_tensor[entry_tensor]
                event_flat = event_flat.scatter_reduce(
                    0,
                    rows.reshape(-1, 1).expand(-1, self.num_events),
                    z_event,
                    reduce="prod",
                    include_self=True,
                )
                grid_flat = grid_flat.scatter_reduce(
                    0,
                    rows.reshape(-1, 1).expand(-1, self.grid_len),
                    z_grid,
                    reduce="prod",
                    include_self=True,
                )
            event_features = event_flat.view(batch_count, rule_count, self.num_events)
            grid_features = grid_flat.view(batch_count, rule_count, self.grid_len)
            return torch.sum(event_features, dim=2), grid_features

        for step in range(max(1, self.steps)):
            opt.zero_grad(set_to_none=True)
            event_masses, grid_matrix = build_features()
            beta_fit, _mu_fit, _event_sum, _signed_grid = rd._solve_beta_block_canonical_batch(
                beta_init=beta_current,
                sign_tensor=sign_tensor,
                event_masses=event_masses.detach(),
                grid_matrix=grid_matrix.detach(),
                grid_weights=self.grid_weights,
                num_events=int(self.num_events),
                max_iter=10,
            )
            beta_current = beta_fit.detach()
            signed_beta = (beta_current * sign_tensor).detach()
            signed_event_sum = torch.sum(signed_beta * event_masses, dim=1)
            signed_grid = torch.einsum("br,brg->bg", signed_beta, grid_matrix)
            mu_fit, exposure = rd._profile_mu_canonical_torch_batch(
                signed_grid=signed_grid,
                grid_weights=self.grid_weights,
                num_events=int(self.num_events),
            )
            loss = torch.sum(-(torch.log(torch.clamp(mu_fit, min=1e-8)) * float(self.num_events) + signed_event_sum) + mu_fit * exposure)
            loss.backward()
            prev_raw = raw.detach().clone()
            opt.step()
            with torch.no_grad():
                raw.clamp_(min=0.0)
                peak = torch.amax(raw, dim=1, keepdim=True)
                good = peak.squeeze(1) > 1e-8
                raw[good] = raw[good] / torch.clamp(peak[good], min=1e-8)
                raw[~good] = prev_raw[~good]

            if step == self.steps - 1:
                with torch.no_grad():
                    event_eval, grid_eval = build_features()
                    beta_eval, mu_eval, event_sum_eval, signed_grid_eval = rd._solve_beta_block_canonical_batch(
                        beta_init=beta_current,
                        sign_tensor=sign_tensor,
                        event_masses=event_eval,
                        grid_matrix=grid_eval,
                        grid_weights=self.grid_weights,
                        num_events=int(self.num_events),
                        max_iter=16,
                    )
                    exposure_eval = torch.sum(
                        torch.exp(torch.clamp(signed_grid_eval, min=-40.0, max=40.0)) * self.grid_weights.reshape(1, -1),
                        dim=1,
                    )
                    nll = -(torch.log(torch.clamp(mu_eval, min=1e-8)) * float(self.num_events) + event_sum_eval) + mu_eval * exposure_eval
                    dim_values = torch.as_tensor(
                        [
                            float(rd.model_param_dim(rules, self.subsets, int(self.num_knots)))
                            for rules in active_rule_lists
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    bic = (
                        2.0 * nll
                        + dim_values * math.log(float(rd.bic_sample_size(int(self.num_sequences))))
                    ).detach().cpu().numpy().astype(np.float64, copy=False)
                    improved = bic < best_bic
                    if bool(np.any(improved)):
                        improved_tensor = torch.as_tensor(improved, dtype=torch.bool, device=self.device)
                        best_bic[improved] = bic[improved]
                        best_beta[improved_tensor] = beta_eval.detach()[improved_tensor]
                        best_raw[improved_tensor[entry_support_tensor]] = raw.detach()[improved_tensor[entry_support_tensor]]

        with torch.no_grad():
            raw.copy_(best_raw)
            event_final, grid_final = build_features()
            beta_final, mu_final, event_sum_final, signed_grid_final = rd._solve_beta_block_canonical_batch(
                beta_init=best_beta,
                sign_tensor=sign_tensor,
                event_masses=event_final,
                grid_matrix=grid_final,
                grid_weights=self.grid_weights,
                num_events=int(self.num_events),
                max_iter=16,
            )
            exposure_final = torch.sum(
                torch.exp(torch.clamp(signed_grid_final, min=-40.0, max=40.0)) * self.grid_weights.reshape(1, -1),
                dim=1,
            )
            nll_final = -(torch.log(torch.clamp(mu_final, min=1e-8)) * float(self.num_events) + event_sum_final) + mu_final * exposure_final
            dim_final = torch.as_tensor(
                [
                    float(rd.model_param_dim(rules, self.subsets, int(self.num_knots)))
                    for rules in active_rule_lists
                ],
                dtype=torch.float32,
                device=self.device,
            )
            bic_final = (
                2.0 * nll_final
                + dim_final * math.log(float(rd.bic_sample_size(int(self.num_sequences))))
            ).detach().cpu().numpy().astype(np.float64, copy=False)
            beta_np = beta_final.detach().cpu().numpy().astype(np.float64, copy=False)
            mu_np = mu_final.detach().cpu().numpy().astype(np.float64, copy=False)
            heights_np = normalize_rows(raw).detach().cpu().numpy().astype(np.float64, copy=False)

        out: dict[rd.SupportKey, rd.SupportEvalResult] = {}
        for batch_row, rules in enumerate(active_rule_lists):
            exc_params: dict[int, float] = {}
            inh_params: dict[int, float] = {}
            for rule_row, rule in enumerate(rules):
                beta_val = float(beta_np[int(batch_row), int(rule_row)])
                if str(rule.sign) == "exc":
                    exc_params[int(rule.idx)] = beta_val
                else:
                    inh_params[int(rule.idx)] = beta_val
            rule_heights: dict[tuple[int, int], np.ndarray] = {}
            for entry_id, row in enumerate(entry_support):
                if int(row) == int(batch_row):
                    rule_heights[(int(entry_rule_idx[entry_id]), int(entry_source[entry_id]))] = heights_np[int(entry_id)].copy()
            result = rd.SupportEvalResult(
                bic=float(bic_final[int(batch_row)]),
                mu=float(mu_np[int(batch_row)]),
                exc_params=exc_params,
                inh_params=inh_params,
                rule_heights=rule_heights,
                arrays_out=None,
                active_rules=list(rules),
            )
            out[self._support_key(list(rules))] = result
        return out



def _pricing_violations_for_support(
    *,
    current_result: rd.SupportEvalResult,
    evaluator: _SieveSupportEvaluator,
    subsets,
    source_ids: tuple[int, ...],
) -> list[tuple[float, rd.ActiveRule]]:
    current = rd.sort_unique_sign_exclusive_rules(list(current_result.active_rules))
    current_idx = {int(rule.idx) for rule in current}
    candidate_rules = [
        rd.ActiveRule(idx=int(idx), sign=str(sign))
        for idx in range(len(subsets))
        for sign in ("exc", "inh")
        if int(idx) not in current_idx
    ]
    current_penalty = _support_identity_bic_scale_penalty(
        active_rules=current,
        subsets=subsets,
    )
    current_dim = rd.model_param_dim(current, subsets, evaluator.num_knots)
    log_n = math.log(float(rd.bic_sample_size(int(evaluator.num_sequences))))

    def penalty_delta(rule: rd.ActiveRule) -> float:
        trial = rd.sort_unique_sign_exclusive_rules(current + [rule])
        trial_penalty = _support_identity_bic_scale_penalty(
            active_rules=trial,
            subsets=subsets,
        )
        trial_dim = rd.model_param_dim(trial, subsets, evaluator.num_knots)
        return float(trial_dim - current_dim) * float(log_n) + float(trial_penalty - current_penalty)

    return evaluator.score_pricing_blocks(
        current_result=current_result,
        candidate_rules=candidate_rules,
        penalty_delta=penalty_delta,
    )


def _exact_pricing_rescue(
    *,
    current_result: rd.SupportEvalResult,
    evaluator: _SieveSupportEvaluator,
    subsets,
    source_ids: tuple[int, ...],
    basis_cache: SourceBasisCache,
    template_rule_heights,
    grid_weights: np.ndarray,
    device: torch.device,
    torch_basis_cache: rd.TorchBasisCache | None,
    num_sequences: int,
    penalty_scale: float,
    kernel_smoothness_ridge: float,
    support_cache: dict[rd.ExactSupportCacheKey, rd.SupportEvalResult],
) -> tuple[rd.SupportEvalResult, dict[str, int]]:
    stats = {
        "passes": 0,
        "pricing_evals": 0,
        "pricing_violations": 0,
        "fallback_exact_evals": 0,
        "fallback_accepts": 0,
        "accepted_adds": 0,
        "admissibility_pruned": 0,
    }

    best = current_result

    while True:
        stats["passes"] += 1
        current = rd.sort_unique_sign_exclusive_rules(list(best.active_rules))
        violations = _pricing_violations_for_support(
            current_result=best,
            evaluator=evaluator,
            subsets=subsets,
            source_ids=source_ids,
        )
        stats["pricing_evals"] += int(max(0, 2 * len(subsets) - 2 * len(current)))
        stats["pricing_violations"] += int(len(violations))
        if not violations:
            break

        best_by_stratum: dict[tuple[int, str], tuple[float, rd.ActiveRule]] = {}
        margin_by_rule = {
            (int(rule.idx), str(rule.sign)): float(margin)
            for margin, rule in violations
        }
        for _margin, rule in violations:
            key = (int(len(subsets[int(rule.idx)])), str(rule.sign))
            margin = float(margin_by_rule.get((int(rule.idx), str(rule.sign)), 0.0))
            old = best_by_stratum.get(key)
            if old is None or margin > float(old[0]):
                best_by_stratum[key] = (margin, rule)
        # One reduced-cost winner per order/sign stratum.
        candidate_rules = [
            rule
            for _margin, rule in sorted(
                best_by_stratum.values(),
                key=lambda item: (-float(item[0]), int(item[1].idx), str(item[1].sign)),
            )
        ]
        if not candidate_rules:
            break
        fallback_rule_lists = [
            rd.sort_unique_sign_exclusive_rules(current + [rule])
            for rule in candidate_rules
        ]
        fallback_results = (
            rd.evaluate_support_exact_batch(
                support_rule_lists=fallback_rule_lists,
                subsets=subsets,
                basis_cache=basis_cache,
                base_rule_heights=best.rule_heights,
                template_rule_heights=template_rule_heights,
                grid_weights_train=grid_weights,
                grid_weights_val=grid_weights,
                device=device,
                torch_basis_cache=torch_basis_cache,
                opt_steps=60,
                lr=0.05,
                penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
                penalty_scale=float(penalty_scale),
                num_val_sequences=int(num_sequences),
                kernel_smoothness_ridge=float(kernel_smoothness_ridge),
                support_cache=support_cache,
                warm_start_result=best,
                score_only=True,
                exact_batch_size=int(FINAL_EXACT_BATCH_SIZE),
            )
            if fallback_rule_lists
            else {}
        )
        stats["fallback_exact_evals"] += int(len(fallback_rule_lists))
        all_results = dict(fallback_results)
        best_fallback = best
        for result in all_results.values():
            if not _amplitude_admissible_addition(
                base_result=best,
                candidate_result=result,
                num_sequences=int(num_sequences),
                signed_universe_size=int(evaluator.signed_universe_size),
            ):
                stats["admissibility_pruned"] += 1
                continue
            if _final_support_better(
                result,
                best_fallback,
                signed_universe_size=int(evaluator.signed_universe_size),
            ):
                best_fallback = result
        if best_fallback is best:
            break
        best = best_fallback
        stats["fallback_accepts"] += 1
        stats["accepted_adds"] += 1
        _progress(
            "exact_pricing_rescue:accepted "
            f"pass={stats['passes']} mode=fallback "
            f"rules={len(best.active_rules)} violations={len(violations)}"
        )

    return best, stats


def _exact_drop_certificate(
    *,
    current_result: rd.SupportEvalResult,
    subsets,
    basis_cache: SourceBasisCache,
    template_rule_heights,
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: rd.TorchBasisCache | None,
    num_sequences: int,
    penalty_scale: float,
    kernel_smoothness_ridge: float,
    support_cache: dict[rd.ExactSupportCacheKey, rd.SupportEvalResult],
    signed_universe_size: int,
) -> tuple[rd.SupportEvalResult, dict[str, int], dict[rd.SupportKey, rd.SupportEvalResult]]:
    stats = {"evals": 0, "accepts": 0}
    best = current_result
    last_drop_results: dict[rd.SupportKey, rd.SupportEvalResult] = {}
    while True:
        current_rules = rd.sort_unique_sign_exclusive_rules(list(best.active_rules))
        if not current_rules:
            break
        drop_pairs = [
            (current_rules[drop_pos], current_rules[:drop_pos] + current_rules[drop_pos + 1 :])
            for drop_pos in range(len(current_rules))
        ]
        drop_trials = [trial for _dropped, trial in drop_pairs]
        non_empty_trials = [trial for trial in drop_trials if trial]
        drop_results: dict[rd.SupportKey, rd.SupportEvalResult] = {}
        if non_empty_trials:
            drop_results.update(
                rd.evaluate_support_exact_batch(
                    support_rule_lists=non_empty_trials,
                    subsets=subsets,
                    basis_cache=basis_cache,
                    base_rule_heights=best.rule_heights,
                    template_rule_heights=template_rule_heights,
                    grid_weights_train=grid_weights_train,
                    grid_weights_val=grid_weights_val,
                    device=device,
                    torch_basis_cache=torch_basis_cache,
                    opt_steps=60,
                    lr=0.05,
                    penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
                    penalty_scale=float(penalty_scale),
                    num_val_sequences=int(num_sequences),
                    kernel_smoothness_ridge=float(kernel_smoothness_ridge),
                    support_cache=support_cache,
                    warm_start_result=best,
                    score_only=True,
                    exact_batch_size=int(FINAL_EXACT_BATCH_SIZE),
                )
            )
        if any(not trial for trial in drop_trials):
            empty_result = rd.evaluate_support_exact(
                active_rules=[],
                subsets=subsets,
                basis_cache=basis_cache,
                base_rule_heights=best.rule_heights,
                template_rule_heights=template_rule_heights,
                grid_weights_train=grid_weights_train,
                grid_weights_val=grid_weights_val,
                device=device,
                torch_basis_cache=torch_basis_cache,
                opt_steps=60,
                lr=0.05,
                penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
                penalty_scale=float(penalty_scale),
                num_val_sequences=int(num_sequences),
                kernel_smoothness_ridge=float(kernel_smoothness_ridge),
                support_cache=support_cache,
            )
            drop_results[rd.support_key_from_rules([])] = empty_result
        stats["evals"] += int(len(drop_trials))
        last_drop_results = dict(drop_results)
        best_drop = best
        for dropped_rule, trial in drop_pairs:
            _ = dropped_rule
            trial_key = rd.support_key_from_rules(rd.sort_unique_sign_exclusive_rules(trial))
            trial_result = drop_results.get(trial_key)
            if trial_result is None:
                continue
            if _final_support_better(
                trial_result,
                best_drop,
                signed_universe_size=int(signed_universe_size),
            ):
                best_drop = trial_result
        if best_drop is best:
            break

        best = best_drop
        stats["accepts"] += 1
        _progress(
            "final_exact_drop_certificate:accepted "
            f"rules={len(best.active_rules)} evals={stats['evals']}"
        )
    return best, stats, last_drop_results


def _exact_swap_certificate(
    *,
    current_result: rd.SupportEvalResult,
    evaluator: _SieveSupportEvaluator,
    subsets,
    source_ids: tuple[int, ...],
    basis_cache: SourceBasisCache,
    template_rule_heights,
    grid_weights: np.ndarray,
    device: torch.device,
    torch_basis_cache: rd.TorchBasisCache | None,
    num_sequences: int,
    penalty_scale: float,
    kernel_smoothness_ridge: float,
    support_cache: dict[rd.ExactSupportCacheKey, rd.SupportEvalResult],
    signed_universe_size: int,
    precomputed_drop_results: dict[rd.SupportKey, rd.SupportEvalResult] | None = None,
) -> tuple[rd.SupportEvalResult, dict[str, int]]:
    """Exact one-drop/one-add replacement certificate.

    Drop-only cannot recover a true lower-order rule when the current support
    contains a high-order proxy: dropping the proxy alone may hurt, while
    replacing it with the true rule improves the final objective.  This stage
    evaluates the deterministic steepest pricing replacement after each exact
    one-rule deletion.
    """
    stats = {
        "drop_base_evals": 0,
        "pricing_evals": 0,
        "pricing_violations": 0,
        "swap_evals": 0,
        "bound_pruned": 0,
        "drop_base_reused": 0,
        "accepts": 0,
        "admissibility_pruned": 0,
    }
    current = rd.sort_unique_sign_exclusive_rules(list(current_result.active_rules))
    if not current:
        return current_result, stats

    current_key = rd.support_key_from_rules(current)
    expected_drop_keys = {
        rd.support_key_from_rules(current[:drop_pos] + current[drop_pos + 1 :])
        for drop_pos in range(len(current))
    }
    if precomputed_drop_results is not None and expected_drop_keys.issubset(set(precomputed_drop_results)):
        drop_results = {
            key: precomputed_drop_results[key]
            for key in expected_drop_keys
        }
        stats["drop_base_reused"] = int(len(drop_results))
    else:
        drop_trials = [
            current[:drop_pos] + current[drop_pos + 1 :]
            for drop_pos in range(len(current))
        ]
        drop_results = rd.evaluate_support_exact_batch(
            support_rule_lists=drop_trials,
            subsets=subsets,
            basis_cache=basis_cache,
            base_rule_heights=current_result.rule_heights,
            template_rule_heights=template_rule_heights,
            grid_weights_train=grid_weights,
            grid_weights_val=grid_weights,
            device=device,
            torch_basis_cache=torch_basis_cache,
            opt_steps=60,
            lr=0.05,
            penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
            penalty_scale=float(penalty_scale),
            num_val_sequences=int(num_sequences),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=support_cache,
            warm_start_result=current_result,
            score_only=True,
            exact_batch_size=int(FINAL_EXACT_BATCH_SIZE),
        )
        stats["drop_base_evals"] = int(len(drop_trials))

    swap_trials: dict[rd.SupportKey, list[rd.ActiveRule]] = {}
    swap_bases: dict[rd.SupportKey, rd.SupportEvalResult] = {}
    current_score = _final_support_score(current_result, signed_universe_size=int(signed_universe_size))
    for drop_result in drop_results.values():
        base_rules = rd.sort_unique_sign_exclusive_rules(list(drop_result.active_rules))
        drop_base_score = _support_base_mdl_score(drop_result, signed_universe_size=int(signed_universe_size))
        violations = _pricing_violations_for_support(
            current_result=drop_result,
            evaluator=evaluator,
            subsets=subsets,
            source_ids=source_ids,
        )
        stats["pricing_evals"] += int(max(0, 2 * len(subsets) - 2 * len(base_rules)))
        stats["pricing_violations"] += int(len(violations))
        if not violations:
            continue
        best_by_stratum: dict[tuple[int, str], tuple[float, rd.ActiveRule]] = {}
        for margin, rule in violations:
            key = (int(len(subsets[int(rule.idx)])), str(rule.sign))
            old = best_by_stratum.get(key)
            if old is None or float(margin) > float(old[0]):
                best_by_stratum[key] = (float(margin), rule)
        for _margin, rule in sorted(
            best_by_stratum.values(),
            key=lambda item: (-float(item[0]), int(item[1].idx), str(item[1].sign)),
        ):
            # Skip swaps whose one-column bound cannot beat the incumbent.
            optimistic_score = float(drop_base_score) - float(_margin)
            if float(optimistic_score) >= float(current_score) - 1e-8:
                stats["bound_pruned"] += 1
                continue
            try:
                trial = rd.sort_unique_sign_exclusive_rules(base_rules + [rule])
            except ValueError:
                continue
            trial_key = rd.support_key_from_rules(trial)
            if trial_key == current_key:
                continue
            swap_trials.setdefault(trial_key, trial)
            swap_bases.setdefault(trial_key, drop_result)

    if not swap_trials:
        return current_result, stats

    swap_results = rd.evaluate_support_exact_batch(
        support_rule_lists=list(swap_trials.values()),
        subsets=subsets,
        basis_cache=basis_cache,
        base_rule_heights=current_result.rule_heights,
        template_rule_heights=template_rule_heights,
        grid_weights_train=grid_weights,
        grid_weights_val=grid_weights,
        device=device,
        torch_basis_cache=torch_basis_cache,
        opt_steps=60,
        lr=0.05,
        penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
        penalty_scale=float(penalty_scale),
        num_val_sequences=int(num_sequences),
        kernel_smoothness_ridge=float(kernel_smoothness_ridge),
        support_cache=support_cache,
        warm_start_result=current_result,
        score_only=True,
        exact_batch_size=int(FINAL_EXACT_BATCH_SIZE),
    )
    stats["swap_evals"] = int(len(swap_trials))

    best = current_result
    for key, result in swap_results.items():
        base_result = swap_bases.get(key)
        if base_result is None:
            continue
        if not _amplitude_admissible_addition(
            base_result=base_result,
            candidate_result=result,
            num_sequences=int(num_sequences),
            signed_universe_size=int(signed_universe_size),
        ):
            stats["admissibility_pruned"] += 1
            continue
        if _final_support_better(result, best, signed_universe_size=int(signed_universe_size)):
            best = result
    if best is not current_result:
        stats["accepts"] = 1
        _progress(
            "final_exact_swap_certificate:accepted "
            f"rules={len(best.active_rules)} evals={stats['swap_evals']}"
        )
    return best, stats


def _run_exact_local_closure(
    *,
    current_result: rd.SupportEvalResult,
    evaluator: _SieveSupportEvaluator,
    subsets,
    source_ids: tuple[int, ...],
    basis_cache: SourceBasisCache,
    template_rule_heights,
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: rd.TorchBasisCache | None,
    num_sequences: int,
    penalty_scale: float,
    kernel_smoothness_ridge: float,
    support_cache: dict[rd.ExactSupportCacheKey, rd.SupportEvalResult],
    signed_universe_size: int,
    reason: str,
) -> tuple[rd.SupportEvalResult, dict[str, float], dict[str, int]]:
    timings = {
        "exact_pricing_rescue": 0.0,
        "final_exact_drop_certificate": 0.0,
        "final_exact_swap_certificate": 0.0,
    }
    counts = {
        "exact_local_closure_rounds": 0,
        "exact_pricing_rescue_passes": 0,
        "exact_pricing_rescue_pricing_evals": 0,
        "exact_pricing_rescue_pricing_violations": 0,
        "exact_pricing_rescue_fallback_exact_evals": 0,
        "exact_pricing_rescue_fallback_accepts": 0,
        "exact_pricing_rescue_accepted_adds": 0,
        "final_exact_drop_certificate_evals": 0,
        "final_exact_drop_certificate_accepts": 0,
        "final_exact_swap_certificate_drop_base_evals": 0,
        "final_exact_swap_certificate_pricing_evals": 0,
        "final_exact_swap_certificate_pricing_violations": 0,
        "final_exact_swap_certificate_swap_evals": 0,
        "final_exact_swap_certificate_bound_pruned": 0,
        "final_exact_swap_certificate_accepts": 0,
        "final_exact_swap_certificate_drop_base_reused": 0,
    }
    result = current_result
    while True:
        counts["exact_local_closure_rounds"] += 1
        closure_round = int(counts["exact_local_closure_rounds"])
        _progress(
            f"exact_pricing_rescue:start "
            f"reason={reason} round={closure_round} rules={len(result.active_rules)}"
        )
        t_exact_rescue = time.perf_counter()
        result, rescue_stats = _exact_pricing_rescue(
            current_result=result,
            evaluator=evaluator,
            subsets=subsets,
            source_ids=source_ids,
            basis_cache=basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights=grid_weights_train,
            device=device,
            torch_basis_cache=torch_basis_cache,
            num_sequences=num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=support_cache,
        )
        timings["exact_pricing_rescue"] += float(time.perf_counter() - t_exact_rescue)
        for key, value in rescue_stats.items():
            counts[f"exact_pricing_rescue_{key}"] = (
                int(counts.get(f"exact_pricing_rescue_{key}", 0)) + int(value)
            )
        _progress(
            f"exact_pricing_rescue:end "
            f"reason={reason} round={closure_round} "
            f"sec={timings['exact_pricing_rescue']:.3f} "
            f"passes={rescue_stats.get('passes', 0)} "
            f"accepted={rescue_stats.get('accepted_adds', 0)}"
        )

        _progress(
            f"final_exact_drop_certificate:start "
            f"reason={reason} round={closure_round} rules={len(result.active_rules)}"
        )
        t_final_drop = time.perf_counter()
        result, drop_stats, drop_results_for_swap = _exact_drop_certificate(
            current_result=result,
            subsets=subsets,
            basis_cache=basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            num_sequences=num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=support_cache,
            signed_universe_size=int(signed_universe_size),
        )
        timings["final_exact_drop_certificate"] += float(time.perf_counter() - t_final_drop)
        counts["final_exact_drop_certificate_evals"] += int(drop_stats.get("evals", 0))
        counts["final_exact_drop_certificate_accepts"] += int(drop_stats.get("accepts", 0))
        _progress(
            f"final_exact_drop_certificate:end "
            f"reason={reason} round={closure_round} "
            f"sec={timings['final_exact_drop_certificate']:.3f} "
            f"accepts={drop_stats.get('accepts', 0)} "
            f"evals={drop_stats.get('evals', 0)}"
        )

        _progress(
            f"final_exact_swap_certificate:start "
            f"reason={reason} round={closure_round} rules={len(result.active_rules)}"
        )
        t_final_swap = time.perf_counter()
        result, swap_stats = _exact_swap_certificate(
            current_result=result,
            evaluator=evaluator,
            subsets=subsets,
            source_ids=source_ids,
            basis_cache=basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights=grid_weights_train,
            device=device,
            torch_basis_cache=torch_basis_cache,
            num_sequences=num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=support_cache,
            signed_universe_size=int(signed_universe_size),
            precomputed_drop_results=drop_results_for_swap,
        )
        timings["final_exact_swap_certificate"] += float(time.perf_counter() - t_final_swap)
        for key, value in swap_stats.items():
            counts[f"final_exact_swap_certificate_{key}"] = (
                int(counts.get(f"final_exact_swap_certificate_{key}", 0)) + int(value)
            )
        _progress(
            f"final_exact_swap_certificate:end "
            f"reason={reason} round={closure_round} "
            f"sec={timings['final_exact_swap_certificate']:.3f} "
            f"accepts={swap_stats.get('accepts', 0)} "
            f"evals={swap_stats.get('swap_evals', 0)} "
            f"drop_reused={swap_stats.get('drop_base_reused', 0)}"
        )

        if int(drop_stats.get("accepts", 0)) == 0 and int(swap_stats.get("accepts", 0)) == 0:
            break
    return result, timings, counts


def _sieve_priced_column_generation(
    *,
    start_result: rd.SupportEvalResult,
    subsets,
    source_ids: tuple[int, ...],
    signed_universe_size: int,
    evaluator: _SieveSupportEvaluator,
) -> tuple[rd.SupportEvalResult, dict[str, int]]:
    source_ids = tuple(int(src) for src in source_ids)
    stats = {
        "enabled": 1,
        "passes": 0,
        "pricing_evals": 0,
        "score_pricing_evals": 0,
        "score_pricing_violations": 0,
        "pricing_blocks_evaluated": 0,
        "pricing_violations": 0,
        "pricing_block_violations": 0,
        "pricing_columns_added": 0,
        "scheduled_blocks": 0,
        "block_state_evals": 0,
        "drop_state_evals": 0,
        "add_state_evals": 0,
        "swap_base_count": 0,
        "swap_pricing_evals": 0,
        "swap_state_evals": 0,
        "accepted_blocks": 0,
        "final_rule_count": 0,
    }

    def normalized_rules(rules: list[rd.ActiveRule]) -> list[rd.ActiveRule]:
        return rd.sort_unique_sign_exclusive_rules(list(rules))

    def support_key(rules: list[rd.ActiveRule]) -> rd.SupportKey:
        return rd.support_key_from_rules(normalized_rules(rules))

    def support_score(result: rd.SupportEvalResult) -> float:
        return _support_base_mdl_score(result, signed_universe_size=int(signed_universe_size))

    best = start_result
    seen_supports = {support_key(list(best.active_rules))}
    all_signed_rules = [
        rd.ActiveRule(idx=int(idx), sign=str(sign))
        for idx in range(len(subsets))
        for sign in ("exc", "inh")
    ]

    def pricing_violations_for_result(
        current_result: rd.SupportEvalResult,
        *,
        stat_prefix: str = "pricing",
    ) -> list[tuple[float, rd.ActiveRule]]:
        current = normalized_rules(list(current_result.active_rules))
        current_idx = {int(rule.idx) for rule in current}
        pricing_rules: list[rd.ActiveRule] = []
        for rule in all_signed_rules:
            if int(rule.idx) in current_idx:
                continue
            pricing_rules.append(rule)

        current_penalty = _support_identity_bic_scale_penalty(
            active_rules=current,
            subsets=subsets,
        )
        current_dim = rd.model_param_dim(current, subsets, evaluator.num_knots)
        log_n = math.log(float(rd.bic_sample_size(int(evaluator.num_sequences))))

        def penalty_delta(rule: rd.ActiveRule) -> float:
            trial = normalized_rules(current + [rule])
            trial_penalty = _support_identity_bic_scale_penalty(
                active_rules=trial,
                subsets=subsets,
            )
            trial_dim = rd.model_param_dim(trial, subsets, evaluator.num_knots)
            return float(trial_dim - current_dim) * float(log_n) + float(trial_penalty - current_penalty)

        violations = evaluator.score_pricing_blocks(
            current_result=current_result,
            candidate_rules=pricing_rules,
            penalty_delta=penalty_delta,
        )
        if str(stat_prefix) == "pricing":
            stats["pricing_evals"] += int(len(pricing_rules))
            stats["score_pricing_evals"] += int(len(pricing_rules))
            stats["score_pricing_violations"] += int(len(violations))
            stats["pricing_violations"] += int(len(violations))
        else:
            stats["swap_pricing_evals"] += int(len(pricing_rules))
        return violations

    def queue_candidate(
        queued: dict[rd.SupportKey, list[rd.ActiveRule]],
        rules: list[rd.ActiveRule],
        *,
        current_key: rd.SupportKey,
    ) -> None:
        normalized = normalized_rules(list(rules))
        key = support_key(normalized)
        if key == current_key:
            return
        queued.setdefault(key, normalized)

    def best_improving_result(
        candidate_results: dict[rd.SupportKey, rd.SupportEvalResult],
        *,
        current_best: rd.SupportEvalResult,
        seen_supports: set[rd.SupportKey],
    ) -> tuple[rd.SupportKey | None, rd.SupportEvalResult | None]:
        best_key: rd.SupportKey | None = None
        best_result = current_best
        for key, result in candidate_results.items():
            if key in seen_supports:
                continue
            if support_score(result) + 1e-8 < support_score(best_result):
                best_key = key
                best_result = result
        if best_key is None or best_result is current_best:
            return None, None
        return best_key, best_result

    def stratified_pricing_columns(
        violations: list[tuple[float, rd.ActiveRule]],
    ) -> list[tuple[float, rd.ActiveRule]]:
        # Deterministic order/sign strata avoid a tunable top-K shortlist.
        best_by_stratum: dict[tuple[int, str], tuple[float, rd.ActiveRule]] = {}
        for margin, rule in violations:
            key = (int(len(subsets[int(rule.idx)])), str(rule.sign))
            old = best_by_stratum.get(key)
            if old is None or float(margin) > float(old[0]):
                best_by_stratum[key] = (float(margin), rule)
        return sorted(
            best_by_stratum.values(),
            key=lambda item: (-float(item[0]), int(item[1].idx), str(item[1].sign)),
        )

    while True:
        stats["passes"] += 1
        current = normalized_rules(list(best.active_rules))
        current_key = support_key(current)
        queued: dict[rd.SupportKey, list[rd.ActiveRule]] = {}

        drop_trials = [
            current[:drop_pos] + current[drop_pos + 1 :]
            for drop_pos in range(len(current))
        ]
        for trial in drop_trials:
            queue_candidate(queued, trial, current_key=current_key)
        stats["drop_state_evals"] += int(len(drop_trials))

        violations = pricing_violations_for_result(best, stat_prefix="pricing")
        add_columns = stratified_pricing_columns(violations)
        stats["pricing_columns_added"] += int(len(add_columns))
        for _margin, rule in add_columns:
            queue_candidate(queued, current + [rule], current_key=current_key)
        stats["add_state_evals"] += int(len(add_columns))
        stats["block_state_evals"] += int(len(drop_trials) + len(add_columns))

        candidate_results = evaluator.evaluate_many(list(queued.values())) if queued else {}
        accepted_key, accepted = best_improving_result(
            candidate_results,
            current_best=best,
            seen_supports=seen_supports,
        )
        move_type = "drop/add"

        if accepted is None or accepted_key is None:
            swap_queued: dict[rd.SupportKey, list[rd.ActiveRule]] = {}
            drop_results = {
                key: result
                for key, result in candidate_results.items()
                if len(result.active_rules) == max(len(current) - 1, 0)
            }
            for drop_base in drop_results.values():
                stats["swap_base_count"] += 1
                base_rules = normalized_rules(list(drop_base.active_rules))
                drop_violations = pricing_violations_for_result(drop_base, stat_prefix="swap")
                for _margin, rule in stratified_pricing_columns(drop_violations):
                    try:
                        queue_candidate(swap_queued, base_rules + [rule], current_key=current_key)
                    except ValueError:
                        continue
            stats["swap_state_evals"] += int(len(swap_queued))
            stats["block_state_evals"] += int(len(swap_queued))
            swap_results = evaluator.evaluate_many(list(swap_queued.values())) if swap_queued else {}
            accepted_key, accepted = best_improving_result(
                swap_results,
                current_best=best,
                seen_supports=seen_supports,
            )
            move_type = "drop+add"

        if accepted is None or accepted_key is None:
            break

        best = accepted
        seen_supports.add(accepted_key)
        stats["accepted_blocks"] += 1
        _progress(
            "sieve_priced_column_generation:accepted "
            f"pass={stats['passes']} move={stats['accepted_blocks']} "
            f"type={move_type} rules={len(best.active_rules)} "
            f"pricing={len(violations)} states={stats['block_state_evals']}"
        )

    stats["final_rule_count"] = int(len(best.active_rules))
    return best, stats


BENCHMARKS = [
    {"name": "logical_clean_plus", "config": FINAL_SYNTHETIC_CONFIGS / "logical_clean_plus.yaml"},
    {"name": "logical_shared", "config": FINAL_SYNTHETIC_CONFIGS / "logical_shared.yaml"},
    {"name": "logical_context", "config": FINAL_SYNTHETIC_CONFIGS / "logical_context.yaml"},
    {"name": "kernel_triangular", "config": FINAL_SYNTHETIC_CONFIGS / "kernel_triangular.yaml"},
    {"name": "kernel_exponential", "config": FINAL_SYNTHETIC_CONFIGS / "kernel_exponential.yaml"},
    {"name": "kernel_gaussian", "config": FINAL_SYNTHETIC_CONFIGS / "kernel_gaussian.yaml"},
    {"name": "num_predicates_10", "config": FINAL_SYNTHETIC_CONFIGS / "num_predicates_10.yaml"},
    {"name": "num_predicates_20", "config": FINAL_SYNTHETIC_CONFIGS / "num_predicates_20.yaml"},
    {"name": "ablation_excitation_only", "config": FINAL_SYNTHETIC_CONFIGS / "ablation_excitation_only.yaml"},
    {"name": "ablation_inhibition_only", "config": FINAL_SYNTHETIC_CONFIGS / "ablation_inhibition_only.yaml"},
    {"name": "ablation_mixed_sign", "config": FINAL_SYNTHETIC_CONFIGS / "ablation_mixed_sign.yaml"},
]


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def _config_with_dataset_seed(cfg: dict, dataset_seed: int | None) -> dict:
    if dataset_seed is None:
        return cfg
    out = copy.deepcopy(cfg)
    seed = int(dataset_seed)
    out["seed"] = seed
    path_cfg = dict(out.get("path", {}))
    output_path = str(path_cfg.get("output_path", "")).strip()
    if output_path:
        path_cfg["output_path"] = re.sub(r"seed_\d+", f"seed_{seed}", output_path)
    out["path"] = path_cfg
    return out


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        workspace_prefix = Path("/workspace")
        try:
            rel = path.relative_to(workspace_prefix)
        except ValueError:
            return path
        return ROOT / rel
    return ROOT / path


def resolve_compute_device(requested: str) -> torch.device:
    req = str(requested).strip().lower()
    if req in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(requested))


def _normalize_config_for_compare(obj):
    if isinstance(obj, dict):
        return {str(key): _normalize_config_for_compare(value) for key, value in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, list):
        return [_normalize_config_for_compare(value) for value in obj]
    return obj


def _rule_sign(rule_cfg: dict) -> str | None:
    w_pos = float(rule_cfg.get("W_pos", 0.0))
    w_neg = float(rule_cfg.get("W_neg", 0.0))
    if w_pos > w_neg:
        return "exc"
    if w_neg > w_pos:
        return "inh"
    return None


def _truth_rule_lookup(config: dict) -> dict[tuple[tuple[int, ...], str, int], dict]:
    lookup: dict[tuple[tuple[int, ...], str, int], dict] = {}
    for rule_cfg in config.get("rules", []):
        sign = _rule_sign(rule_cfg)
        if sign is None:
            continue
        key = (
            tuple(sorted(int(s) for s in rule_cfg["condition"].keys())),
            sign,
            int(rule_cfg["target"]),
        )
        lookup[key] = rule_cfg
    return lookup


def _config_max_lag(config: dict) -> float:
    kernel_type = str(config.get("kernel", "triangular"))
    max_lag = 0.0
    for rule_cfg in config.get("rules", []):
        for source_condition in rule_cfg.get("condition", {}).values():
            peaks = list(source_condition.get("peaks", []))
            widths = list(source_condition.get("widths", []))
            support_mults = list(source_condition.get("support_mults", []))
            if not support_mults:
                support_mults = [3.0] * len(peaks)
            for peak, width, support_mult in zip(peaks, widths, support_mults):
                if kernel_type in {"gaussian", "exponential"}:
                    support = float(peak) + max(float(support_mult), 0.0) * float(width)
                else:
                    support = float(width)
                max_lag = max(max_lag, support)
    return max(float(max_lag), 1.0)


def _triangular_kernel_values(knots: np.ndarray, peak: float, width: float, mix_weight: float) -> np.ndarray:
    out = np.zeros_like(knots, dtype=np.float64)
    eps = 1e-12
    width = max(float(width), eps)
    peak = float(np.clip(float(peak), eps, width - eps))
    mask = (knots > 0.0) & (knots <= width)
    if not np.any(mask):
        return out
    dt = knots[mask]
    vals = np.where(
        dt <= peak,
        float(mix_weight) * (dt / peak),
        float(mix_weight) * ((width - dt) / max(width - peak, eps)),
    )
    out[mask] = np.maximum(vals, 0.0)
    return out


def _gaussian_kernel_values(
    knots: np.ndarray,
    peak: float,
    sigma: float,
    mix_weight: float,
    support_mult: float,
) -> np.ndarray:
    out = np.zeros_like(knots, dtype=np.float64)
    eps = 1e-12
    sigma = max(float(sigma), eps)
    peak = max(float(peak), eps)
    support = peak + max(float(support_mult), 0.0) * sigma
    mask = (knots > 0.0) & (knots <= support)
    if not np.any(mask):
        return out
    z = (knots[mask] - peak) / sigma
    out[mask] = float(mix_weight) * np.exp(-0.5 * z * z)
    return out


def _exponential_kernel_values(
    knots: np.ndarray,
    peak: float,
    tau: float,
    mix_weight: float,
    support_mult: float,
) -> np.ndarray:
    out = np.zeros_like(knots, dtype=np.float64)
    eps = 1e-12
    tau = max(float(tau), eps)
    peak = max(float(peak), eps)
    support = peak + max(float(support_mult), 0.0) * tau
    mask = (knots >= peak) & (knots <= support)
    if not np.any(mask):
        return out
    z = (knots[mask] - peak) / tau
    out[mask] = float(mix_weight) * np.exp(-z)
    return out


def _truth_kernel_heights(
    *,
    kernel_type: str,
    source_condition: dict,
    knots: np.ndarray,
) -> np.ndarray:
    peaks = list(source_condition.get("peaks", []))
    widths = list(source_condition.get("widths", []))
    mix_weights = list(source_condition.get("mix_weights", []))
    support_mults = list(source_condition.get("support_mults", []))
    n_comp = len(peaks)
    if len(widths) != n_comp or len(mix_weights) != n_comp:
        raise ValueError("kernel condition expects peaks/widths/mix_weights with equal lengths")
    if not support_mults:
        support_mults = [3.0] * n_comp
    elif len(support_mults) != n_comp:
        raise ValueError("kernel condition support_mults length must match peaks length")

    params_array = np.asarray(
        list(zip(peaks, widths, mix_weights, support_mults)),
        dtype=np.float64,
    )
    if params_array.shape[0] == 0:
        return np.zeros_like(knots, dtype=np.float64)
    params_array[:, 1] = np.maximum(params_array[:, 1], 1e-6)
    if kernel_type == "triangular":
        params_array[:, 0] = np.clip(params_array[:, 0], 1e-6, params_array[:, 1] - 1e-6)
    else:
        params_array[:, 0] = np.maximum(params_array[:, 0], 1e-6)
    params_array[:, 3] = np.maximum(params_array[:, 3], 0.0)
    params_array[:, 2] = np.maximum(params_array[:, 2], 0.0)
    if params_array.shape[0] != 1:
        raise NotImplementedError(
            "kernel recovery for product_max_witness currently assumes one "
            "fixed-shape component per rule-source kernel, matching the "
            "active benchmark suite"
        )
    params_array[:, 2] = 1.0

    vals = np.zeros_like(knots, dtype=np.float64)
    for peak, width, mix_weight, support_mult in params_array:
        if kernel_type == "gaussian":
            vals += _gaussian_kernel_values(knots, peak, width, mix_weight, support_mult)
        elif kernel_type == "exponential":
            vals += _exponential_kernel_values(knots, peak, width, mix_weight, support_mult)
        else:
            vals += _triangular_kernel_values(knots, peak, width, mix_weight)
    return normalize_piecewise_score(vals)


def _compute_kernel_recovery(
    *,
    config: dict,
    target: int,
    matched: list[tuple[tuple[int, ...], str, int]],
    subsets: list[tuple[int, ...]],
    rule_heights: dict[tuple[int, int], np.ndarray],
    knots: np.ndarray,
) -> dict:
    subset_to_idx = {tuple(int(s) for s in subset): int(idx) for idx, subset in enumerate(subsets)}
    truth_lookup = _truth_rule_lookup(config)
    details: list[dict] = []
    l1_vals: list[float] = []
    l2_vals: list[float] = []
    knots = np.asarray(knots, dtype=np.float64)

    for rule in sorted(matched):
        subset, _sign, _target = rule
        idx = subset_to_idx.get(tuple(int(s) for s in subset))
        truth_rule = truth_lookup.get(rule)
        if idx is None or truth_rule is None:
            continue
        kernel_type = str(truth_rule.get("kernel", config.get("kernel", "triangular")))
        for src in subset:
            src = int(src)
            est_raw = np.asarray(rule_heights[(int(idx), src)], dtype=np.float64)
            est = normalize_piecewise_score(est_raw)
            source_condition = truth_rule["condition"].get(src)
            if source_condition is None:
                source_condition = truth_rule["condition"].get(str(src))
            if source_condition is None:
                raise KeyError(f"missing source {src} condition for truth rule {rule}")
            truth = _truth_kernel_heights(
                kernel_type=kernel_type,
                source_condition=source_condition,
                knots=knots,
            )
            l1 = float(np.trapz(np.abs(est - truth), x=knots))
            l2 = float(math.sqrt(float(np.trapz((est - truth) ** 2, x=knots))))
            l1_vals.append(l1)
            l2_vals.append(l2)
            details.append(
                {
                    "rule": format_rule(rule, target),
                    "source": int(src),
                    "kernel_type": str(kernel_type),
                    "l1": l1,
                    "l2": l2,
                    "knots": [float(x) for x in knots],
                    "estimated": [float(x) for x in est],
                    "truth": [float(x) for x in truth],
                }
            )

    return {
        "num_rule_source_pairs": int(len(details)),
        "mean_l1": float(np.mean(l1_vals)) if l1_vals else 0.0,
        "mean_l2": float(np.mean(l2_vals)) if l2_vals else 0.0,
        "max_l1": float(np.max(l1_vals)) if l1_vals else 0.0,
        "max_l2": float(np.max(l2_vals)) if l2_vals else 0.0,
        "details": details,
    }


def _learned_rule_parameter_details(
    *,
    active_rules: list[rd.ActiveRule],
    subsets: list[tuple[int, ...]],
    target: int,
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    rule_heights: dict[tuple[int, int], np.ndarray],
    knots: np.ndarray,
) -> list[dict]:
    knots = np.asarray(knots, dtype=np.float64)
    details: list[dict] = []
    for rule in rd.sort_unique_sign_exclusive_rules(list(active_rules)):
        subset = tuple(int(src) for src in subsets[int(rule.idx)])
        if str(rule.sign) == "exc":
            beta = float(exc_params.get(int(rule.idx), 0.0))
            sign_name = "excitation"
        else:
            beta = float(inh_params.get(int(rule.idx), 0.0))
            sign_name = "inhibition"
        kernels: list[dict] = []
        for src in subset:
            raw = rule_heights.get((int(rule.idx), int(src)))
            if raw is None:
                continue
            heights = normalize_piecewise_score(np.asarray(raw, dtype=np.float64))
            area = float(np.trapz(heights, x=knots)) if knots.size == heights.size else 0.0
            if area > 1e-12:
                density = heights / float(area)
                mean_lag = float(np.trapz(knots * density, x=knots))
            else:
                density = np.zeros_like(heights)
                mean_lag = None
            peak_lag = float(knots[int(np.argmax(heights))]) if knots.size else None
            kernels.append(
                {
                    "source": int(src),
                    "knots": [float(x) for x in knots],
                    "shape_heights": [float(x) for x in heights],
                    "area_normalized_density": [float(x) for x in density],
                    "area": float(area),
                    "peak_lag": peak_lag,
                    "mean_lag": mean_lag,
                }
            )
        details.append(
            {
                "rule": format_rule((subset, str(rule.sign), int(target)), int(target)),
                "sources": [int(src) for src in subset],
                "sign": sign_name,
                "target": int(target),
                "beta": float(beta),
                "kernel_distribution_by_source": kernels,
            }
        )
    return details


def generate_dataset(config_path: Path, *, dataset_seed: int | None = None) -> Path:
    cfg = _config_with_dataset_seed(load_yaml(config_path), dataset_seed)
    output_path = resolve_repo_path(cfg["path"]["output_path"])
    rules = create_rules_from_config(cfg)
    generation_model = str(cfg.get("synthetic_generation_model", "")).strip()
    if generation_model != "canonical_loglink":
        raise NotImplementedError(
            "The active benchmark suite is canonical-only. "
            f"Config {config_path} requests synthetic_generation_model={generation_model!r}."
        )
    data = generate_canonical_loglink_data(
        rules=rules,
        num_samples=cfg.get("num_samples", 5000),
        time_horizon=cfg.get("time_horizon", 100.0),
        base_intensities=cfg.get("base_intensity", {}),
        max_len=cfg.get("max_len", 1024),
        seed=cfg.get("seed"),
    )
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    rng.shuffle(data)
    n = len(data)
    dataset = {
        "train": data,
        "val": [],
        "test": [],
        "metadata": {
            "num_types": cfg["num_event_types"],
            "config": cfg,
            "split_strategy": "all_sequences_for_synthetic_rule_discovery",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    return output_path


def prepare_config_state(
    config_path: Path,
    device_name: str,
    *,
    regenerate_dataset: bool = True,
    dataset_seed: int | None = None,
) -> dict:
    cfg = _config_with_dataset_seed(load_yaml(config_path), dataset_seed)
    data_path = (
        generate_dataset(config_path, dataset_seed=dataset_seed)
        if regenerate_dataset
        else resolve_repo_path(cfg["path"]["output_path"])
    )

    train, val, metadata = _load_rule_discovery_dataset(data_path)
    dataset_config = metadata.get("config", {})
    if not regenerate_dataset:
        lhs = _normalize_config_for_compare(cfg)
        rhs = _normalize_config_for_compare(dataset_config)
        if lhs != rhs:
            raise ValueError(
                "dataset metadata config does not match the current YAML config for "
                f"{config_path}. Regenerate the dataset before reuse."
            )
    config = dataset_config or cfg
    activation_mode = str(config.get("activation_mode", "product_max_witness")).strip()
    if activation_mode != "product_max_witness":
        raise NotImplementedError(
            "The active-set learner currently supports only activation_mode='product_max_witness'. "
            f"Config {config_path} requests activation_mode={activation_mode!r}. "
            "This guard is intentional to prevent silent synthetic/learner mismatch."
        )
    intensity_model = str(config.get("intensity_model", "canonical_loglink")).strip()
    if intensity_model != "canonical_loglink":
        raise NotImplementedError(
            "The active benchmark suite is canonical-loglink only. "
            f"Config {config_path} requests intensity_model={intensity_model!r}."
        )
    target = int(config["rules"][0]["target"])
    gt = gt_rules_from_config(config)
    num_types = int(metadata["num_types"])
    if train or val:
        observed_max_time = max((max(seq["time"]) for seq in train + val if seq["time"]), default=0.0)
    else:
        observed_max_time = 0.0
    time_horizon = float(config.get("time_horizon", observed_max_time))
    source_ids = tuple(sorted(k for k in range(num_types) if k != target))
    max_lag = _config_max_lag(config)

    train_arrays = build_seq_event_arrays(train, num_types)
    val_arrays = build_seq_event_arrays(val, num_types)
    global_kernels_exc = estimate_source_kernels(
        train_arrays,
        source_ids=source_ids,
        target=target,
        max_lag=float(max_lag),
        num_bins=40,
        num_knots=7,
        time_horizon=float(time_horizon),
        sign="exc",
    )
    global_kernels_inh = estimate_source_kernels(
        train_arrays,
        source_ids=source_ids,
        target=target,
        max_lag=float(max_lag),
        num_bins=40,
        num_knots=7,
        time_horizon=float(time_horizon),
        sign="inh",
    )
    grid_step = auto_grid_step(global_kernels_exc)

    tr_event_seq, tr_event_times = collect_target_events(train, target=target)
    va_event_seq, va_event_times = collect_target_events(val, target=target)
    tr_grid_seq, tr_grid_times, tr_grid_w = build_midpoint_grid(train, time_horizon=float(time_horizon), step=grid_step)
    va_grid_seq, va_grid_times, va_grid_w = build_midpoint_grid(val, time_horizon=float(time_horizon), step=grid_step)

    basis_cache = SourceBasisCache(
        source_ids=source_ids,
        knots=next(iter(global_kernels_exc.values())).knots,
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        train_event_seq_ids=tr_event_seq,
        train_event_times=tr_event_times,
        train_grid_seq_ids=tr_grid_seq,
        train_grid_times=tr_grid_times,
        val_event_seq_ids=va_event_seq,
        val_event_times=va_event_times,
        val_grid_seq_ids=va_grid_seq,
        val_grid_times=va_grid_times,
    )
    subsets, source_components, source_presence_masks = feasible_subset_list(
        source_ids=source_ids,
        max_order=int(FINAL_MAX_RULE_ORDER),
        basis_cache=basis_cache,
    )
    rule_heights_exc = rd.initialize_rule_specific_heights(
        subsets=subsets,
        source_ids=source_ids,
        global_kernels=global_kernels_exc,
    )
    rule_heights_inh = rd.initialize_rule_specific_heights(
        subsets=subsets,
        source_ids=source_ids,
        global_kernels=global_kernels_inh,
    )

    device = resolve_compute_device(device_name)
    torch_basis_cache = rd.TorchBasisCache(basis_cache, device)
    return {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "config": config,
        "target": int(target),
        "gt": gt,
        "subsets": subsets,
        "rule_heights": rule_heights_exc,
        "template_rule_heights": {
            "exc": rule_heights_exc,
            "inh": rule_heights_inh,
        },
        "basis_cache": basis_cache,
        "grid_weights_train": np.asarray(tr_grid_w, dtype=np.float64),
        "grid_weights_val": np.asarray(va_grid_w, dtype=np.float64),
        "grid_step": float(grid_step),
        "time_horizon": float(time_horizon),
        "device": device,
        "torch_basis_cache": torch_basis_cache,
        "num_val_sequences": int(len(val)),
        "knots": np.asarray(next(iter(global_kernels_exc.values())).knots, dtype=np.float64),
        "source_components": source_components,
        "source_presence_masks": source_presence_masks,
    }


def evaluate_prepared_state(
    prepared: dict,
) -> dict:
    t_start = time.perf_counter()
    penalty_scale = 1.0
    kernel_smoothness_ridge = FINAL_KERNEL_SMOOTHNESS_RIDGE
    config = prepared["config"]
    target = int(prepared["target"])
    gt = prepared["gt"]
    subsets = prepared["subsets"]
    source_ids = tuple(sorted(int(k) for k in range(int(config["num_event_types"])) if int(k) != int(target)))
    _set_final_support_score_context(
        subsets=subsets,
        source_ids=source_ids,
        num_knots=int(len(prepared["knots"])),
    )
    device = prepared["device"]
    rule_heights = {
        key: np.asarray(val, dtype=np.float64).copy()
        for key, val in prepared["rule_heights"].items()
    }
    template_rule_heights = {
        str(sign): {
            key: np.asarray(val, dtype=np.float64).copy()
            for key, val in height_map.items()
        }
        for sign, height_map in prepared["template_rule_heights"].items()
    }
    final_support_cache: dict[rd.ExactSupportCacheKey, rd.SupportEvalResult] = {}
    final_selection_ctx = _build_full_likelihood_selection_context(prepared)
    final_basis_cache = final_selection_ctx["basis_cache"]
    final_torch_basis_cache = final_selection_ctx["torch_basis_cache"]
    final_grid_weights_train = final_selection_ctx["grid_weights"]
    final_grid_weights_val = final_selection_ctx["grid_weights"]
    final_num_sequences = int(final_selection_ctx["num_sequences"])
    _set_final_support_score_context(
        subsets=subsets,
        source_ids=source_ids,
        num_knots=int(len(prepared["knots"])),
    )
    final_stage_timings: dict[str, float] = {}
    final_stage_counts: dict[str, int] = {}
    signed_universe_size = 2 * int(len(subsets))
    final_stage_counts["final_full_likelihood_selection"] = 1
    final_stage_counts["final_selection_sequences"] = int(final_num_sequences)
    final_stage_counts["final_signed_universe_size"] = int(signed_universe_size)
    final_stage_counts["final_support_cache_size_start"] = int(len(final_support_cache))

    _progress("sieve_feature_precompute:start")
    t_sieve_precompute = time.perf_counter()
    sieve_evaluator = _SieveSupportEvaluator(
        prepared=prepared,
        basis_cache=final_basis_cache,
        torch_basis_cache=final_torch_basis_cache,
        grid_weights=final_grid_weights_train,
        source_ids=source_ids,
        num_sequences=final_num_sequences,
        signed_universe_size=int(signed_universe_size),
        steps=int(FINAL_SIEVE_STEPS),
        batch_size=int(FINAL_SIEVE_BATCH_SIZE),
    )
    final_stage_timings["sieve_feature_precompute"] = float(time.perf_counter() - t_sieve_precompute)
    _progress(f"sieve_feature_precompute:end sec={final_stage_timings['sieve_feature_precompute']:.3f}")

    null_key = rd.support_key_from_rules([])
    final_result = sieve_evaluator.evaluate_many([[]])[null_key]

    _progress("sieve_priced_column_generation:start rules=0")
    t_sieve_search = time.perf_counter()
    sieve_best, sieve_stats = _sieve_priced_column_generation(
        start_result=final_result,
        subsets=subsets,
        source_ids=source_ids,
        signed_universe_size=int(signed_universe_size),
        evaluator=sieve_evaluator,
    )
    final_stage_timings["sieve_priced_column_generation"] = float(time.perf_counter() - t_sieve_search)
    for key, value in sieve_stats.items():
        final_stage_counts[f"sieve_priced_column_generation_{key}"] = int(value)
    _progress(
        f"sieve_priced_column_generation:end "
        f"sec={final_stage_timings['sieve_priced_column_generation']:.3f} "
        f"passes={sieve_stats.get('passes', 0)} "
        f"accepted={sieve_stats.get('accepted_blocks', 0)} "
        f"pricing={sieve_stats.get('pricing_evals', 0)} "
        f"states={sieve_stats.get('block_state_evals', 0)}"
    )

    _progress(f"final_free_kernel_polish:start rules={len(sieve_best.active_rules)}")
    t_final_polish = time.perf_counter()
    final_result = rd.evaluate_support_exact(
        active_rules=list(sieve_best.active_rules),
        subsets=subsets,
        basis_cache=final_basis_cache,
        base_rule_heights=rule_heights,
        template_rule_heights=template_rule_heights,
        grid_weights_train=final_grid_weights_train,
        grid_weights_val=final_grid_weights_val,
        device=device,
        torch_basis_cache=final_torch_basis_cache,
        opt_steps=60,
        lr=0.05,
        penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
        penalty_scale=float(penalty_scale),
        num_val_sequences=final_num_sequences,
        kernel_smoothness_ridge=float(kernel_smoothness_ridge),
        support_cache=final_support_cache,
        warm_start_result=sieve_best,
    )
    final_stage_timings["final_free_kernel_polish"] = float(time.perf_counter() - t_final_polish)
    _progress(f"final_free_kernel_polish:end sec={final_stage_timings['final_free_kernel_polish']:.3f}")

    total_rescue_time = 0.0
    total_drop_time = 0.0
    total_rescue_stats = {
        "passes": 0,
        "pricing_evals": 0,
        "pricing_violations": 0,
        "fallback_exact_evals": 0,
        "fallback_accepts": 0,
        "accepted_adds": 0,
    }
    total_drop_stats = {"evals": 0, "accepts": 0}
    total_swap_time = 0.0
    total_swap_stats = {
        "drop_base_evals": 0,
        "pricing_evals": 0,
        "pricing_violations": 0,
        "swap_evals": 0,
        "bound_pruned": 0,
        "accepts": 0,
    }
    exact_local_closure_rounds = 0
    while True:
        exact_local_closure_rounds += 1
        _progress(
            f"exact_pricing_rescue:start "
            f"round={exact_local_closure_rounds} rules={len(final_result.active_rules)}"
        )
        t_exact_rescue = time.perf_counter()
        final_result, rescue_stats = _exact_pricing_rescue(
            current_result=final_result,
            evaluator=sieve_evaluator,
            subsets=subsets,
            source_ids=source_ids,
            basis_cache=final_basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights=final_grid_weights_train,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            num_sequences=final_num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=final_support_cache,
        )
        total_rescue_time += float(time.perf_counter() - t_exact_rescue)
        for key, value in rescue_stats.items():
            total_rescue_stats[key] = int(total_rescue_stats.get(key, 0)) + int(value)
        _progress(
            f"exact_pricing_rescue:end "
            f"round={exact_local_closure_rounds} "
            f"sec={total_rescue_time:.3f} "
            f"passes={rescue_stats.get('passes', 0)} "
            f"accepted={rescue_stats.get('accepted_adds', 0)}"
        )

        _progress(
            f"final_exact_drop_certificate:start "
            f"round={exact_local_closure_rounds} rules={len(final_result.active_rules)}"
        )
        t_final_drop = time.perf_counter()
        final_result, drop_stats, drop_results_for_swap = _exact_drop_certificate(
            current_result=final_result,
            subsets=subsets,
            basis_cache=final_basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights_train=final_grid_weights_train,
            grid_weights_val=final_grid_weights_val,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            num_sequences=final_num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=final_support_cache,
            signed_universe_size=int(signed_universe_size),
        )
        total_drop_time += float(time.perf_counter() - t_final_drop)
        total_drop_stats["evals"] += int(drop_stats.get("evals", 0))
        total_drop_stats["accepts"] += int(drop_stats.get("accepts", 0))
        _progress(
            f"final_exact_drop_certificate:end "
            f"round={exact_local_closure_rounds} "
            f"sec={total_drop_time:.3f} "
            f"accepts={drop_stats.get('accepts', 0)} "
            f"evals={drop_stats.get('evals', 0)}"
        )

        _progress(
            f"final_exact_swap_certificate:start "
            f"round={exact_local_closure_rounds} rules={len(final_result.active_rules)}"
        )
        t_final_swap = time.perf_counter()
        final_result, swap_stats = _exact_swap_certificate(
            current_result=final_result,
            evaluator=sieve_evaluator,
            subsets=subsets,
            source_ids=source_ids,
            basis_cache=final_basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights=final_grid_weights_train,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            num_sequences=final_num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=final_support_cache,
            signed_universe_size=int(signed_universe_size),
            precomputed_drop_results=drop_results_for_swap,
        )
        total_swap_time += float(time.perf_counter() - t_final_swap)
        for key, value in swap_stats.items():
            total_swap_stats[key] = int(total_swap_stats.get(key, 0)) + int(value)
        _progress(
            f"final_exact_swap_certificate:end "
            f"round={exact_local_closure_rounds} "
            f"sec={total_swap_time:.3f} "
            f"accepts={swap_stats.get('accepts', 0)} "
            f"evals={swap_stats.get('swap_evals', 0)} "
            f"drop_reused={swap_stats.get('drop_base_reused', 0)}"
        )

        if int(drop_stats.get("accepts", 0)) == 0 and int(swap_stats.get("accepts", 0)) == 0:
            break

    final_stage_timings["exact_pricing_rescue"] = float(total_rescue_time)
    final_stage_timings["final_exact_drop_certificate"] = float(total_drop_time)
    final_stage_timings["final_exact_swap_certificate"] = float(total_swap_time)
    final_stage_counts["exact_local_closure_rounds"] = int(exact_local_closure_rounds)
    for key, value in total_rescue_stats.items():
        final_stage_counts[f"exact_pricing_rescue_{key}"] = int(value)
    final_stage_counts["final_exact_drop_certificate_evals"] = int(total_drop_stats["evals"])
    final_stage_counts["final_exact_drop_certificate_accepts"] = int(total_drop_stats["accepts"])
    for key, value in total_swap_stats.items():
        final_stage_counts[f"final_exact_swap_certificate_{key}"] = int(value)
    final_stage_counts["pricing_atom_cache_entries"] = int(len(sieve_evaluator._pricing_atom_cache))
    final_stage_counts["pricing_atom_cache_bytes"] = int(sieve_evaluator._pricing_atom_cache_bytes)
    final_stage_counts["pricing_atom_cache_budget_bytes"] = int(sieve_evaluator._pricing_atom_cache_max_bytes)

    if final_result.active_rules and final_result.arrays_out is None:
        _progress(f"final_exact_refit_for_certificate:start rules={len(final_result.active_rules)}")
        t_certificate_refit = time.perf_counter()
        final_result = rd.evaluate_support_exact(
            active_rules=list(final_result.active_rules),
            subsets=subsets,
            basis_cache=final_basis_cache,
            base_rule_heights=final_result.rule_heights,
            template_rule_heights=template_rule_heights,
            grid_weights_train=final_grid_weights_train,
            grid_weights_val=final_grid_weights_val,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            opt_steps=60,
            lr=0.05,
            penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
            penalty_scale=float(penalty_scale),
            num_val_sequences=final_num_sequences,
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=None,
            warm_start_result=final_result,
        )
        final_stage_timings["final_exact_refit_for_certificate"] = float(
            time.perf_counter() - t_certificate_refit
        )
        _progress(
            f"final_exact_refit_for_certificate:end "
            f"sec={final_stage_timings['final_exact_refit_for_certificate']:.3f}"
        )

    bic = float(final_result.bic)
    mu = float(final_result.mu)
    exc_params = dict(final_result.exc_params)
    inh_params = dict(final_result.inh_params)
    rule_heights = final_result.rule_heights
    active_rules = rd.threshold_active_rules(
        active_rules=list(final_result.active_rules),
        exc_params=exc_params,
        inh_params=inh_params,
        beta_threshold=float(FINAL_SUPPORT_BETA_THRESHOLD),
    )
    if len(active_rules) < len(final_result.active_rules):
        _progress(f"final_exact_refit_after_threshold:start rules={len(active_rules)}")
        t_threshold_refit = time.perf_counter()
        final_result = rd.evaluate_support_exact(
            active_rules=active_rules,
            subsets=subsets,
            basis_cache=final_basis_cache,
            base_rule_heights=rule_heights,
            template_rule_heights=template_rule_heights,
            grid_weights_train=final_grid_weights_train,
            grid_weights_val=final_grid_weights_val,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            opt_steps=60,
            lr=0.05,
            penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
            penalty_scale=float(penalty_scale),
            num_val_sequences=final_num_sequences,
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=None,
            warm_start_result=final_result,
        )
        final_stage_timings["final_exact_refit_after_threshold"] = float(
            time.perf_counter() - t_threshold_refit
        )
        _progress(
            f"final_exact_refit_after_threshold:end "
            f"sec={final_stage_timings['final_exact_refit_after_threshold']:.3f}"
        )
        bic = float(final_result.bic)
        mu = float(final_result.mu)
        exc_params = dict(final_result.exc_params)
        inh_params = dict(final_result.inh_params)
        rule_heights = final_result.rule_heights
        active_rules = list(final_result.active_rules)

    conditional_rows: list[dict] = []
    total_conditional_stats = {
        "passes": 0,
        "drop_evals": 0,
        "rejects": 0,
        "kept_by_amplitude": 0,
    }
    final_stage_timings["conditional_necessity_certificate"] = 0.0
    admissibility_projection_rounds = 0
    while True:
        _progress(
            f"conditional_necessity_certificate:start "
            f"projection_round={admissibility_projection_rounds} "
            f"rules={len(final_result.active_rules)}"
        )
        t_conditional_certificate = time.perf_counter()
        final_result, conditional_stats, rows = _conditional_necessity_certificate(
            current_result=final_result,
            subsets=subsets,
            basis_cache=final_basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights=final_grid_weights_val,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            num_sequences=final_num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            signed_universe_size=int(signed_universe_size),
        )
        final_stage_timings["conditional_necessity_certificate"] += float(
            time.perf_counter() - t_conditional_certificate
        )
        conditional_rows.extend(rows)
        for key, value in conditional_stats.items():
            total_conditional_stats[key] = int(total_conditional_stats.get(key, 0)) + int(value)
        _progress(
            f"conditional_necessity_certificate:end "
            f"projection_round={admissibility_projection_rounds} "
            f"sec={final_stage_timings['conditional_necessity_certificate']:.3f} "
            f"rejects={conditional_stats.get('rejects', 0)} "
            f"evals={conditional_stats.get('drop_evals', 0)}"
        )
        if int(conditional_stats.get("rejects", 0)) == 0:
            break

        admissibility_projection_rounds += 1
        final_result, projection_timings, projection_counts = _run_exact_local_closure(
            current_result=final_result,
            evaluator=sieve_evaluator,
            subsets=subsets,
            source_ids=source_ids,
            basis_cache=final_basis_cache,
            template_rule_heights=template_rule_heights,
            grid_weights_train=final_grid_weights_train,
            grid_weights_val=final_grid_weights_val,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            num_sequences=final_num_sequences,
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=final_support_cache,
            signed_universe_size=int(signed_universe_size),
            reason=f"admissibility_projection_{admissibility_projection_rounds}",
        )
        for key, value in projection_timings.items():
            final_stage_timings[key] = float(final_stage_timings.get(key, 0.0)) + float(value)
        for key, value in projection_counts.items():
            final_stage_counts[key] = int(final_stage_counts.get(key, 0)) + int(value)

        exc_params = dict(final_result.exc_params)
        inh_params = dict(final_result.inh_params)
        rule_heights = final_result.rule_heights
        active_rules = rd.threshold_active_rules(
            active_rules=list(final_result.active_rules),
            exc_params=exc_params,
            inh_params=inh_params,
            beta_threshold=float(FINAL_SUPPORT_BETA_THRESHOLD),
        )
        if len(active_rules) < len(final_result.active_rules):
            _progress(
                f"final_exact_refit_after_projection_threshold:start "
                f"projection_round={admissibility_projection_rounds} rules={len(active_rules)}"
            )
            t_projection_threshold_refit = time.perf_counter()
            final_result = rd.evaluate_support_exact(
                active_rules=active_rules,
                subsets=subsets,
                basis_cache=final_basis_cache,
                base_rule_heights=rule_heights,
                template_rule_heights=template_rule_heights,
                grid_weights_train=final_grid_weights_train,
                grid_weights_val=final_grid_weights_val,
                device=device,
                torch_basis_cache=final_torch_basis_cache,
                opt_steps=60,
                lr=0.05,
                penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
                penalty_scale=float(penalty_scale),
                num_val_sequences=final_num_sequences,
                kernel_smoothness_ridge=float(kernel_smoothness_ridge),
                support_cache=None,
                warm_start_result=final_result,
            )
            final_stage_timings["final_exact_refit_after_projection_threshold"] = (
                float(final_stage_timings.get("final_exact_refit_after_projection_threshold", 0.0))
                + float(time.perf_counter() - t_projection_threshold_refit)
            )
            _progress(
                f"final_exact_refit_after_projection_threshold:end "
                f"sec={final_stage_timings['final_exact_refit_after_projection_threshold']:.3f}"
            )

    for key, value in total_conditional_stats.items():
        final_stage_counts[f"conditional_necessity_certificate_{key}"] = int(value)
    final_stage_counts["admissibility_projection_rounds"] = int(admissibility_projection_rounds)

    bic = float(final_result.bic)
    mu = float(final_result.mu)
    exc_params = dict(final_result.exc_params)
    inh_params = dict(final_result.inh_params)
    rule_heights = final_result.rule_heights
    active_rules = rd.threshold_active_rules(
        active_rules=list(final_result.active_rules),
        exc_params=exc_params,
        inh_params=inh_params,
        beta_threshold=float(FINAL_SUPPORT_BETA_THRESHOLD),
    )
    if len(active_rules) < len(final_result.active_rules):
        _progress(f"final_exact_refit_after_admissibility_threshold:start rules={len(active_rules)}")
        t_admissibility_threshold_refit = time.perf_counter()
        final_result = rd.evaluate_support_exact(
            active_rules=active_rules,
            subsets=subsets,
            basis_cache=final_basis_cache,
            base_rule_heights=rule_heights,
            template_rule_heights=template_rule_heights,
            grid_weights_train=final_grid_weights_train,
            grid_weights_val=final_grid_weights_val,
            device=device,
            torch_basis_cache=final_torch_basis_cache,
            opt_steps=60,
            lr=0.05,
            penalize_kernel_df=bool(FINAL_REPORT_KERNEL_DF),
            penalty_scale=float(penalty_scale),
            num_val_sequences=final_num_sequences,
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=None,
            warm_start_result=final_result,
        )
        final_stage_timings["final_exact_refit_after_admissibility_threshold"] = float(
            time.perf_counter() - t_admissibility_threshold_refit
        )
        _progress(
            f"final_exact_refit_after_admissibility_threshold:end "
            f"sec={final_stage_timings['final_exact_refit_after_admissibility_threshold']:.3f}"
        )
        bic = float(final_result.bic)
        mu = float(final_result.mu)
        exc_params = dict(final_result.exc_params)
        inh_params = dict(final_result.inh_params)
        rule_heights = final_result.rule_heights

    preds = summarize_results(
        subsets=subsets,
        exc_params=exc_params,
        inh_params=inh_params,
        target=target,
        beta_threshold=float(FINAL_SUPPORT_BETA_THRESHOLD),
    )
    matched = sorted(gt & preds)
    missing = sorted(gt - preds)
    extra = sorted(preds - gt)
    kernel_recovery = _compute_kernel_recovery(
        config=config,
        target=target,
        matched=matched,
        subsets=subsets,
        rule_heights=rule_heights,
        knots=np.asarray(prepared["knots"], dtype=np.float64),
    )
    learned_rule_parameter_details = _learned_rule_parameter_details(
        active_rules=list(final_result.active_rules),
        subsets=subsets,
        target=int(target),
        exc_params=exc_params,
        inh_params=inh_params,
        rule_heights=rule_heights,
        knots=np.asarray(prepared["knots"], dtype=np.float64),
    )
    final_support_score = float(_final_support_score(final_result, signed_universe_size=int(signed_universe_size)))
    return {
        "benchmark": str(Path(str(prepared["config_path"])).stem).replace(".yaml", ""),
        "data_path": str(prepared["data_path"]),
        "config_path": str(prepared["config_path"]),
        "target": int(target),
        "device": str(device),
        "algorithm_profile": "constrained_block_mdl_amplitude_necessity_rdk_tpp",
        "intensity_model": "canonical_loglink",
        "bic": float(bic),
        "elapsed_sec": float(time.perf_counter() - t_start),
        "final_kernel_df": bool(FINAL_REPORT_KERNEL_DF),
        "final_full_likelihood_selection": True,
        "final_support_criterion": "constrained_block_mdl_conditional_amplitude_necessity",
        "final_support_score": float(final_support_score),
        "final_selection_sequences": int(final_num_sequences),
        "conditional_necessity_certificate": {
            "type": "finite_family_one_sided_amplitude",
            "delta_n": float(rd.bic_sample_size(int(final_num_sequences)) ** -2),
            "signed_universe_size": int(signed_universe_size),
            "rows": conditional_rows,
        },
        "mu": float(mu),
        "learned_rule_parameter_details": learned_rule_parameter_details,
        "beta_threshold": float(FINAL_SUPPORT_BETA_THRESHOLD),
        "kernel_smoothness_ridge": float(kernel_smoothness_ridge),
        "runtime_profile": {
            "final_stage_timings_sec": final_stage_timings,
            "final_stage_counts": final_stage_counts,
        },
        "subset_component_count": int(len(prepared.get("source_components", ()))),
        "subset_count": int(len(subsets)),
        "selected_rule_count": int(len(preds)),
        "true_rule_count": int(len(gt)),
        "matched": [format_rule(r, target) for r in matched],
        "missing": [format_rule(r, target) for r in missing],
        "extra": [format_rule(r, target) for r in extra],
        "predicted": [format_rule(r, target) for r in sorted(preds)],
        "recall": float(len(matched)) / max(len(gt), 1),
        "precision": float(len(matched)) / max(len(preds), 1),
        "kernel_recovery": kernel_recovery,
    }


def evaluate_config(
    config_path: Path,
    device_name: str,
    *,
    regenerate_dataset: bool = True,
    dataset_seed: int | None = None,
) -> dict:
    prepared = prepare_config_state(
        config_path,
        device_name,
        regenerate_dataset=bool(regenerate_dataset),
        dataset_seed=dataset_seed,
    )
    return evaluate_prepared_state(prepared)


def parse_name_filter(spec: str) -> set[str]:
    return {part.strip() for part in str(spec).split(",") if part.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cpu_threads", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--reuse_dataset", action="store_true")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--result_path", default="")
    args = ap.parse_args()

    cpu_threads = configure_runtime_resources(None if int(args.cpu_threads) <= 0 else int(args.cpu_threads))
    requested = parse_name_filter(args.only)
    benchmarks = BENCHMARKS
    if requested:
        benchmarks = [item for item in BENCHMARKS if item["name"] in requested]
        missing = sorted(requested - {item["name"] for item in benchmarks})
        if missing:
            raise ValueError(f"unknown benchmark names: {missing}")

    print(
        json.dumps(
            {
                "cpu_threads": int(cpu_threads),
                "default_cpu_threads": int(default_cpu_threads()),
                "device": str(args.device),
                "reuse_dataset": bool(args.reuse_dataset),
                "dataset_seed": int(args.dataset_seed) if int(args.dataset_seed) > 0 else None,
                "benchmarks": [item["name"] for item in benchmarks],
            },
            indent=2,
        ),
        flush=True,
    )

    out = {}
    results_path = RESULTS_PATH if not str(args.result_path).strip() else resolve_repo_path(str(args.result_path).strip())
    results_path.parent.mkdir(parents=True, exist_ok=True)
    for item in benchmarks:
        name = item["name"]
        print("RUNNING", name, flush=True)
        out[name] = evaluate_config(
            item["config"],
            str(args.device),
            regenerate_dataset=not bool(args.reuse_dataset),
            dataset_seed=int(args.dataset_seed) if int(args.dataset_seed) > 0 else None,
        )
        print(json.dumps({name: out[name]}, indent=2), flush=True)
        results_path.write_text(json.dumps(out, indent=2))
    print("WROTE", results_path)


if __name__ == "__main__":
    main()
