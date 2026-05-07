"""Rule-dependent kernel support refit utilities for the paper runner."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from conjunctive_rule_initializer import (
    SourceBasisCache,
    WitnessQueryData,
    base_rate_fit,
    normalize_piecewise_score,
)


@dataclass(frozen=True)
class ActiveRule:
    idx: int
    sign: str


@dataclass
class SupportEvalResult:
    bic: float
    mu: float
    exc_params: dict[int, float]
    inh_params: dict[int, float]
    rule_heights: dict[tuple[int, int], np.ndarray]
    arrays_out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None
    active_rules: list[ActiveRule]


SupportKey = tuple[tuple[int, str], ...]
ExactSupportCacheKey = tuple[object, ...]


@dataclass(frozen=True)
class TorchWitnessQueryData:
    num_queries: int
    query_index: torch.Tensor
    left_index: torch.Tensor
    right_index: torch.Tensor
    left_weight: torch.Tensor
    right_weight: torch.Tensor


class TorchBasisCache:
    def __init__(self, basis_cache: SourceBasisCache, device: torch.device):
        self._basis_cache = basis_cache
        self._device = device
        self._cache: dict[tuple[int, str], TorchWitnessQueryData] = {}
        self._combo_cache: dict[tuple[int, tuple[str, ...]], tuple[TorchWitnessQueryData, tuple[int, ...]]] = {}

    def _to_torch(self, data: WitnessQueryData) -> TorchWitnessQueryData:
        return TorchWitnessQueryData(
            num_queries=int(data.num_queries),
            query_index=torch.as_tensor(data.query_index, dtype=torch.int64, device=self._device),
            left_index=torch.as_tensor(data.left_index, dtype=torch.int64, device=self._device),
            right_index=torch.as_tensor(data.right_index, dtype=torch.int64, device=self._device),
            left_weight=torch.as_tensor(data.left_weight, dtype=torch.float32, device=self._device),
            right_weight=torch.as_tensor(data.right_weight, dtype=torch.float32, device=self._device),
        )

    def arrays_for_rule_source(
        self,
        rule_idx: int | None,
        src: int,
    ) -> tuple[TorchWitnessQueryData, TorchWitnessQueryData, TorchWitnessQueryData, TorchWitnessQueryData]:
        src = int(src)
        rule_key = None if rule_idx is None else int(rule_idx)
        keys = [
            (rule_key, src, "tr_ev"),
            (rule_key, src, "tr_gr"),
            (rule_key, src, "va_ev"),
            (rule_key, src, "va_gr"),
        ]
        missing = [key for key in keys if key not in self._cache]
        if missing:
            b_tr_ev, b_tr_gr, b_va_ev, b_va_gr = self._basis_cache.arrays_for_rule_source(rule_key, src)
            self._cache[(rule_key, src, "tr_ev")] = self._to_torch(b_tr_ev)
            self._cache[(rule_key, src, "tr_gr")] = self._to_torch(b_tr_gr)
            self._cache[(rule_key, src, "va_ev")] = self._to_torch(b_va_ev)
            self._cache[(rule_key, src, "va_gr")] = self._to_torch(b_va_gr)
        return (
            self._cache[(rule_key, src, "tr_ev")],
            self._cache[(rule_key, src, "tr_gr")],
            self._cache[(rule_key, src, "va_ev")],
            self._cache[(rule_key, src, "va_gr")],
        )

    def arrays(self, src: int) -> tuple[TorchWitnessQueryData, TorchWitnessQueryData, TorchWitnessQueryData, TorchWitnessQueryData]:
        return self.arrays_for_rule_source(None, int(src))

    def combined_arrays(
        self,
        src: int,
        parts: tuple[str, ...],
        rule_idx: int | None = None,
    ) -> tuple[TorchWitnessQueryData, tuple[int, ...]]:
        src = int(src)
        parts = tuple(str(part) for part in parts)
        rule_key = None if rule_idx is None else int(rule_idx)
        key = (rule_key, src, parts)
        cached = self._combo_cache.get(key)
        if cached is not None:
            return cached

        tr_ev, tr_gr, va_ev, va_gr = self.arrays_for_rule_source(rule_key, src)
        lookup = {
            "tr_ev": tr_ev,
            "tr_gr": tr_gr,
            "va_ev": va_ev,
            "va_gr": va_gr,
        }
        arrays = [lookup[part] for part in parts]
        sizes = tuple(int(arr.num_queries) for arr in arrays)
        offsets = np.cumsum((0,) + sizes[:-1], dtype=np.int64)
        query_index = torch.cat(
            [arr.query_index + int(offset) for arr, offset in zip(arrays, offsets)],
            dim=0,
        )
        combined = TorchWitnessQueryData(
            num_queries=int(sum(sizes)),
            query_index=query_index,
            left_index=torch.cat([arr.left_index for arr in arrays], dim=0),
            right_index=torch.cat([arr.right_index for arr in arrays], dim=0),
            left_weight=torch.cat([arr.left_weight for arr in arrays], dim=0),
            right_weight=torch.cat([arr.right_weight for arr in arrays], dim=0),
        )
        out = (combined, sizes)
        self._combo_cache[key] = out
        return out


def normalize_score_heights_torch(raw_heights: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(raw_heights, min=0.0)
    peak = torch.amax(x)
    if bool((peak > 1e-8).item()):
        return x / torch.clamp(peak, min=1e-8)
    out = torch.zeros_like(x)
    if int(out.numel()) > 0:
        out[0] = 1.0
    return out


def normalize_area_heights_torch(raw_heights: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(raw_heights, min=0.0)
    if int(x.numel()) <= 1:
        return torch.ones_like(x)
    area = torch.trapz(x, knots)
    if bool((area > 1e-8).item()):
        return x / torch.clamp(area, min=1e-8)
    span = torch.clamp(knots[-1] - knots[0], min=1e-8)
    return torch.ones_like(x) / span


def normalize_kernel_heights_torch(
    raw_heights: torch.Tensor,
    *,
    knots: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    mode = str(mode)
    if mode == "peak":
        return normalize_score_heights_torch(raw_heights)
    if mode == "area":
        if knots is None:
            raise ValueError("area-normalized kernels require knot locations")
        return normalize_area_heights_torch(raw_heights, knots)
    raise ValueError(f"unknown kernel normalization mode: {mode}")


def ensure_feasible_score_heights_numpy(raw_heights: np.ndarray) -> np.ndarray:
    x = normalize_piecewise_score(np.asarray(raw_heights, dtype=np.float64))
    if x.size > 0 and float(np.max(x)) <= 1e-12:
        x = np.zeros_like(x, dtype=np.float64)
        x[0] = 1.0
    return x


def second_difference_smoothness_torch(heights: torch.Tensor) -> torch.Tensor:
    if int(heights.numel()) < 3:
        return torch.zeros((), dtype=torch.float32, device=heights.device)
    d2 = heights[2:] - 2.0 * heights[1:-1] + heights[:-2]
    return torch.sum(d2 * d2)


def witness_response_torch_batched(
    data: TorchWitnessQueryData,
    heights_batch: torch.Tensor,
    *,
    clip_upper: bool = True,
) -> torch.Tensor:
    batch_size = int(heights_batch.shape[0])
    num_queries = int(data.num_queries)
    out = torch.zeros((batch_size, num_queries), dtype=torch.float32, device=heights_batch.device)
    if batch_size == 0 or int(data.query_index.numel()) == 0:
        return out
    vals = (
        data.left_weight.unsqueeze(0) * heights_batch[:, data.left_index]
        + data.right_weight.unsqueeze(0) * heights_batch[:, data.right_index]
    )
    vals = torch.clamp(vals, min=0.0)
    if bool(clip_upper):
        vals = torch.clamp(vals, max=1.0)
    flat_out = torch.zeros((batch_size * num_queries,), dtype=torch.float32, device=heights_batch.device)
    offsets = torch.arange(batch_size, device=heights_batch.device, dtype=torch.int64).unsqueeze(1) * int(num_queries)
    flat_index = (data.query_index.unsqueeze(0) + offsets).reshape(-1)
    flat_vals = vals.reshape(-1)
    flat_out.scatter_reduce_(0, flat_index, flat_vals, reduce="amax", include_self=True)
    return flat_out.view(batch_size, num_queries)


def kernel_param_key(
    *,
    idx: int,
    sign: str,
    src: int,
    subsets,
):
    return ("rule_src", int(idx), int(src))


def kernel_group_count(
    active_rules: list[ActiveRule],
    subsets,
) -> int:
    groups = {
        kernel_param_key(
            idx=int(ar.idx),
            sign=str(ar.sign),
            src=int(src),
            subsets=subsets,
        )
        for ar in active_rules
        for src in subsets[int(ar.idx)]
    }
    return int(len(groups))


def model_param_dim(
    active_rules: list[ActiveRule],
    subsets,
    num_knots: int,
) -> float:
    _ = int(num_knots)
    # Count each selected rule-source kernel block once.
    kernel_df = 1.0
    return float(1 + len(active_rules) + kernel_group_count(active_rules, subsets) * float(kernel_df))


def bic_sample_size(num_sequences: int) -> int:
    # BIC is scaled by independent trajectories, not quadrature points.
    return max(int(num_sequences), 2)


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 1e-8)
    out = np.empty_like(x, dtype=np.float64)
    mask = x > 20.0
    out[mask] = x[mask]
    out[~mask] = np.log(np.expm1(x[~mask]))
    return out


def initialize_rule_specific_heights(
    *,
    subsets,
    source_ids,
    global_kernels,
):
    out: dict[tuple[int, int], np.ndarray] = {}
    for idx, subset in enumerate(subsets):
        subset = tuple(int(s) for s in subset)
        for src in subset:
            out[(int(idx), int(src))] = np.asarray(global_kernels[int(src)].heights, dtype=np.float64).copy()
    return out


def support_key_from_rules(active_rules: list[ActiveRule]) -> SupportKey:
    return tuple((int(ar.idx), str(ar.sign)) for ar in active_rules)


def exact_support_cache_key(
    *,
    active_rules: list[ActiveRule],
    basis_cache: SourceBasisCache,
    base_rule_heights: dict[tuple[int, int], np.ndarray],
    template_rule_heights: dict[str, dict[tuple[int, int], np.ndarray]] | None,
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    opt_steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float,
    kernel_smoothness_ridge: float,
    num_val_sequences: int | None,
    kernel_normalization: str = "peak",
) -> ExactSupportCacheKey:
    return (
        support_key_from_rules(active_rules),
        int(opt_steps),
        float(lr),
        bool(penalize_kernel_df),
        float(penalty_scale),
        float(kernel_smoothness_ridge),
        str(kernel_normalization),
        int(num_val_sequences if num_val_sequences is not None else -1),
        id(basis_cache),
        id(base_rule_heights),
        id(template_rule_heights),
        id(grid_weights_train),
        id(grid_weights_val),
    )

def minimal_cache_result(result: SupportEvalResult) -> SupportEvalResult:
    return SupportEvalResult(
        bic=float(result.bic),
        mu=float(result.mu),
        exc_params=dict(result.exc_params),
        inh_params=dict(result.inh_params),
        rule_heights=result.rule_heights,
        arrays_out=None,
        active_rules=list(result.active_rules),
    )


def _build_support_warm_start(
    *,
    target_rules: list[ActiveRule],
    subsets,
    base_rule_heights: dict[tuple[int, int], np.ndarray],
    template_rule_heights: dict[str, dict[tuple[int, int], np.ndarray]] | None,
    warm_start_result: SupportEvalResult | None = None,
    fallback: float = 0.1,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, str], float]]:
    init_rule_heights = dict(base_rule_heights)
    init_coef_map = {
        (int(ar.idx), str(ar.sign)): float(fallback)
        for ar in target_rules
    }
    warm_rule_keys = set()
    if warm_start_result is not None:
        warm_rule_keys = {(int(ar.idx), str(ar.sign)) for ar in warm_start_result.active_rules}
    for ar in target_rules:
        rule_sources = tuple(int(src) for src in subsets[int(ar.idx)])
        rule_key = (int(ar.idx), str(ar.sign))
        if warm_start_result is not None and rule_key in warm_rule_keys:
            beta = (
                float(warm_start_result.exc_params.get(int(ar.idx), 0.0))
                if str(ar.sign) == "exc"
                else float(warm_start_result.inh_params.get(int(ar.idx), 0.0))
            )
            init_coef_map[rule_key] = max(beta, 1e-6)
            for src in rule_sources:
                key = (int(ar.idx), int(src))
                warm_h = warm_start_result.rule_heights.get(key)
                if warm_h is not None:
                    init_rule_heights[key] = np.asarray(warm_h, dtype=np.float64).copy()
                else:
                    init_rule_heights[key] = np.asarray(base_rule_heights[key], dtype=np.float64).copy()
            continue
        sign_templates = template_rule_heights.get(str(ar.sign), {}) if template_rule_heights is not None else {}
        for src in rule_sources:
            key = (int(ar.idx), int(src))
            tpl = sign_templates.get(key)
            if tpl is not None:
                init_rule_heights[key] = np.asarray(tpl, dtype=np.float64).copy()
            else:
                init_rule_heights[key] = np.asarray(base_rule_heights[key], dtype=np.float64).copy()
    return init_rule_heights, init_coef_map

def sort_unique_sign_exclusive_rules(
    rules: list[ActiveRule],
) -> list[ActiveRule]:
    seen_sign_by_idx: dict[int, str] = {}
    unique: dict[tuple[int, str], ActiveRule] = {}
    for ar in rules:
        idx = int(ar.idx)
        sign = str(ar.sign)
        old_sign = seen_sign_by_idx.get(idx)
        if old_sign is not None and old_sign != sign:
            raise ValueError(f"sign-exclusive support violated for subset {idx}: {old_sign} vs {sign}")
        seen_sign_by_idx[idx] = sign
        unique[(idx, sign)] = ActiveRule(idx=idx, sign=sign)
    return sorted(unique.values(), key=lambda ar: (int(ar.idx), str(ar.sign)))


def threshold_active_rules(
    *,
    active_rules: list[ActiveRule],
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    beta_threshold: float,
) -> list[ActiveRule]:
    kept: list[ActiveRule] = []
    for ar in active_rules:
        beta = float(exc_params.get(int(ar.idx), 0.0) if ar.sign == "exc" else inh_params.get(int(ar.idx), 0.0))
        if beta > float(beta_threshold):
            kept.append(ActiveRule(idx=int(ar.idx), sign=str(ar.sign)))
    return sort_unique_sign_exclusive_rules(kept)


def _profile_mu_canonical_torch(
    *,
    signed_grid: torch.Tensor,
    grid_weights: torch.Tensor,
    num_events: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expo = torch.exp(torch.clamp(signed_grid, min=-40.0, max=40.0))
    exposure = torch.dot(grid_weights, expo)
    mu = torch.clamp(
        torch.tensor(float(num_events), dtype=torch.float32, device=signed_grid.device)
        / torch.clamp(exposure, min=1e-8),
        min=1e-8,
    )
    return mu, exposure


def _canonical_nll_from_signed_terms(
    *,
    signed_event_sum: torch.Tensor,
    signed_grid: torch.Tensor,
    grid_weights: torch.Tensor,
    num_events: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, exposure = _profile_mu_canonical_torch(
        signed_grid=signed_grid,
        grid_weights=grid_weights,
        num_events=int(num_events),
    )
    n_events = float(num_events)
    nll = -(torch.log(mu) * n_events + signed_event_sum) + mu * exposure
    return nll, mu, exposure


def _solve_beta_block_canonical(
    *,
    beta_init: torch.Tensor,
    sign_tensor: torch.Tensor,
    event_masses: torch.Tensor,
    grid_matrix: torch.Tensor,
    grid_weights: torch.Tensor,
    num_events: int,
    max_iter: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(beta_init.numel()) == 0:
        signed_grid = torch.zeros((grid_matrix.shape[1],), dtype=torch.float32, device=grid_matrix.device)
        signed_event_sum = torch.zeros((), dtype=torch.float32, device=grid_matrix.device)
        mu, _ = _profile_mu_canonical_torch(
            signed_grid=signed_grid,
            grid_weights=grid_weights,
            num_events=int(num_events),
        )
        return beta_init.detach().clone(), mu.detach(), signed_event_sum, signed_grid

    raw_beta = torch.nn.Parameter(
        torch.tensor(
            inverse_softplus(np.asarray(beta_init.detach().cpu().numpy(), dtype=np.float64)),
            dtype=torch.float32,
            device=beta_init.device,
        )
    )
    opt_beta = torch.optim.LBFGS(
        [raw_beta],
        lr=1.0,
        max_iter=max(1, int(max_iter)),
        history_size=max(3, min(10, int(beta_init.numel()))),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-5,
        tolerance_change=1e-8,
    )

    def closure():
        opt_beta.zero_grad(set_to_none=True)
        beta = F.softplus(raw_beta) + 1e-8
        signed_beta = beta * sign_tensor
        signed_event_sum = torch.dot(signed_beta, event_masses)
        signed_grid = torch.matmul(signed_beta, grid_matrix)
        nll, _mu, _exposure = _canonical_nll_from_signed_terms(
            signed_event_sum=signed_event_sum,
            signed_grid=signed_grid,
            grid_weights=grid_weights,
            num_events=int(num_events),
        )
        nll.backward()
        return nll

    opt_beta.step(closure)
    beta = (F.softplus(raw_beta.detach()) + 1e-8).detach()
    signed_beta = beta * sign_tensor
    signed_event_sum = torch.dot(signed_beta, event_masses).detach()
    signed_grid = torch.matmul(signed_beta, grid_matrix).detach()
    mu, _exposure = _profile_mu_canonical_torch(
        signed_grid=signed_grid,
        grid_weights=grid_weights,
        num_events=int(num_events),
    )
    return beta, mu.detach(), signed_event_sum, signed_grid


def optimize_active_set_torch(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    rule_heights: dict[tuple[int, int], np.ndarray],
    init_coef_map: dict[tuple[int, str], float],
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None = None,
    steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    bic_num_sequences: int | None = None,
    kernel_smoothness_ridge: float = 0.0,
    kernel_normalization: str = "peak",
) -> tuple[float, float, dict[int, float], dict[int, float], dict[tuple[int, int], np.ndarray], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    # Keep untouched rule-source kernels as shared references and only replace
    # active entries with newly fitted arrays. This avoids repeatedly deep
    # copying the full candidate universe on every exact refit.
    rule_heights = {
        (int(idx), int(src)): np.asarray(val, dtype=np.float64)
        for (idx, src), val in rule_heights.items()
    }
    if not active_rules:
        mu = base_rate_fit(0, float(np.sum(grid_weights_train)))
        return float("inf"), mu, {}, {}, rule_heights, {}

    kernel_normalization = str(kernel_normalization)
    if kernel_normalization not in {"peak", "area"}:
        raise ValueError(f"unknown kernel normalization mode: {kernel_normalization}")
    clip_witness_upper = kernel_normalization == "peak"

    knots = basis_cache.knots
    gw_tr = torch.as_tensor(grid_weights_train, dtype=torch.float32, device=device)
    gw_va = torch.as_tensor(grid_weights_val, dtype=torch.float32, device=device)

    if torch_basis_cache is None:
        torch_basis_cache = TorchBasisCache(basis_cache, device)

    raw_heights: dict[tuple[object, ...], torch.nn.Parameter] = {}
    def param_key_for(ar: ActiveRule, src: int):
        return kernel_param_key(
            idx=int(ar.idx),
            sign=str(ar.sign),
            src=int(src),
            subsets=subsets,
        )

    beta_init = []
    for ar in active_rules:
        coef0 = float(init_coef_map.get((int(ar.idx), str(ar.sign)), 0.1))
        beta_init.append(max(coef0, 1e-6))
        for src in subsets[int(ar.idx)]:
            key = (int(ar.idx), int(src))
            pkey = param_key_for(ar, int(src))
            if pkey not in raw_heights:
                h0 = ensure_feasible_score_heights_numpy(rule_heights[key])
                raw_heights[pkey] = torch.nn.Parameter(
                    torch.tensor(h0, dtype=torch.float32, device=device)
                )

    rule_specs = [
        (
            ar,
            tuple(int(src) for src in subsets[int(ar.idx)]),
            tuple(param_key_for(ar, int(src)) for src in subsets[int(ar.idx)]),
        )
        for ar in active_rules
    ]
    knot_tensors: dict[tuple[object, ...], torch.Tensor] = {}
    for ar, rule_sources, rule_pkeys in rule_specs:
        for src, pkey in zip(rule_sources, rule_pkeys):
            if pkey in knot_tensors:
                continue
            knot_tensors[pkey] = torch.as_tensor(
                basis_cache.knots_for_rule_source(int(ar.idx), int(src)),
                dtype=torch.float32,
                device=device,
            )
    use_rule_source_knots = bool(getattr(basis_cache, "rule_source_knots", {}))
    source_users: dict[tuple[int | None, int], list[tuple[int, tuple[object, ...], int]]] = {}
    for rule_row, (ar, rule_sources, rule_pkeys) in enumerate(rule_specs):
        for src, pkey in zip(rule_sources, rule_pkeys):
            group_key = (int(ar.idx), int(src)) if use_rule_source_knots else (None, int(src))
            source_users.setdefault(group_key, []).append((int(rule_row), pkey, int(ar.idx)))
    first_rule_idx = int(rule_specs[0][0].idx) if use_rule_source_knots else None
    first_src = int(rule_specs[0][1][0])
    first_tr_ev, first_tr_gr, first_va_ev, first_va_gr = torch_basis_cache.arrays_for_rule_source(first_rule_idx, first_src)
    tr_event_len = int(first_tr_ev.num_queries)
    tr_grid_len = int(first_tr_gr.num_queries)
    va_event_len = int(first_va_ev.num_queries)
    va_grid_len = int(first_va_gr.num_queries)
    sign_tensor = torch.tensor(
        [1.0 if ar.sign == "exc" else -1.0 for ar in active_rules],
        dtype=torch.float32,
        device=device,
    )
    beta_current = torch.tensor(beta_init, dtype=torch.float32, device=device)
    params = list(raw_heights.values())
    opt = torch.optim.Adam(params, lr=float(lr))

    best_bic = float("inf")
    best_snapshot = None

    def project_raw_heights_(prev_params: dict[tuple[object, ...], torch.Tensor] | None = None):
        with torch.no_grad():
            for key, param in raw_heights.items():
                param.clamp_(min=0.0)
                norm_mass = (
                    torch.amax(param)
                    if kernel_normalization == "peak"
                    else torch.trapz(param, knot_tensors[key])
                )
                if float(norm_mass.detach().cpu().item()) <= 1e-8 and prev_params is not None:
                    param.copy_(
                        normalize_kernel_heights_torch(
                            prev_params[key],
                            knots=knot_tensors.get(key),
                            mode=kernel_normalization,
                        )
                    )
                else:
                    param.copy_(
                        normalize_kernel_heights_torch(
                            param,
                            knots=knot_tensors.get(key),
                            mode=kernel_normalization,
                        )
                    )
                if kernel_normalization == "peak" and float(torch.amax(param).detach().cpu().item()) <= 1e-8:
                    if prev_params is not None:
                        prev = torch.clamp(prev_params[key], min=0.0)
                        prev_peak = torch.amax(prev)
                        if float(prev_peak.detach().cpu().item()) > 1e-8:
                            param.copy_(prev / prev_peak)
                            continue
                    param.fill_(0.0)
                    param[0] = 1.0

    def build_feature_state(
        *,
        emit_numpy: bool = False,
        include_train: bool = True,
        include_val: bool = True,
    ):
        train_event_factors = [[] for _ in rule_specs] if include_train else None
        train_grid_factors = [[] for _ in rule_specs] if include_train else None
        val_event_factors = [[] for _ in rule_specs] if include_val else None
        val_grid_factors = [[] for _ in rule_specs] if include_val else None
        arrays_out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = {} if emit_numpy else None
        heights_out: dict[tuple[int, int], np.ndarray] | None = {} if emit_numpy else None
        kernel_smooth_reg = torch.zeros((), dtype=torch.float32, device=device)
        normalized_heights: dict[tuple[object, ...], torch.Tensor] = {}

        for (rule_idx_for_knots, src), users in source_users.items():
            rule_rows = [int(rule_row) for rule_row, _pkey, _idx in users]
            pkeys = [pkey for _rule_row, pkey, _idx in users]
            idxs = [int(idx) for _rule_row, _pkey, idx in users]
            h_batch = []
            for pkey in pkeys:
                h = normalized_heights.get(pkey)
                if h is None:
                    h = normalize_kernel_heights_torch(
                        raw_heights[pkey],
                        knots=knot_tensors.get(pkey),
                        mode=kernel_normalization,
                    )
                    normalized_heights[pkey] = h
                    if float(kernel_smoothness_ridge) > 0.0:
                        kernel_smooth_reg = kernel_smooth_reg + second_difference_smoothness_torch(h)
                h_batch.append(h)
            h_batch_t = torch.stack(h_batch, dim=0)
            if heights_out is not None:
                for rule_row, idx, h in zip(rule_rows, idxs, h_batch):
                    key = (int(idx), int(src))
                    if key not in heights_out:
                        heights_out[key] = h.detach().cpu().numpy().astype(np.float64, copy=True)
            if include_train and include_val:
                combined, sizes = torch_basis_cache.combined_arrays(
                    int(src),
                    ("tr_ev", "tr_gr", "va_ev", "va_gr"),
                    rule_idx=rule_idx_for_knots,
                )
                z_all = witness_response_torch_batched(combined, h_batch_t, clip_upper=clip_witness_upper)
                n_tr_ev, n_tr_gr, n_va_ev, n_va_gr = sizes
                cut0 = n_tr_ev
                cut1 = cut0 + n_tr_gr
                cut2 = cut1 + n_va_ev
                z_tr_ev = z_all[:, :cut0]
                z_tr_gr = z_all[:, cut0:cut1]
                z_va_ev = z_all[:, cut1:cut2]
                z_va_gr = z_all[:, cut2:]
                for local_row, rule_row in enumerate(rule_rows):
                    train_event_factors[rule_row].append(z_tr_ev[local_row])
                    train_grid_factors[rule_row].append(z_tr_gr[local_row])
                    val_event_factors[rule_row].append(z_va_ev[local_row])
                    val_grid_factors[rule_row].append(z_va_gr[local_row])
            elif include_train:
                combined, sizes = torch_basis_cache.combined_arrays(
                    int(src),
                    ("tr_ev", "tr_gr"),
                    rule_idx=rule_idx_for_knots,
                )
                z_all = witness_response_torch_batched(combined, h_batch_t, clip_upper=clip_witness_upper)
                n_tr_ev, n_tr_gr = sizes
                z_tr_ev = z_all[:, :n_tr_ev]
                z_tr_gr = z_all[:, n_tr_ev:n_tr_ev + n_tr_gr]
                for local_row, rule_row in enumerate(rule_rows):
                    train_event_factors[rule_row].append(z_tr_ev[local_row])
                    train_grid_factors[rule_row].append(z_tr_gr[local_row])
            elif include_val:
                combined, sizes = torch_basis_cache.combined_arrays(
                    int(src),
                    ("va_ev", "va_gr"),
                    rule_idx=rule_idx_for_knots,
                )
                z_all = witness_response_torch_batched(combined, h_batch_t, clip_upper=clip_witness_upper)
                n_va_ev, n_va_gr = sizes
                z_va_ev = z_all[:, :n_va_ev]
                z_va_gr = z_all[:, n_va_ev:n_va_ev + n_va_gr]
                for local_row, rule_row in enumerate(rule_rows):
                    val_event_factors[rule_row].append(z_va_ev[local_row])
                    val_grid_factors[rule_row].append(z_va_gr[local_row])

        train_event_masses: list[torch.Tensor] = []
        train_grid_features: list[torch.Tensor] = []
        val_event_masses: list[torch.Tensor] = []
        val_grid_features: list[torch.Tensor] = []

        for rule_row, (ar, _rule_sources, _rule_pkeys) in enumerate(rule_specs):
            if include_train:
                feat_tr_ev = (
                    torch.prod(torch.stack(train_event_factors[rule_row], dim=0), dim=0)
                    if train_event_factors[rule_row]
                    else torch.ones((tr_event_len,), dtype=torch.float32, device=device)
                )
                feat_tr_gr = (
                    torch.prod(torch.stack(train_grid_factors[rule_row], dim=0), dim=0)
                    if train_grid_factors[rule_row]
                    else torch.ones((tr_grid_len,), dtype=torch.float32, device=device)
                )
                train_event_masses.append(feat_tr_ev.sum())
                train_grid_features.append(feat_tr_gr)
            else:
                feat_tr_ev = None
                feat_tr_gr = None
            if include_val:
                feat_va_ev = (
                    torch.prod(torch.stack(val_event_factors[rule_row], dim=0), dim=0)
                    if val_event_factors[rule_row]
                    else torch.ones((va_event_len,), dtype=torch.float32, device=device)
                )
                feat_va_gr = (
                    torch.prod(torch.stack(val_grid_factors[rule_row], dim=0), dim=0)
                    if val_grid_factors[rule_row]
                    else torch.ones((va_grid_len,), dtype=torch.float32, device=device)
                )
                val_event_masses.append(feat_va_ev.sum())
                val_grid_features.append(feat_va_gr)
            else:
                feat_va_ev = None
                feat_va_gr = None
            if arrays_out is not None:
                arrays_out[int(ar.idx)] = (
                    feat_tr_ev.detach().cpu().numpy().astype(np.float64, copy=False) if include_train else np.zeros((0,), dtype=np.float64),
                    feat_tr_gr.detach().cpu().numpy().astype(np.float64, copy=False) if include_train else np.zeros((0,), dtype=np.float64),
                    feat_va_ev.detach().cpu().numpy().astype(np.float64, copy=False) if include_val else np.zeros((0,), dtype=np.float64),
                    feat_va_gr.detach().cpu().numpy().astype(np.float64, copy=False) if include_val else np.zeros((0,), dtype=np.float64),
                )

        return {
            "train_event_masses": torch.stack(train_event_masses) if include_train else torch.zeros((0,), dtype=torch.float32, device=device),
            "train_grid_matrix": torch.stack(train_grid_features) if include_train else torch.zeros((len(active_rules), 0), dtype=torch.float32, device=device),
            "val_event_masses": torch.stack(val_event_masses) if include_val else torch.zeros((0,), dtype=torch.float32, device=device),
            "val_grid_matrix": torch.stack(val_grid_features) if include_val else torch.zeros((len(active_rules), 0), dtype=torch.float32, device=device),
            "kernel_smooth_reg": kernel_smooth_reg,
            "heights_out": heights_out,
            "arrays_out": arrays_out,
        }

    def as_param_maps(beta_values: torch.Tensor) -> tuple[dict[int, float], dict[int, float]]:
        exc_params: dict[int, float] = {}
        inh_params: dict[int, float] = {}
        for ar, beta in zip(active_rules, beta_values.detach().cpu().numpy().tolist()):
            if ar.sign == "exc":
                exc_params[int(ar.idx)] = float(beta)
            else:
                inh_params[int(ar.idx)] = float(beta)
        return exc_params, inh_params

    beta_block_iters = max(6, min(16, int(steps) // 4 if int(steps) > 0 else 8))
    eval_every = max(1, min(10, int(steps)))
    min_steps_for_stop = min(max(0, int(steps) - 1), max(10, 2 * eval_every))
    bic_tol = 1e-5
    param_tol = 1e-4
    patience = 2
    stagnant_evals = 0
    prev_eval_beta: torch.Tensor | None = None
    prev_eval_raw: dict[tuple[object, ...], torch.Tensor] | None = None

    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        train_state = build_feature_state(include_train=True, include_val=False, emit_numpy=False)
        beta_fit, mu_fit, signed_event_sum, signed_grid = _solve_beta_block_canonical(
            beta_init=beta_current,
            sign_tensor=sign_tensor,
            event_masses=train_state["train_event_masses"].detach(),
            grid_matrix=train_state["train_grid_matrix"].detach(),
            grid_weights=gw_tr,
            num_events=tr_event_len,
            max_iter=beta_block_iters,
        )
        beta_current = beta_fit.detach()
        signed_beta_detached = (beta_current * sign_tensor).detach()
        signed_event_sum_h = torch.dot(signed_beta_detached, train_state["train_event_masses"])
        signed_grid_h = torch.matmul(signed_beta_detached, train_state["train_grid_matrix"])
        train_nll, _mu_h, _exposure_h = _canonical_nll_from_signed_terms(
            signed_event_sum=signed_event_sum_h,
            signed_grid=signed_grid_h,
            grid_weights=gw_tr,
            num_events=tr_event_len,
        )
        if float(kernel_smoothness_ridge) > 0.0:
            train_nll = train_nll + float(kernel_smoothness_ridge) * train_state["kernel_smooth_reg"]
        train_nll.backward()
        prev_params = {key: param.detach().clone() for key, param in raw_heights.items()}
        opt.step()
        project_raw_heights_(prev_params)

        if step % eval_every == 0 or step == int(steps) - 1:
            with torch.no_grad():
                eval_state = build_feature_state(include_train=True, include_val=True, emit_numpy=False)
            beta_eval, mu_eval, _signed_event_train, _signed_grid_train = _solve_beta_block_canonical(
                beta_init=beta_current,
                sign_tensor=sign_tensor,
                event_masses=eval_state["train_event_masses"].detach(),
                grid_matrix=eval_state["train_grid_matrix"].detach(),
                grid_weights=gw_tr,
                num_events=tr_event_len,
                max_iter=beta_block_iters,
            )
            beta_current = beta_eval.detach()
            signed_beta_eval = beta_current * sign_tensor
            signed_event_val = torch.dot(signed_beta_eval, eval_state["val_event_masses"])
            signed_grid_val = torch.matmul(signed_beta_eval, eval_state["val_grid_matrix"])
            ll_val = (
                torch.log(mu_eval) * float(va_event_len)
                + signed_event_val
                - torch.dot(gw_va, mu_eval * torch.exp(torch.clamp(signed_grid_val, min=-40.0, max=40.0)))
            )
            n_eff = bic_sample_size(int(bic_num_sequences) if bic_num_sequences is not None else int(va_grid_len))
            bic = float(
                    (-2.0 * ll_val).detach().cpu().item()
                    + (
                    float(penalty_scale) * float(model_param_dim(active_rules, subsets, int(len(knots)))) * math.log(float(n_eff))
                    if penalize_kernel_df
                    else float(penalty_scale) * float(1 + len(active_rules)) * math.log(float(n_eff))
                )
            )
            improved = bool(bic < best_bic - bic_tol)
            if bic < best_bic:
                best_bic = bic
                best_snapshot = {
                    "raw_heights": {key: param.detach().clone() for key, param in raw_heights.items()},
                    "beta": beta_current.detach().clone(),
                }
            max_param_change = float("inf")
            if prev_eval_beta is not None and prev_eval_raw is not None:
                beta_change = float(torch.max(torch.abs(beta_current - prev_eval_beta)).detach().cpu().item())
                height_change = 0.0
                for key, param in raw_heights.items():
                    cur_change = float(torch.max(torch.abs(param.detach() - prev_eval_raw[key])).detach().cpu().item())
                    if cur_change > height_change:
                        height_change = cur_change
                max_param_change = max(beta_change, height_change)
            prev_eval_beta = beta_current.detach().clone()
            prev_eval_raw = {key: param.detach().clone() for key, param in raw_heights.items()}
            if step >= int(min_steps_for_stop):
                if improved:
                    stagnant_evals = 0
                elif max_param_change <= float(param_tol):
                    stagnant_evals += 1
                else:
                    stagnant_evals = 0
                if stagnant_evals >= int(patience):
                    break

    if best_snapshot is None:
        raise RuntimeError("Active-set optimization failed to produce a valid state")

    with torch.no_grad():
        for key, param in raw_heights.items():
            param.copy_(best_snapshot["raw_heights"][key])
    beta_current = best_snapshot["beta"].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        final_state = build_feature_state(include_train=True, include_val=True, emit_numpy=True)
    beta_final, mu_final, _signed_event_final, _signed_grid_final = _solve_beta_block_canonical(
        beta_init=beta_current,
        sign_tensor=sign_tensor,
        event_masses=final_state["train_event_masses"].detach(),
        grid_matrix=final_state["train_grid_matrix"].detach(),
        grid_weights=gw_tr,
        num_events=tr_event_len,
        max_iter=max(beta_block_iters, 12),
    )
    exc_params, inh_params = as_param_maps(beta_final)
    for key, h in (final_state["heights_out"] or {}).items():
        rule_heights[key] = np.asarray(h, dtype=np.float64)
    final_arrays = final_state["arrays_out"] or {}
    final_bic = validation_bic_from_arrays(
        active_rules=active_rules,
        arrays_out=final_arrays,
        exc_params=exc_params,
        inh_params=inh_params,
        mu=float(mu_final.detach().cpu().item()),
        subsets=subsets,
        num_knots=int(len(knots)),
        grid_weights_val=grid_weights_val,
        penalize_kernel_df=bool(penalize_kernel_df),
        penalty_scale=float(penalty_scale),
        num_sequences_val=bic_num_sequences,
    )
    return float(final_bic), float(mu_final.detach().cpu().item()), exc_params, inh_params, rule_heights, final_arrays


def _solve_beta_block_canonical_batch(
    *,
    beta_init: torch.Tensor,
    sign_tensor: torch.Tensor,
    event_masses: torch.Tensor,
    grid_matrix: torch.Tensor,
    grid_weights: torch.Tensor,
    num_events: int,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(beta_init.ndim) != 2:
        raise ValueError("batched beta solve expects a (batch, rule) beta tensor")
    batch_count = int(beta_init.shape[0])
    rule_count = int(beta_init.shape[1])
    grid_len = int(grid_matrix.shape[2]) if int(grid_matrix.ndim) == 3 else 0
    if rule_count == 0:
        signed_grid = torch.zeros((batch_count, grid_len), dtype=torch.float32, device=grid_matrix.device)
        signed_event_sum = torch.zeros((batch_count,), dtype=torch.float32, device=grid_matrix.device)
        mu, _ = _profile_mu_canonical_torch_batch(
            signed_grid=signed_grid,
            grid_weights=grid_weights,
            num_events=int(num_events),
        )
        return beta_init.detach().clone(), mu.detach(), signed_event_sum, signed_grid

    x_grid = sign_tensor.reshape(batch_count, rule_count, 1) * grid_matrix
    signed_event_masses = sign_tensor * event_masses
    gw = grid_weights.reshape(1, grid_len)
    eye = torch.eye(rule_count, dtype=torch.float32, device=grid_matrix.device).reshape(1, rule_count, rule_count)
    beta = torch.clamp(beta_init.detach().to(dtype=torch.float32), min=1e-8)

    def value_grad_hess(beta_value: torch.Tensor):
        signed_beta = beta_value * sign_tensor
        signed_event_sum = torch.sum(signed_beta * event_masses, dim=1)
        signed_grid = torch.einsum("br,brg->bg", signed_beta, grid_matrix)
        expo = torch.exp(torch.clamp(signed_grid, min=-40.0, max=40.0))
        exposure = torch.sum(expo * gw, dim=1)
        mu = torch.clamp(
            torch.full((batch_count,), float(num_events), dtype=torch.float32, device=grid_matrix.device)
            / torch.clamp(exposure, min=1e-8),
            min=1e-8,
        )
        nll = -(torch.log(mu) * float(num_events) + signed_event_sum) + mu * exposure
        weights = expo * gw
        mean_x = torch.einsum("bg,brg->br", weights, x_grid) / torch.clamp(exposure.reshape(batch_count, 1), min=1e-8)
        second_x = torch.einsum("bg,brg,bsg->brs", weights, x_grid, x_grid) / torch.clamp(
            exposure.reshape(batch_count, 1, 1),
            min=1e-8,
        )
        grad = float(num_events) * mean_x - signed_event_masses
        hess = float(num_events) * (
            second_x - mean_x.reshape(batch_count, rule_count, 1) * mean_x.reshape(batch_count, 1, rule_count)
        )
        hess = 0.5 * (hess + torch.transpose(hess, 1, 2))
        return nll, grad, hess, mu, signed_event_sum, signed_grid

    for _iter in range(max(1, int(max_iter))):
        nll, grad, hess, _mu, _signed_event_sum, _signed_grid = value_grad_hess(beta)
        free = (beta > 1e-7) | (grad < -1e-6)
        if not bool(torch.any(free)):
            break
        free_f = free.to(dtype=torch.float32)
        masked_grad = grad * free_f
        masked_hess = hess * free_f.reshape(batch_count, rule_count, 1) * free_f.reshape(batch_count, 1, rule_count)
        masked_hess = masked_hess + eye * (1.0 - free_f).reshape(batch_count, rule_count, 1)
        masked_hess = masked_hess + eye * 1e-5
        try:
            delta = -torch.linalg.solve(masked_hess, masked_grad.reshape(batch_count, rule_count, 1)).reshape(
                batch_count,
                rule_count,
            )
        except RuntimeError:
            delta = -torch.linalg.pinv(masked_hess) @ masked_grad.reshape(batch_count, rule_count, 1)
            delta = delta.reshape(batch_count, rule_count)
        delta = delta * free_f
        if float(torch.max(torch.abs(delta)).detach().cpu().item()) <= 1e-6:
            break

        neg_delta = delta < 0.0
        feasible_ratio = torch.where(
            neg_delta,
            -0.99 * beta / torch.clamp(delta, max=-1e-12),
            torch.full_like(delta, float("inf")),
        )
        max_step = torch.clamp(torch.amin(feasible_ratio, dim=1), max=1.0)
        max_step = torch.where(torch.isfinite(max_step), max_step, torch.ones_like(max_step))
        step = torch.clamp(max_step, min=1e-6, max=1.0)
        best_beta = beta
        accepted = torch.zeros((batch_count,), dtype=torch.bool, device=grid_matrix.device)
        for _ls in range(12):
            cand_beta = torch.clamp(beta + step.reshape(batch_count, 1) * delta, min=1e-8)
            cand_nll, _cand_grad, _cand_hess, _cand_mu, _cand_event, _cand_grid = value_grad_hess(cand_beta)
            ok = cand_nll <= nll + 1e-7
            newly_ok = ok & (~accepted)
            if bool(torch.any(newly_ok)):
                best_beta = torch.where(newly_ok.reshape(batch_count, 1), cand_beta, best_beta)
                accepted = accepted | newly_ok
            if bool(torch.all(accepted)):
                break
            step = torch.where(accepted, step, step * 0.5)
        beta_next = torch.where(accepted.reshape(batch_count, 1), best_beta, beta)
        max_update = float(torch.max(torch.abs(beta_next - beta)).detach().cpu().item())
        beta = beta_next.detach()
        if max_update <= 1e-6:
            break

    _nll, grad, _hess, mu, signed_event_sum, signed_grid = value_grad_hess(beta)
    interior = beta > 1e-5
    kkt_residual = torch.maximum(
        torch.max(torch.where(interior, torch.abs(grad), torch.zeros_like(grad)), dim=1).values,
        torch.max(torch.where(interior, torch.zeros_like(grad), torch.relu(-grad)), dim=1).values,
    )
    fallback_rows = torch.nonzero(kkt_residual > 5e-2, as_tuple=False).reshape(-1).detach().cpu().numpy().tolist()
    if fallback_rows:
        beta_rows = [beta[int(row)] for row in range(batch_count)]
        mu_rows = [mu[int(row)] for row in range(batch_count)]
        event_rows = [signed_event_sum[int(row)] for row in range(batch_count)]
        grid_rows = [signed_grid[int(row)] for row in range(batch_count)]
        for row in fallback_rows:
            row_beta, row_mu, row_event, row_grid = _solve_beta_block_canonical(
                beta_init=beta_init[int(row)],
                sign_tensor=sign_tensor[int(row)],
                event_masses=event_masses[int(row)],
                grid_matrix=grid_matrix[int(row)],
                grid_weights=grid_weights,
                num_events=int(num_events),
                max_iter=int(max_iter),
            )
            beta_rows[int(row)] = row_beta
            mu_rows[int(row)] = row_mu.reshape(())
            event_rows[int(row)] = row_event.reshape(())
            grid_rows[int(row)] = row_grid
        beta = torch.stack(beta_rows, dim=0)
        mu = torch.stack(mu_rows, dim=0)
        signed_event_sum = torch.stack(event_rows, dim=0)
        signed_grid = torch.stack(grid_rows, dim=0)
    return beta.detach(), mu.detach(), signed_event_sum.detach(), signed_grid.detach()


def _profile_mu_canonical_torch_batch(
    *,
    signed_grid: torch.Tensor,
    grid_weights: torch.Tensor,
    num_events: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expo = torch.exp(torch.clamp(signed_grid, min=-40.0, max=40.0))
    exposure = torch.sum(expo * grid_weights.reshape(1, -1), dim=1)
    mu = torch.clamp(
        torch.full(
            (int(signed_grid.shape[0]),),
            float(num_events),
            dtype=torch.float32,
            device=signed_grid.device,
        )
        / torch.clamp(exposure, min=1e-8),
        min=1e-8,
    )
    return mu, exposure


def _normalize_height_rows_torch(raw: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(raw, min=0.0)
    peak = torch.amax(x, dim=1, keepdim=True)
    return x / torch.clamp(peak, min=1e-8)


def optimize_active_set_torch_batch_same_size(
    *,
    active_rule_lists: list[list[ActiveRule]],
    subsets,
    basis_cache: SourceBasisCache,
    base_rule_heights: dict[tuple[int, int], np.ndarray],
    template_rule_heights: dict[str, dict[tuple[int, int], np.ndarray]] | None,
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None,
    steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    bic_num_sequences: int | None = None,
    kernel_smoothness_ridge: float = 0.0,
    warm_start_result: SupportEvalResult | None = None,
    score_only: bool = False,
) -> dict[SupportKey, SupportEvalResult]:
    if not active_rule_lists:
        return {}
    active_rule_lists = [
        sort_unique_sign_exclusive_rules(list(rules))
        for rules in active_rule_lists
    ]
    rule_count = int(len(active_rule_lists[0]))
    if any(int(len(rules)) != rule_count for rules in active_rule_lists):
        raise ValueError("batched exact refit requires equal-size support groups")
    if rule_count == 0:
        out: dict[SupportKey, SupportEvalResult] = {}
        for rules in active_rule_lists:
            bic, mu = fit_constant_bic(
                n_events_val=int(basis_cache.val_event_times.size),
                grid_weights_val=grid_weights_val,
                n_events_train=int(basis_cache.train_event_times.size),
                grid_weights_train=grid_weights_train,
                n_val_sequences=int(bic_num_sequences if bic_num_sequences is not None else 1),
            )
            out[support_key_from_rules(rules)] = SupportEvalResult(
                bic=float(bic),
                mu=float(mu),
                exc_params={},
                inh_params={},
                rule_heights=base_rule_heights,
                arrays_out=None,
                active_rules=[],
            )
        return out

    if torch_basis_cache is None:
        torch_basis_cache = TorchBasisCache(basis_cache, device)

    batch_count = int(len(active_rule_lists))
    num_knots = int(len(basis_cache.knots))
    gw_tr = torch.as_tensor(grid_weights_train, dtype=torch.float32, device=device)
    gw_va = torch.as_tensor(grid_weights_val, dtype=torch.float32, device=device)

    first_src = int(subsets[int(active_rule_lists[0][0].idx)][0])
    first_tr_ev, first_tr_gr, first_va_ev, first_va_gr = torch_basis_cache.arrays(first_src)
    tr_event_len = int(first_tr_ev.num_queries)
    tr_grid_len = int(first_tr_gr.num_queries)
    va_event_len = int(first_va_ev.num_queries)
    va_grid_len = int(first_va_gr.num_queries)
    same_train_val_queries = (
        tr_event_len == va_event_len
        and tr_grid_len == va_grid_len
        and np.array_equal(basis_cache.train_event_seq_ids, basis_cache.val_event_seq_ids)
        and np.array_equal(basis_cache.train_event_times, basis_cache.val_event_times)
        and np.array_equal(basis_cache.train_grid_seq_ids, basis_cache.val_grid_seq_ids)
        and np.array_equal(basis_cache.train_grid_times, basis_cache.val_grid_times)
        and np.array_equal(np.asarray(grid_weights_train), np.asarray(grid_weights_val))
    )

    init_height_maps: list[dict[tuple[int, int], np.ndarray]] = []
    beta_init_np = np.zeros((batch_count, rule_count), dtype=np.float32)
    sign_np = np.zeros((batch_count, rule_count), dtype=np.float32)
    entry_support: list[int] = []
    entry_rule_row: list[int] = []
    entry_rule_idx: list[int] = []
    entry_source: list[int] = []
    entry_init_heights: list[np.ndarray] = []

    for batch_row, rules in enumerate(active_rule_lists):
        init_rule_heights, init_coef_map = _build_support_warm_start(
            target_rules=rules,
            subsets=subsets,
            base_rule_heights=base_rule_heights,
            template_rule_heights=template_rule_heights,
            warm_start_result=warm_start_result,
            fallback=0.1,
        )
        init_height_maps.append(init_rule_heights)
        for rule_row, rule in enumerate(rules):
            beta_init_np[int(batch_row), int(rule_row)] = max(
                float(init_coef_map.get((int(rule.idx), str(rule.sign)), 0.1)),
                1e-6,
            )
            sign_np[int(batch_row), int(rule_row)] = 1.0 if str(rule.sign) == "exc" else -1.0
            for src in subsets[int(rule.idx)]:
                key = (int(rule.idx), int(src))
                entry_support.append(int(batch_row))
                entry_rule_row.append(int(rule_row))
                entry_rule_idx.append(int(rule.idx))
                entry_source.append(int(src))
                entry_init_heights.append(ensure_feasible_score_heights_numpy(init_rule_heights[key]))

    raw = torch.nn.Parameter(
        torch.as_tensor(
            np.stack(entry_init_heights, axis=0).astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        )
    )
    sign_tensor = torch.as_tensor(sign_np, dtype=torch.float32, device=device)
    beta_current = torch.as_tensor(beta_init_np, dtype=torch.float32, device=device)
    entry_support_tensor = torch.as_tensor(entry_support, dtype=torch.int64, device=device)
    entry_rule_flat_tensor = torch.as_tensor(
        [
            int(batch_row) * int(rule_count) + int(rule_row)
            for batch_row, rule_row in zip(entry_support, entry_rule_row)
        ],
        dtype=torch.int64,
        device=device,
    )
    source_to_entries: dict[int, list[int]] = {}
    for entry_id, src in enumerate(entry_source):
        source_to_entries.setdefault(int(src), []).append(int(entry_id))
    source_to_entry_tensors = {
        int(src): torch.as_tensor(entry_ids, dtype=torch.int64, device=device)
        for src, entry_ids in source_to_entries.items()
    }

    opt = torch.optim.Adam([raw], lr=float(lr))
    beta_block_iters = max(6, min(16, int(steps) // 4 if int(steps) > 0 else 8))
    eval_every = max(1, min(10, int(steps)))
    min_steps_for_stop = min(max(0, int(steps) - 1), max(10, 2 * eval_every))
    bic_tol = 1e-5
    param_tol = 1e-4
    patience = 2
    best_bic = np.full((batch_count,), np.inf, dtype=np.float64)
    best_raw = raw.detach().clone()
    best_beta = beta_current.detach().clone()
    prev_eval_beta: torch.Tensor | None = None
    prev_eval_raw: torch.Tensor | None = None
    stagnant_evals = np.zeros((batch_count,), dtype=np.int64)
    active_support = np.ones((batch_count,), dtype=bool)

    def project_raw_heights_(prev_params: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            raw.clamp_(min=0.0)
            peak = torch.amax(raw, dim=1, keepdim=True)
            good = peak.squeeze(1) > 1e-8
            if bool(torch.any(good).item()):
                raw[good] = raw[good] / torch.clamp(peak[good], min=1e-8)
            if bool(torch.any(~good).item()):
                if prev_params is not None:
                    prev = torch.clamp(prev_params[~good], min=0.0)
                    prev_peak = torch.amax(prev, dim=1, keepdim=True)
                    prev_good = prev_peak.squeeze(1) > 1e-8
                    if bool(torch.any(prev_good).item()):
                        bad_rows = torch.nonzero(~good, as_tuple=False).flatten()
                        raw[bad_rows[prev_good]] = prev[prev_good] / torch.clamp(
                            prev_peak[prev_good],
                            min=1e-8,
                        )
                    if bool(torch.any(~prev_good).item()):
                        bad_rows = torch.nonzero(~good, as_tuple=False).flatten()
                        raw[bad_rows[~prev_good]].fill_(0.0)
                        raw[bad_rows[~prev_good], 0] = 1.0
                else:
                    raw[~good].fill_(0.0)
                    raw[~good, 0] = 1.0

    def freeze_inactive_optimizer_rows_() -> None:
        inactive = torch.as_tensor(~active_support, dtype=torch.bool, device=device)
        if not bool(torch.any(inactive).item()):
            return
        inactive_entries = inactive[entry_support_tensor]
        if not bool(torch.any(inactive_entries).item()):
            return
        with torch.no_grad():
            raw[inactive_entries] = best_raw[inactive_entries]
            state = opt.state.get(raw)
            if state is not None:
                for name in ("exp_avg", "exp_avg_sq"):
                    buf = state.get(name)
                    if buf is not None:
                        buf[inactive_entries] = 0.0

    def build_feature_state(
        *,
        emit_numpy: bool = False,
        include_train: bool = True,
        include_val: bool = True,
    ):
        reuse_train_as_val = bool(include_train and include_val and same_train_val_queries)
        arrays_out = [dict() for _ in range(batch_count)] if emit_numpy else None
        heights_out = [dict() for _ in range(batch_count)] if emit_numpy else None
        smooth_by_support = torch.zeros((batch_count,), dtype=torch.float32, device=device)
        normalized = _normalize_height_rows_torch(raw)
        if float(kernel_smoothness_ridge) > 0.0 and int(num_knots) >= 3:
            d2 = normalized[:, 2:] - 2.0 * normalized[:, 1:-1] + normalized[:, :-2]
            smooth_entry = torch.sum(d2 * d2, dim=1)
            smooth_by_support.scatter_add_(0, entry_support_tensor, smooth_entry)

        flat_rule_count = int(batch_count) * int(rule_count)
        if include_train:
            train_event_flat = torch.ones(
                (flat_rule_count, tr_event_len),
                dtype=torch.float32,
                device=device,
            )
            train_grid_flat = torch.ones(
                (flat_rule_count, tr_grid_len),
                dtype=torch.float32,
                device=device,
            )
        else:
            train_event_flat = None
            train_grid_flat = None
        if include_val and not reuse_train_as_val:
            val_event_flat = torch.ones(
                (flat_rule_count, va_event_len),
                dtype=torch.float32,
                device=device,
            )
            val_grid_flat = torch.ones(
                (flat_rule_count, va_grid_len),
                dtype=torch.float32,
                device=device,
            )
        else:
            val_event_flat = None
            val_grid_flat = None

        def scatter_prod(dest: torch.Tensor, rows: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            if int(values.numel()) == 0:
                return dest
            index = rows.reshape(-1, 1).expand(-1, int(values.shape[1]))
            return dest.scatter_reduce(0, index, values, reduce="prod", include_self=True)

        for src, entry_tensor in source_to_entry_tensors.items():
            h_batch = normalized[entry_tensor]
            if heights_out is not None:
                for local_pos, entry_id in enumerate(entry_tensor.detach().cpu().numpy().tolist()):
                    batch_row = int(entry_support[int(entry_id)])
                    key = (int(entry_rule_idx[int(entry_id)]), int(entry_source[int(entry_id)]))
                    if key not in heights_out[batch_row]:
                        heights_out[batch_row][key] = h_batch[int(local_pos)].detach().cpu().numpy().astype(
                            np.float64,
                            copy=True,
                        )
            flat_rows = entry_rule_flat_tensor[entry_tensor]
            if include_train and include_val and reuse_train_as_val:
                combined, sizes = torch_basis_cache.combined_arrays(int(src), ("tr_ev", "tr_gr"))
                z_all = witness_response_torch_batched(combined, h_batch)
                n_tr_ev, n_tr_gr = sizes
                train_event_flat = scatter_prod(train_event_flat, flat_rows, z_all[:, : int(n_tr_ev)])
                train_grid_flat = scatter_prod(
                    train_grid_flat,
                    flat_rows,
                    z_all[:, int(n_tr_ev): int(n_tr_ev) + int(n_tr_gr)],
                )
            elif include_train and include_val:
                combined, sizes = torch_basis_cache.combined_arrays(int(src), ("tr_ev", "tr_gr", "va_ev", "va_gr"))
                z_all = witness_response_torch_batched(combined, h_batch)
                n_tr_ev, n_tr_gr, n_va_ev, _n_va_gr = sizes
                cut0 = n_tr_ev
                cut1 = cut0 + n_tr_gr
                cut2 = cut1 + n_va_ev
                train_event_flat = scatter_prod(train_event_flat, flat_rows, z_all[:, :cut0])
                train_grid_flat = scatter_prod(train_grid_flat, flat_rows, z_all[:, cut0:cut1])
                val_event_flat = scatter_prod(val_event_flat, flat_rows, z_all[:, cut1:cut2])
                val_grid_flat = scatter_prod(val_grid_flat, flat_rows, z_all[:, cut2:])
            elif include_train:
                combined, sizes = torch_basis_cache.combined_arrays(int(src), ("tr_ev", "tr_gr"))
                z_all = witness_response_torch_batched(combined, h_batch)
                n_tr_ev, n_tr_gr = sizes
                train_event_flat = scatter_prod(train_event_flat, flat_rows, z_all[:, : int(n_tr_ev)])
                train_grid_flat = scatter_prod(
                    train_grid_flat,
                    flat_rows,
                    z_all[:, int(n_tr_ev): int(n_tr_ev) + int(n_tr_gr)],
                )
            elif include_val:
                combined, sizes = torch_basis_cache.combined_arrays(int(src), ("va_ev", "va_gr"))
                z_all = witness_response_torch_batched(combined, h_batch)
                n_va_ev, n_va_gr = sizes
                val_event_flat = scatter_prod(val_event_flat, flat_rows, z_all[:, : int(n_va_ev)])
                val_grid_flat = scatter_prod(
                    val_grid_flat,
                    flat_rows,
                    z_all[:, int(n_va_ev): int(n_va_ev) + int(n_va_gr)],
                )

        if include_train:
            train_event_features = train_event_flat.view(batch_count, rule_count, tr_event_len)
            train_grid_features = train_grid_flat.view(batch_count, rule_count, tr_grid_len)
            train_event_masses = torch.sum(train_event_features, dim=2)
        else:
            train_event_features = None
            train_grid_features = None
            train_event_masses = torch.zeros((batch_count, rule_count), dtype=torch.float32, device=device)
        if include_val and reuse_train_as_val:
            val_event_features = train_event_features
            val_grid_features = train_grid_features
            val_event_masses = train_event_masses
        elif include_val:
            val_event_features = val_event_flat.view(batch_count, rule_count, va_event_len)
            val_grid_features = val_grid_flat.view(batch_count, rule_count, va_grid_len)
            val_event_masses = torch.sum(val_event_features, dim=2)
        else:
            val_event_features = None
            val_grid_features = None
            val_event_masses = torch.zeros((batch_count, rule_count), dtype=torch.float32, device=device)

        if arrays_out is not None:
            for b, rules in enumerate(active_rule_lists):
                for r, rule in enumerate(rules):
                    arrays_out[b][int(rule.idx)] = (
                        train_event_features[int(b), int(r)].detach().cpu().numpy().astype(np.float64, copy=False)
                        if include_train
                        else np.zeros((0,), dtype=np.float64),
                        train_grid_features[int(b), int(r)].detach().cpu().numpy().astype(np.float64, copy=False)
                        if include_train
                        else np.zeros((0,), dtype=np.float64),
                        val_event_features[int(b), int(r)].detach().cpu().numpy().astype(np.float64, copy=False)
                        if include_val
                        else np.zeros((0,), dtype=np.float64),
                        val_grid_features[int(b), int(r)].detach().cpu().numpy().astype(np.float64, copy=False)
                        if include_val
                        else np.zeros((0,), dtype=np.float64),
                    )

        return {
            "train_event_masses": train_event_masses,
            "train_grid_matrix": train_grid_features
            if include_train
            else torch.zeros((batch_count, rule_count, 0), dtype=torch.float32, device=device),
            "val_event_masses": val_event_masses,
            "val_grid_matrix": val_grid_features
            if include_val
            else torch.zeros((batch_count, rule_count, 0), dtype=torch.float32, device=device),
            "kernel_smooth_reg": smooth_by_support,
            "heights_out": heights_out,
            "arrays_out": arrays_out,
        }

    for step in range(int(steps)):
        if not bool(np.any(active_support)):
            break
        opt.zero_grad(set_to_none=True)
        train_state = build_feature_state(include_train=True, include_val=False, emit_numpy=False)
        beta_fit, _mu_fit, _signed_event_sum, _signed_grid = _solve_beta_block_canonical_batch(
            beta_init=beta_current,
            sign_tensor=sign_tensor,
            event_masses=train_state["train_event_masses"].detach(),
            grid_matrix=train_state["train_grid_matrix"].detach(),
            grid_weights=gw_tr,
            num_events=tr_event_len,
            max_iter=beta_block_iters,
        )
        beta_current = beta_fit.detach()
        signed_beta = (beta_current * sign_tensor).detach()
        signed_event_sum = torch.sum(signed_beta * train_state["train_event_masses"], dim=1)
        signed_grid = torch.einsum("br,brg->bg", signed_beta, train_state["train_grid_matrix"])
        mu_train, exposure_train = _profile_mu_canonical_torch_batch(
            signed_grid=signed_grid,
            grid_weights=gw_tr,
            num_events=tr_event_len,
        )
        train_nll = -(torch.log(mu_train) * float(tr_event_len) + signed_event_sum) + mu_train * exposure_train
        active_mask = torch.as_tensor(active_support, dtype=torch.float32, device=device)
        loss_by_support = train_nll
        if float(kernel_smoothness_ridge) > 0.0:
            loss_by_support = loss_by_support + float(kernel_smoothness_ridge) * train_state["kernel_smooth_reg"]
        loss = torch.sum(loss_by_support * active_mask)
        loss.backward()
        prev_params = raw.detach().clone()
        opt.step()
        project_raw_heights_(prev_params)
        freeze_inactive_optimizer_rows_()

        if step % eval_every == 0 or step == int(steps) - 1:
            with torch.no_grad():
                eval_state = build_feature_state(include_train=True, include_val=True, emit_numpy=False)
            beta_eval, mu_eval, _signed_event_train, _signed_grid_train = _solve_beta_block_canonical_batch(
                beta_init=beta_current,
                sign_tensor=sign_tensor,
                event_masses=eval_state["train_event_masses"].detach(),
                grid_matrix=eval_state["train_grid_matrix"].detach(),
                grid_weights=gw_tr,
                num_events=tr_event_len,
                max_iter=beta_block_iters,
            )
            beta_current = beta_eval.detach()
            signed_beta_eval = beta_current * sign_tensor
            signed_event_val = torch.sum(signed_beta_eval * eval_state["val_event_masses"], dim=1)
            signed_grid_val = torch.einsum("br,brg->bg", signed_beta_eval, eval_state["val_grid_matrix"])
            exposure_val = torch.sum(
                torch.exp(torch.clamp(signed_grid_val, min=-40.0, max=40.0)) * gw_va.reshape(1, -1),
                dim=1,
            )
            ll_val = torch.log(mu_eval) * float(va_event_len) + signed_event_val - mu_eval * exposure_val
            n_eff = bic_sample_size(int(bic_num_sequences) if bic_num_sequences is not None else int(va_grid_len))
            dim_values = torch.as_tensor(
                [
                    float(model_param_dim(rules, subsets, int(num_knots)))
                    if bool(penalize_kernel_df)
                    else float(1 + len(rules))
                    for rules in active_rule_lists
                ],
                dtype=torch.float32,
                device=device,
            )
            bic_values_t = -2.0 * ll_val + float(penalty_scale) * dim_values * math.log(float(n_eff))
            bic_values = bic_values_t.detach().cpu().numpy().astype(np.float64, copy=False)
            improved = (bic_values < best_bic - bic_tol) | np.isinf(best_bic)
            if bool(np.any(improved)):
                improved_support = torch.as_tensor(improved, dtype=torch.bool, device=device)
                improved_entries = improved_support[entry_support_tensor]
                best_raw[improved_entries] = raw.detach()[improved_entries]
                best_beta[improved_support] = beta_current.detach()[improved_support]
                best_bic[improved] = bic_values[improved]

            max_param_change = np.full((batch_count,), np.inf, dtype=np.float64)
            if prev_eval_beta is not None and prev_eval_raw is not None:
                beta_change = torch.max(torch.abs(beta_current - prev_eval_beta), dim=1).values.detach().cpu().numpy()
                entry_change = torch.max(torch.abs(raw.detach() - prev_eval_raw), dim=1).values.detach().cpu().numpy()
                height_change = np.zeros((batch_count,), dtype=np.float64)
                for entry_id, batch_row in enumerate(entry_support):
                    height_change[int(batch_row)] = max(
                        float(height_change[int(batch_row)]),
                        float(entry_change[int(entry_id)]),
                    )
                max_param_change = np.maximum(beta_change.astype(np.float64), height_change)
            prev_eval_beta = beta_current.detach().clone()
            prev_eval_raw = raw.detach().clone()

            if step >= int(min_steps_for_stop):
                for batch_row in range(batch_count):
                    if not bool(active_support[int(batch_row)]):
                        continue
                    if bool(improved[int(batch_row)]):
                        stagnant_evals[int(batch_row)] = 0
                    elif float(max_param_change[int(batch_row)]) <= float(param_tol):
                        stagnant_evals[int(batch_row)] += 1
                    else:
                        stagnant_evals[int(batch_row)] = 0
                    if int(stagnant_evals[int(batch_row)]) >= int(patience):
                        active_support[int(batch_row)] = False
                freeze_inactive_optimizer_rows_()

    if np.any(np.isinf(best_bic)):
        raise RuntimeError("Batched active-set optimization failed to produce a valid state")

    with torch.no_grad():
        raw.copy_(best_raw)
    beta_current = best_beta.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        final_state = build_feature_state(include_train=True, include_val=True, emit_numpy=not bool(score_only))
    beta_final, mu_final, _signed_event_final, _signed_grid_final = _solve_beta_block_canonical_batch(
        beta_init=beta_current,
        sign_tensor=sign_tensor,
        event_masses=final_state["train_event_masses"].detach(),
        grid_matrix=final_state["train_grid_matrix"].detach(),
        grid_weights=gw_tr,
        num_events=tr_event_len,
        max_iter=max(beta_block_iters, 12),
    )
    signed_beta_final = (beta_final * sign_tensor).to(dtype=torch.float64)
    mu_final_d = mu_final.to(dtype=torch.float64)
    gw_va_d = torch.as_tensor(grid_weights_val, dtype=torch.float64, device=device)
    signed_event_val = torch.sum(signed_beta_final * final_state["val_event_masses"].to(dtype=torch.float64), dim=1)
    signed_grid_val = torch.einsum(
        "br,brg->bg",
        signed_beta_final,
        final_state["val_grid_matrix"].to(dtype=torch.float64),
    )
    exposure_val = torch.sum(
        torch.exp(torch.clamp(signed_grid_val, min=-40.0, max=40.0)) * gw_va_d.reshape(1, -1),
        dim=1,
    )
    ll_val = torch.log(torch.clamp(mu_final_d, min=1e-8)) * float(va_event_len) + signed_event_val - mu_final_d * exposure_val
    n_eff = bic_sample_size(int(bic_num_sequences) if bic_num_sequences is not None else int(va_grid_len))
    dim_values = torch.as_tensor(
        [
            float(model_param_dim(rules, subsets, int(num_knots)))
            if bool(penalize_kernel_df)
            else float(1 + len(rules))
            for rules in active_rule_lists
        ],
        dtype=torch.float64,
        device=device,
    )
    final_bic_values = (
        -2.0 * ll_val + float(penalty_scale) * dim_values * math.log(float(n_eff))
    ).detach().cpu().numpy().astype(np.float64, copy=False)
    score_only_heights_by_row: list[dict[tuple[int, int], np.ndarray]] | None = None
    if bool(score_only):
        normalized_np = _normalize_height_rows_torch(raw).detach().cpu().numpy().astype(np.float64, copy=False)
        score_only_heights_by_row = [dict(base_rule_heights) for _ in range(batch_count)]
        for entry_id, batch_row in enumerate(entry_support):
            key = (int(entry_rule_idx[int(entry_id)]), int(entry_source[int(entry_id)]))
            score_only_heights_by_row[int(batch_row)][key] = normalized_np[int(entry_id)].copy()

    out: dict[SupportKey, SupportEvalResult] = {}
    for batch_row, rules in enumerate(active_rule_lists):
        exc_params: dict[int, float] = {}
        inh_params: dict[int, float] = {}
        for rule_row, rule in enumerate(rules):
            beta_val = float(beta_final[int(batch_row), int(rule_row)].detach().cpu().item())
            if str(rule.sign) == "exc":
                exc_params[int(rule.idx)] = beta_val
            else:
                inh_params[int(rule.idx)] = beta_val
        if bool(score_only):
            if score_only_heights_by_row is None:
                raise RuntimeError("score-only refit did not materialize optimized kernel heights")
            trial_heights = score_only_heights_by_row[int(batch_row)]
            arrays_fit = None
        else:
            trial_heights = dict(base_rule_heights)
            for key, h in (final_state["heights_out"][int(batch_row)] or {}).items():
                trial_heights[key] = np.asarray(h, dtype=np.float64)
            arrays_fit = final_state["arrays_out"][int(batch_row)] or {}
        out[support_key_from_rules(rules)] = SupportEvalResult(
            bic=float(final_bic_values[int(batch_row)]),
            mu=float(mu_final[int(batch_row)].detach().cpu().item()),
            exc_params=exc_params,
            inh_params=inh_params,
            rule_heights=trial_heights,
            arrays_out=arrays_fit,
            active_rules=list(rules),
        )
    return out


def fit_constant_bic(
    n_events_val: int,
    grid_weights_val: np.ndarray,
    n_events_train: int,
    grid_weights_train: np.ndarray,
    *,
    n_val_sequences: int,
):
    mu = base_rate_fit(n_events_train, float(np.sum(grid_weights_train)))
    ll_val = float(n_events_val) * math.log(max(mu, 1e-8)) - float(np.sum(grid_weights_val)) * mu
    bic = -2.0 * ll_val + math.log(float(bic_sample_size(int(n_val_sequences))))
    return float(bic), float(mu)


def validation_bic_from_arrays(
    *,
    active_rules: list[ActiveRule],
    arrays_out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    exc_params: dict[int, float],
    inh_params: dict[int, float],
    mu: float,
    subsets,
    num_knots: int,
    grid_weights_val: np.ndarray,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    num_sequences_val: int | None = None,
) -> float:
    if not active_rules:
        raise ValueError("validation_bic_from_arrays requires a non-empty active_rules list")

    va_event_len = int(next(iter(arrays_out.values()))[2].size)
    exc_ev = np.zeros((va_event_len,), dtype=np.float64)
    inh_ev = np.zeros((va_event_len,), dtype=np.float64)
    exc_gr = np.zeros((grid_weights_val.size,), dtype=np.float64)
    inh_gr = np.zeros((grid_weights_val.size,), dtype=np.float64)
    for ar in active_rules:
        arr = arrays_out[int(ar.idx)]
        if ar.sign == "exc":
            coef = float(exc_params.get(int(ar.idx), 0.0))
            exc_ev += coef * arr[2]
            exc_gr += coef * arr[3]
        else:
            coef = float(inh_params.get(int(ar.idx), 0.0))
            inh_ev += coef * arr[2]
            inh_gr += coef * arr[3]
    ll_val = float(
        va_event_len * math.log(max(float(mu), 1e-8))
        + np.sum(exc_ev)
        - np.sum(inh_ev)
        - np.dot(grid_weights_val, float(mu) * np.exp(np.clip(exc_gr - inh_gr, -40.0, 40.0)))
    )
    n_eff = bic_sample_size(int(num_sequences_val) if num_sequences_val is not None else int(grid_weights_val.size))
    return float(
        -2.0 * ll_val
        + (
            float(penalty_scale) * float(model_param_dim(active_rules, subsets, int(num_knots))) * math.log(float(n_eff))
            if penalize_kernel_df
            else float(penalty_scale) * float(1 + len(active_rules)) * math.log(float(n_eff))
        )
    )


def evaluate_support_exact(
    *,
    active_rules: list[ActiveRule],
    subsets,
    basis_cache: SourceBasisCache,
    base_rule_heights: dict[tuple[int, int], np.ndarray],
    template_rule_heights: dict[str, dict[tuple[int, int], np.ndarray]] | None,
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None,
    opt_steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    num_val_sequences: int | None = None,
    kernel_smoothness_ridge: float = 0.0,
    kernel_normalization: str = "peak",
    support_cache: dict[ExactSupportCacheKey, SupportEvalResult] | None = None,
    warm_start_result: SupportEvalResult | None = None,
) -> SupportEvalResult:
    kernel_normalization = str(kernel_normalization)
    if kernel_normalization not in {"peak", "area"}:
        raise ValueError(f"unknown kernel normalization mode: {kernel_normalization}")
    target_rules = sorted(active_rules, key=lambda ar: (int(ar.idx), str(ar.sign)))
    target_key = exact_support_cache_key(
        active_rules=target_rules,
        basis_cache=basis_cache,
        base_rule_heights=base_rule_heights,
        template_rule_heights=template_rule_heights,
        grid_weights_train=grid_weights_train,
        grid_weights_val=grid_weights_val,
        opt_steps=int(opt_steps),
        lr=float(lr),
        penalize_kernel_df=bool(penalize_kernel_df),
        penalty_scale=float(penalty_scale),
        kernel_smoothness_ridge=float(kernel_smoothness_ridge),
        num_val_sequences=num_val_sequences,
        kernel_normalization=kernel_normalization,
    )
    # A finite-step refit is not invariant to warm-start parameters/heights.
    # Cache only cold-start support refits, and bind them to the exact
    # full/validation context plus current base height map.
    cache_enabled = support_cache is not None and warm_start_result is None
    if cache_enabled:
        cached = support_cache.get(target_key)
        if cached is not None:
            return cached

    if not target_rules:
        n_train = int(basis_cache.train_event_times.size)
        n_val = int(basis_cache.val_event_times.size)
        bic, mu = fit_constant_bic(
            n_events_val=n_val,
            grid_weights_val=grid_weights_val,
            n_events_train=n_train,
            grid_weights_train=grid_weights_train,
            n_val_sequences=int(num_val_sequences if num_val_sequences is not None else 1),
        )
        result = SupportEvalResult(
            bic=float(bic),
            mu=float(mu),
            exc_params={},
            inh_params={},
            rule_heights=base_rule_heights,
            arrays_out=None,
            active_rules=[],
        )
        if cache_enabled:
            support_cache[target_key] = minimal_cache_result(result)
        return result

    init_rule_heights, init_coef_map = _build_support_warm_start(
        target_rules=target_rules,
        subsets=subsets,
        base_rule_heights=base_rule_heights,
        template_rule_heights=template_rule_heights,
        warm_start_result=warm_start_result,
        fallback=0.1,
    )
    bic, mu_fit, exc_fit, inh_fit, trial_heights, arrays_fit = optimize_active_set_torch(
        active_rules=target_rules,
        subsets=subsets,
        basis_cache=basis_cache,
        rule_heights=init_rule_heights,
        init_coef_map=init_coef_map,
        grid_weights_train=grid_weights_train,
        grid_weights_val=grid_weights_val,
        device=device,
        torch_basis_cache=torch_basis_cache,
        steps=int(opt_steps),
        lr=float(lr),
        penalize_kernel_df=bool(penalize_kernel_df),
        penalty_scale=float(penalty_scale),
        bic_num_sequences=num_val_sequences,
        kernel_smoothness_ridge=float(kernel_smoothness_ridge),
        kernel_normalization=kernel_normalization,
    )
    result = SupportEvalResult(
        bic=float(bic),
        mu=float(mu_fit),
        exc_params=dict(exc_fit),
        inh_params=dict(inh_fit),
        rule_heights=trial_heights,
        arrays_out=arrays_fit,
        active_rules=list(target_rules),
    )
    if cache_enabled:
        support_cache[target_key] = minimal_cache_result(result)
    return result


def evaluate_support_exact_batch(
    *,
    support_rule_lists: list[list[ActiveRule]],
    subsets,
    basis_cache: SourceBasisCache,
    base_rule_heights: dict[tuple[int, int], np.ndarray],
    template_rule_heights: dict[str, dict[tuple[int, int], np.ndarray]] | None,
    grid_weights_train: np.ndarray,
    grid_weights_val: np.ndarray,
    device: torch.device,
    torch_basis_cache: TorchBasisCache | None,
    opt_steps: int,
    lr: float,
    penalize_kernel_df: bool,
    penalty_scale: float = 1.0,
    num_val_sequences: int | None = None,
    kernel_smoothness_ridge: float = 0.0,
    support_cache: dict[ExactSupportCacheKey, SupportEvalResult] | None = None,
    warm_start_result: SupportEvalResult | None = None,
    score_only: bool = False,
    exact_batch_size: int = 8,
) -> dict[SupportKey, SupportEvalResult]:
    out: dict[SupportKey, SupportEvalResult] = {}
    pending_by_size: dict[int, list[list[ActiveRule]]] = {}
    cache_enabled = support_cache is not None and warm_start_result is None
    # Implementation-only batching parameter: it changes GPU utilization, not
    # the evaluated support objective or accepted support path.
    exact_batch_size = max(1, int(exact_batch_size))
    for rules in support_rule_lists:
        rules = sort_unique_sign_exclusive_rules(list(rules))
        target_support_key = support_key_from_rules(rules)
        if target_support_key in out:
            continue
        cache_key = exact_support_cache_key(
            active_rules=rules,
            basis_cache=basis_cache,
            base_rule_heights=base_rule_heights,
            template_rule_heights=template_rule_heights,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            opt_steps=int(opt_steps),
            lr=float(lr),
            penalize_kernel_df=bool(penalize_kernel_df),
            penalty_scale=float(penalty_scale),
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            num_val_sequences=num_val_sequences,
        )
        if cache_enabled:
            cached = support_cache.get(cache_key)
            if cached is not None:
                out[target_support_key] = cached
                continue
        pending_by_size.setdefault(int(len(rules)), []).append(rules)

    def eval_serial(rules: list[ActiveRule]) -> SupportEvalResult:
        return evaluate_support_exact(
            active_rules=rules,
            subsets=subsets,
            basis_cache=basis_cache,
            base_rule_heights=base_rule_heights,
            template_rule_heights=template_rule_heights,
            grid_weights_train=grid_weights_train,
            grid_weights_val=grid_weights_val,
            device=device,
            torch_basis_cache=torch_basis_cache,
            opt_steps=int(opt_steps),
            lr=float(lr),
            penalize_kernel_df=bool(penalize_kernel_df),
            penalty_scale=float(penalty_scale),
            num_val_sequences=num_val_sequences,
            kernel_smoothness_ridge=float(kernel_smoothness_ridge),
            support_cache=support_cache,
            warm_start_result=warm_start_result,
        )

    for _support_size, rules_group in sorted(pending_by_size.items(), key=lambda item: int(item[0])):
        if len(rules_group) < 2 and not bool(score_only):
            for rules in rules_group:
                out[support_key_from_rules(rules)] = eval_serial(rules)
            continue
        for start in range(0, len(rules_group), int(exact_batch_size)):
            chunk = rules_group[start: start + int(exact_batch_size)]
            if len(chunk) < 2 and not bool(score_only):
                rules = chunk[0]
                out[support_key_from_rules(rules)] = eval_serial(rules)
                continue
            chunk_results = optimize_active_set_torch_batch_same_size(
                active_rule_lists=chunk,
                subsets=subsets,
                basis_cache=basis_cache,
                base_rule_heights=base_rule_heights,
                template_rule_heights=template_rule_heights,
                grid_weights_train=grid_weights_train,
                grid_weights_val=grid_weights_val,
                device=device,
                torch_basis_cache=torch_basis_cache,
                steps=int(opt_steps),
                lr=float(lr),
                penalize_kernel_df=bool(penalize_kernel_df),
                penalty_scale=float(penalty_scale),
                bic_num_sequences=num_val_sequences,
                kernel_smoothness_ridge=float(kernel_smoothness_ridge),
                warm_start_result=warm_start_result,
                score_only=bool(score_only),
            )
            out.update(chunk_results)
            if cache_enabled:
                for rules in chunk:
                    result = chunk_results.get(support_key_from_rules(rules))
                    if result is None:
                        continue
                    cache_key = exact_support_cache_key(
                        active_rules=rules,
                        basis_cache=basis_cache,
                        base_rule_heights=base_rule_heights,
                        template_rule_heights=template_rule_heights,
                        grid_weights_train=grid_weights_train,
                        grid_weights_val=grid_weights_val,
                        opt_steps=int(opt_steps),
                        lr=float(lr),
                        penalize_kernel_df=bool(penalize_kernel_df),
                        penalty_scale=float(penalty_scale),
                        kernel_smoothness_ridge=float(kernel_smoothness_ridge),
                        num_val_sequences=num_val_sequences,
                    )
                    support_cache[cache_key] = minimal_cache_result(result)
    return out
