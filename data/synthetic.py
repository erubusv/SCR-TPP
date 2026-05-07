"""Synthetic data generator for the active logical TPP benchmark line.

Supported target intensity:
    canonical log-link:
        lambda_k(t) = b_k * exp(E_k(t) - I_k(t)) + eps

where
    E_k(t) = sum_{r:target(r)=k} W_pos[r] * p_r(t)
    I_k(t) = sum_{r:target(r)=k} W_neg[r] * p_r(t)

Supported rule activation:
    product_max_witness:
        p_r(t) = prod_{src in cond(r)} max_{i:x_i=src,t_i<t} g_{r,src}(t-t_i)

The `product_max_witness` mode treats each rule-source kernel as a
pointwise-bounded witness score with peak normalized to 1:

    0 <= g_{r,s}(tau) <= 1,   max_tau g_{r,s}(tau) = 1.

A source contributes only its strongest historical witness, so repeated
occurrences of the same source do not accumulate.
"""

import os
import math
import numpy as np
from collections import defaultdict
from numba import njit
from tqdm import tqdm


@njit(fastmath=True, cache=True, inline='always')
def triangular_kernel_func(dt, peak, width, mix_weight):
    """Asymmetric triangular kernel on [0, width] with peak at `peak`."""
    if dt <= 0.0 or dt > width:
        return 0.0

    eps = 1e-12
    if peak <= eps:
        return mix_weight * (width - dt) / max(width, eps)
    if peak >= width - eps:
        return mix_weight * (dt / max(width, eps))

    if dt <= peak:
        return mix_weight * (dt / peak)
    return mix_weight * ((width - dt) / (width - peak))


@njit(fastmath=True, cache=True, inline='always')
def gaussian_kernel_func(dt, peak, sigma, mix_weight, support_mult):
    """Causal truncated Gaussian with peak at `peak` and std `sigma`.

    The kernel is zero outside [0, peak + support_mult * sigma].
    Width in config is interpreted as sigma for gaussian kernels.
    """
    if dt <= 0.0:
        return 0.0

    eps = 1e-12
    sigma = max(sigma, eps)
    support = peak + max(support_mult, 0.0) * sigma
    if dt > support:
        return 0.0
    z = (dt - peak) / sigma
    return mix_weight * np.exp(-0.5 * z * z)


@njit(fastmath=True, cache=True, inline='always')
def exponential_kernel_func(dt, peak, tau, mix_weight, support_mult):
    """Shifted truncated exponential with onset/peak at `peak` and scale `tau`.

    The kernel is zero outside [peak, peak + support_mult * tau].
    Width in config is interpreted as tau for exponential kernels.
    """
    if dt < peak:
        return 0.0

    eps = 1e-12
    tau = max(tau, eps)
    support = peak + max(support_mult, 0.0) * tau
    if dt > support:
        return 0.0
    z = (dt - peak) / tau
    return mix_weight * np.exp(-z)


@njit(fastmath=True, cache=True, inline='always')
def flat_kernel_func(dt, width, mix_weight):
    """Flat piecewise-linear kernel on [0, width]."""
    if dt <= 0.0 or dt > width:
        return 0.0
    return mix_weight / max(width, 1e-12)


class Rule:
    """Rule container used by the synthetic generator."""

    def __init__(
        self,
        rule_id,
        target,
        W_pos,
        W_neg,
        condition,
        kernel_params,
        kernel_type='triangular',
        bias=0.0,
        activation_mode='product_max_witness',
        softmin_tau=0.15,
    ):
        self.rule_id = int(rule_id)
        self.target = int(target)
        self.W_pos = max(0.0, float(W_pos))
        self.W_neg = max(0.0, float(W_neg))
        self.condition = [int(c) for c in condition]
        self.kernel_type = kernel_type
        self.bias = float(bias)
        self.activation_mode = str(activation_mode)
        self.softmin_tau = max(float(softmin_tau), 1e-6)

        self.kernel_params_arrays = {}
        self.max_support = {}
        for src_type, params in kernel_params.items():
            src_type = int(src_type)
            params_array = np.array(params, dtype=np.float64)
            if params_array.shape[0] == 0:
                params_array = np.array([[0.5, 1.0, 1.0, 3.0]], dtype=np.float64)
            if params_array.shape[1] == 3:
                support_mult = np.full((params_array.shape[0], 1), 3.0, dtype=np.float64)
                params_array = np.concatenate([params_array, support_mult], axis=1)

            # Ensure peak <= width to keep triangular shape valid.
            params_array[:, 1] = np.maximum(params_array[:, 1], 1e-6)  # width or sigma
            if self.kernel_type == 'triangular':
                params_array[:, 0] = np.clip(params_array[:, 0], 1e-6, params_array[:, 1] - 1e-6)  # peak
            elif self.kernel_type == 'flat':
                params_array[:, 0] = 0.5 * params_array[:, 1]
            else:
                params_array[:, 0] = np.maximum(params_array[:, 0], 1e-6)
            params_array[:, 3] = np.maximum(params_array[:, 3], 0.0)

            mix_weights = np.maximum(params_array[:, 2], 0.0)
            if self.activation_mode == 'product_max_witness':
                # Max-witness kernels are peak-normalized.
                if params_array.shape[0] != 1:
                    raise NotImplementedError(
                        "product_max_witness synthetic generation currently "
                        "supports exact peak normalization only for single-"
                        "component rule-source kernels"
                    )
                params_array[:, 2] = 1.0
            else:
                mix_sum = mix_weights.sum()
                if mix_sum > 0:
                    params_array[:, 2] = mix_weights / mix_sum
                else:
                    params_array[:, 2] = mix_weights

            self.kernel_params_arrays[src_type] = params_array
            if self.kernel_type == 'triangular':
                self.max_support[src_type] = float(np.max(params_array[:, 1]))
            else:
                self.max_support[src_type] = float(np.max(params_array[:, 0] + params_array[:, 3] * params_array[:, 1]))

    def _kernel_value(self, dt, params):
        total = 0.0
        for i in range(len(params)):
            peak, width, mix_weight, support_mult = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
            if self.kernel_type == 'triangular':
                val = triangular_kernel_func(dt, peak, width, mix_weight)
            elif self.kernel_type == 'flat':
                val = flat_kernel_func(dt, width, mix_weight)
            elif self.kernel_type == 'gaussian':
                val = gaussian_kernel_func(dt, peak, width, mix_weight, support_mult)
            elif self.kernel_type == 'exponential':
                val = exponential_kernel_func(dt, peak, width, mix_weight, support_mult)
            else:
                val = triangular_kernel_func(dt, peak, width, mix_weight)
            total += val
            if total >= 1.0:
                return 1.0
        if total <= 0.0:
            return 0.0
        if total >= 1.0:
            return 1.0
        return float(total)

    def compute_source_kernel_maxes(self, t, event_history_arrays, event_counts):
        source_maxes = []
        count_window = 0
        for src_type in self.condition:
            count = event_counts[src_type]
            if count == 0:
                source_maxes.append(0.0)
                continue

            max_support = self.max_support.get(src_type, 0.0)
            idx = count - 1
            params = self.kernel_params_arrays.get(src_type)
            if params is None:
                source_maxes.append(0.0)
                continue

            src_max = 0.0
            while idx >= 0:
                t_event = event_history_arrays[src_type, idx]
                dt = t - t_event
                if dt <= 0.0:
                    idx -= 1
                    continue
                if dt > max_support:
                    break

                count_window += 1
                src_max = max(src_max, self._kernel_value(dt, params))
                idx -= 1
                if src_max >= 1.0:
                    src_max = 1.0
                    break
            if src_max <= 0.0:
                source_maxes.append(0.0)
            elif src_max >= 1.0:
                source_maxes.append(1.0)
            else:
                source_maxes.append(float(src_max))
        return source_maxes, count_window

    def _kernel_component_future_max(self, dt, peak, width, mix_weight, support_mult):
        dt = float(dt)
        peak = float(peak)
        width = float(width)
        mix_weight = float(mix_weight)
        support_mult = float(support_mult)

        if self.kernel_type == 'triangular':
            if dt <= 0.0:
                return max(mix_weight, 0.0)
            if dt > width:
                return 0.0
            if dt <= peak:
                return max(mix_weight, 0.0)
            return max(triangular_kernel_func(dt, peak, width, mix_weight), 0.0)

        if self.kernel_type == 'flat':
            if dt <= 0.0:
                return max(flat_kernel_func(0.5 * width, width, mix_weight), 0.0)
            if dt > width:
                return 0.0
            return max(flat_kernel_func(dt, width, mix_weight), 0.0)

        if self.kernel_type == 'gaussian':
            support = peak + max(support_mult, 0.0) * width
            if dt <= 0.0:
                return max(mix_weight, 0.0)
            if dt > support:
                return 0.0
            if dt <= peak:
                return max(mix_weight, 0.0)
            return max(gaussian_kernel_func(dt, peak, width, mix_weight, support_mult), 0.0)

        if self.kernel_type == 'exponential':
            support = peak + max(support_mult, 0.0) * width
            if dt <= peak:
                return max(mix_weight, 0.0)
            if dt > support:
                return 0.0
            return max(exponential_kernel_func(dt, peak, width, mix_weight, support_mult), 0.0)

        if dt <= 0.0:
            return max(mix_weight, 0.0)
        if dt > width:
            return 0.0
        if dt <= peak:
            return max(mix_weight, 0.0)
        return max(triangular_kernel_func(dt, peak, width, mix_weight), 0.0)

    def future_source_kernel_max_upper_bound(self, t, event_history_arrays, event_counts):
        source_upper_maxes = []
        count_window = 0
        for src_type in self.condition:
            count = event_counts[src_type]
            if count == 0:
                source_upper_maxes.append(0.0)
                continue

            max_support = self.max_support.get(src_type, 0.0)
            idx = count - 1
            params = self.kernel_params_arrays.get(src_type)
            if params is None:
                source_upper_maxes.append(0.0)
                continue

            src_upper = 0.0
            while idx >= 0:
                t_event = event_history_arrays[src_type, idx]
                dt = t - t_event
                if dt <= 0.0:
                    idx -= 1
                    continue
                if dt > max_support:
                    break

                count_window += 1
                event_upper = 0.0
                for i in range(len(params)):
                    peak, width, mix_weight, support_mult = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
                    event_upper += self._kernel_component_future_max(dt, peak, width, mix_weight, support_mult)
                if event_upper >= 1.0:
                    src_upper = 1.0
                    break
                if event_upper > src_upper:
                    src_upper = float(event_upper)
                idx -= 1
            if src_upper <= 0.0:
                source_upper_maxes.append(0.0)
            elif src_upper >= 1.0:
                source_upper_maxes.append(1.0)
            else:
                source_upper_maxes.append(float(src_upper))
        return source_upper_maxes, count_window

    def _kernel_component_interval_max(self, dt_lo, dt_hi, peak, width, mix_weight, support_mult):
        dt_lo = float(dt_lo)
        dt_hi = float(dt_hi)
        peak = float(peak)
        width = float(width)
        mix_weight = float(mix_weight)
        support_mult = float(support_mult)

        if dt_hi <= dt_lo:
            return 0.0

        if self.kernel_type == 'triangular':
            left = max(dt_lo, 0.0)
            right = min(dt_hi, width)
            if right <= left:
                return 0.0
            if left <= peak <= right:
                return max(mix_weight, 0.0)
            return max(
                triangular_kernel_func(left, peak, width, mix_weight),
                triangular_kernel_func(right, peak, width, mix_weight),
                0.0,
            )

        if self.kernel_type == 'flat':
            left = max(dt_lo, 0.0)
            right = min(dt_hi, width)
            if right <= left:
                return 0.0
            return max(flat_kernel_func(0.5 * (left + right), width, mix_weight), 0.0)

        if self.kernel_type == 'gaussian':
            support = peak + max(support_mult, 0.0) * width
            left = max(dt_lo, 0.0)
            right = min(dt_hi, support)
            if right <= left:
                return 0.0
            if left <= peak <= right:
                return max(mix_weight, 0.0)
            return max(
                gaussian_kernel_func(left, peak, width, mix_weight, support_mult),
                gaussian_kernel_func(right, peak, width, mix_weight, support_mult),
                0.0,
            )

        if self.kernel_type == 'exponential':
            support = peak + max(support_mult, 0.0) * width
            if dt_hi <= peak:
                return 0.0
            left = max(dt_lo, peak)
            right = min(dt_hi, support)
            if right <= left:
                return 0.0
            if dt_lo <= peak < dt_hi:
                return max(mix_weight, 0.0)
            return max(exponential_kernel_func(left, peak, width, mix_weight, support_mult), 0.0)

        left = max(dt_lo, 0.0)
        right = min(dt_hi, width)
        if right <= left:
            return 0.0
        if left <= peak <= right:
            return max(mix_weight, 0.0)
        return max(
            triangular_kernel_func(left, peak, width, mix_weight),
            triangular_kernel_func(right, peak, width, mix_weight),
            0.0,
        )

    def interval_source_kernel_max_upper_bound(self, t_lo, t_hi, event_history_arrays, event_counts):
        source_upper_maxes = []
        count_window = 0
        t_lo = float(t_lo)
        t_hi = float(t_hi)
        for src_type in self.condition:
            count = event_counts[src_type]
            if count == 0:
                source_upper_maxes.append(0.0)
                continue

            max_support = self.max_support.get(src_type, 0.0)
            idx = count - 1
            params = self.kernel_params_arrays.get(src_type)
            if params is None:
                source_upper_maxes.append(0.0)
                continue

            src_upper = 0.0
            while idx >= 0:
                t_event = event_history_arrays[src_type, idx]
                dt_lo = t_lo - t_event
                dt_hi = t_hi - t_event
                if dt_hi <= 0.0:
                    idx -= 1
                    continue
                if dt_lo > max_support:
                    break

                count_window += 1
                event_upper = 0.0
                for i in range(len(params)):
                    peak, width, mix_weight, support_mult = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
                    event_upper += self._kernel_component_interval_max(dt_lo, dt_hi, peak, width, mix_weight, support_mult)
                if event_upper >= 1.0:
                    src_upper = 1.0
                    break
                if event_upper > src_upper:
                    src_upper = float(event_upper)
                idx -= 1
            if src_upper <= 0.0:
                source_upper_maxes.append(0.0)
            elif src_upper >= 1.0:
                source_upper_maxes.append(1.0)
            else:
                source_upper_maxes.append(float(src_upper))
        return source_upper_maxes, count_window

    def compute_activation(self, t, event_history_arrays, event_counts):
        if self.activation_mode != 'product_max_witness':
            raise NotImplementedError(
                "Synthetic generation supports only activation_mode='product_max_witness' "
                f"but received {self.activation_mode!r}."
            )
        src_maxes, _ = self.compute_source_kernel_maxes(t, event_history_arrays, event_counts)
        if not src_maxes:
            return 0.0
        out = 1.0
        for src_max in src_maxes:
            if src_max <= 0.0:
                return 0.0
            if src_max < 1.0:
                out *= float(src_max)
        if out <= 0.0:
            return 0.0
        if out >= 1.0:
            return 1.0
        return float(out)

    def activation_upper_bound(self, t, event_history_arrays, event_counts):
        """Safe future upper bound for p_r(u), u >= t, used by thinning."""
        if self.activation_mode != 'product_max_witness':
            raise NotImplementedError(
                "Synthetic generation supports only activation_mode='product_max_witness' "
                f"but received {self.activation_mode!r}."
            )
        src_uppers, _ = self.future_source_kernel_max_upper_bound(t, event_history_arrays, event_counts)
        if not src_uppers:
            return 0.0
        out = 1.0
        for src_upper in src_uppers:
            if src_upper <= 0.0:
                return 0.0
            if src_upper < 1.0:
                out *= float(src_upper)
        if out <= 0.0:
            return 0.0
        if out >= 1.0:
            return 1.0
        return float(out)

    def activation_interval_upper_bound(self, t_lo, t_hi, event_history_arrays, event_counts):
        """Safe upper bound for p_r(u) on the interval [t_lo, t_hi]."""
        if float(t_hi) <= float(t_lo):
            return 0.0
        if self.activation_mode == 'product_max_witness':
            src_uppers, _ = self.interval_source_kernel_max_upper_bound(t_lo, t_hi, event_history_arrays, event_counts)
            if not src_uppers:
                return 0.0
            out = 1.0
            for src_upper in src_uppers:
                if src_upper <= 0.0:
                    return 0.0
                if src_upper < 1.0:
                    out *= float(src_upper)
            if out <= 0.0:
                return 0.0
            if out >= 1.0:
                return 1.0
            return float(out)
        return self.activation_upper_bound(t_lo, event_history_arrays, event_counts)


def _single_target_fast_path_info(rules, event_types):
    """Return exact single-target fast-path info when benchmark structure allows it.

    Fast path is valid when:
    - every rule points to the same target
    - that target is never used as a source
    - all other event types have no incoming rules (baseline-only sources)
    """
    if not rules:
        return None

    targets = {int(rule.target) for rule in rules}
    if len(targets) != 1:
        return None

    target = int(next(iter(targets)))
    for rule in rules:
        if target in set(int(src) for src in rule.condition):
            return None

    return {
        "target": target,
        "non_target_types": [int(k) for k in event_types if int(k) != target],
        "target_rules": [rule for rule in rules if int(rule.target) == target],
    }


def _sample_homogeneous_poisson_times(rate, time_horizon, rng):
    """Exact homogeneous Poisson process on [0, time_horizon] via count + sorted uniforms."""
    rate = float(rate)
    if rate <= 0.0:
        return np.empty(0, dtype=np.float64)
    n = int(rng.poisson(rate * float(time_horizon)))
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    return np.sort(rng.uniform(0.0, float(time_horizon), size=n)).astype(np.float64)


def _build_fixed_history_arrays(event_types, source_times_by_type):
    """Pack pre-generated source histories into the dense arrays expected by Rule helpers."""
    max_type = max(int(k) for k in event_types)
    max_count = 1
    for arr in source_times_by_type.values():
        max_count = max(max_count, int(len(arr)))
    event_history = np.zeros((max_type + 1, max_count), dtype=np.float64)
    full_counts = np.zeros(max_type + 1, dtype=np.int32)
    for event_type, times in source_times_by_type.items():
        event_type = int(event_type)
        count = int(len(times))
        if count > 0:
            event_history[event_type, :count] = times
            full_counts[event_type] = count
    return event_history, full_counts


def _past_event_counts_at_time(source_times_by_type, max_type, t):
    """Counts of source events with timestamp < t for each event type."""
    counts = np.zeros(max_type + 1, dtype=np.int32)
    t = float(t)
    for event_type, times in source_times_by_type.items():
        counts[int(event_type)] = int(np.searchsorted(times, t, side='left'))
    return counts


def _advance_past_event_counts_in_place(source_times_by_type, counts, t):
    """Advance monotone source-event counts to the new candidate time t."""
    t = float(t)
    for event_type, times in source_times_by_type.items():
        event_type = int(event_type)
        idx = int(counts[event_type])
        n = int(len(times))
        while idx < n and float(times[idx]) < t:
            idx += 1
        counts[event_type] = idx
    return counts


def _advance_past_event_counts_through_time_in_place(source_times_by_type, counts, t):
    """Advance counts to include source events with timestamp <= t."""
    t = float(t)
    for event_type, times in source_times_by_type.items():
        counts[int(event_type)] = int(np.searchsorted(times, t, side='right'))
    return counts


def _next_source_event_time(source_times_by_type, counts, t_curr, time_horizon):
    """Earliest source-event time strictly greater than t_curr, else time_horizon."""
    next_t = float(time_horizon)
    t_curr = float(t_curr)
    for event_type, times in source_times_by_type.items():
        idx = int(counts[int(event_type)])
        n = int(len(times))
        while idx < n and float(times[idx]) <= t_curr:
            idx += 1
        if idx < n:
            next_t = min(next_t, float(times[idx]))
    return float(next_t)


def _proxy_target_cause_label(target_rules, t, event_history_arrays, event_counts, rng, precomputed_activations=None):
    """Auxiliary label for canonical log-link data.

    There is no exact branching decomposition for the log-link model, so we keep the
    existing baseline-plus-positive-factor proxy labeling used by the benchmark.
    """
    num_causes = 1 + len(target_rules)
    cause_weights = np.zeros(num_causes, dtype=np.float64)
    cause_ids = np.zeros(num_causes, dtype=np.int32)
    cause_weights[0] = 1.0
    cause_ids[0] = -1
    for idx, rule in enumerate(target_rules, start=1):
        if precomputed_activations is None:
            p_r = rule.compute_activation(t, event_history_arrays, event_counts)
        else:
            p_r = float(precomputed_activations[idx - 1])
        cause_weights[idx] = max(0.0, math.exp(min(rule.W_pos * p_r, 40.0)) - 1.0)
        cause_ids[idx] = rule.rule_id
    cause_sum = float(np.sum(cause_weights))
    if cause_sum <= 1e-10:
        return -1
    cause_probs = cause_weights / cause_sum
    cause_idx = int(rng.choice(num_causes, p=cause_probs))
    return int(cause_ids[cause_idx])


def _generate_single_target_canonical_loglink_data(
    rules,
    num_samples,
    time_horizon,
    base_intensities,
    max_len=1024,
    eps=1e-8,
    seed=None,
):
    """Exact fast path for the benchmark structure with one target and baseline-only sources.

    Source event types are independent homogeneous Poisson processes.
    Conditional on those source histories, the target is a one-dimensional NHPP with
    deterministic intensity lambda_T(t)=b_T exp(E_T(t)-I_T(t)).
    """
    data = []
    max_len_cap = None if max_len is None else int(max_len)
    if max_len_cap is not None and max_len_cap <= 0:
        max_len_cap = None

    event_types = sorted(int(k) for k in base_intensities.keys())
    fast = _single_target_fast_path_info(rules, event_types)
    if fast is None:
        raise ValueError("single-target canonical fast path called on incompatible rule structure")

    target = int(fast["target"])
    target_rules = list(fast["target_rules"])
    non_target_types = list(fast["non_target_types"])
    max_type = max(int(k) for k in event_types)
    b_target = max(float(base_intensities[target]), float(eps))
    use_interval_thinning = all(str(rule.activation_mode) == 'product_max_witness' for rule in target_rules)

    rng = np.random.default_rng(seed)
    disable_tqdm = str(os.environ.get("TQDM_DISABLE", "")).lower() in {"1", "true", "yes", "y"}

    for _ in tqdm(range(num_samples), leave=False, disable=disable_tqdm):
        source_times_by_type = {}
        for event_type in non_target_types:
            source_times_by_type[int(event_type)] = _sample_homogeneous_poisson_times(
                rate=float(base_intensities[event_type]),
                time_horizon=float(time_horizon),
                rng=rng,
            )

        event_history, full_counts = _build_fixed_history_arrays(event_types, source_times_by_type)

        target_times = []
        target_labels = []
        t_curr = 0.0
        counts_curr = np.zeros(max_type + 1, dtype=np.int32)
        while t_curr < float(time_horizon):
            interval_end = float(time_horizon)
            if use_interval_thinning:
                interval_end = _next_source_event_time(
                    source_times_by_type=source_times_by_type,
                    counts=counts_curr,
                    t_curr=t_curr,
                    time_horizon=time_horizon,
                )

            e_bound = 0.0
            for rule in target_rules:
                if use_interval_thinning:
                    p_bound = rule.activation_interval_upper_bound(t_curr, interval_end, event_history, counts_curr)
                else:
                    p_bound = rule.activation_upper_bound(t_curr, event_history, counts_curr)
                e_bound += rule.W_pos * float(p_bound)
            lambda_max = b_target * math.exp(min(float(e_bound), 40.0)) + float(eps)
            lambda_max = max(lambda_max, 0.1)

            dt = -np.log(rng.random()) / lambda_max
            t_cand = t_curr + dt
            if t_cand > interval_end:
                if interval_end >= float(time_horizon):
                    break
                t_curr = interval_end
                counts_curr = _advance_past_event_counts_through_time_in_place(source_times_by_type, counts_curr, t_curr)
                continue

            if not use_interval_thinning:
                counts_curr = _advance_past_event_counts_in_place(source_times_by_type, counts_curr, t_cand)
            E_t = 0.0
            I_t = 0.0
            p_vals = []
            for rule in target_rules:
                p_r = rule.compute_activation(t_cand, event_history, counts_curr)
                p_vals.append(float(p_r))
                E_t += rule.W_pos * p_r
                I_t += rule.W_neg * p_r
            eta_true = E_t - I_t
            if eta_true > 40.0:
                eta_true = 40.0
            elif eta_true < -40.0:
                eta_true = -40.0
            lambda_true = b_target * math.exp(eta_true) + float(eps)

            if rng.random() < (lambda_true / lambda_max):
                target_times.append(float(t_cand))
                target_labels.append(
                    _proxy_target_cause_label(
                        target_rules=target_rules,
                        t=t_cand,
                        event_history_arrays=event_history,
                        event_counts=counts_curr,
                        rng=rng,
                        precomputed_activations=p_vals,
                    )
                )
            t_curr = t_cand

        all_times = []
        all_events = []
        all_labels = []
        for event_type in non_target_types:
            times = source_times_by_type[event_type]
            if len(times) == 0:
                continue
            all_times.extend(times.tolist())
            all_events.extend([int(event_type)] * len(times))
            all_labels.extend([-1] * len(times))
        if target_times:
            all_times.extend(target_times)
            all_events.extend([target] * len(target_times))
            all_labels.extend(target_labels)

        if not all_times:
            continue

        order = np.argsort(np.asarray(all_times, dtype=np.float64), kind='mergesort')
        if max_len_cap is not None and len(order) > max_len_cap:
            order = order[:max_len_cap]

        seq_times = np.asarray(all_times, dtype=np.float64)[order]
        seq_events = np.asarray(all_events, dtype=np.int32)[order]
        seq_labels = np.asarray(all_labels, dtype=np.int32)[order]

        data.append({
            'time': seq_times.tolist(),
            'event': seq_events.tolist(),
            'label': seq_labels.tolist(),
        })

    return data


def generate_canonical_loglink_data(rules, num_samples, time_horizon, base_intensities, max_len=1024, eps=1e-8, seed=None):
    """Generate synthetic event sequences for lambda(t)=b*exp(E-I)."""
    event_types = sorted(int(k) for k in base_intensities.keys())
    if _single_target_fast_path_info(rules, event_types) is not None:
        return _generate_single_target_canonical_loglink_data(
            rules=rules,
            num_samples=num_samples,
            time_horizon=time_horizon,
            base_intensities=base_intensities,
            max_len=max_len,
            eps=eps,
            seed=seed,
        )

    data = []
    max_len_cap = None if max_len is None else int(max_len)
    if max_len_cap is not None and max_len_cap <= 0:
        max_len_cap = None

    rules_by_target = defaultdict(list)
    for rule in rules:
        rules_by_target[rule.target].append(rule)

    max_type = max(event_types)
    num_types = len(event_types)
    base_intensity_array = np.array([float(base_intensities[k]) for k in event_types], dtype=np.float64)

    initial_capacity = max_len_cap if max_len_cap is not None else 256
    event_history = np.zeros((max_type + 1, initial_capacity), dtype=np.float64)
    lambda_k_array = np.zeros(num_types, dtype=np.float64)

    rng = np.random.default_rng(seed)

    disable_tqdm = str(os.environ.get("TQDM_DISABLE", "")).lower() in {"1", "true", "yes", "y"}
    for _ in tqdm(range(num_samples), leave=False, disable=disable_tqdm):
        t_curr = 0.0
        event_history.fill(0.0)
        event_counts_dict = {k: 0 for k in event_types}

        full_sequence_times = np.zeros(initial_capacity, dtype=np.float64)
        full_sequence_events = np.zeros(initial_capacity, dtype=np.int32)
        full_sequence_labels = np.zeros(initial_capacity, dtype=np.int32)
        seq_len = 0

        while t_curr < time_horizon:
            if max_len_cap is not None and seq_len >= max_len_cap:
                break
            lambda_max = 0.0
            for k_idx, k in enumerate(event_types):
                b_k = max(base_intensity_array[k_idx], eps)
                e_bound = 0.0
                for rule in rules_by_target[k]:
                    p_bound = rule.activation_upper_bound(t_curr, event_history, event_counts_dict)
                    e_bound += rule.W_pos * float(p_bound)
                lambda_max += b_k * np.exp(min(e_bound, 40.0)) + eps

            if lambda_max < 0.1:
                lambda_max = 0.1

            u = rng.random()
            dt = -np.log(u) / lambda_max
            t_cand = t_curr + dt

            if t_cand > time_horizon:
                break

            lambda_k_array.fill(0.0)
            for k_idx, k in enumerate(event_types):
                b_k = max(base_intensity_array[k_idx], eps)
                E_k = 0.0
                I_k = 0.0

                for rule in rules_by_target[k]:
                    p_r = rule.compute_activation(t_cand, event_history, event_counts_dict)
                    E_k += rule.W_pos * p_r
                    I_k += rule.W_neg * p_r

                lambda_k_array[k_idx] = b_k * np.exp(np.clip(E_k - I_k, -40.0, 40.0)) + eps

            lambda_total = np.sum(lambda_k_array)
            if rng.random() >= lambda_total / lambda_max:
                t_curr = t_cand
                continue
            if lambda_total < 1e-10:
                t_curr = t_cand
                continue

            type_probs = lambda_k_array / lambda_total
            selected_type_idx = rng.choice(num_types, p=type_probs)
            selected_type = event_types[selected_type_idx]

            target_rules = rules_by_target[selected_type]
            num_causes = 1 + len(target_rules)

            cause_weights = np.zeros(num_causes, dtype=np.float64)
            cause_ids = np.zeros(num_causes, dtype=np.int32)

            # Auxiliary label only: no exact branching decomposition exists for
            # the log-link model, so we use baseline-plus-positive-factor proxy
            # weights. These labels are not used by the benchmark learner.
            cause_weights[0] = 1.0
            cause_ids[0] = -1

            for idx, rule in enumerate(target_rules, start=1):
                p_r = rule.compute_activation(t_cand, event_history, event_counts_dict)
                cause_weights[idx] = max(0.0, np.exp(min(rule.W_pos * p_r, 40.0)) - 1.0)
                cause_ids[idx] = rule.rule_id

            cause_sum = np.sum(cause_weights)
            if cause_sum > 1e-10:
                cause_probs = cause_weights / cause_sum
                cause_idx = rng.choice(num_causes, p=cause_probs)
                cause_rule = int(cause_ids[cause_idx])
            else:
                cause_rule = -1

            if seq_len >= full_sequence_times.shape[0]:
                new_capacity = max(full_sequence_times.shape[0] * 2, seq_len + 1)
                new_times = np.zeros(new_capacity, dtype=np.float64)
                new_events = np.zeros(new_capacity, dtype=np.int32)
                new_labels = np.zeros(new_capacity, dtype=np.int32)
                new_times[:seq_len] = full_sequence_times[:seq_len]
                new_events[:seq_len] = full_sequence_events[:seq_len]
                new_labels[:seq_len] = full_sequence_labels[:seq_len]
                full_sequence_times = new_times
                full_sequence_events = new_events
                full_sequence_labels = new_labels

            full_sequence_times[seq_len] = t_cand
            full_sequence_events[seq_len] = selected_type
            full_sequence_labels[seq_len] = cause_rule

            history_idx = event_counts_dict[selected_type]
            if history_idx >= event_history.shape[1]:
                new_capacity = max(event_history.shape[1] * 2, history_idx + 1)
                new_history = np.zeros((max_type + 1, new_capacity), dtype=np.float64)
                new_history[:, : event_history.shape[1]] = event_history
                event_history = new_history
            event_history[selected_type, history_idx] = t_cand
            event_counts_dict[selected_type] += 1

            seq_len += 1
            t_curr = t_cand

        if seq_len > 0:
            data.append({
                'time': full_sequence_times[:seq_len].tolist(),
                'event': full_sequence_events[:seq_len].tolist(),
                'label': full_sequence_labels[:seq_len].tolist(),
            })

    return data


def create_rules_from_config(config):
    """Create Rule objects from config dict."""
    rules = []

    default_activation_mode = config.get('activation_mode', 'product_max_witness')
    default_kernel_type = config.get('kernel', 'triangular')
    for rule_cfg in config.get('rules', []):
        rule_id = rule_cfg['id']
        target = rule_cfg['target']
        W_pos = rule_cfg.get('W_pos', 1.0)
        W_neg = rule_cfg.get('W_neg', 0.0)
        kernel_type = rule_cfg.get('kernel', default_kernel_type)
        bias = rule_cfg.get('bias', 0.0)
        activation_mode = rule_cfg.get('activation_mode', default_activation_mode)
        if str(activation_mode) != 'product_max_witness':
            raise NotImplementedError(
                f"Rule {rule_id} requests activation_mode={activation_mode!r}, "
                "but the active benchmark suite supports only 'product_max_witness'."
            )
        if abs(float(bias)) > 1e-12:
            raise ValueError(
                f"Rule {rule_id} sets bias={bias} under activation_mode={activation_mode!r}, "
                "but product_max_witness does not use bias."
            )

        condition = set(int(k) for k in rule_cfg['condition'].keys())

        kernel_params = {}
        for src_type, params in rule_cfg['condition'].items():
            src_type = int(src_type)
            peaks = params.get('peaks', [0.8])
            widths = params.get('widths', [1.6])
            mix_weights = params.get('mix_weights', [1.0])
            support_mults = params.get('support_mults', [3.0] * len(peaks))
            kernel_params[src_type] = list(zip(peaks, widths, mix_weights, support_mults))

        rules.append(
            Rule(
                rule_id=rule_id,
                target=target,
                W_pos=W_pos,
                W_neg=W_neg,
                condition=condition,
                kernel_params=kernel_params,
                kernel_type=kernel_type,
                bias=bias,
                activation_mode=activation_mode,
                softmin_tau=0.15,
            )
        )

    return rules
