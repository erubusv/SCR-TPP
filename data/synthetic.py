"""Synthetic data generator for logical TPP benchmarks.

Supported target intensities:
    multiplicative:
        lambda_k(t) = (b_k + E_k(t)) * exp(-I_k(t)) + eps
    canonical log-link:
        lambda_k(t) = b_k * exp(E_k(t) - I_k(t)) + eps

where
    E_k(t) = sum_{r:target(r)=k} W_pos[r] * p_r(t)
    I_k(t) = sum_{r:target(r)=k} W_neg[r] * p_r(t)

Supported rule activations:
    sum_relu:
        p_r(t) = ReLU(sum_{src in cond(r)} K_src(t) - bias_r)
    softmin_relu:
        p_r(t) = ReLU(softmin_{src in cond(r)} K_src(t) - bias_r)
    product_bounded:
        p_r(t) = prod_{src in cond(r)} (1 - exp(-K_src(t)))

The `product_bounded` mode is a smooth conjunction-style activation: all
sources must be active for the rule to contribute, but each source still
retains its own temporal kernel.
"""

import numpy as np
import os
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
    if dt <= peak:
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
        activation_mode='sum_relu',
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

            mix_weights = params_array[:, 2]
            mix_sum = mix_weights.sum()
            if mix_sum > 0:
                params_array[:, 2] = mix_weights / mix_sum

            self.kernel_params_arrays[src_type] = params_array
            if self.kernel_type == 'triangular':
                self.max_support[src_type] = float(np.max(params_array[:, 1]))
            else:
                self.max_support[src_type] = float(np.max(params_array[:, 0] + params_array[:, 3] * params_array[:, 1]))

    def compute_kernel_sum_and_count(self, t, event_history_arrays, event_counts):
        """Compute raw kernel sum and #events in support windows."""
        kernel_sum = 0.0
        count_window = 0

        for src_type in self.condition:
            count = event_counts[src_type]
            if count == 0:
                continue

            max_support = self.max_support.get(src_type, 0.0)
            idx = count - 1
            params = self.kernel_params_arrays.get(src_type)
            if params is None:
                continue

            while idx >= 0:
                t_event = event_history_arrays[src_type, idx]
                dt = t - t_event
                if dt <= 0.0:
                    idx -= 1
                    continue
                if dt > max_support:
                    break

                count_window += 1
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
                    kernel_sum += val

                idx -= 1

        return kernel_sum, count_window

    def compute_source_kernel_sums(self, t, event_history_arrays, event_counts):
        source_sums = []
        count_window = 0
        for src_type in self.condition:
            count = event_counts[src_type]
            if count == 0:
                source_sums.append(0.0)
                continue

            max_support = self.max_support.get(src_type, 0.0)
            idx = count - 1
            params = self.kernel_params_arrays.get(src_type)
            if params is None:
                source_sums.append(0.0)
                continue

            src_sum = 0.0
            while idx >= 0:
                t_event = event_history_arrays[src_type, idx]
                dt = t - t_event
                if dt <= 0.0:
                    idx -= 1
                    continue
                if dt > max_support:
                    break

                count_window += 1
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
                    src_sum += val
                idx -= 1
            source_sums.append(float(src_sum))
        return source_sums, count_window

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

    def future_source_kernel_sums_upper_bound(self, t, event_history_arrays, event_counts):
        source_upper_sums = []
        count_window = 0
        for src_type in self.condition:
            count = event_counts[src_type]
            if count == 0:
                source_upper_sums.append(0.0)
                continue

            max_support = self.max_support.get(src_type, 0.0)
            idx = count - 1
            params = self.kernel_params_arrays.get(src_type)
            if params is None:
                source_upper_sums.append(0.0)
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
                for i in range(len(params)):
                    peak, width, mix_weight, support_mult = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
                    src_upper += self._kernel_component_future_max(dt, peak, width, mix_weight, support_mult)
                idx -= 1

            source_upper_sums.append(float(src_upper))
        return source_upper_sums, count_window

    def _softmin(self, values):
        vals = np.asarray(values, dtype=np.float64)
        if vals.size == 0:
            return 0.0
        tau = self.softmin_tau
        z = -vals / tau
        zmax = np.max(z)
        lse = zmax + np.log(np.mean(np.exp(z - zmax)))
        return float(-tau * lse)

    def compute_activation(self, t, event_history_arrays, event_counts):
        if self.activation_mode == 'product_bounded':
            src_sums, _ = self.compute_source_kernel_sums(t, event_history_arrays, event_counts)
            if not src_sums:
                return 0.0
            out = 1.0
            for src_sum in src_sums:
                out *= (1.0 - np.exp(-max(0.0, float(src_sum))))
            return float(out)
        if self.activation_mode == 'softmin_relu':
            src_sums, _ = self.compute_source_kernel_sums(t, event_history_arrays, event_counts)
            agg = self._softmin(src_sums)
            return max(0.0, agg - self.bias)
        kernel_sum, _ = self.compute_kernel_sum_and_count(t, event_history_arrays, event_counts)
        return max(0.0, kernel_sum - self.bias)

    def activation_upper_bound(self, t, event_history_arrays, event_counts):
        """Safe future upper bound for p_r(u), u >= t, used by thinning."""
        if self.activation_mode == 'product_bounded':
            src_uppers, _ = self.future_source_kernel_sums_upper_bound(t, event_history_arrays, event_counts)
            if not src_uppers:
                return 0.0
            out = 1.0
            for src_upper in src_uppers:
                out *= (1.0 - np.exp(-max(0.0, float(src_upper))))
            return float(out)
        if self.activation_mode == 'softmin_relu':
            src_uppers, _ = self.future_source_kernel_sums_upper_bound(t, event_history_arrays, event_counts)
            agg = self._softmin(src_uppers)
            return max(0.0, agg - self.bias)
        src_uppers, _ = self.future_source_kernel_sums_upper_bound(t, event_history_arrays, event_counts)
        return max(0.0, float(np.sum(src_uppers)) - self.bias)


def generate_multiplicative_data(rules, num_samples, time_horizon, base_intensities, max_len=1024, eps=1e-8, seed=None):
    """Generate synthetic event sequences with Ogata thinning."""
    data = []

    rules_by_target = defaultdict(list)
    for rule in rules:
        rules_by_target[rule.target].append(rule)

    event_types = sorted(int(k) for k in base_intensities.keys())
    num_types = len(event_types)
    type_to_idx = {k: i for i, k in enumerate(event_types)}
    base_intensity_array = np.array([float(base_intensities[k]) for k in event_types], dtype=np.float64)

    event_history = np.zeros((num_types, max_len), dtype=np.float64)
    lambda_k_array = np.zeros(num_types, dtype=np.float64)

    rng = np.random.default_rng(seed)

    disable_tqdm = str(os.environ.get("TQDM_DISABLE", "")).lower() in {"1", "true", "yes", "y"}
    for _ in tqdm(range(num_samples), leave=False, disable=disable_tqdm):
        t_curr = 0.0
        event_history.fill(0.0)
        event_counts_dict = {k: 0 for k in event_types}

        full_sequence_times = np.zeros(max_len, dtype=np.float64)
        full_sequence_events = np.zeros(max_len, dtype=np.int32)
        full_sequence_labels = np.zeros(max_len, dtype=np.int32)
        seq_len = 0

        while t_curr < time_horizon and seq_len < max_len:
            # Upper bound (ignoring inhibition gives safe upper bound)
            lambda_max = 0.0
            for k_idx, k in enumerate(event_types):
                b_k = base_intensity_array[k_idx]
                e_bound = 0.0
                for rule in rules_by_target[k]:
                    p_bound = rule.activation_upper_bound(t_curr, event_history, event_counts_dict)
                    e_bound += rule.W_pos * float(p_bound)
                lambda_max += max(0.0, b_k + e_bound) + eps

            if lambda_max < 0.1:
                lambda_max = 0.1

            u = rng.random()
            dt = -np.log(u) / lambda_max
            t_cand = t_curr + dt

            if t_cand > time_horizon:
                break

            # True intensity
            lambda_k_array.fill(0.0)
            for k_idx, k in enumerate(event_types):
                b_k = base_intensity_array[k_idx]
                E_k = 0.0
                I_k = 0.0

                for rule in rules_by_target[k]:
                    p_r = rule.compute_activation(t_cand, event_history, event_counts_dict)
                    E_k += rule.W_pos * p_r
                    I_k += rule.W_neg * p_r

                lambda_k_array[k_idx] = (b_k + E_k) * np.exp(-I_k) + eps

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

            # Cause labels: base + positive rule causes only.
            target_rules = rules_by_target[selected_type]
            num_causes = 1 + len(target_rules)

            cause_weights = np.zeros(num_causes, dtype=np.float64)
            cause_ids = np.zeros(num_causes, dtype=np.int32)

            cause_weights[0] = max(0.0, base_intensities[selected_type])
            cause_ids[0] = -1

            for idx, rule in enumerate(target_rules, start=1):
                p_r = rule.compute_activation(t_cand, event_history, event_counts_dict)
                cause_weights[idx] = max(0.0, rule.W_pos * p_r)
                cause_ids[idx] = rule.rule_id

            cause_sum = np.sum(cause_weights)
            if cause_sum > 1e-10:
                cause_probs = cause_weights / cause_sum
                cause_idx = rng.choice(num_causes, p=cause_probs)
                cause_rule = int(cause_ids[cause_idx])
            else:
                cause_rule = -1

            full_sequence_times[seq_len] = t_cand
            full_sequence_events[seq_len] = selected_type
            full_sequence_labels[seq_len] = cause_rule

            event_history[selected_type_idx, event_counts_dict[selected_type]] = t_cand
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
        cause_weights[idx] = max(0.0, np.exp(min(rule.W_pos * p_r, 40.0)) - 1.0)
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

    event_types = sorted(int(k) for k in base_intensities.keys())
    fast = _single_target_fast_path_info(rules, event_types)
    if fast is None:
        raise ValueError("single-target canonical fast path called on incompatible rule structure")

    target = int(fast["target"])
    target_rules = list(fast["target_rules"])
    non_target_types = list(fast["non_target_types"])
    max_type = max(int(k) for k in event_types)
    b_target = max(float(base_intensities[target]), float(eps))

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
            e_bound = 0.0
            for rule in target_rules:
                e_bound += rule.W_pos * float(rule.activation_upper_bound(t_curr, event_history, counts_curr))
            lambda_max = b_target * np.exp(min(float(e_bound), 40.0)) + float(eps)
            lambda_max = max(lambda_max, 0.1)

            dt = -np.log(rng.random()) / lambda_max
            t_cand = t_curr + dt
            if t_cand > float(time_horizon):
                break

            counts_curr = _advance_past_event_counts_in_place(source_times_by_type, counts_curr, t_cand)
            E_t = 0.0
            I_t = 0.0
            p_vals = []
            for rule in target_rules:
                p_r = rule.compute_activation(t_cand, event_history, counts_curr)
                p_vals.append(float(p_r))
                E_t += rule.W_pos * p_r
                I_t += rule.W_neg * p_r
            lambda_true = b_target * np.exp(np.clip(E_t - I_t, -40.0, 40.0)) + float(eps)

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
        if len(order) > int(max_len):
            order = order[: int(max_len)]

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

    rules_by_target = defaultdict(list)
    for rule in rules:
        rules_by_target[rule.target].append(rule)

    num_types = len(event_types)
    type_to_idx = {k: i for i, k in enumerate(event_types)}
    base_intensity_array = np.array([float(base_intensities[k]) for k in event_types], dtype=np.float64)

    event_history = np.zeros((num_types, max_len), dtype=np.float64)
    lambda_k_array = np.zeros(num_types, dtype=np.float64)

    rng = np.random.default_rng(seed)

    disable_tqdm = str(os.environ.get("TQDM_DISABLE", "")).lower() in {"1", "true", "yes", "y"}
    for _ in tqdm(range(num_samples), leave=False, disable=disable_tqdm):
        t_curr = 0.0
        event_history.fill(0.0)
        event_counts_dict = {k: 0 for k in event_types}

        full_sequence_times = np.zeros(max_len, dtype=np.float64)
        full_sequence_events = np.zeros(max_len, dtype=np.int32)
        full_sequence_labels = np.zeros(max_len, dtype=np.int32)
        seq_len = 0

        while t_curr < time_horizon and seq_len < max_len:
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

            full_sequence_times[seq_len] = t_cand
            full_sequence_events[seq_len] = selected_type
            full_sequence_labels[seq_len] = cause_rule

            event_history[selected_type_idx, event_counts_dict[selected_type]] = t_cand
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

    default_activation_mode = config.get('activation_mode', 'sum_relu')
    default_kernel_type = config.get('kernel', 'triangular')
    default_softmin_tau = config.get('softmin_tau', 0.15)
    for rule_cfg in config.get('rules', []):
        rule_id = rule_cfg['id']
        target = rule_cfg['target']
        W_pos = rule_cfg.get('W_pos', 1.0)
        W_neg = rule_cfg.get('W_neg', 0.0)
        kernel_type = rule_cfg.get('kernel', default_kernel_type)
        bias = rule_cfg.get('bias', 0.0)
        activation_mode = rule_cfg.get('activation_mode', default_activation_mode)
        softmin_tau = rule_cfg.get('softmin_tau', default_softmin_tau)

        if str(activation_mode) == 'product_bounded' and abs(float(bias)) > 1e-12:
            raise ValueError(
                f"Rule {rule_id} sets bias={bias} under activation_mode='product_bounded', "
                "but product_bounded does not use bias. Remove the bias field or switch "
                "to a bias-aware activation mode."
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
                softmin_tau=softmin_tau,
            )
        )

    return rules


def generate_synthetic_exp(config):
    """Legacy alias entrypoint."""
    return generate_multiplicative_data(
        create_rules_from_config(config),
        config['num_samples'],
        config['time_horizon'],
        config['base_intensity'],
        max_len=config.get('max_len', 1024),
        seed=config.get('seed'),
    )


# Compatibility aliases expected by other scripts.
generate_exp_data = generate_multiplicative_data
generate_additive_data = generate_multiplicative_data


def generate_exact_log_rewrite_data(rules, num_samples, time_horizon, base_intensities, max_len=1024, seed=None):
    """Generate data for lambda(t)=exp(log(b+E(t)) - I(t)).

    Algebraically this is identical to the existing multiplicative generator
    lambda(t)=(b+E(t))exp(-I(t)), so we intentionally reuse the battle-tested
    implementation to preserve rate and attribution semantics exactly while
    keeping config values on the intuitive raw scale.
    """

    return generate_multiplicative_data(
        rules=rules,
        num_samples=num_samples,
        time_horizon=time_horizon,
        base_intensities=base_intensities,
        max_len=max_len,
        seed=seed,
    )
