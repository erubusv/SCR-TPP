"""Synthetic data generator for HNSTPP-Refactored (multiplicative inhibition).

Intensity:
    lambda_k(t) = (b_k + E_k(t)) * exp(-I_k(t)) + eps
where
    E_k(t) = sum_{r:target(r)=k} W_pos[r] * p_r(t)
    I_k(t) = sum_{r:target(r)=k} W_neg[r] * p_r(t)
    p_r(t) = ReLU(sum_{src in cond(r)} K_src(t) - bias_r)
"""

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


class Rule:
    """Rule container used by the synthetic generator."""

    def __init__(self, rule_id, target, W_pos, W_neg, condition, kernel_params, kernel_type='triangular', bias=0.0):
        self.rule_id = int(rule_id)
        self.target = int(target)
        self.W_pos = max(0.0, float(W_pos))
        self.W_neg = max(0.0, float(W_neg))
        self.condition = [int(c) for c in condition]
        self.kernel_type = kernel_type
        self.bias = float(bias)

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
                    elif self.kernel_type == 'gaussian':
                        val = gaussian_kernel_func(dt, peak, width, mix_weight, support_mult)
                    else:
                        val = triangular_kernel_func(dt, peak, width, mix_weight)
                    kernel_sum += val

                idx -= 1

        return kernel_sum, count_window

    def compute_activation(self, t, event_history_arrays, event_counts):
        kernel_sum, _ = self.compute_kernel_sum_and_count(t, event_history_arrays, event_counts)
        return max(0.0, kernel_sum - self.bias)


def generate_multiplicative_data(rules, num_samples, time_horizon, base_intensities, max_len=1024, eps=1e-8):
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

    rng = np.random.default_rng()

    for _ in tqdm(range(num_samples), leave=False):
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
                    _, count_window = rule.compute_kernel_sum_and_count(t_curr, event_history, event_counts_dict)
                    e_bound += rule.W_pos * float(count_window)
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


def create_rules_from_config(config):
    """Create Rule objects from config dict."""
    rules = []

    for rule_cfg in config.get('rules', []):
        rule_id = rule_cfg['id']
        target = rule_cfg['target']
        W_pos = rule_cfg.get('W_pos', 1.0)
        W_neg = rule_cfg.get('W_neg', 0.0)
        kernel_type = rule_cfg.get('kernel', 'triangular')
        bias = rule_cfg.get('bias', 0.0)

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
    )


# Compatibility aliases expected by other scripts.
generate_exp_data = generate_multiplicative_data
generate_additive_data = generate_multiplicative_data
