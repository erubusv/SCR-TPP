"""HNSTPP-Refactored: rule-specific kernels + exact sweep-line integration.

Implements model_refactoring.md v1 core requirements:
1. Multiplicative dual-channel intensity: lambda_k(t) = (b_k + E_k(t)) * exp(-I_k(t)) + eps
2. Rule-specific temporal shape via direct (rule,type,component) kernels
3. Event-level attribution + responsibility regularization for base absorbing mitigation
4. Exact integral with sweep-line breakpoints + ReLU root insertion for p_r(t)
"""

from .base import BaseTPP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HNSTPP(BaseTPP):
    """HNSTPP with exact closed-form integration via sweep-line algorithm."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_types = config['num_types']
        self.num_rules = config['num_rules']
        self.num_components = int(config.get('num_components', 3))
        self.tau = float(config.get('start_tau', 5.0))
        self.pad_token_id = config.get('pad_token_id', config['num_types'])
        self.eps = float(config.get('epsilon', 1e-6))
        self.i_max = float(config.get('i_max', 20.0))
        # Optional Gaussian jitter on theta logits during training only.
        self.theta_noise_std = float(config.get('theta_noise_std', 0.0))

        # Max cap for kernel support (set from training data)
        self.max_cap = config.get('max_cap', None)

        # ============== Rule Structure Learning ==============
        # H[k, r] = 1 if type k is in source set of rule r
        init_std = 1.0 / np.sqrt(self.num_types)
        self.theta = nn.Parameter(torch.randn(self.num_types, self.num_rules) * init_std)
        # Head[r, k] = 1 if rule r targets type k
        self.rule_target_logits = nn.Parameter(torch.randn(self.num_rules, self.num_types) * 0.5)
        # alpha[k, r] controls relative source contribution inside rule r
        self.source_alpha_logits = nn.Parameter(
            torch.randn(self.num_types, self.num_rules) * init_std
        )

        # ============== Weights (Excitation/Inhibition) ==============
        # V8: Fully separated excitation/inhibition weights (Neural Hawkes inspired)
        #   W_pos[r] = softplus(w_exc_raw[r])   (excitatory magnitude)
        #   W_neg[r] = softplus(w_inh_raw[r])   (inhibitory magnitude)
        #   rule_sign[r] = sigmoid(sign_logits[r] / tau_sign)  (1=exc, 0=inh)
        #   Effective: exc_contrib = W_pos * rule_sign, inh_contrib = W_neg * (1-rule_sign)
        init_w_exc = float(config.get('init_w_exc', -1.0))
        init_w_inh = float(config.get('init_w_inh', -1.0))
        self.w_exc_raw = nn.Parameter(torch.full((self.num_rules,), init_w_exc))
        self.w_inh_raw = nn.Parameter(torch.full((self.num_rules,), init_w_inh))
        init_sign_std = float(config.get('init_sign_std', 0.0))
        if init_sign_std > 0:
            self.sign_logits = nn.Parameter(torch.randn(self.num_rules) * init_sign_std)
        else:
            self.sign_logits = nn.Parameter(torch.zeros(self.num_rules))

        # ============== Rule Bias ==============
        init_bias_raw = float(config.get('init_bias_raw', 0.0))
        self.rule_bias_raw = nn.Parameter(torch.full((self.num_rules,), init_bias_raw))

        # ============== Rule-Type Kernel Parameters ==============
        # width_{r,k,c} = W_cap * sigmoid(alpha_{r,k,c})
        # peak_{r,k,c}  = width_{r,k,c} * sigmoid(beta_{r,k,c})
        # mix_{r,k,c}   = softmax_c(gamma_{r,k,c})
        self.raw_width_logits = nn.Parameter(
            torch.zeros(self.num_rules, self.num_types, self.num_components)
        )
        self.raw_peak_ratio_logits = nn.Parameter(
            torch.zeros(self.num_rules, self.num_types, self.num_components)
        )
        self.raw_mix_logits = nn.Parameter(
            torch.zeros(self.num_rules, self.num_types, self.num_components)
        )

        # ============== Base Intensity (per type) ==============
        init_b0 = float(config.get('init_b0', -3.0))
        self.b0 = nn.Parameter(torch.full((self.num_types,), init_b0))

    # ============== Parameter Accessors ==============

    def set_max_cap(self, time_diffs, event_types=None):
        """Set max_cap from training data: multiplier * percentile(interarrival)."""
        percentile = self.config.get('max_cap_percentile', 0.95)
        multiplier = self.config.get('max_cap_multiplier', 5.0)

        if event_types is not None:
            valid_mask = (event_types != self.pad_token_id)
            valid_time_diffs = time_diffs[valid_mask]
        else:
            valid_time_diffs = time_diffs.flatten()

        valid_time_diffs = valid_time_diffs[valid_time_diffs > 0]

        if len(valid_time_diffs) == 0:
            self.max_cap = 10.0
            print("Warning: No valid time differences. Using default max_cap=10.0")
            return

        pctl_value = torch.quantile(valid_time_diffs.float(), percentile).item()
        self.max_cap = multiplier * pctl_value
        print(f"Set max_cap = {self.max_cap:.4f} (P{percentile*100:.0f} = {pctl_value:.4f} x {multiplier})")

    def get_kernel_params(self):
        """Get rule-type-component kernel parameters."""
        if self.max_cap is None:
            W_cap = 10.0
        else:
            W_cap = self.max_cap

        widths = W_cap * torch.sigmoid(self.raw_width_logits)             # (R, K, C)
        peaks = widths * torch.sigmoid(self.raw_peak_ratio_logits)        # (R, K, C)
        mix_weights = F.softmax(self.raw_mix_logits, dim=-1)              # (R, K, C)

        return peaks, widths, mix_weights, W_cap

    def get_rule_matrices(self, tau=None, theta_noise_std=None, use_ste=True):
        """Get rule source mask H and target matrix Head."""
        if tau is None:
            tau = self.tau
        if theta_noise_std is None:
            theta_noise_std = self.theta_noise_std

        theta_logits = self.theta
        if self.training and float(theta_noise_std) > 0.0:
            theta_logits = theta_logits + torch.randn_like(theta_logits) * float(theta_noise_std)

        H_soft = self._temperature_sigmoid(theta_logits, tau)
        Head_soft = self._temperature_softmax(self.rule_target_logits, tau)
        H_hard = (H_soft > 0.5).float()
        Head_hard = F.one_hot(Head_soft.argmax(dim=-1), num_classes=self.num_types).float()

        # Straight-through estimator in training:
        # forward uses hard assignments to align train/eval behavior,
        # backward uses soft assignments for gradients.
        if self.training:
            if use_ste:
                H_st = H_hard + H_soft - H_soft.detach()
                Head_st = Head_hard + Head_soft - Head_soft.detach()
                return H_st, Head_st, H_soft, Head_soft
            else:
                # Phase 1: use soft assignments directly for smoother optimization
                return H_soft, Head_soft, H_soft, Head_soft

        return H_hard, Head_hard, H_soft, Head_soft

    def get_source_alpha(self, source_mask=None):
        """Get per-rule source mixture alpha_{k,r} with sum_k alpha_{k,r}=1."""
        alpha = F.softmax(self.source_alpha_logits, dim=0)  # (K, R)
        if source_mask is None:
            return alpha

        masked = alpha * source_mask
        denom = masked.sum(dim=0, keepdim=True)
        # If no active source exists for a rule, keep the unmasked alpha.
        return torch.where(denom > self.eps, masked / (denom + self.eps), alpha)

    def get_weights(self):
        """Get non-negative excitation and inhibition weights.
        
        V8/V10: Fully separated W_pos and W_neg with binary sign commitment.
        W_pos[r] = softplus(w_exc_raw[r]) * sign_gate[r]     (active when exc)
        W_neg[r] = softplus(w_inh_raw[r]) * (1 - sign_gate[r])  (active when inh)
        This ensures gradient for W_pos only flows when rule is excitatory,
        and gradient for W_neg only flows when rule is inhibitory.
        """
        tau_sign = max(float(self.config.get('sign_tau', 1.0)), 1e-6)
        w_exc = F.softplus(self.w_exc_raw)
        w_inh = F.softplus(self.w_inh_raw)
        gate = torch.sigmoid(self.sign_logits / tau_sign)
        return w_exc * gate, w_inh * (1.0 - gate), gate

    def get_rule_bias(self):
        """Get non-negative rule bias."""
        return F.softplus(self.rule_bias_raw)

    def get_base_intensity(self, b_max=None):
        """Get base intensity b_k for each type."""
        b = F.softplus(self.b0)
        if b_max is not None and b_max > 0:
            b = b.clamp(max=b_max)
        return b

    # ============== Temperature Relaxation ==============

    def _temperature_sigmoid(self, logits, tau):
        """Temperature-scaled sigmoid used for relaxed rule-source assignment."""
        tau = max(float(tau), 1e-6)
        return torch.sigmoid(logits / tau)

    def _temperature_softmax(self, logits, tau):
        """Temperature-scaled softmax used for relaxed rule-target assignment."""
        tau = max(float(tau), 1e-6)
        return F.softmax(logits / tau, dim=-1)

    # ============== Forward ==============

    def forward(self, x, tau=None, theta_noise_std=None, b_max=None, use_ste=True):
        """Compute all model parameters for loss computation."""
        if tau is None:
            tau = self.tau
        if theta_noise_std is None:
            theta_noise_std = self.theta_noise_std

        H, Head, H_soft, Head_soft = self.get_rule_matrices(tau, theta_noise_std=theta_noise_std, use_ste=use_ste)
        source_alpha = self.get_source_alpha(H_soft)
        peaks, widths, mix_weights, W_cap = self.get_kernel_params()
        W_pos, W_neg, sign_gate = self.get_weights()
        rule_bias = self.get_rule_bias()
        b_k = self.get_base_intensity(b_max=b_max)

        return {
            'tau': tau,
            'theta_noise_std': float(theta_noise_std),
            'H': H,
            'Head': Head,
            'H_soft': H_soft,
            'Head_soft': Head_soft,
            'source_alpha': source_alpha,
            'peaks': peaks,
            'widths': widths,
            'mix_weights': mix_weights,
            'W_cap': W_cap,
            'W_pos': W_pos,
            'W_neg': W_neg,
            'sign_gate': sign_gate,
            'rule_bias': rule_bias,
            'b_k': b_k,
        }

    # ============== Core Computation: Type-Basis Signal Aggregation ==============

    def _slice_events_by_window(self, event_times, event_types, window_start, window_end):
        """Slice events to a time window per batch to avoid full-history dt construction."""
        B = event_times.shape[0]
        device = event_times.device

        valid = (event_types != self.pad_token_id)
        in_window = valid & (event_times >= window_start.unsqueeze(1)) & (event_times <= window_end.unsqueeze(1))

        times_list = []
        types_list = []
        max_len = 0
        for b in range(B):
            idx = torch.nonzero(in_window[b], as_tuple=False).squeeze(1)
            times_b = event_times[b, idx]
            types_b = event_types[b, idx]
            max_len = max(max_len, int(idx.numel()))
            times_list.append(times_b)
            types_list.append(types_b)

        if max_len == 0:
            return (
                event_times.new_zeros(B, 1),
                event_types.new_full((B, 1), self.pad_token_id),
                torch.zeros(B, 1, device=device, dtype=torch.bool),
            )

        times_trim = event_times.new_zeros(B, max_len)
        types_trim = event_types.new_full((B, max_len), self.pad_token_id)
        valid_trim = torch.zeros(B, max_len, device=device, dtype=torch.bool)

        for b in range(B):
            n = int(times_list[b].numel())
            if n > 0:
                times_trim[b, :n] = times_list[b]
                types_trim[b, :n] = types_list[b]
                valid_trim[b, :n] = True

        return times_trim, types_trim, valid_trim

    def _aggregate_type_component(self, kernel_vals, event_types_clamped):
        """Aggregate per-event kernel values to S[..., K, C] via scatter_add."""
        K = self.num_types
        C = self.num_components

        # kernel_vals: (..., L, C)
        prefix = kernel_vals.shape[:-2]
        L = kernel_vals.shape[-2]

        flat_size = K * C
        S_flat = kernel_vals.new_zeros(*prefix, flat_size)

        comp_ids = torch.arange(C, device=kernel_vals.device).view(*([1] * event_types_clamped.dim()), C)
        type_ids = event_types_clamped.unsqueeze(-1) * C
        idx_flat = (type_ids + comp_ids).reshape(*event_types_clamped.shape[:-1], L * C)

        val_flat = kernel_vals.reshape(*kernel_vals.shape[:-2], L * C)
        S_flat.scatter_add_(-1, idx_flat, val_flat)

        return S_flat.view(*prefix, K, C)

    def _triangular_unit_area(self, dt, peaks, widths):
        """Triangular kernel normalized to unit area over [0, width]."""
        p_safe = peaks.clamp(min=self.eps)
        w_minus_p = (widths - peaks).clamp(min=self.eps)
        mask0 = (dt >= 0) & (dt <= peaks)
        mask1 = (dt > peaks) & (dt <= widths)
        kernel0 = dt / p_safe
        kernel1 = (widths - dt) / w_minus_p
        tri = kernel0 * mask0 + kernel1 * mask1
        # Base triangular area is width/2; normalize area to 1.
        return tri * (2.0 / widths.clamp(min=self.eps))

    def compute_rule_type_signal(self, eval_times, event_times, event_types, model_output, chunk_size=64):
        """Compute S_{r,k,c}(t) = sum_{i:type(i)=k} Tri_{r,k,c}(t - t_i)."""
        B, Q = eval_times.shape
        R = self.num_rules
        K = self.num_types
        C = self.num_components
        device = eval_times.device

        S_rules = []
        for r in range(R):
            S_r = self._compute_rule_type_signal_single(
                eval_times, event_times, event_types, model_output, rule_idx=r, chunk_size=chunk_size
            )
            S_rules.append(S_r)
        if not S_rules:
            return torch.zeros(B, Q, 0, K, C, device=device, dtype=eval_times.dtype)
        return torch.stack(S_rules, dim=2)  # (B, Q, R, K, C)

    def _compute_rule_type_signal_single(
        self, eval_times, event_times, event_types, model_output, rule_idx, chunk_size=64
    ):
        """Compute S_{k,c}^{(r)}(t) for a single rule r."""
        B, Q = eval_times.shape
        K = self.num_types
        C = self.num_components
        device = eval_times.device

        peaks_r = model_output['peaks'][rule_idx]              # (K, C)
        widths_r = model_output['widths'][rule_idx]            # (K, C)
        mix_r = model_output['mix_weights'][rule_idx]          # (K, C)
        W_cap = model_output['W_cap']

        t_min = eval_times.min(dim=1).values
        t_max = eval_times.max(dim=1).values
        window_start = t_min - W_cap
        window_end = t_max

        event_times, event_types, valid_event = self._slice_events_by_window(
            event_times, event_types, window_start, window_end
        )
        if not torch.any(valid_event):
            return torch.zeros(B, Q, K, C, device=device, dtype=eval_times.dtype)

        L = event_times.shape[1]
        event_types_clamped = event_types.clamp(0, K - 1)
        peaks_ev = peaks_r[event_types_clamped]       # (B, L, C)
        widths_ev = widths_r[event_types_clamped]     # (B, L, C)
        mix_ev = mix_r[event_types_clamped]           # (B, L, C)

        S_chunks = []
        for q_start in range(0, Q, chunk_size):
            q_end = min(q_start + chunk_size, Q)
            Q_chunk = q_end - q_start

            eval_chunk = eval_times[:, q_start:q_end]            # (B, Q_chunk)
            dt = eval_chunk.unsqueeze(2) - event_times.unsqueeze(1)  # (B, Q_chunk, L)
            valid_dt = valid_event.unsqueeze(1)                  # (B, Q_chunk, L)

            dt_exp = dt.unsqueeze(-1)
            peaks_exp = peaks_ev.unsqueeze(1)
            widths_exp = widths_ev.unsqueeze(1)
            mix_exp = mix_ev.unsqueeze(1)

            kernel_vals = mix_exp * self._triangular_unit_area(dt_exp, peaks_exp, widths_exp)
            kernel_vals = kernel_vals * valid_dt.unsqueeze(-1)

            S_chunk = self._aggregate_type_component(
                kernel_vals,
                event_types_clamped.unsqueeze(1).expand(B, Q_chunk, L),
            )
            S_chunks.append(S_chunk)

        return torch.cat(S_chunks, dim=1)  # (B, Q, K, C)

    def compute_rule_type_signal_at_events(self, times, event_types, model_output):
        """Compute S_{r,k,c}(t_i) using only events within [t_i - W_cap, t_i)."""
        B, L = times.shape
        R = self.num_rules
        K = self.num_types
        C = self.num_components
        device = times.device

        peaks = model_output['peaks']              # (R, K, C)
        widths = model_output['widths']            # (R, K, C)
        mix_weights = model_output['mix_weights']  # (R, K, C)
        W_cap = model_output['W_cap']

        S_out = torch.zeros(B, L, R, K, C, device=device, dtype=times.dtype)

        for b in range(B):
            valid = event_types[b] != self.pad_token_id
            if not torch.any(valid):
                continue
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            L_valid = int(valid_idx.numel())
            if L_valid <= 1:
                continue

            times_b = times[b, valid_idx]
            types_b = event_types[b, valid_idx].clamp(0, K - 1)

            window_start = torch.searchsorted(times_b, times_b - W_cap, right=False)
            idx_i = torch.arange(L_valid, device=device)
            lengths = idx_i - window_start
            max_len = int(lengths.max().item())
            if max_len <= 0:
                continue

            r = torch.arange(max_len, device=device)
            idx = window_start.unsqueeze(1) + r.unsqueeze(0)  # (L_valid, max_len)
            mask = r.unsqueeze(0) < lengths.unsqueeze(1)

            idx_clamped = idx.clamp(max=L_valid - 1)
            times_window = times_b[idx_clamped]
            dt = times_b.unsqueeze(1) - times_window
            types_window = types_b[idx_clamped]

            dt_exp = dt.unsqueeze(-1)                     # (L_valid, max_len, 1)
            type_idx = types_window
            valid_mask_exp = mask.unsqueeze(-1)
            S_rule_list = []

            for r_idx in range(R):
                peaks_ev = peaks[r_idx][type_idx]        # (L_valid, max_len, C)
                widths_ev = widths[r_idx][type_idx]
                mix_ev = mix_weights[r_idx][type_idx]

                kernel_vals = mix_ev * self._triangular_unit_area(dt_exp, peaks_ev, widths_ev)
                kernel_vals = kernel_vals * valid_mask_exp

                S_r = self._aggregate_type_component(kernel_vals, types_window)  # (L_valid, K, C)
                S_rule_list.append(S_r)

            S_valid = torch.stack(S_rule_list, dim=1)  # (L_valid, R, K, C)
            S_out[b, valid_idx] = S_valid

        return S_out

    # ============== Rule Feature Computation ==============

    def compute_rule_preactivation(self, S, model_output):
        """Compute u_r(t) = sum_{k,c} H_{k,r}*alpha_{k,r}*S_{r,k,c}(t) - bias_r."""
        # S: (..., R, K, C)
        H = model_output['H']                                   # (K, M)
        source_alpha = model_output['source_alpha']             # (K, M)
        rule_bias = model_output['rule_bias']                   # (M,)

        source_gate = (H * source_alpha).t().unsqueeze(-1)     # (R, K, 1)
        u = (S * source_gate).sum(dim=(-2, -1))                # (..., R)
        u = u - rule_bias.view(*([1] * (u.dim() - 1)), -1)
        return u

    def compute_rule_feature(self, S, model_output):
        """Compute p_r(t) = ReLU(u_r(t))."""
        return F.relu(self.compute_rule_preactivation(S, model_output))

    def _get_sequence_end(self, event_times, event_types):
        """Get last valid event time per sequence."""
        B, L = event_times.shape
        device = event_times.device
        valid = (event_types != self.pad_token_id)
        if not torch.any(valid):
            return torch.zeros(B, device=device, dtype=event_times.dtype)
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        last_idx = (valid * idx).max(dim=1).values.long()
        t_end = event_times.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        t_end = torch.where(valid.any(dim=1), t_end, torch.zeros_like(t_end))
        return t_end

    def _compute_exc_inh(self, p_r, model_output):
        """Compute E_k(t) and I_k(t) from rule features p_r(t)."""
        Head = model_output['Head']      # (M, K)
        W_pos = model_output['W_pos']    # (M,)
        W_neg = model_output['W_neg']    # (M,)

        weighted_pos = p_r * W_pos.view(*([1] * (p_r.dim() - 1)), -1)
        weighted_neg = p_r * W_neg.view(*([1] * (p_r.dim() - 1)), -1)

        E_k = torch.matmul(weighted_pos, Head)
        I_k = torch.matmul(weighted_neg, Head)
        return E_k, I_k

    def compute_intensity(self, eval_times, event_times, event_types, model_output, p_r=None):
        """Compute lambda_k(t) = (b_k + E_k) * exp(-I_k) + eps."""
        if p_r is None:
            S = self.compute_rule_type_signal(eval_times, event_times, event_types, model_output)
            p_r = self.compute_rule_feature(S, model_output)

        E_k, I_k = self._compute_exc_inh(p_r, model_output)
        I_k = I_k.clamp(min=0.0, max=self.i_max)
        b_k = model_output['b_k']
        lambda_k = (b_k.view(1, 1, -1) + E_k) * torch.exp(-I_k) + self.eps
        return lambda_k

    # ============== Sweep-Line Exact Integration ==============

    def _collect_breakpoints(self, t_start, t_end, event_times, event_types, model_output):
        """Collect breakpoints from type-basis triangular kernels."""
        if t_start.dim() == 1:
            t_start = t_start.view(-1, 1)
            t_end = t_end.view(-1, 1)

        B, Q = t_start.shape
        R = self.num_rules
        K = self.num_types
        C = self.num_components
        device = event_times.device

        peaks = model_output['peaks']      # (R, K, C)
        widths = model_output['widths']    # (R, K, C)
        max_support_type = widths.amax(dim=(0, 2))  # (K,)

        bp_per_batch = []
        max_bp = 0
        eps = 1e-8

        for b in range(B):
            valid = event_types[b] != self.pad_token_id
            if not torch.any(valid):
                bp_b = torch.stack([t_start[b], t_end[b]], dim=-1)
                bp_per_batch.append(bp_b)
                max_bp = max(max_bp, bp_b.shape[1])
                continue

            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            times_b = event_times[b, idx]
            types_b = event_types[b, idx].clamp(0, K - 1)

            peaks_ev = peaks[:, types_b, :]    # (R, L_valid, C)
            widths_ev = widths[:, types_b, :]  # (R, L_valid, C)
            max_support_event = max_support_type[types_b]

            t_event = times_b.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,L_valid,1)
            bp_start = t_event.expand(Q, R, -1, C)
            bp_peak = t_event + peaks_ev.unsqueeze(0)
            bp_end = t_event + widths_ev.unsqueeze(0)

            t_event_exp = times_b.unsqueeze(0)
            valid_window = (t_event_exp <= t_end[b].unsqueeze(-1)) & \
                           (t_event_exp + max_support_event.unsqueeze(0) >= t_start[b].unsqueeze(-1))

            t_start_exp = t_start[b].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            t_end_exp = t_end[b].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            valid_exp = valid_window.unsqueeze(1).unsqueeze(-1)

            in_range_start = valid_exp & (bp_start > t_start_exp) & (bp_start < t_end_exp)
            in_range_peak = valid_exp & (bp_peak > t_start_exp) & (bp_peak < t_end_exp)
            in_range_end = valid_exp & (bp_end > t_start_exp) & (bp_end < t_end_exp)

            all_bp_list = [
                t_start[b].unsqueeze(-1),
                t_end[b].unsqueeze(-1),
                torch.where(in_range_start, bp_start, t_start_exp).reshape(Q, -1),
                torch.where(in_range_peak, bp_peak, t_start_exp).reshape(Q, -1),
                torch.where(in_range_end, bp_end, t_start_exp).reshape(Q, -1),
            ]

            all_bp = torch.cat(all_bp_list, dim=-1)
            all_bp = all_bp.clamp(t_start[b].unsqueeze(-1), t_end[b].unsqueeze(-1))
            all_bp_sorted, _ = torch.sort(all_bp, dim=-1)

            diffs = all_bp_sorted[:, 1:] - all_bp_sorted[:, :-1]
            non_dup = diffs > eps
            keep_mask = torch.cat([torch.ones(Q, 1, device=device, dtype=torch.bool), non_dup], dim=-1)
            all_bp_sorted = torch.where(keep_mask, all_bp_sorted, t_end[b].unsqueeze(-1))
            all_bp_sorted, _ = torch.sort(all_bp_sorted, dim=-1)

            bp_per_batch.append(all_bp_sorted)
            max_bp = max(max_bp, all_bp_sorted.shape[1])

        bp_padded = []
        for b in range(B):
            bp_b = bp_per_batch[b]
            if bp_b.shape[1] < max_bp:
                pad = t_end[b].unsqueeze(-1).expand(Q, max_bp - bp_b.shape[1])
                bp_b = torch.cat([bp_b, pad], dim=-1)
            bp_padded.append(bp_b)

        return torch.stack(bp_padded, dim=0)

    def _compute_u_at_breakpoints(self, breakpoints, event_times, event_types, model_output):
        """Compute rule pre-activation u_r(t) at breakpoints."""
        B, Q, S_bp = breakpoints.shape
        chunk_size = int(self.config.get('breakpoint_chunk_size', 128))

        eval_flat = breakpoints.reshape(B, Q * S_bp)
        H = model_output['H']               # (K, R)
        source_alpha = model_output['source_alpha']  # (K, R)
        rule_bias = model_output['rule_bias']  # (R,)

        u_rules = []
        for r in range(self.num_rules):
            S_r = self._compute_rule_type_signal_single(
                eval_flat, event_times, event_types, model_output, rule_idx=r, chunk_size=chunk_size
            )  # (B, Q*S_bp, K, C)
            h_r = (H[:, r] * source_alpha[:, r]).view(1, 1, self.num_types, 1)
            u_r = (S_r * h_r).sum(dim=(-2, -1)) - rule_bias[r]
            u_rules.append(u_r)

        u_flat = torch.stack(u_rules, dim=-1)  # (B, Q*S_bp, R)
        return u_flat.view(B, Q, S_bp, self.num_rules)

    def _compute_exc_inh_at_breakpoints(self, breakpoints, event_times, event_types, model_output, u_at_bp=None):
        """Compute E_k(t) and I_k(t) at breakpoints."""
        if u_at_bp is None:
            u_at_bp = self._compute_u_at_breakpoints(breakpoints, event_times, event_types, model_output)
        p_r = F.relu(u_at_bp)

        Head = model_output['Head']
        W_pos = model_output['W_pos']
        W_neg = model_output['W_neg']

        weighted_pos = p_r * W_pos.view(1, 1, 1, -1)
        weighted_neg = p_r * W_neg.view(1, 1, 1, -1)

        E_k = torch.matmul(weighted_pos, Head)
        I_k = torch.matmul(weighted_neg, Head)
        return E_k, I_k

    def _add_rule_roots(self, breakpoints, u_at_bp, t_start, t_end):
        """Add roots where rule pre-activation u_r(t) = 0."""
        B, Q, S_bp = breakpoints.shape
        device = breakpoints.device
        eps = 1e-10

        seg_starts = breakpoints[:, :, :-1]
        seg_ends = breakpoints[:, :, 1:]
        seg_valid = (seg_ends > seg_starts + eps)

        u_start = u_at_bp[:, :, :-1, :]
        u_end = u_at_bp[:, :, 1:, :]

        sign_start = (u_start > 0).float()
        sign_end = (u_end > 0).float()
        crossing = (sign_start != sign_end) & seg_valid.unsqueeze(-1)

        seg_len = (seg_ends - seg_starts).unsqueeze(-1)
        slope = (u_end - u_start) / (seg_len + eps)
        root_t = seg_starts.unsqueeze(-1) - u_start / (slope + eps)

        valid_root = crossing & (root_t > seg_starts.unsqueeze(-1) + eps) & (root_t < seg_ends.unsqueeze(-1) - eps)
        root_t = torch.where(valid_root, root_t, t_start.unsqueeze(-1).unsqueeze(-1))

        roots_flat = root_t.reshape(B, Q, -1)
        all_bp = torch.cat([breakpoints, roots_flat], dim=-1)
        all_bp = all_bp.clamp(t_start.unsqueeze(-1), t_end.unsqueeze(-1))
        all_bp_sorted, _ = torch.sort(all_bp, dim=-1)

        diffs = all_bp_sorted[:, :, 1:] - all_bp_sorted[:, :, :-1]
        non_dup = diffs > eps
        keep_mask = torch.cat([torch.ones(B, Q, 1, device=device, dtype=torch.bool), non_dup], dim=-1)
        all_bp_sorted = torch.where(keep_mask, all_bp_sorted, t_end.unsqueeze(-1))
        all_bp_sorted, _ = torch.sort(all_bp_sorted, dim=-1)

        return all_bp_sorted

    def compute_integral_exact(self, t_start, t_end, event_times, event_types, model_output):
        """Compute exact integral of multiplicative intensity."""
        squeeze_out = False
        if t_start.dim() == 1:
            t_start = t_start.unsqueeze(1)
            t_end = t_end.unsqueeze(1)
            squeeze_out = True

        breakpoints = self._collect_breakpoints(t_start, t_end, event_times, event_types, model_output)

        u_at_bp = self._compute_u_at_breakpoints(breakpoints, event_times, event_types, model_output)
        breakpoints = self._add_rule_roots(breakpoints, u_at_bp, t_start, t_end)

        u_at_bp = self._compute_u_at_breakpoints(breakpoints, event_times, event_types, model_output)
        E_k, I_k = self._compute_exc_inh_at_breakpoints(
            breakpoints, event_times, event_types, model_output, u_at_bp=u_at_bp
        )
        I_k = I_k.clamp(min=0.0, max=self.i_max)

        seg_starts = breakpoints[:, :, :-1]
        seg_ends = breakpoints[:, :, 1:]
        seg_len = (seg_ends - seg_starts).clamp(min=0.0)  # (B, Q, S-1)
        seg_valid = seg_len > 1e-10

        L = seg_len.unsqueeze(-1)  # (B, Q, S-1, 1)
        E0 = E_k[:, :, :-1, :]
        E1 = E_k[:, :, 1:, :]
        I0 = I_k[:, :, :-1, :]
        I1 = I_k[:, :, 1:, :]

        eps = 1e-10
        slope_E = (E1 - E0) / (L + eps)
        slope_I = (I1 - I0) / (L + eps)
        b_k = model_output['b_k'].view(1, 1, 1, -1)

        # Local segment form (s in [0, L]):
        # E(s)=E0 + slope_E*s, I(s)=I0 + slope_I*s
        A = b_k + E0
        B_coef = slope_E
        C = I0
        D = slope_I

        DL = D * L
        exp_neg_C = torch.exp(-C)
        exp_neg_DL = torch.exp(-DL.clamp(min=-50.0, max=50.0))

        use_linear = D.abs() < 1e-8
        safe_D = torch.where(use_linear, torch.ones_like(D), D)
        inv_D = 1.0 / safe_D
        inv_D2 = inv_D * inv_D

        phi1 = (1.0 - exp_neg_DL) * inv_D
        phi2 = (1.0 - exp_neg_DL * (1.0 + DL)) * inv_D2
        integral_general = exp_neg_C * (A * phi1 + B_coef * phi2)

        integral_linear = exp_neg_C * (A * L + 0.5 * B_coef * L * L)
        segment_integral = torch.where(use_linear, integral_linear, integral_general)
        segment_integral = segment_integral + self.eps * L
        segment_integral = segment_integral * seg_valid.unsqueeze(-1).float()

        total_integral = segment_integral.sum(dim=(2, 3))
        return total_integral.squeeze(1) if squeeze_out else total_integral

    def compute_integral_approx(self, t_start, t_end, event_times, event_types, model_output, num_points=64):
        """Approximate integral of total intensity with trapezoidal rule."""
        squeeze_out = False
        if t_start.dim() == 1:
            t_start = t_start.unsqueeze(1)
            t_end = t_end.unsqueeze(1)
            squeeze_out = True

        B, Q = t_start.shape
        num_points = max(int(num_points), 2)

        u = torch.linspace(0.0, 1.0, steps=num_points + 1, device=t_start.device, dtype=t_start.dtype)
        span = (t_end - t_start).clamp(min=0.0)
        eval_times = t_start.unsqueeze(-1) + span.unsqueeze(-1) * u.view(1, 1, -1)  # (B,Q,S)

        eval_flat = eval_times.reshape(B, Q * (num_points + 1))
        lambda_k = self.compute_intensity(eval_flat, event_times, event_types, model_output)  # (B,Q*S,K)
        lambda_total = lambda_k.sum(dim=-1).view(B, Q, num_points + 1)

        delta = span.unsqueeze(-1) / float(num_points)
        trap = 0.5 * (lambda_total[..., :-1] + lambda_total[..., 1:]) * delta
        integral = trap.sum(dim=-1)
        return integral.squeeze(1) if squeeze_out else integral

    # ============== Loss / Attribution ==============

    def _compute_dual_channel_factors(self, p_r, input_ids, model_output):
        """Compute event-wise excitation/inhibition factors for the observed type."""
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids == self.pad_token_id] = 0

        Head_km = model_output['Head'].t()  # (K, M)
        head_event = Head_km[safe_input_ids]  # (B, L, M)
        c_pos = p_r * model_output['W_pos'].view(1, 1, -1) * head_event
        c_neg = p_r * model_output['W_neg'].view(1, 1, -1) * head_event
        b_target = model_output['b_k'][safe_input_ids]
        return c_pos, c_neg, b_target, safe_input_ids

    def compute_event_attributions(self, batch, model_output=None, tau=None, normalize=True):
        """Compute event-level dual-channel attribution and responsibilities."""
        if model_output is None:
            model_output = self.forward(batch, tau=tau)
        if tau is None:
            tau = float(model_output.get('tau', self.tau))
        tau = max(float(tau), 1e-6)

        time_diffs = batch['time_diffs']
        input_ids = batch['input_ids']
        mask = batch.get('attention_mask', torch.ones_like(time_diffs)).float()
        mask = mask * (input_ids != self.pad_token_id).float()
        times = torch.cumsum(time_diffs, dim=1)

        S = self.compute_rule_type_signal_at_events(times, input_ids, model_output)
        p_r = self.compute_rule_feature(S, model_output)
        c_pos, c_neg, b_target, _ = self._compute_dual_channel_factors(p_r, input_ids, model_output)

        factors = torch.cat([b_target.unsqueeze(-1), c_pos], dim=-1)
        gamma = F.softmax(factors / tau, dim=-1)

        out = {
            'mask': mask,
            'times': times,
            'event_types': input_ids,
            'c_pos': c_pos,
            'c_neg': c_neg,
            'gamma': gamma,
            'base_factor': b_target,
        }
        if normalize:
            out['c_pos_norm'] = c_pos / (c_pos.sum(dim=-1, keepdim=True) + b_target.unsqueeze(-1) + self.eps)
            out['c_neg_norm'] = c_neg / (c_neg.sum(dim=-1, keepdim=True) + self.eps)
        return out

    def compute_loss(self, batch, model_output, tau=None, include_aux_loss=True):
        """Compute NLL + responsibility/structure regularization."""

        time_diffs = batch['time_diffs']
        input_ids = batch['input_ids']
        mask = batch.get('attention_mask', torch.ones_like(time_diffs)).float()

        device = time_diffs.device
        times = torch.cumsum(time_diffs, dim=1)

        pad_mask = (input_ids != self.pad_token_id).float()
        mask = mask * pad_mask

        t_end = self._get_sequence_end(times, input_ids)
        t_start = torch.zeros_like(t_end)

        S = self.compute_rule_type_signal_at_events(times, input_ids, model_output)
        p_r = self.compute_rule_feature(S, model_output)

        E_k, I_k = self._compute_exc_inh(p_r, model_output)

        c_pos, c_neg, b_target, safe_input_ids = self._compute_dual_channel_factors(
            p_r, input_ids, model_output
        )
        E_target = E_k.gather(2, safe_input_ids.unsqueeze(-1)).squeeze(-1)
        I_target = I_k.gather(2, safe_input_ids.unsqueeze(-1)).squeeze(-1)

        I_target = I_target.clamp(min=0.0, max=self.i_max)
        lambda_target = (b_target + E_target) * torch.exp(-I_target) + self.eps
        event_ll = (torch.log(lambda_target) * mask).sum()

        num_events = mask.sum() + 1e-9
        integral_method = str(self.config.get('integral_method', 'approx')).lower()
        if integral_method == 'exact':
            integrals = self.compute_integral_exact(t_start, t_end, times, input_ids, model_output)
        else:
            num_points = int(self.config.get('integral_num_points', 64))
            integrals = self.compute_integral_approx(
                t_start, t_end, times, input_ids, model_output, num_points=num_points
            )
        integral_loss = integrals.sum()
        nll = (-event_ll + integral_loss) / num_events

        if tau is None:
            tau = float(model_output.get('tau', self.tau))
        tau = max(float(tau), 1e-6)

        factors = torch.cat([b_target.unsqueeze(-1), c_pos], dim=-1)
        gamma = F.softmax(factors / tau, dim=-1)

        entropy = -(gamma * torch.log(gamma + self.eps)).sum(dim=-1)
        entropy_mean = (entropy * mask).sum() / num_events

        mean_gamma_base = (gamma[..., 0] * mask).sum() / num_events
        base_resp_target = float(self.config.get('base_resp_target', 0.3))
        use_base_hinge = bool(self.config.get('use_base_hinge', False))
        if use_base_hinge:
            resp_base_penalty = F.relu(mean_gamma_base - base_resp_target)
        else:
            resp_base_penalty = mean_gamma_base

        lambda_ent = float(self.config.get('lambda_ent', 0.0))
        lambda_resp_base = float(self.config.get('lambda_resp_base', 0.0))
        resp_loss = lambda_ent * entropy_mean + lambda_resp_base * resp_base_penalty

        reg_loss = self._compute_regularization(model_output)

        # V10: Inhibition gradient boost - auxiliary loss to ensure inhibitory rules
        # produce meaningful attenuation at their target events
        inh_aux_loss = torch.tensor(0.0, device=device)
        lambda_inh_boost = float(self.config.get('lambda_inh_boost', 0.0))
        if lambda_inh_boost > 0.0 and include_aux_loss:
            sign_gate = model_output['sign_gate']
            inh_mask_rule = (sign_gate < 0.5).float()  # (R,)
            if inh_mask_rule.sum() > 0:
                inh_p_r = p_r * model_output['W_neg'].view(1, 1, -1) * inh_mask_rule.view(1, 1, -1)
                Head_km = model_output['Head'].t()  # (K, M)
                head_event = Head_km[safe_input_ids]  # (B, L, M)
                inh_at_target = (inh_p_r * head_event).sum(dim=-1)  # (B, L)
                inh_margin = float(self.config.get('inh_margin', 0.1))
                inh_deficit = F.relu(inh_margin - inh_at_target)
                inh_aux_loss = (inh_deficit * mask).sum() / num_events

        total_loss = nll + reg_loss + (resp_loss if include_aux_loss else 0.0) + lambda_inh_boost * inh_aux_loss

        masked_lambda = lambda_target[mask.bool()]
        lambda_std = masked_lambda.std() if masked_lambda.numel() > 1 else torch.tensor(0.0, device=device)
        p_r_mean = (p_r * mask.unsqueeze(-1)).sum() / (mask.sum() * p_r.size(-1) + 1e-9)

        c_pos_sum = c_pos.sum(dim=-1)
        c_neg_sum = c_neg.sum(dim=-1)
        exc_top1 = torch.where(
            c_pos_sum > 0,
            c_pos.max(dim=-1).values / (c_pos_sum + self.eps),
            torch.zeros_like(c_pos_sum),
        )
        inh_top1 = torch.where(
            c_neg_sum > 0,
            c_neg.max(dim=-1).values / (c_neg_sum + self.eps),
            torch.zeros_like(c_neg_sum),
        )
        exc_top1_mean = (exc_top1 * mask).sum() / num_events
        inh_top1_mean = (inh_top1 * mask).sum() / num_events

        return {
            'total_loss': total_loss,
            'nll_loss': nll,
            'reg_loss': reg_loss,
            'resp_loss': resp_loss,
            'event_ll': event_ll.item(),
            'integral_loss': integral_loss.item(),
            'num_events': num_events.item(),
            'lambda_std_at_events': lambda_std.item(),
            'b_k_mean': model_output['b_k'].mean().item(),
            'W_pos_mean': model_output['W_pos'].mean().item(),
            'W_neg_mean': model_output['W_neg'].mean().item(),
            'sign_gate_mean': model_output['sign_gate'].mean().item(),
            'source_card_mean': model_output['H_soft'].sum(dim=0).mean().item(),
            'p_r_mean': p_r_mean.item(),
            'mean_gamma_base': mean_gamma_base.item(),
            'resp_entropy': entropy_mean.item(),
            'resp_base_penalty': resp_base_penalty.item(),
            'exc_top1_ratio': exc_top1_mean.item(),
            'inh_top1_ratio': inh_top1_mean.item(),
            'attenuation_mean': (torch.exp(-I_target) * mask).sum().div(num_events).item(),
            'inh_aux_loss': inh_aux_loss.item() if isinstance(inh_aux_loss, torch.Tensor) else float(inh_aux_loss),
        }

    def _compute_regularization(self, model_output):
        """Compute regularization losses."""
        H_soft = model_output['H_soft']
        Head_soft = model_output['Head_soft']
        W_pos = model_output['W_pos']
        W_neg = model_output['W_neg']
        sign_gate = model_output['sign_gate']
        source_alpha = model_output['source_alpha']

        lambda_h_sparse = float(self.config.get('lambda_h_sparse', self.config.get('lambda_sparse', 1e-3)))
        lambda_h_binary = float(self.config.get('lambda_h_binary', 0.0))
        lambda_h_card = float(self.config.get('lambda_h_card', 0.0))
        h_card_target = float(self.config.get('h_card_target', 1.0))
        lambda_w_sparse = float(self.config.get('lambda_w_sparse', 0.0))
        lambda_overlap = float(self.config.get('lambda_overlap', 1e-2))
        lambda_sign_binary = float(self.config.get('lambda_sign_binary', 0.0))
        lambda_ortho = float(self.config.get('lambda_ortho', 1e-3))
        lambda_head = float(self.config.get('lambda_head_entropy', 1e-3))
        lambda_head_ortho = float(self.config.get('lambda_head_ortho', 0.0))
        lambda_base_l1 = float(self.config.get('lambda_base_l1', 0.0))
        lambda_alpha_entropy = float(self.config.get('lambda_alpha_entropy', 0.0))

        H_norm = H_soft / (H_soft.norm(dim=0, keepdim=True) + 1e-9)
        gram = torch.matmul(H_norm.t(), H_norm)
        eye = torch.eye(self.num_rules, device=H_soft.device)
        reg_ortho = ((gram - eye) ** 2).mean()

        reg_h_sparse = H_soft.mean()
        reg_h_binary = (H_soft * (1.0 - H_soft)).mean()
        reg_h_card = ((H_soft.sum(dim=0) - h_card_target) ** 2).mean()

        reg_L1 = W_pos.sum() + W_neg.sum()
        # V10: Overlap penalty within each rule (W_pos*W_neg per-rule)
        reg_overlap = (W_pos * W_neg).sum()
        # Sign commitment for binary specialization
        reg_sign_binary = (sign_gate * (1.0 - sign_gate)).mean()
        lambda_sign_mag = float(self.config.get('lambda_sign_magnitude', 0.0))
        if lambda_sign_mag > 0:
            sign_mag_penalty = F.relu(2.0 - self.sign_logits.abs()).mean()
            reg_sign_binary = reg_sign_binary + lambda_sign_mag * sign_mag_penalty

        b_k = model_output['b_k']
        reg_base = b_k.abs().sum()
        reg_head = -(Head_soft * torch.log(Head_soft + self.eps)).sum(dim=-1).mean()
        gram_head = torch.matmul(Head_soft, Head_soft.t())
        reg_head_ortho = ((gram_head - eye) ** 2).mean()
        reg_alpha_entropy = -(source_alpha * torch.log(source_alpha + self.eps)).sum(dim=0).mean()

        total_reg = (
            lambda_ortho * reg_ortho +
            lambda_h_sparse * reg_h_sparse +
            lambda_h_binary * reg_h_binary +
            lambda_h_card * reg_h_card +
            lambda_w_sparse * reg_L1 +
            lambda_overlap * reg_overlap +
            lambda_sign_binary * reg_sign_binary +
            lambda_head * reg_head +
            lambda_head_ortho * reg_head_ortho +
            lambda_alpha_entropy * reg_alpha_entropy +
            lambda_base_l1 * reg_base
        )

        return total_reg

    # ============== Structure Extraction ==============

    def get_structure(self):
        """Extract learned rule structures for interpretation."""
        with torch.no_grad():
            model_output = self.forward(None, tau=0.1)

            H = (model_output['H'] > 0.5).float()
            Head = F.one_hot(model_output['Head'].argmax(dim=-1), num_classes=self.num_types).float()

            return {
                'H': H.cpu(),
                'Head': Head.cpu(),
                'source_alpha': model_output['source_alpha'].cpu(),
                'W_pos': model_output['W_pos'].cpu(),
                'W_neg': model_output['W_neg'].cpu(),
                'sign_gate': model_output['sign_gate'].cpu(),
                'peaks': model_output['peaks'].cpu(),
                'widths': model_output['widths'].cpu(),
                'mix_weights': model_output['mix_weights'].cpu(),
                'b_k': model_output['b_k'].cpu(),
                'rule_bias': model_output['rule_bias'].cpu(),
            }
