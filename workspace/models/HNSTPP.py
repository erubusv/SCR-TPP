"""HNSTPP: Hypergraph Neural Spatio-Temporal Point Process.

Piecewise-linear temporal kernel with uniform trapezoidal integral approximation.

Intensity:  λ_k(t) = (b_k + E_k(t)) · exp(-I_k(t)) + ε
    E_k(t) = Σ_r W⁺_r · Head_{r,k} · p_r(t)
    I_k(t) = Σ_r W⁻_r · Head_{r,k} · p_r(t)
    p_r(t) = ReLU(S_r(t) − bias_r)
    S_r(t) = Σ_{j: t_j<t} H_{k_j,r} · g_r(t − t_j)      ← binary mask only
    g_r(·) = piecewise-linear kernel on [0, max_cap]  (M bins, M−1 heights)
"""

from .base import BaseTPP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HNSTPP(BaseTPP):
    """HNSTPP with piecewise-linear kernels and trapezoidal integration."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_types = config['num_types']
        self.num_rules = config['num_rules']
        self.num_bins = int(config.get('num_bins', 20))
        self.tau = float(config.get('start_tau', 5.0))
        self.pad_token_id = config.get('pad_token_id', config['num_types'])
        self.eps = float(config.get('epsilon', 1e-6))
        self.i_max = float(config.get('i_max', 20.0))
        self.max_cap = config.get('max_cap', None)

        R, K, M = self.num_rules, self.num_types, self.num_bins

        # --- Rule structure (binary source mask H, target Head) ---
        init_std = 1.0 / np.sqrt(K)
        self.theta = nn.Parameter(torch.randn(K, R) * init_std)
        self.rule_target_logits = nn.Parameter(torch.randn(R, K) * 0.5)

        # --- Excitation / inhibition weights ---
        self.w_exc_raw = nn.Parameter(torch.full((R,), float(config.get('init_w_exc', -1.0))))
        self.w_inh_raw = nn.Parameter(torch.full((R,), float(config.get('init_w_inh', -1.0))))
        sign_std = float(config.get('init_sign_std', 0.0))
        self.sign_logits = nn.Parameter(torch.randn(R) * sign_std if sign_std > 0 else torch.zeros(R))

        # --- Rule bias ---
        self.rule_bias_raw = nn.Parameter(torch.full((R,), float(config.get('init_bias_raw', 0.0))))

        # --- Piecewise-linear temporal kernel ---
        # M bins → M+1 grid points; endpoints fixed at 0 → M−1 learnable heights
        self.kernel_height_raw = nn.Parameter(torch.zeros(R, M - 1))

        # --- Base intensity ---
        self.b0 = nn.Parameter(torch.full((K,), float(config.get('init_b0', -3.0))))

        # Runtime-only inhibition family hypotheses.
        self.register_buffer('family_hyp_active', torch.zeros(R, dtype=torch.float32))
        self.register_buffer('family_hyp_target', torch.full((R,), -1, dtype=torch.long))
        self.register_buffer('family_hyp_order', torch.zeros(R, dtype=torch.long))
        self.register_buffer('family_hyp_mode', torch.full((R,), -1, dtype=torch.long))  # 0=proxy, 1=exclusive
        self.register_buffer('family_hyp_sources', torch.full((R, 3), -1, dtype=torch.long))
        self.register_buffer('family_hyp_weight', torch.zeros(R, 7, dtype=torch.float32))

    def clear_family_hypotheses(self):
        self.family_hyp_active.zero_()
        self.family_hyp_target.fill_(-1)
        self.family_hyp_order.zero_()
        self.family_hyp_mode.fill_(-1)
        self.family_hyp_sources.fill_(-1)
        self.family_hyp_weight.zero_()
    def register_family_hypothesis(self, slot: int, target: int, sources, mode: int, weights):
        srcs = tuple(int(s) for s in sources)
        if len(srcs) not in (2, 3):
            raise ValueError("family hypothesis expects order 2 or 3")
        self.family_hyp_active[slot] = 1.0
        self.family_hyp_target[slot] = int(target)
        self.family_hyp_order[slot] = len(srcs)
        self.family_hyp_mode[slot] = int(mode)
        self.family_hyp_sources[slot].fill_(-1)
        for i, s in enumerate(srcs):
            self.family_hyp_sources[slot, i] = int(s)
        w = torch.as_tensor(weights, dtype=self.family_hyp_weight.dtype, device=self.family_hyp_weight.device)
        self.family_hyp_weight[slot].zero_()
        self.family_hyp_weight[slot, : w.numel()] = w

    def set_max_cap(self, time_diffs: torch.Tensor, event_types: torch.Tensor):
        """Compute and set ``max_cap`` from a batch of time-difference data.

        Filters out padding tokens, then uses
        ``max_cap_percentile`` × ``max_cap_multiplier`` of the empirical
        distribution of positive inter-event times.
        """
        valid = event_types != self.pad_token_id
        td = time_diffs[valid].cpu().float().numpy()
        td = td[td > 0]
        if len(td) == 0:
            self.max_cap = 1.0
        else:
            pct = float(self.config.get('max_cap_percentile', 0.95))
            mul = float(self.config.get('max_cap_multiplier', 5.0))
            self.max_cap = float(np.percentile(td, pct * 100) * mul)
        print(f"max_cap = {self.max_cap:.4f}")

    def get_kernel_heights(self):
        """Non-negative interior heights via softplus.  (R, M−1)."""
        return F.softplus(self.kernel_height_raw)

    def get_rule_matrices(self, tau=None):
        """H (source mask) and Head (target) with straight-through estimator."""
        tau = max(float(tau or self.tau), 1e-6)
        H_soft = torch.sigmoid(self.theta / tau)
        Head_soft = F.softmax(self.rule_target_logits / tau, dim=-1)

        H_hard = (H_soft > 0.5).float()
        Head_hard = F.one_hot(Head_soft.argmax(-1), num_classes=self.num_types).float()

        if self.training:
            H = H_hard + H_soft - H_soft.detach()
            Head = Head_hard + Head_soft - Head_soft.detach()
        else:
            H, Head = H_hard, Head_hard
        return H, Head, H_soft, Head_soft

    def get_weights(self):
        """W⁺, W⁻, sign_gate   each (R,)."""
        tau_s = max(float(self.config.get('sign_tau', 1.0)), 1e-6)
        w_exc = F.softplus(self.w_exc_raw)
        w_inh = F.softplus(self.w_inh_raw)
        gate = torch.sigmoid(self.sign_logits / tau_s)
        return w_exc * gate, w_inh * (1.0 - gate), gate

    def get_rule_bias(self):
        return F.softplus(self.rule_bias_raw)

    def get_base_intensity(self):
        return F.softplus(self.b0)

    def forward(self, x=None, tau=None):
        tau = tau or self.tau
        H, Head, H_soft, Head_soft = self.get_rule_matrices(tau)
        W_pos, W_neg, sign_gate = self.get_weights()
        return {
            'H': H, 'Head': Head,
            'H_soft': H_soft, 'Head_soft': Head_soft,
            'kernel_heights': self.get_kernel_heights(),
            'W_pos': W_pos, 'W_neg': W_neg, 'sign_gate': sign_gate,
            'rule_bias': self.get_rule_bias(),
            'b_k': self.get_base_intensity(),
        }

    def _eval_kernel(self, dt, heights):
        """Evaluate g_r(dt) for all rules.

        Grid : 0 = t_0 < t_1 < … < t_M = max_cap
        Heights: h_0 = 0, h_1…h_{M−1} (learnable), h_M = 0

        Args:  dt (...)  |  heights (R, M−1)
        Returns: (R, ...)
        """
        M = self.num_bins
        h_full = F.pad(heights, (1, 1), value=0.0)             # (R, M+1)
        bin_w = self.max_cap / M
        valid = (dt >= 0) & (dt < self.max_cap)
        dt_norm = dt.clamp(0, self.max_cap * (1 - 1e-7)) / bin_w
        idx = dt_norm.long().clamp(max=M - 1)
        frac = dt_norm - idx.float()
        flat = idx.reshape(-1)
        R = heights.shape[0]
        h_l = h_full[:, flat].reshape(R, *dt.shape)
        h_r = h_full[:, flat + 1].reshape(R, *dt.shape)
        return (h_l + (h_r - h_l) * frac.unsqueeze(0)) * valid.unsqueeze(0).float()

    # ================================================================== #
    #  Signal computation  (no source_alpha — H binary mask only)          #
    # ================================================================== #

    def _compute_signal_at_events(self, times, event_types, model_output):
        """Causal S_r(t_i) = Σ_{j<i} H_{k_j,r} · g_r(t_i − t_j).

        Returns: (B, L, R)
        """
        B, L = times.shape
        R, K = self.num_rules, self.num_types
        device = times.device
        heights = model_output['kernel_heights']
        H = model_output['H']                                      # (K, R)

        S = times.new_zeros(B, L, R)
        for b in range(B):
            valid = event_types[b] != self.pad_token_id
            vi = torch.nonzero(valid, as_tuple=False).squeeze(1)
            Lv = vi.numel()
            if Lv <= 1:
                continue

            t_b = times[b, vi]
            k_b = event_types[b, vi].clamp(0, K - 1)

            ws = torch.searchsorted(t_b, t_b - self.max_cap, right=False)   # Start indices for valid history within max_cap
            lens = torch.arange(Lv, device=device) - ws
            W = int(lens.max().item())
            if W <= 0:
                continue

            off = torch.arange(W, device=device)
            idx = (ws.unsqueeze(1) + off.unsqueeze(0)).clamp(max=Lv - 1)
            mask = off.unsqueeze(0) < lens.unsqueeze(1)            # (Lv, W)

            dt = t_b.unsqueeze(1) - t_b[idx]                       # (Lv, W)
            kv = self._eval_kernel(dt, heights)                     # (R, Lv, W)
            kv = kv * mask.unsqueeze(0).float()
            h_mask = H[k_b[idx]].permute(2, 0, 1)                  # (R, Lv, W)
            S[b, vi] = (kv * h_mask).sum(dim=2).t()

        return S

    def _compute_signal(self, eval_times, event_times, event_types,
                        model_output):
        """Vectorised S_r(t) at arbitrary eval_times.  Returns (B, Q, R)."""
        heights = model_output['kernel_heights']
        H = model_output['H']                                      # (K, R)
        K = self.num_types

        valid = (event_types != self.pad_token_id)                  # (B, L)
        types = event_types.clamp(0, K - 1)

        dt = eval_times.unsqueeze(2) - event_times.unsqueeze(1)    # (B, Q, L)
        causal = valid.unsqueeze(1) & (dt > 0) & (dt <= self.max_cap)

        kv = self._eval_kernel(dt, heights) * causal.unsqueeze(0).float()  # (R,B,Q,L)
        h_mask = H[types].permute(2, 0, 1).unsqueeze(2)            # (R, B, 1, L)
        return (kv * h_mask).sum(dim=-1).permute(1, 2, 0)          # (B, Q, R)

    def _compute_source_signal(self, eval_times, event_times, event_types, model_output, src_idx: int, slot: int):
        heights = model_output['kernel_heights'][slot:slot + 1]
        valid = (event_types != self.pad_token_id)
        src_mask = (event_types == int(src_idx)) & valid
        dt = eval_times.unsqueeze(2) - event_times.unsqueeze(1)
        causal = src_mask.unsqueeze(1) & (dt > 0) & (dt <= self.max_cap)
        kv = self._eval_kernel(dt, heights)[0]
        return (kv * causal.float()).sum(dim=-1)

    def _compute_family_hyp_inh(self, eval_times, event_times, event_types, model_output):
        active_slots = torch.nonzero(self.family_hyp_active > 0.5, as_tuple=False).squeeze(1)
        if active_slots.numel() == 0:
            return eval_times.new_zeros((*eval_times.shape, self.num_types))

        corr = eval_times.new_zeros((*eval_times.shape, self.num_types))
        single_bias = 0.25

        for slot_t in active_slots.tolist():
            tgt = int(self.family_hyp_target[slot_t].item())
            order = int(self.family_hyp_order[slot_t].item())
            mode = int(self.family_hyp_mode[slot_t].item())
            srcs = [int(x) for x in self.family_hyp_sources[slot_t].tolist() if int(x) >= 0]
            if tgt < 0 or order not in (2, 3) or len(srcs) != order:
                continue

            ps = [
                F.relu(self._compute_source_signal(eval_times, event_times, event_types, model_output, src, slot_t) - single_bias)
                for src in srcs
            ]
            w = self.family_hyp_weight[slot_t]

            if order == 2:
                pa, pb = ps
                if mode == 0:
                    phi = w[0] * pa + w[1] * pb
                else:
                    ab = torch.minimum(pa, pb)
                    a_only = F.relu(pa - pb)
                    b_only = F.relu(pb - pa)
                    phi = w[0] * a_only + w[1] * b_only + w[2] * ab
            else:
                pa, pb, pc = ps
                if mode == 0:
                    phi = w[0] * pa + w[1] * pb + w[2] * pc
                else:
                    abc = torch.minimum(torch.minimum(pa, pb), pc)
                    ab = F.relu(torch.minimum(pa, pb) - pc)
                    ac = F.relu(torch.minimum(pa, pc) - pb)
                    bc = F.relu(torch.minimum(pb, pc) - pa)
                    a_only = F.relu(pa - torch.maximum(pb, pc))
                    b_only = F.relu(pb - torch.maximum(pa, pc))
                    c_only = F.relu(pc - torch.maximum(pa, pb))
                    phi = (
                        w[0] * a_only + w[1] * b_only + w[2] * c_only +
                        w[3] * ab + w[4] * ac + w[5] * bc + w[6] * abc
                    )
            corr[..., tgt] += phi
        return corr


    def _compute_exc_inh(self, p_r, model_output):
        E_k = torch.matmul(p_r * model_output['W_pos'], model_output['Head'])
        I_k = torch.matmul(p_r * model_output['W_neg'], model_output['Head'])
        return E_k, I_k

    def compute_intensity(self, eval_times, event_times, event_types,
                          model_output):
        """λ_k(t) = (b_k + E_k) · exp(−I_k) + ε."""
        S = self._compute_signal(eval_times, event_times, event_types,
                                 model_output)
        p_r = F.relu(S - model_output['rule_bias'])
        E_k, I_k = self._compute_exc_inh(p_r, model_output)
        I_k = I_k + self._compute_family_hyp_inh(eval_times, event_times, event_types, model_output)
        I_k = I_k.clamp(0, self.i_max)
        return (model_output['b_k'] + E_k) * torch.exp(-I_k) + self.eps
    

    def compute_integral(self, t_start, t_end, event_times, event_types,
                         model_output, num_points=64):
        squeeze = False
        if t_start.dim() == 1:
            t_start, t_end = t_start.unsqueeze(1), t_end.unsqueeze(1)
            squeeze = True

        B, Q = t_start.shape
        N = num_points + 1
        u = torch.linspace(0, 1, N, device=t_start.device, dtype=t_start.dtype)
        span = (t_end - t_start).clamp(min=0)
        grid = t_start.unsqueeze(-1) + span.unsqueeze(-1) * u

        lam = self.compute_intensity(
            grid.reshape(B, Q * N), event_times, event_types, model_output
        ).sum(dim=-1).reshape(B, Q, N)

        delta = span / num_points
        trap = 0.5 * (lam[..., :-1] + lam[..., 1:])
        integral = (trap * delta.unsqueeze(-1)).sum(dim=-1)
        return integral.squeeze(1) if squeeze else integral

    def _get_sequence_end(self, event_times, event_types):
        B, L = event_times.shape
        valid = (event_types != self.pad_token_id)
        idx = torch.arange(L, device=event_times.device).unsqueeze(0).expand(B, L)
        last = (valid.long() * idx).max(dim=1).values.long()
        t_end = event_times.gather(1, last.unsqueeze(1)).squeeze(1)
        return torch.where(valid.any(dim=1), t_end, torch.zeros_like(t_end))

    def compute_loss(self, batch, model_output, **kwargs):
        time_diffs = batch['time_diffs']
        input_ids = batch['input_ids']
        mask = batch.get('attention_mask', torch.ones_like(time_diffs)).float()
        mask = mask * (input_ids != self.pad_token_id).float()

        times = torch.cumsum(time_diffs, dim=1)
        t_end = self._get_sequence_end(times, input_ids)
        t_start = torch.zeros_like(t_end)

        # Event path shares the same intensity semantics as runtime.
        lam_all = self.compute_intensity(times, times, input_ids, model_output)
        safe_ids = input_ids.clamp(0, self.num_types - 1)
        lam_tgt = lam_all.gather(2, safe_ids.unsqueeze(-1)).squeeze(-1)
        event_ll = (torch.log(lam_tgt) * mask).sum()

        # Integral
        n_pts = int(self.config.get('integral_num_points', 64))
        integral = self.compute_integral(
            t_start, t_end, times, input_ids, model_output, n_pts).sum()

        num_ev = mask.sum() + 1e-9
        nll = (-event_ll + integral) / num_ev
        reg = self._compute_regularization(model_output)

        return {
            'total_loss': nll + reg,
            'nll_loss': nll, 'reg_loss': reg,
            'event_ll': event_ll.item(),
            'integral_loss': integral.item(),
            'num_events': num_ev.item(),
            'b_k_mean': model_output['b_k'].mean().item(),
            'W_pos_mean': model_output['W_pos'].mean().item(),
            'W_neg_mean': model_output['W_neg'].mean().item(),
            'sign_gate_mean': model_output['sign_gate'].mean().item(),
        }

    def _compute_regularization(self, model_output):
        H_soft = model_output['H_soft']
        Head_soft = model_output['Head_soft']
        W_pos, W_neg = model_output['W_pos'], model_output['W_neg']
        sign_gate = model_output['sign_gate']
        heights = model_output['kernel_heights']
        cfg = self.config

        # H orthogonality
        H_n = H_soft / (H_soft.norm(dim=0, keepdim=True) + 1e-9)
        gram = H_n.t() @ H_n
        eye = torch.eye(self.num_rules, device=H_soft.device)
        r_ortho = ((gram - eye) ** 2).mean()

        r_h_sp = H_soft.mean()
        r_h_bin = (H_soft * (1 - H_soft)).mean()
        r_w_sp = W_pos.sum() + W_neg.sum()
        r_overlap = (W_pos * W_neg).sum()
        r_sign = (sign_gate * (1 - sign_gate)).mean()
        r_head = -(Head_soft * torch.log(Head_soft + self.eps)).sum(-1).mean()

        # Kernel smoothness
        h_full = F.pad(heights, (1, 1), value=0.0)
        diffs = h_full[:, 1:] - h_full[:, :-1]
        r_smooth = (diffs ** 2).sum()

        return (
            float(cfg.get('lambda_ortho', 1e-3))          * r_ortho
            + float(cfg.get('lambda_h_sparse', 1e-3))     * r_h_sp
            + float(cfg.get('lambda_h_binary', 0.0))      * r_h_bin
            + float(cfg.get('lambda_w_sparse', 0.0))      * r_w_sp
            + float(cfg.get('lambda_overlap', 1e-2))      * r_overlap
            + float(cfg.get('lambda_sign_binary', 0.0))   * r_sign
            + float(cfg.get('lambda_head_entropy', 1e-3)) * r_head
            + float(cfg.get('lambda_smooth', 1.0))        * r_smooth
        )


    def get_structure(self):
        with torch.no_grad():
            mo = self.forward(tau=0.1)
            return {
                'H': (mo['H'] > 0.5).float().cpu(),
                'Head': F.one_hot(
                    mo['Head'].argmax(-1), self.num_types).float().cpu(),
                'W_pos': mo['W_pos'].cpu(),
                'W_neg': mo['W_neg'].cpu(),
                'sign_gate': mo['sign_gate'].cpu(),
                'kernel_heights': mo['kernel_heights'].cpu(),
                'b_k': mo['b_k'].cpu(),
                'rule_bias': mo['rule_bias'].cpu(),
                'family_hyp_active': (self.family_hyp_active > 0.5).cpu(),
                'family_hyp_target': self.family_hyp_target.cpu(),
                'family_hyp_order': self.family_hyp_order.cpu(),
                'family_hyp_mode': self.family_hyp_mode.cpu(),
                'family_hyp_sources': self.family_hyp_sources.cpu(),
                'family_hyp_weight': self.family_hyp_weight.cpu(),
            }
