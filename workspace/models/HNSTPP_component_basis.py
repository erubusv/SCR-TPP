"""Fixed-target component-basis point process model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTPP


class HNSTPPComponentBasis(BaseTPP):
    """Bounded source evidence + conjunctive subset basis for one fixed target."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_types = int(config["num_types"])
        self.fixed_target = int(config["fixed_target"])
        self.pad_token_id = int(config.get("pad_token_id", self.num_types))
        self.eps = float(config.get("epsilon", 1e-6))
        self.i_max = float(config.get("i_max", 20.0))
        self.max_cap = float(config["max_cap"])

        structure = config["structure"]
        source_defs = structure["source_defs"]
        atoms = structure["atoms"]
        base_rates = np.asarray(structure["base_rates"], dtype=np.float32)

        self.num_sources = len(source_defs)
        self.num_atoms = len(atoms)

        self.register_buffer(
            "source_types",
            torch.as_tensor([int(sd["source"]) for sd in source_defs], dtype=torch.long),
        )
        self.register_buffer(
            "source_peak",
            torch.as_tensor([float(sd["peak"]) for sd in source_defs], dtype=torch.float32),
        )
        self.register_buffer(
            "source_width",
            torch.as_tensor([float(sd["width"]) for sd in source_defs], dtype=torch.float32),
        )
        self.register_buffer(
            "source_beta",
            torch.as_tensor([float(sd["beta"]) for sd in source_defs], dtype=torch.float32),
        )
        self.register_buffer(
            "source_alpha",
            torch.as_tensor([float(sd["alpha"]) for sd in source_defs], dtype=torch.float32),
        )

        atom_mask = torch.zeros((self.num_atoms, self.num_sources), dtype=torch.float32)
        atom_orders = torch.zeros((self.num_atoms,), dtype=torch.long)
        atom_sources = torch.full((self.num_atoms, 3), -1, dtype=torch.long)
        local_by_source = {int(sd["source"]): i for i, sd in enumerate(source_defs)}
        for atom in atoms:
            atom_id = int(atom["atom_id"])
            srcs = tuple(int(s) for s in atom["sources"])
            atom_orders[atom_id] = int(atom["order"])
            for j, src in enumerate(srcs):
                atom_mask[atom_id, local_by_source[src]] = 1.0
                if j < atom_sources.shape[1]:
                    atom_sources[atom_id, j] = int(src)
        self.register_buffer("atom_mask", atom_mask)
        self.register_buffer("atom_order", atom_orders)
        self.register_buffer("atom_sources", atom_sources)

        init_we = np.asarray(structure.get("w_exc", np.zeros((self.num_atoms,), dtype=np.float32)), dtype=np.float32)
        init_wi = np.asarray(structure.get("w_inh", np.zeros((self.num_atoms,), dtype=np.float32)), dtype=np.float32)
        self.w_exc_raw = nn.Parameter(self._softplus_inverse(torch.as_tensor(init_we, dtype=torch.float32).clamp(min=1e-6)))
        self.w_inh_raw = nn.Parameter(self._softplus_inverse(torch.as_tensor(init_wi, dtype=torch.float32).clamp(min=1e-6)))

        self.register_buffer("base_rates_full", torch.as_tensor(base_rates, dtype=torch.float32))
        tgt_rate = max(float(base_rates[self.fixed_target]), 1e-5)
        self.b0_target_raw = nn.Parameter(self._softplus_inverse(torch.tensor(tgt_rate, dtype=torch.float32)))

    @staticmethod
    def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(x.clamp(min=1e-6)))

    def forward(self, x=None, tau=None):
        return {
            "w_exc": F.softplus(self.w_exc_raw),
            "w_inh": F.softplus(self.w_inh_raw),
            "b_target": F.softplus(self.b0_target_raw),
        }

    def _triangular_kernel(self, dt: torch.Tensor, peak: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        valid = (dt > 0.0) & (dt < width)
        peak = peak.clamp(min=1e-6, max=(width - 1e-6).clamp(min=1e-6))
        left = dt <= peak
        vals = torch.zeros_like(dt)
        vals = torch.where(valid & left, dt / peak, vals)
        vals = torch.where(valid & (~left), (width - dt) / (width - peak).clamp(min=1e-6), vals)
        return vals.clamp(min=0.0)

    def _compute_q_sources(self, eval_times: torch.Tensor, event_times: torch.Tensor, event_types: torch.Tensor) -> torch.Tensor:
        dt = eval_times.unsqueeze(2) - event_times.unsqueeze(1)  # (B, Q, L)
        q_list = []
        valid = event_types != self.pad_token_id
        for j in range(self.num_sources):
            src = int(self.source_types[j].item())
            peak = self.source_peak[j]
            width = self.source_width[j]
            beta = self.source_beta[j]
            alpha = self.source_alpha[j]
            src_mask = (event_types == src) & valid
            causal = src_mask.unsqueeze(1) & (dt > 0.0) & (dt < width)
            z = (self._triangular_kernel(dt, peak, width) * causal.float()).sum(dim=-1)
            q = 1.0 - torch.exp(-alpha * torch.relu(z - beta))
            q_list.append(q.clamp(min=0.0, max=1.0))
        return torch.stack(q_list, dim=-1)  # (B, Q, S)

    def _compute_atoms(self, q_sources: torch.Tensor) -> torch.Tensor:
        # q_sources: (B, Q, S), atom_mask: (A, S)
        out = []
        ones = torch.ones_like(q_sources[..., 0])
        for a in range(self.num_atoms):
            idxs = torch.nonzero(self.atom_mask[a] > 0.5, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                out.append(ones)
            else:
                out.append(torch.prod(q_sources[..., idxs], dim=-1))
        return torch.stack(out, dim=-1)  # (B, Q, A)

    def compute_intensity(self, eval_times: torch.Tensor, event_times: torch.Tensor, event_types: torch.Tensor, model_output: dict) -> torch.Tensor:
        q = self._compute_q_sources(eval_times, event_times, event_types)
        phi = self._compute_atoms(q)
        e_t = torch.matmul(phi, model_output["w_exc"])
        i_t = torch.matmul(phi, model_output["w_inh"]).clamp(max=self.i_max)

        B, Q = eval_times.shape
        lam = self.base_rates_full.to(eval_times.device).view(1, 1, self.num_types).expand(B, Q, self.num_types).clone()
        lam_t = (model_output["b_target"] + e_t).clamp(min=1e-8) * torch.exp(-i_t)
        lam[..., self.fixed_target] = lam_t + self.eps
        lam = lam.clamp(min=self.eps)
        return lam

    def compute_integral(self, t_start: torch.Tensor, t_end: torch.Tensor, event_times: torch.Tensor, event_types: torch.Tensor, model_output: dict, num_points: int = 64) -> torch.Tensor:
        squeeze = False
        if t_start.dim() == 1:
            t_start = t_start.unsqueeze(1)
            t_end = t_end.unsqueeze(1)
            squeeze = True
        bsz, num_q = t_start.shape
        n_grid = num_points + 1
        u = torch.linspace(0.0, 1.0, n_grid, device=t_start.device, dtype=t_start.dtype)
        span = (t_end - t_start).clamp(min=0.0)
        grid = t_start.unsqueeze(-1) + span.unsqueeze(-1) * u
        lam = self.compute_intensity(grid.reshape(bsz, num_q * n_grid), event_times, event_types, model_output)
        lam = lam.sum(dim=-1).reshape(bsz, num_q, n_grid)
        delta = span / max(num_points, 1)
        trap = 0.5 * (lam[..., :-1] + lam[..., 1:])
        integral = (trap * delta.unsqueeze(-1)).sum(dim=-1)
        return integral.squeeze(1) if squeeze else integral

    def _get_sequence_end(self, event_times: torch.Tensor, event_types: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = event_times.shape
        valid = event_types != self.pad_token_id
        idx = torch.arange(seq_len, device=event_times.device).unsqueeze(0).expand(bsz, seq_len)
        last = (valid.long() * idx).max(dim=1).values.long()
        t_end = event_times.gather(1, last.unsqueeze(1)).squeeze(1)
        return torch.where(valid.any(dim=1), t_end, torch.zeros_like(t_end))

    def compute_loss(self, batch, model_output, **kwargs):
        time_diffs = batch["time_diffs"]
        input_ids = batch["input_ids"]
        mask = batch.get("attention_mask", torch.ones_like(time_diffs)).float()
        mask = mask * (input_ids != self.pad_token_id).float()

        times = torch.cumsum(time_diffs, dim=1)
        lam_all = self.compute_intensity(times, times, input_ids, model_output)
        safe_ids = input_ids.clamp(0, self.num_types - 1)
        lam_tgt = lam_all.gather(2, safe_ids.unsqueeze(-1)).squeeze(-1)
        event_ll = (torch.log(lam_tgt.clamp(min=self.eps)) * mask).sum()

        t_end = self._get_sequence_end(times, input_ids)
        t_start = torch.zeros_like(t_end)
        integral = self.compute_integral(
            t_start, t_end, times, input_ids, model_output, int(self.config.get("integral_num_points", 64))
        ).sum()

        num_ev = mask.sum() + 1e-9
        nll = (-event_ll + integral) / num_ev
        overlap_pen = float(self.config.get("lambda_overlap", 0.0)) * (model_output["w_exc"] * model_output["w_inh"]).sum()
        return {
            "total_loss": nll + overlap_pen,
            "nll_loss": nll,
            "reg_loss": overlap_pen,
            "event_ll": float(event_ll.item()),
            "integral_loss": float(integral.item()),
            "num_events": float(num_ev.item()),
            "b_k_mean": float(self.base_rates_full.mean().item()),
            "W_pos_mean": float(model_output["w_exc"].mean().item()) if self.num_atoms > 0 else 0.0,
            "W_neg_mean": float(model_output["w_inh"].mean().item()) if self.num_atoms > 0 else 0.0,
            "sign_gate_mean": 0.0,
        }

    def get_structure(self):
        with torch.no_grad():
            mo = self.forward()
            return {
                "fixed_target": int(self.fixed_target),
                "source_types": self.source_types.cpu(),
                "source_peak": self.source_peak.cpu(),
                "source_width": self.source_width.cpu(),
                "source_beta": self.source_beta.cpu(),
                "source_alpha": self.source_alpha.cpu(),
                "atom_sources": self.atom_sources.cpu(),
                "atom_order": self.atom_order.cpu(),
                "w_exc": mo["w_exc"].cpu(),
                "w_inh": mo["w_inh"].cpu(),
                "base_rates_full": self.base_rates_full.cpu(),
                "b_target": mo["b_target"].cpu(),
            }
