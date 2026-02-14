"""
NSTPP: Neuro-Symbolic Temporal Point Process (ICML 2024 Style)

This implements the core ideas from Neuro-Symbolic TPP:
- Pairwise logic rules: source_type → target_type with temporal constraints
- Differentiable rule learning via Gumbel-Softmax
- Combines neural intensity with symbolic rule structure

Key difference from HNSTPP: 
- NSTPP learns pairwise rules (one source → one target)
- HNSTPP learns hyperedge rules (multiple sources → one target)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .base import BaseTPP


class NSTPP(BaseTPP):
    """Neuro-Symbolic Temporal Point Process.
    
    Learns pairwise logic rules of the form:
        "If event type A occurs, then event type B is more likely to occur 
         within time window [mu - sigma, mu + sigma]"
    
    Architecture:
        - Pairwise rule matrix: (num_types, num_types) binary indicators
        - Temporal constraints: Gaussian (mu, sigma) per rule
        - Base intensity: learned per event type
        - Neural context encoder: GRU for sequence history
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_types = config['num_types']
        self.emb_dim = config.get('embed_dim', 32)
        self.num_heads = config.get('num_heads', 3)  # Mixture of Gaussians
        self.tau = config.get('start_tau', 1.0)
        self.mc_samples = config.get('mc_samples', 5)
        
        # Pairwise rule logits: theta[i,j] indicates if type i triggers type j
        # Initialize near zero for balanced exploration
        self.theta = nn.Parameter(torch.zeros(self.num_types, self.num_types))
        
        # Temporal parameters for each pairwise rule
        # mu: expected time lag from source to target
        # sigma: tolerance window
        self.rule_mu = nn.Parameter(torch.ones(self.num_types, self.num_types, self.num_heads) * 0.5)
        self.rule_log_sigma = nn.Parameter(torch.zeros(self.num_types, self.num_types, self.num_heads))
        self.rule_w = nn.Parameter(torch.zeros(self.num_types, self.num_types, self.num_heads))
        
        # Base intensity per event type
        self.b0 = nn.Parameter(torch.zeros(self.num_types))
        
        # Context encoder
        self.history_encoder = nn.GRU(self.emb_dim, self.emb_dim, batch_first=True)
        
        # Intensity weight predictor from context
        self.intensity_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.num_types)
        )
        
        # Learnable threshold for temporal satisfaction
        self.theta_temporal = nn.Parameter(torch.tensor(0.0))
        
    def _get_theta_temporal(self):
        return torch.sigmoid(self.theta_temporal)
    
    def straight_through_estimator(self, logits, tau=None):
        """Gumbel-Sigmoid for differentiable discrete sampling."""
        if tau is None:
            tau = self.tau
            
        if not self.training:
            y_soft = torch.sigmoid(logits / tau)
            y_hard = (y_soft > 0.5).float()
            return y_hard, y_soft
        
        # Gumbel noise for exploration
        u1 = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        u2 = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        
        logistic_noise = -torch.log(-torch.log(u1)) + torch.log(-torch.log(u2))
        
        y_soft = torch.sigmoid((logits + logistic_noise) / tau)
        y_hard = (y_soft > 0.5).float()
        
        return (y_hard - y_soft).detach() + y_soft, y_soft
    
    def _compute_temporal_score(self, delta_t, source_types):
        """
        Compute temporal satisfaction score for pairwise rules.
        
        Args:
            delta_t: (B, L, L) time differences matrix
            source_types: (B, L) event types in history
            
        Returns:
            scores: (B, L, L, num_types) temporal satisfaction per target type
        """
        B, L = source_types.shape
        
        # Get temporal parameters for each source type
        # source_types: (B, L) -> index into rule_mu/sigma
        mu = F.softplus(self.rule_mu[source_types])  # (B, L, num_types, num_heads)
        sigma = torch.exp(self.rule_log_sigma[source_types]) + 1e-5  # (B, L, num_types, num_heads)
        w = F.softmax(self.rule_w[source_types], dim=-1)  # (B, L, num_types, num_heads)
        
        # delta_t: (B, L_query, L_history) -> (B, L_query, L_history, 1, 1)
        dt = delta_t.unsqueeze(-1).unsqueeze(-1)
        
        # mu, sigma, w: (B, L_history, num_types, num_heads) -> expand for query
        mu = mu.unsqueeze(1)  # (B, 1, L_history, num_types, num_heads)
        sigma = sigma.unsqueeze(1)
        w = w.unsqueeze(1)
        
        # Gaussian temporal kernel
        gauss = torch.exp(-0.5 * ((dt - mu) / sigma) ** 2)  # (B, L_q, L_h, num_types, num_heads)
        score = torch.sum(w * gauss, dim=-1)  # (B, L_q, L_h, num_types)
        
        return score
    
    def forward(self, x, tau=None):
        if tau is None:
            tau = self.tau
            
        # Get rule structure
        R_hard, R_soft = self.straight_through_estimator(self.theta, tau)
        
        # Encode history
        input_ids = x['input_ids']
        event_embs = self.event_embedding(input_ids)
        context_h, _ = self.history_encoder(event_embs)
        
        return {
            'R_hard': R_hard,
            'R_soft': R_soft,
            'context_h': context_h,
        }
    
    def compute_intensity(self, delta_t, source_types, R_hard, R_soft, context_h, valid_mask=None):
        """
        Compute intensity for each event type at query positions.
        
        Args:
            delta_t: (B, L_query, L_history) time differences
            source_types: (B, L_history) event types
            R_hard: (num_types, num_types) hard rule matrix
            R_soft: (num_types, num_types) soft rule matrix
            context_h: (B, L, emb_dim) context from GRU
            valid_mask: (B, L_query, L_history) causal mask
            
        Returns:
            lambda_per_type: (B, L_query, num_types) intensity per type
        """
        B, L_q, L_h = delta_t.shape
        
        # Temporal scores: how well each history event satisfies temporal constraint
        temporal_scores = self._compute_temporal_score(delta_t, source_types)  # (B, L_q, L_h, num_types)
        
        # Apply causal mask
        if valid_mask is not None:
            temporal_scores = temporal_scores * valid_mask.unsqueeze(-1)
        
        # One-hot encoding of source types
        source_onehot = F.one_hot(source_types, num_classes=self.num_types + 1).float()
        source_onehot = source_onehot[..., :self.num_types]  # (B, L_h, num_types)
        
        # For each target type, aggregate rule activations from all source types
        # R_hard[i,j] = 1 if source i can trigger target j
        # We want: for target j, sum over sources i where R_hard[i,j]=1 and source occurred
        
        # Weighted temporal contribution: temporal_scores * source_presence * rule_weight
        # temporal_scores: (B, L_q, L_h, num_types_target)
        # source_onehot: (B, L_h, num_types_source)
        # R_hard: (num_types_source, num_types_target)
        
        # For each history position, get which rules are active
        # source_onehot @ R_hard: (B, L_h, num_types_target) - which targets can be triggered
        rule_activation = torch.matmul(source_onehot, R_hard)  # (B, L_h, num_types)
        
        # Combine with temporal satisfaction
        # temporal_scores: (B, L_q, L_h, num_types)
        # rule_activation: (B, L_h, num_types) -> (B, 1, L_h, num_types)
        rule_contribution = temporal_scores * rule_activation.unsqueeze(1)
        
        # Aggregate over history
        if valid_mask is not None:
            rule_contribution = rule_contribution * valid_mask.unsqueeze(-1)
        
        # Sum over history positions
        lambda_from_rules = rule_contribution.sum(dim=2)  # (B, L_q, num_types)
        
        # Add context-based intensity
        context_contribution = self.intensity_mlp(context_h)  # (B, L, num_types)
        
        # Base intensity
        base = F.softplus(self.b0)  # (num_types,)
        
        # Combine
        lambda_per_type = base + F.softplus(lambda_from_rules) + F.softplus(context_contribution)
        
        return lambda_per_type
    
    def _calculate_nll(self, batch, model_output):
        time_diffs = batch['time_diffs']
        input_ids = batch['input_ids']
        mask = batch.get('attention_mask', torch.ones_like(time_diffs))
        
        R_hard = model_output['R_hard']
        R_soft = model_output['R_soft']
        context_h = model_output['context_h']
        
        B, L = time_diffs.shape
        
        # Build time matrix
        times = torch.cumsum(time_diffs, dim=1)
        delta_t_matrix = times.unsqueeze(2) - times.unsqueeze(1)
        
        # Causal mask (can only look at past events)
        causal_mask = torch.tril(torch.ones(L, L, device=times.device), diagonal=-1)
        causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
        valid_mask = causal_mask * mask.unsqueeze(1) * mask.unsqueeze(2)
        
        # Shift context for autoregressive prediction
        context_h_shifted = torch.cat([
            torch.zeros_like(context_h[:, :1, :]),
            context_h[:, :-1, :]
        ], dim=1)
        
        # Compute intensity
        lambda_events = self.compute_intensity(
            delta_t_matrix, input_ids, R_hard, R_soft, context_h_shifted, valid_mask
        )
        
        # Event log-likelihood: log(lambda^*(t_i, k_i))
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids == self.pad_token_id] = 0
        target_lambda = lambda_events.gather(2, safe_input_ids.unsqueeze(-1)).squeeze(-1)
        
        safe_lambda = torch.where(mask > 0, target_lambda, torch.ones_like(target_lambda))
        event_ll = torch.sum(torch.log(safe_lambda.clamp_min(1e-6)) * mask)
        
        # Integral approximation via MC sampling
        total_integral = 0
        for _ in range(self.mc_samples):
            rand_ratios = torch.rand((B, L, 1), device=times.device)
            sample_offsets = time_diffs.unsqueeze(-1) * rand_ratios
            
            t_prev = torch.cat([torch.zeros(B, 1, device=times.device), times[:, :-1]], dim=1)
            sample_times = t_prev.unsqueeze(-1) + sample_offsets
            
            dt_samples = sample_times - times.unsqueeze(1)
            sample_mask = (dt_samples > 0).float() * mask.unsqueeze(1)
            
            lambda_samples = self.compute_intensity(
                dt_samples, input_ids, R_hard, R_soft, context_h_shifted, sample_mask
            )
            
            total_integral += lambda_samples.sum(dim=-1).squeeze(-1)
        
        avg_lambda = total_integral / self.mc_samples
        integral = (avg_lambda * time_diffs).sum() * mask.sum() / mask.sum().clamp_min(1)
        non_event_ll = torch.sum((avg_lambda * time_diffs) * mask)
        
        num_events = mask.sum() + 1e-9
        loss = -(event_ll - non_event_ll) / num_events
        
        return loss
    
    def _calculate_regularization(self, model_output):
        R_soft = model_output['R_soft']
        
        # Sparsity: encourage fewer rules
        lambda_sparse = float(self.config.get('lambda_sparse', 1e-3))
        reg_sparse = torch.sum(torch.abs(R_soft))
        
        return lambda_sparse * reg_sparse
    
    def get_structure(self):
        """Return learned rule structure for interpretation."""
        with torch.no_grad():
            R_hard, R_soft = self.straight_through_estimator(self.theta)
            mu = F.softplus(self.rule_mu)
            sigma = torch.exp(self.rule_log_sigma)
            
        return {
            'rule_matrix': R_hard,
            'rule_matrix_soft': R_soft,
            'rule_mu': mu,
            'rule_sigma': sigma,
        }
    
    def explain_model_parameters(self, event_names=None):
        """Generate human-readable explanation of learned rules."""
        struct = self.get_structure()
        R = struct['rule_matrix'].cpu().numpy()
        mu = struct['rule_mu'].cpu().numpy()
        sigma = struct['rule_sigma'].cpu().numpy()
        
        if event_names is None:
            event_names = [f"Type {i}" for i in range(self.num_types)]
        
        rules = []
        for i in range(self.num_types):
            for j in range(self.num_types):
                if R[i, j] > 0.5:
                    # Find dominant Gaussian head
                    w = F.softmax(self.rule_w[i, j], dim=-1).detach().cpu().numpy()
                    k = np.argmax(w)
                    rules.append({
                        'Source': event_names[i],
                        'Target': event_names[j],
                        'Lag (mu)': f"{mu[i, j, k]:.2f}",
                        'Window (sigma)': f"{sigma[i, j, k]:.2f}",
                        'Confidence': f"{torch.sigmoid(self.theta[i, j]).item():.2f}"
                    })
        
        df = pd.DataFrame(rules)
        return df, pd.DataFrame()  # Return empty interactions df for compatibility
    
    def get_rule_diagnostics(self):
        """Diagnostic info for training monitoring."""
        with torch.no_grad():
            R_hard, R_soft = self.straight_through_estimator(self.theta)
            theta_sigmoid = torch.sigmoid(self.theta)
            
            num_active_rules = (theta_sigmoid > 0.5).sum().item()
            mu_vals = F.softplus(self.rule_mu).cpu().numpy()
            sigma_vals = torch.exp(self.rule_log_sigma).cpu().numpy()
            
        return {
            'num_active_rules': num_active_rules,
            'theta_mean': theta_sigmoid.mean().item(),
            'theta_std': theta_sigmoid.std().item(),
            'mu_mean': mu_vals.mean(),
            'mu_std': mu_vals.std(),
            'sigma_mean': sigma_vals.mean(),
            'sigma_std': sigma_vals.std(),
            'b0_values': F.softplus(self.b0).detach().cpu().numpy().tolist(),
        }
