from workspace.models.base import BaseTPP
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HNSTPP(BaseTPP):
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_types = config['num_types']
        self.num_rules = config['num_rules']
        self.emb_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.tau = config['tau']
        self.mc_samples = config['mc_samples']

        self.pad_token_id = config['pad_token_id']

        self.theta = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.num_types, self.num_rules)))
        self.theta_head = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.num_rules, self.num_types)))
        self.W_msg = nn.Linear(self.emb_dim, self.emb_dim)
        self.W_Q = nn.Linear(self.emb_dim, self.emb_dim)
        self.W_K = nn.Linear(self.emb_dim, self.emb_dim)
        self.gru_cell = nn.GRUCell(self.emb_dim, self.emb_dim)

        self.rule_mu = nn.Parameter(torch.randn(self.num_types + 1, self.num_rules, self.num_heads))
        self.rule_log_sigma = nn.Parameter(torch.randn(self.num_types + 1, self.num_rules, self.num_heads))
        self.rule_w = nn.Parameter(torch.randn(self.num_types + 1, self.num_rules, self.num_heads))

        self.intensity_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
        )
        self.b0 = nn.Parameter(torch.full((self.num_types,), -2.0))

        # Predictors for interactions among rules
        self.alpha_predictor = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
            nn.Tanh()
        )
        self.beta_predictor = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
            nn.Softplus()
        )

    
    def straight_through_estimator(self, logits):
        """
        Gumbel-Sigmoid Sampling using Logistic Noise
        """
        u1 = torch.rand_like(logits)
        u2 = torch.rand_like(logits)
        
        gumbels_1 = -torch.log(-torch.log(u1 + 1e-9) + 1e-9)
        gumbels_2 = -torch.log(-torch.log(u2 + 1e-9) + 1e-9)
        
        logistic_noise = gumbels_1 - gumbels_2
        
        y_soft = torch.sigmoid((logits + logistic_noise) / self.tau)
        y_hard = (y_soft > 0.5).float()

        return (y_hard - y_soft).detach() + y_soft, y_soft
    

    def _learn_interaction_graph(self, rule_embs_static, A_adj):
        Z = rule_embs_static.clone()

        for _ in range(2):
            M = self.W_msg(Z)
            Q = self.W_Q(Z)
            K = self.W_K(Z)

            scores = torch.matmul(Q, K.t()) / (self.emb_dim ** 0.5)
            scores_masked = scores * A_adj + (1 - A_adj) * -1e9
            attention = F.softmax(scores_masked, dim=-1)
            M_agg = torch.matmul(attention, M)
            Z = self.gru_cell(M_agg, Z)
        
        Z_receiver = Z.unsqueeze(1).expand(-1, self.num_rules, -1)
        Z_sender = Z.unsqueeze(0).expand(self.num_rules, -1, -1)
        Z_pair = torch.cat([Z_receiver, Z_sender], dim=-1)
        alpha_matrix = self.alpha_predictor(Z_pair).squeeze(-1)
        beta_matrix = self.beta_predictor(Z_pair).squeeze(-1)

        return alpha_matrix, beta_matrix


    def forward(self, x):
        H_hard, H_soft = self.straight_through_estimator(self.theta)
        Head_hard = F.gumbel_softmax(self.theta_head, tau=self.tau, hard=True, dim=-1)
        pred_embs = self.event_embedding.weight[:self.num_types]
        rule_embs_static = H_t = torch.matmul(H_soft.t(), pred_embs)

        A_raw = torch.matmul(H_hard.t(), H_hard)
        A_mask = A_raw * (1 - torch.eye(self.num_rules, device=H_hard.device))
        A_adj = (A_mask > 0).float()

        alpha, beta = self._learn_interaction_graph(rule_embs_static, A_adj)
        
        return {
            'rule_embs': rule_embs_static,
            'H_soft': H_soft,
            'H_hard': H_hard,
            'Head_hard': Head_hard,
            'interaction_alpha': alpha,
            'interaction_beta': beta
        }
    

    def _compute_temporal_satisfaction(self, delta_t, input_ids, H_soft):
        """
        Calculates temporal satisfaction for each rule based on Type-Specific Multi-head Gaussians.
        """
        mu_selected = self.rule_mu[input_ids] 
        sigma_selected = torch.exp(self.rule_log_sigma[input_ids]) + 1e-5
        w_selected = F.softmax(self.rule_w[input_ids], dim=-1)

        mu_final = mu_selected.unsqueeze(1)
        sigma_final = sigma_selected.unsqueeze(1)
        w_final = w_selected.unsqueeze(1)

        dt_exp = delta_t.unsqueeze(-1).unsqueeze(-1)

        term = -0.5 * ((dt_exp - mu_final) / sigma_final) ** 2
        val_per_head = torch.exp(term)

        time_match_score = torch.sum(w_final * val_per_head, dim=-1)

        return time_match_score
    

    def compute_intensity(self, delta_t, rule_embs, alpha, beta, past_temporal_scores, input_ids, H_soft, Head_hard, valid_mask=None):
        """
        Computes final intensity by applying Interaction.
        """
        B, Q, H_len = delta_t.shape
        M = self.num_rules

        hist_onehot_full = F.one_hot(input_ids, num_classes=self.num_types + 1).float()
        hist_onehot = hist_onehot_full[..., :self.num_types]
        
        weighted_types = past_temporal_scores.unsqueeze(-1) * hist_onehot.unsqueeze(1).unsqueeze(3)

        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)
            weighted_types = weighted_types * mask_expanded
        
        type_activation = torch.sum(weighted_types, dim=2)

        required_pattern = H_soft.t().view(1, 1, M, self.num_types)
        safe_activation = type_activation + 1e-9
        
        log_logic_strength = torch.sum(required_pattern * torch.log(safe_activation), dim=-1)
        rule_logic_strength = torch.exp(log_logic_strength)

        # Computes interaction effect
        dt_exp = delta_t.unsqueeze(-1).unsqueeze(-1)
        dt_exp_safe = dt_exp.clamp(min=0.0)
        beta_exp = beta.view(1, 1, 1, M, M) 
        alpha_exp = alpha.view(1, 1, M, M)

        decay_kernel = torch.exp(-beta_exp * dt_exp_safe)
        if valid_mask is not None:
            decay_kernel = decay_kernel * valid_mask.unsqueeze(-1).unsqueeze(-1)
        
        act_exp = past_temporal_scores.unsqueeze(-2)

        transmitted = torch.sum(decay_kernel * act_exp, dim=2)
        interaction_effect = torch.sum(transmitted * alpha_exp, dim=-1)

        # Final intensity
        if rule_embs.dim() == 2:
            base_weights = self.intensity_mlp(rule_embs).view(1, 1, M).expand(B, Q, -1)
        else:
            base_weights = self.intensity_mlp(rule_embs).view(B, 1, M).expand(-1, Q, -1)

        raw_intensity = F.softplus(base_weights + interaction_effect)
        rule_activation = raw_intensity * rule_logic_strength
        lambda_per_type = torch.matmul(rule_activation, Head_hard)
        
        return lambda_per_type


    def _calculate_nll(self, batch, model_output):
        time_diffs = batch['time_diffs']
        input_ids = batch['input_ids']
        mask = batch.get('attention_mask', torch.ones_like(time_diffs))
        
        rule_embs = model_output['rule_embs']
        H_soft = model_output['H_soft']
        alpha = model_output['interaction_alpha']
        beta = model_output['interaction_beta']
        Head_hard = model_output['Head_hard']

        B, L = time_diffs.shape

        times = torch.cumsum(time_diffs, dim=1)
        delta_t_matrix = times.unsqueeze(2) - times.unsqueeze(1)
        
        causal_mask = torch.tril(torch.ones(L, L, device=times.device), diagonal=-1)
        causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
        valid_mask_event = causal_mask * mask.unsqueeze(1) * mask.unsqueeze(2)

        seq_temporal_scores = self._compute_temporal_satisfaction(delta_t_matrix, input_ids, H_soft)
        
        dynamic_weights = self.compute_intensity(
            delta_t_matrix, rule_embs, alpha, beta, 
            past_temporal_scores=seq_temporal_scores,
            input_ids=input_ids,
            H_soft=H_soft,
            Head_hard=Head_hard,
            valid_mask=valid_mask_event
        )
        
        lambda_events = F.softplus(self.b0) + dynamic_weights
        # Padding index to 0 (will be masked)
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids == self.pad_token_id] = 0
        target_lambda = lambda_events.gather(2, safe_input_ids.unsqueeze(-1)).squeeze(-1)
        # Sum_{i=1}^N log(lambda^*(t_i))
        event_loss = torch.sum(torch.log(target_lambda + 1e-9) * mask)

        # Integral Loss
        total_integral = 0
        num_samples = self.mc_samples

        for _ in range(num_samples):
            rand_ratios = torch.rand((B, L, 1), device=times.device)
            sampled_dt_offsets = time_diffs.unsqueeze(-1) * rand_ratios
            
            t_prev = torch.cat([torch.zeros(B, 1, device=times.device), times[:, :-1]], dim=1)
            sample_abs_times = t_prev.unsqueeze(-1) + sampled_dt_offsets 
            
            dt_samples_iter = sample_abs_times - times.unsqueeze(1)
            valid_mask_sample_iter = (dt_samples_iter > 0).float() * mask.unsqueeze(1)
            
            sample_temporal_scores_iter = self._compute_temporal_satisfaction(
                dt_samples_iter, input_ids, H_soft
            )

            lambda_samples_weights_iter = self.compute_intensity(
                dt_samples_iter, rule_embs, alpha, beta,
                past_temporal_scores=sample_temporal_scores_iter,
                input_ids=input_ids,
                H_soft=H_soft,
                Head_hard=Head_hard,
                valid_mask=valid_mask_sample_iter
            )

            lambda_samples_iter = F.softplus(self.b0).sum() + torch.sum(lambda_samples_weights_iter, dim=-1)
            total_integral += lambda_samples_iter.squeeze(-1)
        
        avg_lambda = total_integral / num_samples
        integral = avg_lambda * time_diffs
        non_event_loss = torch.sum(integral * mask)

        num_events = torch.sum(mask) + 1e-9
        loss = -(event_loss - non_event_loss) / num_events
        
        return loss
        
    
    def _calculate_regularization(self, model_output):
        H_soft = model_output['H_soft']
        alpha = model_output['interaction_alpha']

        lambda_ortho = float(self.config['lambda_ortho'])
        lambda_sparse = float(self.config['lambda_sparse'])
        lambda_interaction = float(self.config['lambda_interaction'])

        gram_matrix = torch.matmul(H_soft.t(), H_soft)
        reg_ortho = torch.norm(gram_matrix - torch.eye(self.num_rules, device=H_soft.device), p='fro')
        reg_sparse = torch.sum(torch.abs(H_soft))
        reg_interaction = torch.mean(torch.abs(alpha))

        return (lambda_ortho * reg_ortho) + (lambda_sparse * reg_sparse) + (lambda_interaction * reg_interaction)

    def get_structure(self):
        with torch.no_grad():
            H_hard, H_soft = self.straight_through_estimator(self.theta)
            Head_hard = F.gumbel_softmax(self.theta_head, tau=self.tau, hard=True, dim=-1)
            pred_embs = self.event_embedding.weight[:self.num_types]
            rule_embs_static = torch.matmul(H_soft.t(), pred_embs)
            
            A_raw = torch.matmul(H_hard.t(), H_hard)
            A_mask = A_raw * (1 - torch.eye(self.num_rules, device=H_hard.device))
            A_adj = (A_mask > 0).float()
            
            alpha, beta = self._learn_interaction_graph(rule_embs_static, A_adj)

        return {
            'rule_definitions': H_hard.t(), 
            'interaction_alpha': alpha.detach(), 
            'interaction_beta': beta.detach(),
            'rule_mu': self.rule_mu.detach(), 
            'rule_sigma': torch.exp(self.rule_log_sigma).detach(),
            'rule_target': Head_hard.detach()
        }