import torch
import pandas as pd
import numpy as np
import torch.nn as nn


class BaseTPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_types = config['num_event_types']
        self.pad_token_id = config['pad_token_id']
        
        self.event_embedding = nn.Embedding(
            self.num_types + 1,  # +1 for Padding
            config['embed_dim'], 
            padding_idx=self.pad_token_id
        )

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_loss(self, batch, model_output):
        nll_loss = self._calculate_nll(batch, model_output)
        reg_loss = self._calculate_regularization(model_output)
        total_loss = nll_loss + reg_loss
        return {'total_loss': total_loss, 'nll_loss': nll_loss.item(), 'reg_loss': reg_loss.item()}

    def _calculate_nll(self, batch, model_output):
        raise NotImplementedError("Subclasses should implement this method.")

    def _calculate_regularization(self, model_output):
        # Default implementation returns zero regularization loss
        return 0.0

    def get_structure(self):
        """
        Black box model; no interpretable structure
        H-NSTPP returns adjacency matrix for interpretability
        """
        return None
    
    def explain_model_parameters(self, event_names=None):
        struct = self.get_structure()

        rule_defs = struct['rule_definitions'].cpu().numpy()
        alphas = struct['interaction_alpha'].cpu().numpy()
        betas = struct['interaction_beta'].cpu().numpy()
        
        raw_mu = struct['rule_mu'].cpu().numpy()
        raw_sigma = struct['rule_sigma'].cpu().numpy()
        rule_targets = struct['rule_target'].cpu().numpy()

        num_rules = self.num_rules
        num_types = self.num_types
        
        if event_names is None:
            event_names = [f"Type {i}" for i in range(num_types)]
        
        rule_list = []
        
        for r_idx in range(num_rules):
            required_type_indices = np.where(rule_defs[r_idx] > 0.5)[0]
            
            if len(required_type_indices) == 0:
                condition_str = "Empty rule"
            else:
                conditions = []
                for t_idx in required_type_indices:
                    t_name = event_names[t_idx]
                    mu_val = raw_mu[t_idx, r_idx, :]
                    sigma_val = raw_sigma[t_idx, r_idx, :]
                    
                    head_strs = []
                    for h in range(self.num_heads):
                        head_strs.append(f"{mu_val[h]:.2f}(±{sigma_val[h]:.2f})")
                    
                    time_info = " | ".join(head_strs)
                    
                    cond = f"[{t_name}: {time_info}]"
                    conditions.append(cond)
                
                condition_str = " AND ".join(conditions)
                
            rule_list.append({
                "Rule ID": f"Rule {r_idx}",
                "Logic & Temporal Constraints": condition_str,
                "Active Types Count": len(required_type_indices),
                "Rule target": f"Event {rule_targets[r_idx]}"
            })
            
        df_rules = pd.DataFrame(rule_list)
        
        interaction_list = []
        
        for target in range(num_rules):
            for source in range(num_rules):
                alpha_val = alphas[target, source]
                beta_val = betas[target, source]
                    
                # Half-life = ln(2) / beta
                duration_val = np.log(2) / (beta_val + 1e-9)
                
                interaction_type = "Excitation " if alpha_val > 0 else "Inhibition "
                
                interaction_list.append({
                    "Source Rule": f"Rule {source}",
                    "Target Rule": f"Rule {target}",
                    "Type": interaction_type,
                    "Strength (Alpha)": f"{alpha_val:.4f}",
                    "Decay Rate (Beta)": f"{beta_val:.4f}",
                    "Effective Duration (Half-life)": f"{duration_val:.2f}"
                })
                
        df_interactions = pd.DataFrame(interaction_list)
        
        if not df_interactions.empty:
            df_interactions['abs_strength'] = df_interactions['Strength (Alpha)'].astype(float).abs()
            df_interactions = df_interactions.sort_values(by='abs_strength', ascending=False).drop(columns=['abs_strength'])

        return df_rules, df_interactions