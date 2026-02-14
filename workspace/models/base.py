import torch
import pandas as pd
import numpy as np
import torch.nn as nn


class BaseTPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_types = config['num_types']
        self.pad_token_id = config['pad_token_id']
        
        embed_dim = config.get('embed_dim')
        if embed_dim is None:
            self.event_embedding = None
        else:
            self.event_embedding = nn.Embedding(
                self.num_types + 1,  # +1 for Padding
                embed_dim,
                padding_idx=self.pad_token_id
            )

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_loss(self, batch, model_output, **kwargs):
        nll_loss = self._calculate_nll(batch, model_output)
        reg_loss = self._calculate_regularization(model_output, **kwargs)
        total_loss = nll_loss + reg_loss
        return {'total_loss': total_loss, 'nll_loss': nll_loss, 'reg_loss': reg_loss}

    def _calculate_nll(self, batch, model_output):
        raise NotImplementedError("Subclasses should implement this method.")

    def _calculate_regularization(self, model_output, **kwargs):
        # Default implementation returns zero regularization loss
        return 0.0

    def get_structure(self):
        """
        Black box model; no interpretable structure
        H-NSTPP returns adjacency matrix for interpretability
        """
        return None
    
    def explain_model_parameters(self):
        """
        Interpretation of the rules & interactions learned
        """
        return None
