import torch
import torch.nn as nn
from .base import BaseTPP


class CLUSTER(BaseTPP):
    """Clustering-based mixture TPP baseline.

    Learns K cluster centroids in embedding space; each cluster has per-type base rates.
    Cluster responsibilities are softmax over similarity to last event embedding.
    """

    def __init__(self, config):
        super().__init__(config)
        num_types = self.num_types
        embed_dim = config.get('embed_dim', 32)
        hidden_dim = config.get('hidden_dim', 64)
        self.k = config.get('n_clusters', 8)
        self.cluster_proj = nn.Linear(embed_dim, self.k)
        self.cluster_rates = nn.Parameter(torch.rand(self.k, num_types))
        self.softplus = nn.Softplus()

    def forward(self, x):
        events = x['events']
        times = x['times']
        b, s = events.shape
        type_logits = []
        time_rates = []
        for t in range(s):
            emb = self.event_embedding(events[:, t])
            sim = self.cluster_proj(emb)
            resp = nn.functional.softmax(sim, dim=-1)
            rates = torch.matmul(resp, self.cluster_rates)
            # rates shape: (b, num_types)
            type_logits.append(rates)
            # aggregate to single-rate for time prediction
            rate_time = torch.clamp(rates.sum(dim=-1), min=1e-6)
            time_rates.append(rate_time)
        return {'type_logits': torch.stack(type_logits, dim=1), 'time_rate': torch.stack(time_rates, dim=1)}

    def _calculate_nll(self, batch, model_output):
        type_logits = model_output['type_logits']
        time_rates = model_output['time_rate']
        target_types = batch['target_types']
        target_dt = batch['target_dt']
        ce = nn.functional.cross_entropy(type_logits.view(-1, type_logits.size(-1)), target_types.view(-1), reduction='mean')
        rate = time_rates.view(-1) + 1e-8
        dt = target_dt.view(-1)
        nll_time = torch.mean(-torch.log(rate) + rate * dt)
        return ce + nll_time

    def _calculate_regularization(self, model_output):
        return 0.0
