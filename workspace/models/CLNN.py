import torch
import torch.nn as nn
from .base import BaseTPP


class CLNN(BaseTPP):
    """Continuous-time LNN-like baseline: decay + GRU update per event.

    Implements hidden-state exponential decay between events then GRU update.
    """

    def __init__(self, config):
        super().__init__(config)
        num_types = self.num_types
        embed_dim = config.get('embed_dim', 32)
        hidden_dim = config.get('hidden_dim', 64)
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self.decay = nn.Parameter(torch.randn(hidden_dim))
        self.out_time = nn.Linear(hidden_dim, 1)
        self.out_type = nn.Linear(hidden_dim, num_types)
        self.softplus = nn.Softplus()

    def decay_hidden(self, h, dt):
        return h * torch.exp(-torch.relu(self.decay) * dt.unsqueeze(-1))

    def forward(self, x):
        events = x['events']
        times = x['times']
        b, s = events.shape
        h = torch.zeros(b, self.gru.hidden_size, device=events.device)
        type_logits = []
        time_rates = []
        prev_t = times[:, 0].clone()
        for t in range(s):
            dt = times[:, t] - prev_t
            if t == 0:
                dt = torch.zeros_like(dt)
            h = self.decay_hidden(h, dt)
            emb = self.event_embedding(events[:, t])
            h = self.gru(emb, h)
            logits = self.out_type(h)
            rate = self.softplus(self.out_time(h)).squeeze(-1) + 1e-6
            type_logits.append(logits)
            time_rates.append(rate)
            prev_t = times[:, t]
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
