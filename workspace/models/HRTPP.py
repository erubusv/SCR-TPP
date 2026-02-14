import torch
import torch.nn as nn
from .base import BaseTPP


class HRTPP(BaseTPP):
    """Hierarchical RNN TPP baseline.

    Two-level GRU: session-level (coarse) + event-level updates.
    Implemented simply as stacked GRUCells with decay.
    """

    def __init__(self, config):
        super().__init__(config)
        num_types = self.num_types
        embed_dim = config.get('embed_dim', 32)
        hidden_dim = config.get('hidden_dim', 64)
        self.gru_coarse = nn.GRUCell(embed_dim, hidden_dim)
        self.gru_fine = nn.GRUCell(hidden_dim, hidden_dim)
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
        h_coarse = torch.zeros(b, self.gru_coarse.hidden_size, device=events.device)
        h = torch.zeros(b, self.gru_fine.hidden_size, device=events.device)
        type_logits = []
        time_rates = []
        prev_t = times[:, 0].clone()
        for t in range(s):
            dt = times[:, t] - prev_t
            if t == 0:
                dt = torch.zeros_like(dt)
            h_coarse = self.decay_hidden(h_coarse, dt)
            emb = self.event_embedding(events[:, t])
            h_coarse = self.gru_coarse(emb, h_coarse)
            h = self.decay_hidden(h, dt)
            h = self.gru_fine(h_coarse, h)
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
