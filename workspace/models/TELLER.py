import torch
import torch.nn as nn
from .base import BaseTPP


class TELLER(BaseTPP):
    """Simplified TELLER-style baseline: logic-inspired feature aggregator.

    Uses counts of recent events in a fixed window as features, feeds MLP.
    """

    def __init__(self, config):
        super().__init__(config)
        num_types = self.num_types
        embed_dim = config.get('embed_dim', 32)
        hidden_dim = config.get('hidden_dim', 64)
        self.window = config.get('tell_window', 5)
        self.mlp = nn.Sequential(
            nn.Linear(num_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_time = nn.Linear(hidden_dim, 1)
        self.out_type = nn.Linear(hidden_dim, num_types)
        self.softplus = nn.Softplus()

    def forward(self, x):
        events = x['events']
        b, s = events.shape
        type_logits = []
        time_rates = []
        for t in range(s):
            # compute one-hot counts over last `window` events
            start = max(0, t - self.window + 1)
            window_e = events[:, start:t+1]
            one_hot = torch.zeros(events.size(0), self.num_types, device=events.device)
            for c in range(self.num_types):
                one_hot[:, c] = (window_e == c).sum(dim=1).float()
            h = self.mlp(one_hot)
            logits = self.out_type(h)
            rate = self.softplus(self.out_time(h)).squeeze(-1) + 1e-6
            type_logits.append(logits)
            time_rates.append(rate)
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
