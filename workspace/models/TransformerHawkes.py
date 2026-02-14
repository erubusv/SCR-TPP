import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerHawkes(nn.Module):
    def __init__(self, num_types, pad_id=0, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_types = num_types
        self.d_model = d_model
        self.pad_id = pad_id

        # event type embedding (padding_idx used)
        self.type_emb = nn.Embedding(num_types + 1, d_model, padding_idx=pad_id)

        # position vector for temporal positional encoding (register as buffer)
        pos = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)], dtype=torch.float)
        self.register_buffer('position_vec', pos)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # type prediction (including pad token)
        self.proj_type = nn.Linear(d_model, num_types + 1)

        # per-type linear projection used to compute base intensity before applying softplus
        self.linear_lambda = nn.Linear(d_model, num_types)

        # time-scaling parameter (alpha) and softplus beta (match THP paper)
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _temporal_enc(self, times, non_pad_mask):
        """Sine/cosine style temporal encoding using absolute event times.

        times: (B, L) absolute event times (float)
        non_pad_mask: (B, L, 1) float mask (1.0 for real positions, 0.0 for padding)
        returns: (B, L, d_model)
        """
        result = times.unsqueeze(-1) / self.position_vec
        result = result.clone()
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def _softplus(self, x, beta, clamp=20.0):
        # numerically stable scaled softplus
        temp = beta * x
        temp = torch.clamp(temp, max=clamp)
        return (1.0 / beta) * torch.log1p(torch.exp(temp))

    def forward(self, types, times, src_key_padding_mask=None):
        """Forward encoder pass.

        types: (B, L) long tensor with event type ids (pad id included)
        times: (B, L) float tensor absolute event times (zeros in padding positions)
        src_key_padding_mask: (B, L) bool mask where True indicates padding (same convention as current collate)

        Returns: logits (B,L,num_types+1), all_hid (B,L,num_types), hidden (B,L,d_model)
        Note: `all_hid` is the per-type linear output used for intensity computation (before softplus).
        """
        device = types.device
        B, L = types.shape

        if src_key_padding_mask is None:
            src_key_padding_mask = (types == self.pad_id)

        non_pad_mask = (~src_key_padding_mask).unsqueeze(-1).float() 

        # embeddings + temporal encoding
        t_emb = self.type_emb(types)
        tem_enc = self._temporal_enc(times, non_pad_mask)
        x = t_emb + tem_enc

        # subsequent (causal) mask to prevent attending to future events
        subsequent_mask = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(x, mask=subsequent_mask, src_key_padding_mask=src_key_padding_mask)

        logits = self.proj_type(hidden)
        all_hid = self.linear_lambda(hidden)  # (B,L,num_types)

        return logits, all_hid, hidden

    def predict_event_rates(self, batch, model_output=None):
        """Compute per-event per-type intensities and normalized type probs for given candidate dt offsets.

        batch: dict with keys 'times' (B,L), 'tgt_dt' (B,L), 'tgt_type' (B,L, optional), 'mask' (B,L)
        model_output: optional tuple returned from forward (logits, all_hid, hidden) to avoid recomputing encoder

        Returns dict with:
          - total_rate: (B,L) total intensity (sum over types) at the candidate dt offsets
          - prob: (B,L,num_types) normalized per-type intensity
          - pred_dt: (B,L) predicted expected inter-event time (1.0 / total_rate)
          - mask: (B,L) same mask as input
          - lambda_events: (B,L,num_types) raw per-type intensities
        """
        device = batch['times'].device
        types = batch['types']
        times = batch['times']
        pad_mask = batch['mask']  # True: padding

        if model_output is None:
            logits, all_hid, hidden = self.forward(types, times, src_key_padding_mask=pad_mask)
        else:
            _, all_hid, _ = model_output

        if 'tgt_dt' in batch:
            dt = batch['tgt_dt']
        else:
            dt = torch.zeros_like(times)

        # scale dt by previous absolute time (t_prev + 1) per THP
        if 'times' in batch:
            times = batch['times']
            # previous times are the times at the same index (t_prev for dt at index j)
            t_prev = times
            # +1 to prevent division by zero
            scaled_dt = dt / (t_prev + 1.0)
        else:
            scaled_dt = dt

        dt_exp = scaled_dt.unsqueeze(-1)
        raw = all_hid + self.alpha * dt_exp
        lambda_events = self._softplus(raw, self.beta)

        # intensity at dt = 0 (previous event instant)
        lambda_prev = self._softplus(all_hid, self.beta)

        total_rate = lambda_events.sum(dim=-1) + 1e-9
        total_rate_prev = lambda_prev.sum(dim=-1) + 1e-9
        prob = lambda_events / total_rate.unsqueeze(-1)
        pred_dt = (1.0 / total_rate)

        return {
            'total_rate': total_rate,
            'total_rate_prev': total_rate_prev,
            'prob': prob,
            'pred_dt': pred_dt,
            'mask': pad_mask,
            'lambda_events': lambda_events,
            'lambda_prev': lambda_prev
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
