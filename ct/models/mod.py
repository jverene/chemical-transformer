"""Mixture-of-Depths baseline (Raposo et al., 2024).

Each layer routes tokens through a learned scalar router; only the top-k
tokens (k = capacity_ratio * S) are processed by the FFN, the rest pass
through the residual stream. Selected tokens' FFN outputs are weighted by the
router probability so the router receives task-loss gradients.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

from ..flops import FLOPsCounter
from .common import make_causal_mask


class MoDLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.router = nn.Linear(cfg.d_model, 1)
        self.n_heads = cfg.n_heads
        self.capacity = cfg.mod_capacity

    def forward(self, x, causal_mask=None, pad_mask=None):
        B, S, D = x.shape
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x2 = self.norm1(x + attn_out)

        scores = self.router(x2).squeeze(-1)  # (B, S)
        k = max(1, int(math.ceil(self.capacity * S)))
        masked = scores if pad_mask is None else scores.masked_fill(~pad_mask, -1e9)
        topv, topi = masked.topk(k, dim=1)    # (B, k)
        w = torch.sigmoid(topv)

        sel = x2.gather(1, topi.unsqueeze(-1).expand(-1, -1, D))
        out = self.ffn(sel) * w.unsqueeze(-1)
        ffn_full = torch.zeros_like(x2)
        ffn_full.scatter_(1, topi.unsqueeze(-1).expand(-1, -1, D), out)
        x3 = self.norm2(x2 + ffn_full)

        gate_map = torch.zeros(B, S, device=x.device, dtype=x.dtype)
        gate_map.scatter_(1, topi, w)

        flops = FLOPsCounter.attention(B, S, D, self.n_heads) + \
            FLOPsCounter.ffn_per_token(D, self.ffn[0].out_features) * float(k * B)
        return x3, flops, gate_map


class MoDTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([MoDLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None,
                meta: Optional[dict] = None, collect_chemical: bool = False):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        causal_mask = make_causal_mask(S, x.device, x.dtype)

        total_flops = 0.0
        gates = []
        for layer in self.layers:
            x, flops, gate_map = layer(x, causal_mask=causal_mask, pad_mask=pad_mask)
            total_flops += flops
            gates.append(gate_map)

        x = self.norm(x)
        logits = self.head(x)
        info = {
            'flops': total_flops,
            'sparsity_penalty': torch.zeros((), device=x.device),
            'gate_mean': torch.stack(gates, dim=0).mean(dim=0),  # (B, S)
            'entropy_mean': None,
            'chemical_states': None,
        }
        return logits, info
