"""Standard transformer: full FFN on every token, no modulation."""
from typing import Optional

import torch
import torch.nn as nn

from ..flops import FLOPsCounter
from .common import make_causal_mask


class BaselineLayer(nn.Module):
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
        self.n_heads = cfg.n_heads

    def forward(self, x, causal_mask=None, pad_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x2 = self.norm1(x + attn_out)
        x3 = self.norm2(x2 + self.ffn(x2))

        B, S, D = x.shape
        real = float(pad_mask.sum().item()) if pad_mask is not None else float(B * S)
        flops = FLOPsCounter.attention(B, S, D, self.n_heads) + \
            FLOPsCounter.ffn_per_token(D, self.ffn[0].out_features) * real
        return x3, flops


class BaselineTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([BaselineLayer(cfg) for _ in range(cfg.n_layers)])
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
        for layer in self.layers:
            x, flops = layer(x, causal_mask=causal_mask, pad_mask=pad_mask)
            total_flops += flops

        x = self.norm(x)
        logits = self.head(x)
        info = {
            'flops': total_flops,
            'sparsity_penalty': torch.zeros((), device=x.device),
            'gate_mean': torch.ones(B, S, device=x.device),
            'entropy_mean': None,
            'chemical_states': None,
        }
        return logits, info
