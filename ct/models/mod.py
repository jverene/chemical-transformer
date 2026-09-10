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
from .common import (make_causal_mask, sdpa_attention, rope_cos_sin)


class MoDLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_impl = getattr(cfg, "attn_impl", "manual")
        self.n_heads = cfg.n_heads
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
        self.capacity = cfg.mod_capacity
        if self.attn_impl == "sdpa":
            self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
        else:
            self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                              dropout=cfg.dropout, batch_first=True)

    def forward(self, x, causal_mask=None, pad_mask=None, cos=None, sin=None):
        B, S, D = x.shape
        if self.attn_impl == "sdpa":
            attn_out = sdpa_attention(x, self.q_proj, self.k_proj,
                                      self.v_proj, self.o_proj,
                                      self.n_heads, causal_mask,
                                      cos=cos, sin=sin)
        else:
            attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x2 = self.norm1(x + attn_out)

        scores = self.router(x2).squeeze(-1)  # (B, S)
        k = max(1, int(math.ceil(self.capacity * S)))
        masked = scores if pad_mask is None else scores.masked_fill(~pad_mask, -1e9)
        topv, topi = masked.topk(k, dim=1)    # (B, k)
        w = torch.sigmoid(topv)

        sel = x2.gather(1, topi.unsqueeze(-1).expand(-1, -1, D))
        # autocast keeps x2 fp32 (LayerNorm) but the FFN runs bf16; the
        # in-place scatter_ requires matching dtypes, so cast explicitly.
        out = (self.ffn(sel) * w.unsqueeze(-1)).to(x2.dtype)
        ffn_full = torch.zeros_like(x2)
        ffn_full.scatter_(1, topi.unsqueeze(-1).expand(-1, -1, D), out)
        x3 = self.norm2(x2 + ffn_full)

        gate_map = torch.zeros(B, S, device=x.device, dtype=x.dtype)
        gate_map.scatter_(1, topi, w.to(gate_map.dtype))

        flops = FLOPsCounter.attention(B, S, D, self.n_heads) + \
            FLOPsCounter.ffn_per_token(D, self.ffn[0].out_features) * float(k * B)
        return x3, flops, gate_map


class MoDTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pos_type = getattr(cfg, "pos_type", "learned")
        self.grad_checkpoint = getattr(cfg, "grad_checkpoint", False)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if self.pos_type == "learned":
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        else:
            self.pos_emb = None
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
                meta: Optional[dict] = None, collect_chemical: bool = False,
                oracle_gates: Optional[list] = None):
        B, S = x.shape
        if self.pos_type == "rope":
            x = self.dropout(self.token_emb(x))
            cos, sin = rope_cos_sin(S, self.cfg.d_model // self.cfg.n_heads,
                                    x.device, x.dtype)
        else:
            pos = torch.arange(S, device=x.device).unsqueeze(0)
            x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
            cos = sin = None
        causal_mask = make_causal_mask(S, x.device, x.dtype)

        total_flops = 0.0
        gates = []
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x, flops, gate_map = checkpoint(layer, x, causal_mask, pad_mask,
                                                cos, sin, use_reentrant=False)
            else:
                x, flops, gate_map = layer(x, causal_mask=causal_mask,
                                           pad_mask=pad_mask, cos=cos, sin=sin)
            total_flops = total_flops + flops
            gates.append(gate_map)

        x = self.norm(x)
        logits = self.head(x)
        info = {
            'flops': total_flops,
            'sparsity_penalty': torch.zeros((), device=x.device),
            'gate_mean': torch.stack(gates, dim=0).mean(dim=0),  # (B, S)
            'gates_per_layer': torch.stack(gates, dim=0),
            'entropy_mean': None,
            'chemical_states': None,
        }
        return logits, info
