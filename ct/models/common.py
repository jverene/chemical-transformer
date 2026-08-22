"""Shared model helpers."""
import math

import torch


def make_causal_mask(S: int, device, dtype) -> torch.Tensor:
    mask = torch.zeros(S, S, device=device, dtype=dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )
    return mask


def manual_attention(x, q_p, k_p, v_p, o_p, n_heads: int, causal_mask):
    """Multi-head attention that also returns weights (for the entropy signal)."""
    B, S, D = x.shape
    H, Dh = n_heads, D // n_heads
    q = q_p(x).view(B, S, H, Dh).transpose(1, 2)
    k = k_p(x).view(B, S, H, Dh).transpose(1, 2)
    v = v_p(x).view(B, S, H, Dh).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
    scores = scores + causal_mask
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).contiguous().view(B, S, D)
    return o_p(out), weights


def normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Per-position attention entropy averaged over heads, normalized by
    log(#attended positions) to remove the trivial position bias. (B, S)"""
    ent = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean(dim=1)
    S = weights.shape[-1]
    norm = torch.clamp(
        torch.log(torch.arange(1, S + 1, device=weights.device, dtype=ent.dtype)),
        min=1e-3,
    )
    return ent / norm
