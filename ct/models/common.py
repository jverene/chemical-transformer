"""Shared model helpers."""
import math

import torch
import torch.nn.functional as F


def make_causal_mask(S: int, device, dtype) -> torch.Tensor:
    mask = torch.zeros(S, S, device=device, dtype=dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )
    return mask


def manual_attention(x, q_p, k_p, v_p, o_p, n_heads: int, causal_mask,
                     cos=None, sin=None):
    """Multi-head attention that also returns weights (for the entropy signal)."""
    B, S, D = x.shape
    H, Dh = n_heads, D // n_heads
    q = q_p(x).view(B, S, H, Dh).transpose(1, 2)
    k = k_p(x).view(B, S, H, Dh).transpose(1, 2)
    v = v_p(x).view(B, S, H, Dh).transpose(1, 2)
    if cos is not None:
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
    scores = scores + causal_mask
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).contiguous().view(B, S, D)
    return o_p(out), weights


def sdpa_attention(x, q_p, k_p, v_p, o_p, n_heads: int, causal_mask,
                   cos=None, sin=None):
    """Flash/mem-efficient attention via F.scaled_dot_product_attention.

    No attention-weight dropout and no attention weights returned (the
    entropy signal needs the manual path). Semantically identical to
    manual_attention with the same mask.
    """
    B, S, D = x.shape
    H, Dh = n_heads, D // n_heads
    q = q_p(x).view(B, S, H, Dh).transpose(1, 2)
    k = k_p(x).view(B, S, H, Dh).transpose(1, 2)
    v = v_p(x).view(B, S, H, Dh).transpose(1, 2)
    if cos is not None:
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
    out = out.transpose(1, 2).contiguous().view(B, S, D)
    return o_p(out)


def rope_cos_sin(S: int, Dh: int, device, dtype, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, Dh, 2, device=device,
                                       dtype=torch.float32) / Dh))
    t = torch.arange(S, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    """Rotate-half RoPE. x: (B, H, S, Dh); cos/sin: (S, Dh/2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


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
