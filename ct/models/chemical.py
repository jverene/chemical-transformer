"""Transformer with a token-level compute-budget signal ("chemical state").

A per-position signal flows layer-to-layer and gates the FFN per token, making
per-token FLOPs input-dependent. Modes:

  full      : signal driven by normalized attention entropy (the method).
  off       : ablation - pathway frozen, gate fixed at 1.0 (full compute).
  random    : ablation - entropy replaced by uniform noise in the same range.
  fixed     : ablation - gate fixed at cfg.fixed_gate for every token.
  tag       : ablation - gate driven only by the difficulty-tag identity
               (static oracle heuristic; requires meta['diffs']).
  predictor : learned difficulty predictor (Option 2) - Stage 2 uses a
               DifficultyHead MLP to predict per-token gate from hidden state.
               Stage 1 trains with gate=1.0 everywhere (no gating).
"""
from typing import List, Optional

import torch
import torch.nn as nn

from ..flops import FLOPsCounter
from .common import make_causal_mask, manual_attention, normalized_entropy

MODES = ("full", "off", "random", "fixed", "tag", "predictor", "predictor-supervised")


class DifficultyHead(nn.Module):
    """Learned difficulty predictor: maps hidden state -> per-token gate.
    For Option 3b: simple MLP -> sigmoid, no tau parameter.
    """
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, hidden):
        # hidden: (B, T, D)
        score = self.mlp(hidden).squeeze(-1)  # (B, T)
        gate = torch.sigmoid(score)  # (B, T)
        return gate, score


class ChemicalLayer(nn.Module):
    def __init__(self, cfg, mode: str = "full", stage: int = 1):
        super().__init__()
        assert mode in MODES
        self.mode = mode
        self.stage = stage
        self.n_heads = cfg.n_heads
        self.chemical_dim = cfg.chemical_dim
        self.update_alpha = cfg.update_alpha
        self.fixed_gate = cfg.fixed_gate
        self.chem_tag_init = cfg.chem_tag_init
        self.chem_hidden_input = cfg.chem_hidden_input

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

        # Predictor mode (Option 2): learned difficulty head
        if mode in ("predictor", "predictor-supervised"):
            self.diff_head = DifficultyHead(cfg.d_model)
        else:
            self.diff_head = None

        # Secretion: signal -> chemical-state update.
        # chem_hidden_input -> also read the hidden state (x2).
        secret_in = 1 + (cfg.d_model if cfg.chem_hidden_input else 0)
        self.secretion = nn.Sequential(
            nn.Linear(secret_in, 32), nn.ReLU(), nn.Linear(32, cfg.chemical_dim)
        )
        # Receptor: chemical state -> per-token FFN gate logit.
        # Zero-init -> initial gate sigmoid(0) = 0.5 (neutral start).
        self.router = nn.Linear(cfg.chemical_dim, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        # Tag-only mode: embedding over {none, E, M, H} replacing the state.
        self.tag_emb = nn.Embedding(4, cfg.chemical_dim)

        if mode == "off":
            for p in self.secretion.parameters():
                p.requires_grad = False
            for p in self.router.parameters():
                p.requires_grad = False
            for p in self.tag_emb.parameters():
                p.requires_grad = False

    def forward(self, x, prev_chemical, causal_mask, pad_mask, meta):
        B, S, D = x.shape

        attn_out, weights = manual_attention(
            x, self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.n_heads, causal_mask,
        )
        x2 = self.norm1(x + attn_out)

        entropy = None
        if self.mode == "predictor":
            # Stage 1: full compute everywhere (gate = 1.0)
            # Stage 2: learned difficulty predictor from hidden state
            if self.stage == 1:
                g = torch.ones(B, S, device=x.device, dtype=x.dtype)
                score = None
            else:
                # Use hidden state x2 (post-attention) to predict gate
                g, score = self.diff_head(x2)
            chemical = prev_chemical
        elif self.mode == "predictor-supervised":
            # Option 3b: supervised difficulty with budget
            if self.stage == 1:
                g = torch.ones(B, S, device=x.device, dtype=x.dtype)
                score = None
            else:
                # Use hidden state x2 (post-attention) to predict gate
                g, score = self.diff_head(x2)
            chemical = prev_chemical
        elif self.mode == "off":
            g = torch.ones(B, S, device=x.device, dtype=x.dtype)
            chemical = prev_chemical
        elif self.mode == "fixed":
            g = torch.full((B, S), self.fixed_gate, device=x.device, dtype=x.dtype)
            chemical = prev_chemical
        elif self.mode == "tag":
            diffs = meta["diffs"] + 1  # -1 (none) -> 0, 0/1/2 -> 1/2/3
            g = torch.sigmoid(self.router(self.tag_emb(diffs)).squeeze(-1))
            chemical = prev_chemical
        else:
            if self.mode == "random":
                # same marginal range as normalized entropy ([0, 1]), no signal
                stress = torch.rand(B, S, 1, device=x.device, dtype=x.dtype)
            else:  # full
                entropy = normalized_entropy(weights)
                stress = entropy.unsqueeze(-1)
            if self.chem_hidden_input:
                stress = torch.cat([stress, x2], dim=-1)
            new_chem = self.secretion(stress)
            if prev_chemical is None:
                if (self.chem_tag_init and meta is not None
                        and "diffs" in meta and meta["diffs"].max().item() <= 2):
                    chemical = self.tag_emb(meta["diffs"] + 1)
                else:
                    chemical = new_chem
            else:
                a = self.update_alpha
                chemical = (1 - a) * prev_chemical + a * new_chem
            g = torch.sigmoid(self.router(chemical).squeeze(-1))

        if chemical is None:
            chemical = torch.zeros(B, S, self.chemical_dim,
                                   device=x.device, dtype=x.dtype)

        ffn_out = self.ffn(x2) * g.unsqueeze(-1)
        x3 = self.norm2(x2 + ffn_out)

        g_real = g[pad_mask] if pad_mask is not None else g.flatten()
        processed = float(g_real.sum().item())
        flops = FLOPsCounter.attention(B, S, D, self.n_heads) + \
            FLOPsCounter.ffn_per_token(D, self.ffn[0].out_features) * processed

        return x3, chemical, g, flops, entropy, score


class ChemicalTransformer(nn.Module):
    def __init__(self, cfg, mode: str = "full", stage: int = 1):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.stage = stage
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [ChemicalLayer(cfg, mode=mode, stage=stage) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "router" not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None,
                meta: Optional[dict] = None, collect_chemical: bool = False):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        causal_mask = make_causal_mask(S, x.device, x.dtype)

        chemical = None
        total_flops = 0.0
        gates: List[torch.Tensor] = []
        ents: List[torch.Tensor] = []
        diff_scores: List[torch.Tensor] = []
        states: List[torch.Tensor] = []
        for layer in self.layers:
            x, chemical, g, flops, ent, score = layer(x, chemical, causal_mask, pad_mask, meta)
            total_flops += flops
            gates.append(g)
            if ent is not None:
                ents.append(ent)
            if self.mode in ("predictor", "predictor-supervised") and self.stage == 2 and score is not None:
                diff_scores.append(score)
            if collect_chemical:
                states.append(chemical)

        x = self.norm(x)
        logits = self.head(x)

        gate_stack = torch.stack(gates, dim=0)  # (L, B, S)
        if self.mode in ("full", "random", "tag", "predictor") and pad_mask is not None:
            sparsity = gate_stack[:, pad_mask].mean()
        else:
            sparsity = torch.zeros((), device=x.device)

        info = {
            'flops': total_flops,
            'sparsity_penalty': sparsity,
            'gate_mean': gate_stack.mean(dim=0),                    # (B, S)
            'entropy_mean': (torch.stack(ents, dim=0).mean(dim=0)
                             if ents else None),                    # (B, S)
            'chemical_states': states if collect_chemical else None,
        }
        if self.mode in ("predictor", "predictor-supervised") and self.stage == 2 and diff_scores:
            info['diff_score_mean'] = torch.stack(diff_scores, dim=0).mean(dim=0)  # (B, S)
        return logits, info
