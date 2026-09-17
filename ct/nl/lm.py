"""Gated causal LM: a pretrained HF model with per-token FFN gating injected.

The gate reads the FFN's input hidden state (the post-attention residual,
same position in the stream as the arithmetic models' DifficultyHead) and
scales the FFN output. Gate sources:
  none  : pass-through (Stage-1 dense / baseline)
  fixed : constant gate (fixed-schedule control)
  head  : DifficultyHead, trained with MSE targets + budget (ours)
  mod   : MoD-style top-k router (baseline method)

Billing uses the same FLOPsCounter accounting as the arithmetic models
(attention full, FFN scaled by the gate), with d_ff = 4*d for Pythia.
"""
import math

import torch
import torch.nn as nn

from ct.models.chemical import DifficultyHead
from ct.flops import FLOPsCounter

GATE_TARGETS = [0.1, 0.2, 0.4, 0.6, 0.8]  # index 0 = floor (irreducible)


class MLPWrapper(nn.Module):
    def __init__(self, inner, dim, capacity: float = 0.5):
        super().__init__()
        self.inner = inner
        self.dim = dim
        self.capacity = capacity
        self.gate_mode = "none"  # none|fixed|head|mod|rotation
        self.head = DifficultyHead(dim)   # created for every mode: stable ckpts
        self.router = nn.Linear(dim, 1)
        self.oracle_gate = None           # leaf (B, S) during oracle passes
        self.last_gate = None             # stashed each forward for losses

    def forward(self, x):
        B, S, D = x.shape
        mode = self.gate_mode
        if mode == "none":
            self.last_gate = torch.ones(B, S, device=x.device, dtype=x.dtype)
            return self.inner(x)
        if mode == "fixed":
            g = torch.full((B, S), 0.5, device=x.device, dtype=x.dtype)
            self.last_gate = g
            return self.inner(x) * g.unsqueeze(-1)
        if mode == "head":
            if self.oracle_gate is not None:
                g = self.oracle_gate
            else:
                g, _ = self.head(x)
            self.last_gate = g
            return self.inner(x) * g.unsqueeze(-1)
        if mode == "rotation":
            # per-step random token mask: active tokens get full FFN (g=1),
            # inactive get none; mean gate = budget. Deploy = dense (g=1).
            mask = (torch.rand(x.shape[:2], device=x.device) < self.capacity).float()
            self.last_gate = mask
            return self.inner(x) * mask.unsqueeze(-1)
        if mode == "rotation_skip":
            # same allocation as rotation, but masked tokens' FFN rows are
            # actually SKIPPED (gather/scatter), so wall-clock scales with
            # the budget. Used by the wall-clock benchmark.
            mask = (torch.rand(x.shape[:2], device=x.device) < self.capacity)
            self.last_gate = mask.float()
            flat = x.reshape(B * S, D)
            idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
            out_flat = torch.zeros_like(flat)
            if idx.numel() > 0:
                sel = flat.index_select(0, idx)
                out_flat = out_flat.index_copy(0, idx, self.inner(sel))
            return out_flat.reshape(B, S, D)
        if mode == "mod":
            scores = self.router(x).squeeze(-1)
            k = max(1, int(math.ceil(self.capacity * S)))
            masked = scores
            topv, topi = masked.topk(k, dim=1)
            w = torch.sigmoid(topv)
            sel = x.gather(1, topi.unsqueeze(-1).expand(-1, -1, D))
            out_sel = self.inner(sel) * w.unsqueeze(-1)
            out = torch.zeros_like(x).scatter(
                1, topi.unsqueeze(-1).expand(-1, -1, D), out_sel)
            self.last_gate = torch.zeros(B, S, device=x.device,
                                         dtype=x.dtype).scatter(1, topi, w)
            return out
        raise ValueError(mode)


class GatedLM:
    """Owns tokenizer + model + wrappers; exposes losses, gates, billing."""

    def __init__(self, model_name: str, device: str, method: str = "baseline",
                 mod_capacity: float = 0.5, fixed_gate: float = 0.5):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.method = method
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # fp32 weights + autocast for speed: transformers>=5 preserves the
        # checkpoint dtype (Pythia ships fp16), and pure-fp16 AdamW NaNs.
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          dtype=torch.float32)
        self.d_model = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        self.d_ff = getattr(self.model.config, "intermediate_size",
                            4 * self.d_model)
        self.n_heads = self.model.config.num_attention_heads
        self.model.to(device)
        self.wrappers = []
        core = getattr(self.model, "model", None) or getattr(self.model, "transformer", None) \
            or self.model.base_model
        layers = getattr(core, "layers", None) or core.h
        for layer in layers:
            mlp = getattr(layer, "mlp", None) or layer.mlp
            w = MLPWrapper(mlp, self.d_model, mod_capacity).to(device)
            w.gate_mode = {"baseline": "none", "fixed": "fixed", "mod": "mod",
                           "rotation": "rotation", "ours-gategrad": "head",
                           "ours-loss": "head", "shuffled": "head"}[method]
            layer.mlp = w
            self.wrappers.append(w)

    # ---------- parameter groups ----------
    def param_groups(self, lr: float, body_lr_frac: float = 1.0):
        head_ids, body = set(), []
        for w in self.wrappers:
            for p in w.head.parameters():
                head_ids.add(id(p))
            if self.method == "mod":
                head_ids.add(id(w.router.weight))
                head_ids.add(id(w.router.bias))
        body = [p for p in self.model.parameters() if id(p) not in head_ids]
        heads = [p for w in self.wrappers for p in w.head.parameters()]
        if self.method == "mod":
            heads += [p for w in self.wrappers for p in w.router.parameters()]
        return [{"params": body, "lr": lr * body_lr_frac},
                {"params": heads, "lr": lr}]

    # ---------- forward + losses ----------
    def loss_and_gates(self, x, y, target=None, target_lay=None, mse_weight=5.0,
                       budget_weight=0.5, budget_target=0.5):
        logits = self.model(x).logits
        # fp32 CE regardless of autocast: Pythia logits reach O(1e2-1e3) and
        # bf16 rounding there degrades (and can destabilize) the loss.
        ce = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce
        gates = torch.stack([w.last_gate for w in self.wrappers], 0)  # (L,B,S)
        gate_mean = gates.mean(0)
        info = {"ce": ce.detach(), "gate_mean": gate_mean.detach(),
                "flops": self.billed_flops(x.shape, gates)}
        if target is not None:
            pred_gate = gate_mean if self.method == "mod" else gate_mean
            mse = torch.nn.functional.mse_loss(pred_gate, target.detach())
            budget = (gate_mean.mean() - budget_target) ** 2
            loss = loss + mse_weight * mse + budget_weight * budget
            info["mse"] = mse.detach()
            info["budget"] = budget.detach()
        elif target_lay is not None:
            mse = torch.nn.functional.mse_loss(gates, target_lay.detach())
            budget = (gate_mean.mean() - budget_target) ** 2
            loss = loss + mse_weight * mse + budget_weight * budget
            info["mse"] = mse.detach()
            info["budget"] = budget.detach()
        return loss, info

    def billed_flops(self, shape, gates):
        B, S = shape
        att = FLOPsCounter.attention(B, S, self.d_model, self.n_heads)
        processed = gates.sum() if gates is not None \
            else torch.tensor(float(B * S))
        return att + FLOPsCounter.ffn_per_token(self.d_model, self.d_ff) \
            * self.n_layers * processed

    def oracle_leaves(self, B, S):
        leaves = [torch.ones(B, S, device=self.device, requires_grad=True)
                  for _ in self.wrappers]
        for w, leaf in zip(self.wrappers, leaves):
            w.oracle_gate = leaf
        return leaves

    def clear_oracle(self):
        for w in self.wrappers:
            w.oracle_gate = None

    def gates_per_layer(self):
        return torch.stack([w.last_gate for w in self.wrappers], 0).detach()
