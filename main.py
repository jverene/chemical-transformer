"""
Allostatic Chemical State Transformer - Training Script
12M parameter proof-of-concept for predictive modulatory allocation.
Optimized for Apple Silicon (M3) with MPS backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================================
# DEVICE DETECTION (M3 Mac friendly)
# ============================================================================

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Model
    vocab_size: int = 22  # 0-9, +, -, *, =, [, ], E, M, H, <PAD>, <EOS>, <BOS>
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_len: int = 512
    dropout: float = 0.1

    # Chemical state
    chemical_dim: int = 16
    update_alpha: float = 0.3
    sparsity_lambda: float = 1e-2   # L1 penalty on mean FFN gate
    mod_threshold: float = 0.5      # threshold for the process-rate readout

    # Training
    batch_size: int = 32
    seq_len: int = 256
    lr: float = 3e-4
    n_steps: int = 10000
    eval_every: int = 500
    device: str = get_device()

    # Task
    problems_target: int = 10
    test_seqs: int = 256
    eval_batches: int = 5

CFG = Config()

# ============================================================================
# TOKENIZER
# ============================================================================

class Tokenizer:
    def __init__(self):
        self.chars = ['0','1','2','3','4','5','6','7','8','9',
                      '+','-','*','=',
                      '[',']','E','M','H',
                      '<PAD>','<EOS>','<BOS>']
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.pad_id = self.stoi['<PAD>']
        self.eos_id = self.stoi['<EOS>']
        self.bos_id = self.stoi['<BOS>']
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.itos[i] for i in ids])

TOKENIZER = Tokenizer()
CFG.vocab_size = TOKENIZER.vocab_size

# ============================================================================
# DATA GENERATION
# ============================================================================
# Difficulty is encoded as a leading tag so the model has an early signal it can
# use to pre-allocate compute for the upcoming answer tokens:
#   Easy:   [E] a op b = ans     a,b in [1,9]        ~162 unique (memorizable)
#   Medium: [M] a op b = ans     a,b in [100,999]    ~1.6M unique
#   Hard:   [H] a * b   = ans    a in [1000,9999], b in [100,999]  ~9M unique
# Hard is far beyond what an 11M-param model can memorize in 10k steps.

def generate_problem(difficulty: int) -> Tuple[List[int], int]:
    if difficulty == 0:
        a, b = random.randint(1, 9), random.randint(1, 9)
        op = random.choice(['+', '-'])
        ans = a + b if op == '+' else a - b
        s = f"[E]{a}{op}{b}={ans}"
    elif difficulty == 1:
        a, b = random.randint(100, 999), random.randint(100, 999)
        op = random.choice(['+', '-'])
        ans = a + b if op == '+' else a - b
        s = f"[M]{a}{op}{b}={ans}"
    else:
        a = random.randint(1000, 9999)
        b = random.randint(100, 999)
        s = f"[H]{a}*{b}={a*b}"
    toks = TOKENIZER.encode(s)
    toks.append(TOKENIZER.eos_id)
    return toks, difficulty

def _pack_sequence(max_len: int, source: str = "train") -> Tuple[List[int], List[int]]:
    tokens: List[int] = [TOKENIZER.bos_id]
    diffs: List[int] = [-1]
    while len(tokens) < max_len - 1:
        if source == "held_out":
            toks, d = random.choice(HELD_OUT_PROBLEMS)
        else:
            r = random.random()
            d = 0 if r < 0.30 else (1 if r < 0.70 else 2)
            toks, d = generate_problem(d)
        if len(tokens) + len(toks) > max_len - 1:
            break
        tokens.extend(toks)
        diffs.extend([d] * len(toks))
    if len(tokens) < max_len:
        pad = max_len - len(tokens)
        tokens += [TOKENIZER.pad_id] * pad
        diffs += [-1] * pad
    return tokens[:max_len], diffs[:max_len]

# Frozen held-out pool: built once at import. Training never samples these.
def _build_held_out():
    pool: List[Tuple[List[int], int]] = []
    per_diff = 4000
    for d in (0, 1, 2):
        for _ in range(per_diff):
            pool.append(generate_problem(d))
    random.Random(0).shuffle(pool)
    return pool

HELD_OUT_PROBLEMS: List[Tuple[List[int], int]] = _build_held_out()
HELD_OUT_SEQS: List[Tuple[List[int], List[int]]] = [
    _pack_sequence(CFG.seq_len, source="held_out") for _ in range(CFG.test_seqs)
]

def get_batch(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_list, diff_list = [], []
    for _ in range(batch_size):
        toks, diffs = _pack_sequence(CFG.seq_len, source="train")
        x_list.append(toks)
        diff_list.append(diffs)
    x = torch.tensor(x_list, dtype=torch.long, device=CFG.device)
    diffs = torch.tensor(diff_list, dtype=torch.long, device=CFG.device)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = TOKENIZER.pad_id
    return x, y, diffs

def get_eval_batch(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = random.sample(range(len(HELD_OUT_SEQS)), batch_size)
    x_list = [HELD_OUT_SEQS[i][0] for i in idx]
    diff_list = [HELD_OUT_SEQS[i][1] for i in idx]
    x = torch.tensor(x_list, dtype=torch.long, device=CFG.device)
    diffs = torch.tensor(diff_list, dtype=torch.long, device=CFG.device)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = TOKENIZER.pad_id
    return x, y, diffs

# ============================================================================
# FLOPS COUNTER (dynamic: counts ops actually performed)
# ============================================================================
# Static shape formulas counted the same FLOPs regardless of input. Now FFN ops
# are scaled by the per-token gate, so skipped/gated tokens contribute less.

class FLOPsCounter:
    @staticmethod
    def attention(B: int, S: int, D: int, H: int) -> int:
        Dh = D // H
        qkv = 3 * B * S * D * D               # QKV projections
        scores = B * H * S * S * Dh           # score matrix
        apply = B * H * S * S * Dh            # weights @ V
        out = B * S * D * D                   # output projection
        return qkv + scores + apply + out

    @staticmethod
    def ffn_per_token(D: int, d_ff: int) -> int:
        # two linear layers, MACs -> 2x
        return 2 * (D * d_ff + d_ff * D) * 2

# ============================================================================
# MASK HELPER
# ============================================================================

def make_causal_mask(S: int, device, dtype) -> torch.Tensor:
    mask = torch.zeros(S, S, device=device, dtype=dtype)
    mask.masked_fill_(torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
                      torch.finfo(dtype).min)
    return mask

# ============================================================================
# MODELS
# ============================================================================

def _manual_attention(x, q_p, k_p, v_p, o_p, n_heads, causal_mask):
    """Multi-head attention returning weights (needed for the entropy signal)."""
    B, S, D = x.shape
    H, Dh = n_heads, D // n_heads
    q = q_p(x).view(B, S, H, Dh).transpose(1, 2)
    k = k_p(x).view(B, S, H, Dh).transpose(1, 2)
    v = v_p(x).view(B, S, H, Dh).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
    scores = scores + causal_mask                 # broadcast (S,S) -> (B,H,S,S)
    weights = torch.softmax(scores, dim=-1)       # (B,H,S,S)
    out = torch.matmul(weights, v)                # (B,H,S,Dh)
    out = out.transpose(1, 2).contiguous().view(B, S, D)
    return o_p(out), weights

class ChemicalLayer(nn.Module):
    """
    Transformer layer with allostatic, per-position chemical state.

    Per-position attention entropy (a "stress" signal) drives a chemical state
    that flows layer-to-layer. The chemical state gates the FFN per token
    (Mixture-of-Depths): low-stress (easy) tokens attenuate/skip the FFN,
    high-stress (hard) tokens keep it. This makes per-token FLOPs input-dependent.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, chemical_dim: int,
                 dropout: float = 0.1, chemical_off: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chemical_off = chemical_off

        # Manual MHA so we can read attention weights for the entropy signal.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Secretion: per-position entropy -> chemical state update
        self.secretion = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, chemical_dim)
        )
        # Receptor: chemical state -> per-token FFN gate logit.
        # Zero-init weights + bias 0 -> initial gate sigmoid(0)=0.5, so the model
        # starts neutral and can learn the easy/hard asymmetry rather than
        # collapsing to all-skip (or all-process) at init.
        self.router = nn.Linear(chemical_dim, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        if chemical_off:
            for p in self.secretion.parameters(): p.requires_grad = False
            for p in self.router.parameters(): p.requires_grad = False

    def forward(self, x, prev_chemical, causal_mask, pad_mask):
        B, S, D = x.shape

        attn_out, weights = _manual_attention(
            x, self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.n_heads, causal_mask
        )
        x2 = self.norm1(x + attn_out)

        if self.chemical_off:
            # Ablation: gate fixed at 1.0 (full compute). Chemical pathway inert.
            g = torch.ones(B, S, device=x.device, dtype=x.dtype)
            chemical = prev_chemical if prev_chemical is not None else \
                torch.zeros(B, S, CFG.chemical_dim, device=x.device, dtype=x.dtype)
        else:
            # Per-position attention entropy, averaged over heads: (B, S)
            ent = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean(dim=1)
            # Normalize by log(num_attended_positions) to remove the trivial
            # "later positions have more candidates -> higher entropy" bias.
            norm = torch.clamp(torch.log(torch.arange(1, S + 1, device=x.device, dtype=ent.dtype)),
                               min=1e-3)
            stress = (ent / norm).unsqueeze(-1)                  # (B, S, 1)
            new_chem = self.secretion(stress)                    # (B, S, chemical_dim)
            if prev_chemical is None:
                chemical = new_chem
            else:
                a = CFG.update_alpha
                chemical = (1 - a) * prev_chemical + a * new_chem
            g = torch.sigmoid(self.router(chemical).squeeze(-1))  # (B, S)

        ffn_out = self.ffn(x2) * g.unsqueeze(-1)
        x3 = self.norm2(x2 + ffn_out)

        # dynamic FLOPs: attention runs for all tokens; FFN only "gated" portion
        g_real = g[pad_mask] if pad_mask is not None else g.flatten()
        processed = g_real.sum().item()
        flops = FLOPsCounter.attention(B, S, D, self.n_heads) + \
                FLOPsCounter.ffn_per_token(D, self.ffn[0].out_features) * processed

        return x3, chemical, g, flops


class ChemicalTransformer(nn.Module):
    """Transformer with allostatic chemical state + MoD FFN gating."""
    def __init__(self, chemical_off: bool = False):
        super().__init__()
        self.chemical_off = chemical_off
        self.token_emb = nn.Embedding(CFG.vocab_size, CFG.d_model)
        self.pos_emb = nn.Embedding(CFG.max_len, CFG.d_model)
        self.dropout = nn.Dropout(CFG.dropout)
        self.layers = nn.ModuleList([
            ChemicalLayer(CFG.d_model, CFG.n_heads, CFG.d_ff, CFG.chemical_dim,
                          CFG.dropout, chemical_off=chemical_off)
            for _ in range(CFG.n_layers)
        ])
        self.norm = nn.LayerNorm(CFG.d_model)
        self.head = nn.Linear(CFG.d_model, CFG.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pad_mask=None):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        causal_mask = make_causal_mask(S, x.device, x.dtype)

        chemical = None
        total_flops = 0
        gates: List[torch.Tensor] = []
        for layer in self.layers:
            x, chemical, g, flops = layer(x, chemical, causal_mask, pad_mask)
            total_flops += flops
            gates.append(g)

        x = self.norm(x)
        logits = self.head(x)

        if self.chemical_off or pad_mask is None:
            sparsity = torch.tensor(0.0, device=x.device)
        else:
            stack = torch.stack(gates, dim=0)            # (L, B, S)
            sparsity = stack[:, pad_mask].mean()          # mean gate over real tokens+layers

        info = {
            'flops': total_flops,
            'sparsity_penalty': sparsity,
            'gate_mean': (None if self.chemical_off else torch.stack(gates, dim=0).mean(dim=0)),  # (B,S)
        }
        return logits, info

    def count_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class BaselineLayer(nn.Module):
    """Standard transformer layer, full FFN on every token, no modulation."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x2 = self.norm1(x + attn_out)
        ffn_out = self.ffn(x2)
        x3 = self.norm2(x2 + ffn_out)

        B, S, D = x.shape
        real = int(pad_mask.sum().item()) if pad_mask is not None else B * S
        flops = FLOPsCounter.attention(B, S, D, self.n_heads) + \
                FLOPsCounter.ffn_per_token(D, self.ffn[0].out_features) * real
        return x3, flops


class BaselineTransformer(nn.Module):
    """Standard transformer for comparison (full compute, no chemical state)."""
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(CFG.vocab_size, CFG.d_model)
        self.pos_emb = nn.Embedding(CFG.max_len, CFG.d_model)
        self.dropout = nn.Dropout(CFG.dropout)

        self.layers = nn.ModuleList([
            BaselineLayer(CFG.d_model, CFG.n_heads, CFG.d_ff, CFG.dropout)
            for _ in range(CFG.n_layers)
        ])

        self.norm = nn.LayerNorm(CFG.d_model)
        self.head = nn.Linear(CFG.d_model, CFG.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        causal_mask = make_causal_mask(S, x.device, x.dtype)

        total_flops = 0
        for layer in self.layers:
            x, flops = layer(x, causal_mask=causal_mask, pad_mask=pad_mask)
            total_flops += flops

        x = self.norm(x)
        logits = self.head(x)
        info = {'flops': total_flops, 'sparsity_penalty': torch.tensor(0.0, device=x.device),
                'gate_mean': None}
        return logits, info

    def count_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model: nn.Module, name: str, steps: int = CFG.n_steps) -> dict:
    """Train a model and return metrics history."""
    model = model.to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    history = defaultdict(list)

    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"Parameters: {model.count_params():,}")
    print(f"{'='*60}")

    start_time = time.time()

    for step in range(steps):
        model.train()
        x, y, diffs = get_batch(CFG.batch_size)
        pad_mask = (x != TOKENIZER.pad_id)

        logits, info = model(x, pad_mask=pad_mask)
        ce = F.cross_entropy(logits.view(-1, CFG.vocab_size), y.view(-1), ignore_index=TOKENIZER.pad_id)
        loss = ce + CFG.sparsity_lambda * info['sparsity_penalty']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % CFG.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_eval, y_eval, diffs_eval = get_eval_batch(CFG.batch_size)
                eval_pad = (x_eval != TOKENIZER.pad_id)
                logits_eval, eval_info = model(x_eval, pad_mask=eval_pad)

                preds = logits_eval.argmax(dim=-1)
                mask = y_eval != TOKENIZER.pad_id
                acc = (preds[mask] == y_eval[mask]).float().mean().item()

                easy_mask = mask & (diffs_eval == 0)
                med_mask = mask & (diffs_eval == 1)
                hard_mask = mask & (diffs_eval == 2)

                easy_acc = (preds[easy_mask] == y_eval[easy_mask]).float().mean().item() if easy_mask.sum() > 0 else 0
                med_acc = (preds[med_mask] == y_eval[med_mask]).float().mean().item() if med_mask.sum() > 0 else 0
                hard_acc = (preds[hard_mask] == y_eval[hard_mask]).float().mean().item() if hard_mask.sum() > 0 else 0

                total_tokens = mask.sum().item()
                flops_per_token = eval_info['flops'] / total_tokens if total_tokens > 0 else 0

                elapsed = time.time() - start_time

                print(f"Step {step:5d} | Loss: {loss.item():.4f} | Acc: {acc:.3f} | "
                      f"Easy: {easy_acc:.3f} | Med: {med_acc:.3f} | Hard: {hard_acc:.3f} | "
                      f"FLOPs/tok: {flops_per_token/1e6:.1f}M | Time: {elapsed:.1f}s")

                history['step'].append(step)
                history['loss'].append(loss.item())
                history['acc'].append(acc)
                history['easy_acc'].append(easy_acc)
                history['med_acc'].append(med_acc)
                history['hard_acc'].append(hard_acc)
                history['flops_per_token'].append(flops_per_token)

    return dict(history)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_comparison(chem_hist: dict, base_hist: dict, save_path: str = "comparison.png"):
    """Plot comparison between chemical and baseline models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(chem_hist['step'], chem_hist['loss'], label='Chemical', linewidth=2)
    ax.plot(base_hist['step'], base_hist['loss'], label='Baseline', linewidth=2)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(chem_hist['step'], chem_hist['acc'], label='Chemical', linewidth=2)
    ax.plot(base_hist['step'], base_hist['acc'], label='Baseline', linewidth=2)
    ax.set_xlabel('Step'); ax.set_ylabel('Accuracy'); ax.set_title('Overall Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(chem_hist['step'], chem_hist['easy_acc'], 'g-', label='Chem Easy', linewidth=2)
    ax.plot(chem_hist['step'], chem_hist['med_acc'], 'b-', label='Chem Med', linewidth=2)
    ax.plot(chem_hist['step'], chem_hist['hard_acc'], 'r-', label='Chem Hard', linewidth=2)
    ax.plot(base_hist['step'], base_hist['easy_acc'], 'g--', label='Base Easy', linewidth=1.5)
    ax.plot(base_hist['step'], base_hist['med_acc'], 'b--', label='Base Med', linewidth=1.5)
    ax.plot(base_hist['step'], base_hist['hard_acc'], 'r--', label='Base Hard', linewidth=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Accuracy'); ax.set_title('Per-Difficulty Accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(chem_hist['step'], [f/1e6 for f in chem_hist['flops_per_token']], label='Chemical', linewidth=2)
    ax.plot(base_hist['step'], [f/1e6 for f in base_hist['flops_per_token']], label='Baseline', linewidth=2)
    ax.set_xlabel('Step'); ax.set_ylabel('FLOPs / Token (M)'); ax.set_title('Compute per Token')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Device: {CFG.device}")
    print(f"Vocab size: {CFG.vocab_size}")

    chem_model = ChemicalTransformer()
    chem_hist = train_model(chem_model, "Chemical Transformer")

    base_model = BaselineTransformer()
    base_hist = train_model(base_model, "Baseline Transformer")

    plot_comparison(chem_hist, base_hist)

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Chemical params: {chem_model.count_params():,}")
    print(f"Baseline params: {base_model.count_params():,}")
    print(f"Param delta: {chem_model.count_params() - base_model.count_params():,} "
          f"({(chem_model.count_params() / base_model.count_params() - 1) * 100:.2f}%)")
    print(f"\nFinal Chemical Acc: {chem_hist['acc'][-1]:.4f}")
    print(f"Final Baseline Acc: {base_hist['acc'][-1]:.4f}")
    print(f"\nFinal Chemical FLOPs/tok: {chem_hist['flops_per_token'][-1]/1e6:.2f}M")
    print(f"Final Baseline FLOPs/tok: {base_hist['flops_per_token'][-1]/1e6:.2f}M")

if __name__ == "__main__":
    main()
