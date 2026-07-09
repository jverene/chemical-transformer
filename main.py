"""
Allostatic Chemical State Transformer - Training Script
12M parameter proof-of-concept for predictive modulatory allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Model
    vocab_size: int = 16  # 0-9, +, -, *, =, <PAD>, <EOS>
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_len: int = 512
    dropout: float = 0.1

    # Chemical state
    chemical_dim: int = 16
    update_alpha: float = 0.3

    # Training
    batch_size: int = 32
    seq_len: int = 128
    lr: float = 3e-4
    n_steps: int = 10000
    eval_every: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Task
    problems_per_seq: int = 8

CFG = Config()

# ============================================================================
# TOKENIZER
# ============================================================================

class Tokenizer:
    def __init__(self):
        self.chars = ['0','1','2','3','4','5','6','7','8','9','+','-','*','=','<PAD>','<EOS>']
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for i,c in enumerate(self.chars)}
        self.pad_id = self.stoi['<PAD>']
        self.eos_id = self.stoi['<EOS>']
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.itos[i] for i in ids])

TOKENIZER = Tokenizer()

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_problem(difficulty: int) -> str:
    """Generate arithmetic problem. Higher difficulty = harder."""
    if difficulty == 0:  # Easy: single digit + single digit
        a, b = random.randint(1, 9), random.randint(1, 9)
        op = random.choice(['+', '-'])
        if op == '+':
            return f"{a}+{b}={a+b}"
        else:
            return f"{a}-{b}={a-b}"
    elif difficulty == 1:  # Medium: 2-digit + 2-digit
        a, b = random.randint(10, 99), random.randint(10, 99)
        op = random.choice(['+', '-'])
        if op == '+':
            return f"{a}+{b}={a+b}"
        else:
            return f"{a}-{b}={a-b}"
    else:  # Hard: 2-digit * 2-digit or 3-digit + 3-digit
        if random.random() < 0.5:
            a, b = random.randint(10, 99), random.randint(10, 99)
            return f"{a}*{b}={a*b}"
        else:
            a, b = random.randint(100, 999), random.randint(100, 999)
            return f"{a}+{b}={a+b}"

def generate_sequence(n_problems: int = 8) -> Tuple[List[int], List[int]]:
    """Generate a sequence of mixed-difficulty problems."""
    # Mix: 30% easy, 40% medium, 30% hard
    difficulties = [0] * (n_problems // 3) + [1] * (n_problems // 3) + [2] * (n_problems - 2*(n_problems//3))
    random.shuffle(difficulties)

    tokens = []
    diff_labels = []
    for d in difficulties:
        prob = generate_problem(d)
        tokens.extend(TOKENIZER.encode(prob))
        tokens.append(TOKENIZER.eos_id)
        diff_labels.extend([d] * (len(prob) + 1))

    # Pad or truncate to seq_len
    if len(tokens) < CFG.seq_len:
        tokens += [TOKENIZER.pad_id] * (CFG.seq_len - len(tokens))
        diff_labels += [-1] * (CFG.seq_len - len(diff_labels))
    else:
        tokens = tokens[:CFG.seq_len]
        diff_labels = diff_labels[:CFG.seq_len]

    return tokens, diff_labels

def get_batch(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get a batch of sequences with difficulty labels."""
    x_list = []
    diff_list = []
    for _ in range(batch_size):
        tokens, diffs = generate_sequence(CFG.problems_per_seq)
        x_list.append(tokens)
        diff_list.append(diffs)

    x = torch.tensor(x_list, dtype=torch.long, device=CFG.device)
    diffs = torch.tensor(diff_list, dtype=torch.long, device=CFG.device)
    # Create target by shifting input by 1
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = TOKENIZER.pad_id

    return x, y, diffs

# ============================================================================
# FLOPS COUNTER
# ============================================================================

class FLOPsCounter:
    """Simple FLOPs counter for transformer operations."""
    @staticmethod
    def count_attention(batch: int, seq: int, d_model: int, n_heads: int) -> int:
        # QKV projection: 3 * batch * seq * d_model * d_model
        qkv = 3 * batch * seq * d_model * d_model
        # Attention scores: batch * n_heads * seq * seq * (d_model // n_heads)
        scores = batch * n_heads * seq * seq * (d_model // n_heads)
        # Attention application: batch * n_heads * seq * seq * (d_model // n_heads)
        apply = batch * n_heads * seq * seq * (d_model // n_heads)
        # Output projection: batch * seq * d_model * d_model
        out = batch * seq * d_model * d_model
        return qkv + scores + apply + out

    @staticmethod
    def count_ffn(batch: int, seq: int, d_model: int, d_ff: int) -> int:
        # Two linear layers
        return 2 * batch * seq * d_model * d_ff * 2

    @staticmethod
    def count_layer(batch: int, seq: int, d_model: int, d_ff: int, n_heads: int) -> int:
        return FLOPsCounter.count_attention(batch, seq, d_model, n_heads) + FLOPsCounter.count_ffn(batch, seq, d_model, d_ff)

# ============================================================================
# MODELS
# ============================================================================

class ChemicalLayer(nn.Module):
    """
    Transformer layer with allostatic chemical state modulation.
    The chemical state flows from layer to layer, updated by attention entropy.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

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

        # Secretion: attention entropy -> predicted load (chemical state update)
        self.secretion = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, CFG.chemical_dim)
        )

        # Receptor: chemical state -> modulation knobs
        self.receptor = nn.Sequential(
            nn.Linear(CFG.chemical_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [temperature, gain]
        )

    def forward(self, x: torch.Tensor, prev_chemical: Optional[torch.Tensor] = None, 
                return_flops: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch, seq, d = x.shape

        # Attention
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        # attn_weights: (batch, n_heads, seq, seq)

        # Compute attention entropy as "stress signal"
        # Average over heads and batch
        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean(dim=(1, 2))  # (batch,)
        mean_entropy = entropy.mean().unsqueeze(0)  # (1,)

        # Secretion: predict load from entropy
        chemical = self.secretion(mean_entropy).unsqueeze(0).expand(batch, -1)  # (batch, chemical_dim)

        # Allostatic update: blend with previous chemical state
        if prev_chemical is not None:
            alpha = CFG.update_alpha
            chemical = (1 - alpha) * prev_chemical + alpha * chemical

        # Receptor: map chemical to modulation
        mods = self.receptor(chemical)  # (batch, 2)
        temp = mods[:, 0].sigmoid() * 2 + 0.5   # attention temperature (0.5 to 2.5)
        gain = mods[:, 1].sigmoid() * 2         # FFN gain (0 to 2)

        # Apply modulation
        # Temperature: divide attention output by temp (sharper = less compute smoothing)
        attn_out = attn_out / temp.view(batch, 1, 1)
        x = self.norm1(x + attn_out)

        # Gain: scale FFN output
        ffn_out = self.ffn(x)
        ffn_out = ffn_out * gain.view(batch, 1, 1)
        x = self.norm2(x + ffn_out)

        flops = 0
        if return_flops:
            flops = FLOPsCounter.count_layer(batch, seq, d, self.ffn[0].out_features, self.n_heads)

        return x, chemical, flops


class ChemicalTransformer(nn.Module):
    """Transformer with allostatic chemical state modulation."""
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(CFG.vocab_size, CFG.d_model)
        self.pos_emb = nn.Embedding(CFG.max_len, CFG.d_model)
        self.dropout = nn.Dropout(CFG.dropout)

        self.layers = nn.ModuleList([
            ChemicalLayer(CFG.d_model, CFG.n_heads, CFG.d_ff, CFG.dropout)
            for _ in range(CFG.n_layers)
        ])

        self.norm = nn.LayerNorm(CFG.d_model)
        self.head = nn.Linear(CFG.d_model, CFG.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, return_flops: bool = False) -> Tuple[torch.Tensor, int]:
        batch, seq = x.shape
        pos = torch.arange(seq, device=x.device).unsqueeze(0)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))

        chemical = None
        total_flops = 0

        for layer in self.layers:
            x, chemical, flops = layer(x, chemical, return_flops=return_flops)
            total_flops += flops

        x = self.norm(x)
        logits = self.head(x)

        if return_flops:
            return logits, total_flops
        return logits, 0

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class BaselineLayer(nn.Module):
    """Standard transformer layer with NO chemical modulation."""
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

    def forward(self, x: torch.Tensor, return_flops: bool = False) -> Tuple[torch.Tensor, int]:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        flops = 0
        if return_flops:
            batch, seq, d = x.shape
            flops = FLOPsCounter.count_layer(batch, seq, d, self.ffn[0].out_features, self.attn.num_heads)
        return x, flops


class BaselineTransformer(nn.Module):
    """Standard transformer for comparison."""
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

    def forward(self, x: torch.Tensor, return_flops: bool = False) -> Tuple[torch.Tensor, int]:
        batch, seq = x.shape
        pos = torch.arange(seq, device=x.device).unsqueeze(0)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos))

        total_flops = 0
        for layer in self.layers:
            x, flops = layer(x, return_flops=return_flops)
            total_flops += flops

        x = self.norm(x)
        logits = self.head(x)

        if return_flops:
            return logits, total_flops
        return logits, 0

    def count_params(self):
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

        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, CFG.vocab_size), y.view(-1), ignore_index=TOKENIZER.pad_id)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % CFG.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on fixed batch
                x_eval, y_eval, diffs_eval = get_batch(CFG.batch_size)
                logits_eval, flops = model(x_eval, return_flops=True)

                # Overall accuracy
                preds = logits_eval.argmax(dim=-1)
                mask = y_eval != TOKENIZER.pad_id
                acc = (preds[mask] == y_eval[mask]).float().mean().item()

                # Per-difficulty accuracy
                easy_mask = mask & (diffs_eval == 0)
                med_mask = mask & (diffs_eval == 1)
                hard_mask = mask & (diffs_eval == 2)

                easy_acc = (preds[easy_mask] == y_eval[easy_mask]).float().mean().item() if easy_mask.sum() > 0 else 0
                med_acc = (preds[med_mask] == y_eval[med_mask]).float().mean().item() if med_mask.sum() > 0 else 0
                hard_acc = (preds[hard_mask] == y_eval[hard_mask]).float().mean().item() if hard_mask.sum() > 0 else 0

                # FLOPs per token
                total_tokens = mask.sum().item()
                flops_per_token = flops / total_tokens if total_tokens > 0 else 0

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

    # Loss
    ax = axes[0, 0]
    ax.plot(chem_hist['step'], chem_hist['loss'], label='Chemical', linewidth=2)
    ax.plot(base_hist['step'], base_hist['loss'], label='Baseline', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Overall Accuracy
    ax = axes[0, 1]
    ax.plot(chem_hist['step'], chem_hist['acc'], label='Chemical', linewidth=2)
    ax.plot(base_hist['step'], base_hist['acc'], label='Baseline', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-difficulty accuracy
    ax = axes[1, 0]
    ax.plot(chem_hist['step'], chem_hist['easy_acc'], 'g-', label='Chem Easy', linewidth=2)
    ax.plot(chem_hist['step'], chem_hist['med_acc'], 'b-', label='Chem Med', linewidth=2)
    ax.plot(chem_hist['step'], chem_hist['hard_acc'], 'r-', label='Chem Hard', linewidth=2)
    ax.plot(base_hist['step'], base_hist['easy_acc'], 'g--', label='Base Easy', linewidth=1.5)
    ax.plot(base_hist['step'], base_hist['med_acc'], 'b--', label='Base Med', linewidth=1.5)
    ax.plot(base_hist['step'], base_hist['hard_acc'], 'r--', label='Base Hard', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Difficulty Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # FLOPs per token
    ax = axes[1, 1]
    ax.plot(chem_hist['step'], [f/1e6 for f in chem_hist['flops_per_token']], label='Chemical', linewidth=2)
    ax.plot(base_hist['step'], [f/1e6 for f in base_hist['flops_per_token']], label='Baseline', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('FLOPs / Token (M)')
    ax.set_title('Compute per Token')
    ax.legend()
    ax.grid(True, alpha=0.3)

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

    # Train chemical model
    chem_model = ChemicalTransformer()
    chem_hist = train_model(chem_model, "Chemical Transformer")

    # Train baseline
    base_model = BaselineTransformer()
    base_hist = train_model(base_model, "Baseline Transformer")

    # Compare
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
