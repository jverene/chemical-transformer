"""Shared configuration for all experiments."""
from dataclasses import dataclass, field, asdict

import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


SIZE_PRESETS = {
    # ~11M params
    "small": dict(d_model=384, n_layers=6, n_heads=6, d_ff=1536),
    # ~42M params (4x width)
    "medium": dict(d_model=768, n_layers=6, n_heads=12, d_ff=3072),
}


@dataclass
class Config:
    # model
    vocab_size: int = 22
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_len: int = 512
    dropout: float = 0.1
    # chemical pathway (token-level compute-budget signal)
    chemical_dim: int = 16
    update_alpha: float = 0.3
    sparsity_lambda: float = 1e-2
    chem_tag_init: bool = False    # init chemical state from difficulty-tag embedding
    chem_hidden_input: bool = False  # secretion also reads the hidden state (x2)
    # baselines
    mod_capacity: float = 0.5   # MoD top-k capacity ratio
    fixed_gate: float = 0.5     # fixed-schedule gate value
    # training
    batch_size: int = 32
    seq_len: int = 256
    lr: float = 3e-4
    n_steps: int = 6000
    eval_every: int = 250
    device: str = field(default_factory=get_device)
    # two-stage (Option 2: predictor)
    stage: int = 1  # 1 = Stage 1 (full FFN), 2 = Stage 2 (DifficultyHead)
    # Option 3b: supervised difficulty + budget
    stage2_mode: str = ""  # "supervised_budget" for Option 3b
    predictor_target_gate_e: float = 0.2
    predictor_target_gate_m: float = 0.5
    predictor_target_gate_h: float = 0.8
    predictor_mse_weight: float = 5.0
    predictor_budget_weight: float = 0.5
    predictor_budget_target: float = 0.5
    predictor_freeze_body: bool = False
    # evaluation
    held_out_seed: int = 12345  # same held-out set for every method/seed
    test_seqs: int = 256
    eval_batches: int = 4
    final_eval_batches: int = 8
    wallclock_batches: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_size(cls, size: str, **overrides) -> "Config":
        cfg = cls(**SIZE_PRESETS[size])
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown config field: {k}")
            setattr(cfg, k, v)
        return cfg
