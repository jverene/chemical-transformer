"""Shared configuration for all experiments."""
from dataclasses import dataclass, field, asdict

import torch


def get_device(device: str = None) -> str:
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Per-layer params ~= 4*d^2 (attn) + 2*d*d_ff (FFN); embeddings/head are
# negligible at vocab 22. "target" is the approximate param count the preset
# name promises; build_model asserts the built model lands within 10%.
# "lr" is per-size: 3e-4 destabilizes >=12-layer post-LN models (P1 diagnosis
# Sep 6: 150m collapses to uniform, 42m fine; verified 1e-4 learns). All
# methods at a given size share the preset LR.
SIZE_PRESETS = {
    # ~11M params
    "small": dict(d_model=384, n_layers=6, n_heads=6, d_ff=1536, lr=3e-4,
                  target=11e6, pos_type="learned", attn_impl="manual"),
    # ~42M params (4x width)
    "medium": dict(d_model=768, n_layers=6, n_heads=12, d_ff=3072, lr=3e-4,
                   target=42e6, pos_type="learned", attn_impl="manual"),
    # ~151M
    "150m": dict(d_model=1024, n_layers=12, n_heads=16, d_ff=4096, lr=1e-4,
                 target=150e6, pos_type="rope", attn_impl="sdpa"),
    # ~393M
    "400m": dict(d_model=1280, n_layers=20, n_heads=20, d_ff=5120, lr=6e-5,
                 target=390e6, pos_type="rope", attn_impl="sdpa"),
    # ~1007M
    "1b": dict(d_model=2048, n_layers=20, n_heads=16, d_ff=8192, lr=3e-5,
               target=1000e6, pos_type="rope", attn_impl="sdpa"),
    # ~2988M
    "3b": dict(d_model=2560, n_layers=38, n_heads=20, d_ff=10240, lr=2e-5,
               target=3000e6, pos_type="rope", attn_impl="sdpa"),
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
    pos_type: str = "learned"   # "learned" | "rope"
    attn_impl: str = "manual"   # "manual" | "sdpa" (sdpa required for rope)
    # scale / throughput
    amp: bool = False                  # bf16 autocast (CUDA only)
    grad_accum: int = 1
    grad_checkpoint: bool = False
    lr_schedule: str = "constant"      # "constant" | "cosine"
    warmup_frac: float = 0.02
    min_lr_frac: float = 0.1
    weight_decay: float = None         # None = torch AdamW default (legacy)
    iso_flop_budget: float = 0.0       # >0: stop when billed train FLOPs hit this
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
    # Option B for mixed task: targets from digit-count bins (0.2/0.4/0.6/0.8)
    predictor_digit_targets: bool = False
    # Stage-2 supervision source: "tag" = difficulty tags (workshop recipe),
    # "gategrad" = measured marginal value of compute from the frozen Stage-1
    # body (score = -sum_l dL/dg, quartile-binned; see ct/gategrad.py).
    predictor_target_source: str = "tag"
    # Control: permute the per-position targets within each sequence
    # (separates "supervision with these targets" from "any varying targets").
    predictor_shuffle_targets: bool = False
    # evaluation
    held_out_seed: int = 12345  # same held-out set for every method/seed
    test_seqs: int = 256
    eval_batches: int = 4
    final_eval_batches: int = 8
    wallclock_batches: int = 5
    # provenance (set by for_size; serialized so run JSONs are self-describing)
    size_name: str = ""
    preset_target_params: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_size(cls, size: str, **overrides) -> "Config":
        preset = dict(SIZE_PRESETS[size])
        target = preset.pop("target", None)
        cfg = cls(**preset)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown config field: {k}")
            setattr(cfg, k, v)
        cfg.preset_target_params = target
        cfg.size_name = size
        return cfg
