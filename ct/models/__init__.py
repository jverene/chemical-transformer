from .baseline import BaselineTransformer
from .chemical import ChemicalTransformer
from .mod import MoDTransformer

METHODS = [
    "baseline",
    "chemical",
    "chemical-off",
    "mod",
    "random-gate",
    "fixed-schedule",
    "tag-only",
    "predictor",
    "predictor-supervised",
]


def build_model(method: str, cfg):
    if method == "baseline":
        return BaselineTransformer(cfg)
    if method == "chemical":
        return ChemicalTransformer(cfg, mode="full")
    if method == "chemical-off":
        return ChemicalTransformer(cfg, mode="off")
    if method == "random-gate":
        return ChemicalTransformer(cfg, mode="random")
    if method == "fixed-schedule":
        return ChemicalTransformer(cfg, mode="fixed")
    if method == "tag-only":
        return ChemicalTransformer(cfg, mode="tag")
    if method == "mod":
        return MoDTransformer(cfg)
    if method == "predictor":
        stage = getattr(cfg, "stage", 1)
        return ChemicalTransformer(cfg, mode="predictor", stage=stage)
    if method == "predictor-supervised":
        stage = getattr(cfg, "stage", 1)
        return ChemicalTransformer(cfg, mode="predictor-supervised", stage=stage)
    raise ValueError(f"unknown method: {method}")
