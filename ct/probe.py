"""Linear probing: predict the difficulty bin from the per-layer chemical state.

After training a Chemical model, freeze it, extract the chemical state at each
layer for tokens with a known difficulty label, and train a logistic-regression
probe per layer. High probe accuracy means the state encodes difficulty.
"""
import json
import os
import random

import torch
import torch.nn as nn

from .tokenizer import TOKENIZER
from .tasks import get_task
from .tasks.base import get_batch
from .models import build_model


def _train_logreg(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                  seed: int = 0, epochs: int = 400) -> float:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_train = int(0.8 * len(X))
    tr, te = perm[:n_train], perm[n_train:]
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mean) / std

    probe = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(probe(Xn[tr]), y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(Xn[te]).argmax(dim=-1)
        return float((pred == y[te]).float().mean().item())


@torch.no_grad()
def run_probe(ckpt_path: str, task_name: str, cfg, out_path: str,
              seed: int = 0, n_batches: int = 20) -> dict:
    task = get_task(task_name)
    held_out = task.build_held_out(cfg)
    model = build_model("chemical", cfg).to(cfg.device)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model.eval()
    rng = random.Random(555)

    feats = [[] for _ in range(cfg.n_layers)]
    labels = []
    for _ in range(n_batches):
        x, y, diffs, ans = get_batch(task, cfg, rng, cfg.batch_size, held_out=held_out)
        pad = x != TOKENIZER.pad_id
        _, info = model(x, pad_mask=pad, meta={"diffs": diffs}, collect_chemical=True)
        m = pad & (diffs >= 0)
        for li, st in enumerate(info["chemical_states"]):
            feats[li].append(st[m].float().cpu())
        labels.append(diffs[m].cpu())

    y_all = torch.cat(labels)
    results = {}
    for li in range(cfg.n_layers):
        X = torch.cat(feats[li])
        acc = _train_logreg(X, y_all, task.n_bins, seed=seed)
        results[f"layer_{li}"] = acc
        print(f"probe layer {li}: acc {acc:.3f}", flush=True)

    out = {
        "task": task_name,
        "ckpt": os.path.basename(ckpt_path),
        "n_tokens": int(len(y_all)),
        "chance": 1.0 / task.n_bins,
        "probe_acc_by_layer": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"saved {out_path}", flush=True)
    return out
