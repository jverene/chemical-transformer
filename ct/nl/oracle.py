"""NL difficulty oracle: offline prepass measuring per-token compute value.

Runs the frozen Stage-1 (dense) checkpoint over a pool of training sequences
plus the held-out split, with gate leaves injected at every FFN:
    score_t = -sum_l dL/dg_{t,l}
Bin assignment:
  grad bins : score <= 0 -> 0 (irreducible, floor target 0.1);
              positive scores quartiled -> bins 1..4 (targets 0.2/0.4/0.6/0.8)
  loss bins : per-token CE quartiled -> bins 0..3 (targets 0.2/0.4/0.6/0.8)
Outputs <out>/<domain>/oracle.npz (arrays over pool/heldout seqs) and
oracle_diag.json with the circularity/position diagnostics:
  - frac_nonpos, score-loss correlation (reducibility)
  - per-layer mean gradients (layer-dominance check)
  - Spearman(bin, position-in-sequence) (position confound; the shuffled
    control does NOT remove this, so it must be measured)
  - token-type breakdown per bin (what lands in Q4?)
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .lm import GatedLM, GATE_TARGETS

GRAD_TARGETS = GATE_TARGETS  # [0.1, 0.2, 0.4, 0.6, 0.8]
LOSS_TARGETS = [0.2, 0.4, 0.6, 0.8]


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _token_type(tok, tid: int) -> str:
    s = tok.decode([tid]).strip()
    if not s:
        return "space"
    if s.isdigit():
        return "digit"
    if s[0].isalpha():
        return "alpha"
    return "punct"


def _structural_features(tok, seq_ids):
    """Per-token structural proxies for triangulating measured difficulty
    (used for the code domain's appendix table).

    Returns per-position token class and indentation depth of the token's
    line — computable, human-meaningful structure that measured value should
    correlate with if it measures anything real.
    """
    text = tok.decode(seq_ids)
    # char offset at the end of each decoded token
    ends, pos = [], 0
    for tid in seq_ids:
        pos += len(tok.decode([tid]))
        ends.append(pos)
    n = len(seq_ids) - 1  # positions t predict token t+1
    cls = [_token_type(tok, tid) for tid in seq_ids[:n]]
    indent, line_start = [], 0
    for i in range(n):
        start = ends[i] - len(tok.decode([seq_ids[i]]))
        if start < line_start:
            start = line_start
        indent.append(text.count("\t", line_start, start) * 4
                      + text.count(" ", line_start, start))
        if i > 0 and "\n" in text[ends[i - 1]:ends[i]]:
            line_start = ends[i] - len(text[ends[i - 1]:ends[i]].rpartition("\n")[2])
    return {"cls": cls, "indent": indent}


def oracle_pass(lm: GatedLM, seqs: np.ndarray, batch_seqs: int,
                collect_types: bool = False):
    """One fwd+bwd per batch with gate leaves -> per-token score and CE."""
    n, S = seqs.shape[0], seqs.shape[1] - 1
    score = np.zeros((n, S), dtype=np.float32)
    ce = np.zeros((n, S), dtype=np.float32)
    layer_grad = np.zeros((lm.n_layers, n, S), dtype=np.float32)
    for i in range(0, n, batch_seqs):
        chunk = torch.from_numpy(seqs[i:i + batch_seqs].astype(np.int64)) \
            .to(lm.device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        leaves = lm.oracle_leaves(*x.shape)
        logits = lm.model(x).logits
        ce_tok = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                 y.reshape(-1), reduction="none")
        ce_tok.mean().backward()
        g = torch.stack([leaf.grad for leaf in leaves], 0)
        score[i:i + x.shape[0]] = (-g.sum(0)).float().cpu().numpy()
        ce[i:i + x.shape[0]] = ce_tok.view_as(y).float().cpu().numpy()
        layer_grad[:, i:i + x.shape[0]] = g.float().cpu().numpy()
        lm.clear_oracle()
        for leaf in leaves:
            leaf.grad = None
    return score, ce, layer_grad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--ckpt", required=True,
                   help="Stage-1 (dense continue-trained) checkpoint .pt")
    p.add_argument("--data-root", default="data/nl")
    p.add_argument("--out-root", default="results-nl")
    p.add_argument("--pool-seqs", type=int, default=20000)
    p.add_argument("--batch-seqs", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    lm = GatedLM(args.model, args.device, method="ours-gategrad")
    for w in lm.wrappers:
        w.gate_mode = "head"
    if args.ckpt and os.path.exists(args.ckpt):
        lm.model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    lm.model.eval()

    data_dir = os.path.join(args.data_root, args.domain)
    out_dir = os.path.join(args.out_root, args.domain)
    os.makedirs(out_dir, exist_ok=True)

    train = np.load(os.path.join(data_dir, "train_tokens.npy"), mmap_mode="r")
    held = np.load(os.path.join(data_dir, "heldout_tokens.npy"), mmap_mode="r")
    pool = np.asarray(train[:args.pool_seqs])
    held_a = np.asarray(held)

    print("oracle pass: pool...")
    s_pool, ce_pool, lg_pool = oracle_pass(lm, pool, args.batch_seqs)
    print("oracle pass: heldout...")
    s_he, ce_he, lg_he = oracle_pass(lm, held_a, args.batch_seqs)

    def grad_bins(s):
        b = np.zeros(s.shape, dtype=np.uint8)
        pos = s > 0
        if pos.any():
            edges = np.quantile(s[pos], [0.25, 0.5, 0.75])
            b[pos] = 1 + np.digitize(s[pos], edges)
        return b

    def loss_bins(c):
        edges = np.quantile(c, [0.25, 0.5, 0.75])
        return np.digitize(c, edges).astype(np.uint8)

    gb_pool, lb_pool = grad_bins(s_pool), loss_bins(ce_pool)
    gb_he, lb_he = grad_bins(s_he), loss_bins(ce_he)

    np.savez_compressed(
        os.path.join(out_dir, "oracle.npz"),
        pool_bins_grad=gb_pool, pool_bins_loss=lb_pool,
        pool_score=s_pool.astype(np.float16), pool_ce=ce_pool.astype(np.float16),
        held_bins_grad=gb_he, held_bins_loss=lb_he,
        held_score=s_he.astype(np.float16), held_ce=ce_he.astype(np.float16),
    )

    # ---------------- diagnostics ----------------
    pos = s_pool > 0
    diag = {
        "frac_nonpos": float((~pos).mean()),
        "score_ce_pearson": float(np.corrcoef(s_pool.ravel(),
                                              ce_pool.ravel())[0, 1]),
        "score_ce_spearman": _spearman(s_pool.ravel(), ce_pool.ravel()),
        "layer_grad_mean": lg_pool.mean(axis=(1, 2)).tolist(),
        "grad_target_hist": np.bincount(gb_pool.ravel(),
                                        minlength=5).tolist(),
        "loss_target_hist": np.bincount(lb_pool.ravel(),
                                        minlength=4).tolist(),
    }
    # position confound: does the bin recapitulate position-in-sequence?
    S = s_pool.shape[1]
    posidx = np.tile(np.arange(S), (s_pool.shape[0], 1))
    diag["binspace_position_spearman"] = _spearman(
        gb_pool.ravel().astype(np.float64), posidx.ravel().astype(np.float64))
    diag["lossbin_position_spearman"] = _spearman(
        lb_pool.ravel().astype(np.float64), posidx.ravel().astype(np.float64))

    # token-type breakdown per grad bin (what lands in Q4 / the floor?)
    # Structural triangulation: per grad bin, token-class distribution and
    # mean line-indentation depth (defensible external proxies — the code
    # domain's "does measured value track real structure" appendix table).
    sample = min(500, s_pool.shape[0])
    feats = [_structural_features(lm.tok, pool[i].tolist()) for i in range(sample)]
    struct = {}
    for b in range(5):
        mask = gb_pool[:sample] == b
        indents, classes = [], Counter()
        for i in range(sample):
            indents += [feats[i]["indent"][j] for j in np.nonzero(mask[i])[0]]
            for j in np.nonzero(mask[i])[0]:
                classes[feats[i]["cls"][j]] += 1
        struct[str(b)] = {
            "indent_mean": float(np.mean(indents)) if indents else None,
            "indent_p90": float(np.percentile(indents, 90)) if indents else None,
            "n": int(mask.sum()),
            "cls": dict(classes.most_common(6)),
        }
    diag["structural_by_grad_bin"] = struct

    with open(os.path.join(out_dir, "oracle_diag.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print(json.dumps({k: v for k, v in diag.items()
                      if k != "token_types_by_grad_bin"}, indent=2))


if __name__ == "__main__":
    main()
