"""Wall-clock realization: does the billed FFN saving survive real execution?

Times full fwd+bwd steps of the pretrained LM under
  dense                : every token through every FFN (g=1)
  rotation (masked)    : compute-then-zero — the training-driver path
  rotation_skip (skip) : gather/scatter — masked tokens' FFN rows skipped
at budgets {0.25, 0.5, 0.75}. The skip variant is what a serving stack
would run; the gap to `rotation (masked)` is the price of the naive path.

Writes JSON {dense_tok_s, rows:[{mode,budget,tok_s,ratio}]}.
"""
import argparse
import json
import time

import torch

from .lm import GatedLM


def _sync(x):
    if x.is_cuda:
        torch.cuda.synchronize()
    elif x.device.type == "mps":
        torch.mps.synchronize()


def bench(lm, x, y, iters=10, warmup=3):
    opt = torch.optim.SGD(lm.model.parameters(), lr=0.0)
    for _ in range(warmup):
        opt.zero_grad()
        logits = lm.model(x).logits
        torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
        ).backward()
    _sync(x)
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        logits = lm.model(x).logits
        torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
        ).backward()
    _sync(x)
    return (x.shape[0] * x.shape[1] * iters) / (time.time() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--batch-seqs", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results-nl/wallclock.json")
    args = p.parse_args()

    lm = GatedLM(args.model, args.device, method="baseline")
    lm.model.train()
    g = torch.Generator(device="cpu").manual_seed(0)
    vocab = lm.tok.vocab_size
    x = torch.randint(0, vocab, (args.batch_seqs, args.seq_len),
                      generator=g).to(args.device)
    y = torch.randint(0, vocab, (args.batch_seqs, args.seq_len),
                      generator=g).to(args.device)

    rows = []
    dense_tps = None
    for mode, budget in [("none", 1.0), ("rotation", 0.5),
                         ("rotation_skip", 0.75), ("rotation_skip", 0.5),
                         ("rotation_skip", 0.25)]:
        for w in lm.wrappers:
            w.gate_mode = mode
            w.capacity = budget
        tps = bench(lm, x, y)
        if mode == "none":
            dense_tps = tps
        rows.append({"mode": mode, "budget": budget, "tok_s": round(tps, 1),
                     "ratio": round(tps / dense_tps, 3)})
        print(rows[-1], flush=True)

    out = {"model": args.model, "batch_seqs": args.batch_seqs,
           "seq_len": args.seq_len, "dense_tok_s": round(dense_tps, 1),
           "rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
