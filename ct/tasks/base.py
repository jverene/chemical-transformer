"""Task base class: sampling, sequence packing, held-out sets, batching."""
import random
from typing import List, Optional, Tuple

import torch

from ..tokenizer import TOKENIZER


class Task:
    name: str = "base"
    bin_names: List[str] = []
    has_tags: bool = False
    pool_per_bin: int = 4000

    @property
    def n_bins(self) -> int:
        return len(self.bin_names)

    def sample(self, rng: random.Random, d: Optional[int] = None):
        """Return (tokens incl. EOS, difficulty bin, answer mask aligned with tokens)."""
        raise NotImplementedError

    def encode(self, s: str) -> Tuple[List[int], List[int]]:
        """Encode a problem string; answer mask marks tokens after '=' plus EOS."""
        eq = s.index('=')
        toks = TOKENIZER.encode(s)
        ans = [1 if i > eq else 0 for i in range(len(toks))]
        toks.append(TOKENIZER.eos_id)
        ans.append(1)
        return toks, ans

    def build_pool(self, rng: random.Random) -> list:
        raise NotImplementedError

    def pack(self, rng: random.Random, max_len: int, pool: Optional[list] = None):
        tokens = [TOKENIZER.bos_id]
        diffs = [-1]
        ans = [0]
        while len(tokens) < max_len - 1:
            if pool is not None:
                toks, d, a = pool[rng.randrange(len(pool))]
            else:
                toks, d, a = self.sample(rng)
            if len(tokens) + len(toks) > max_len - 1:
                break
            tokens.extend(toks)
            diffs.extend([d] * len(toks))
            ans.extend(a)
        pad = max_len - len(tokens)
        tokens += [TOKENIZER.pad_id] * pad
        diffs += [-1] * pad
        ans += [0] * pad
        return tokens[:max_len], diffs[:max_len], ans[:max_len]

    def build_held_out(self, cfg) -> list:
        """Frozen held-out set: identical for every method and seed."""
        rng = random.Random(cfg.held_out_seed)
        pool = self.build_pool(rng)
        pack_rng = random.Random(cfg.held_out_seed + 1)
        return [self.pack(pack_rng, cfg.seq_len, pool) for _ in range(cfg.test_seqs)]


def get_batch(task: Task, cfg, rng: random.Random, batch_size: int,
              held_out: Optional[list] = None):
    if held_out is not None:
        seqs = [held_out[rng.randrange(len(held_out))] for _ in range(batch_size)]
    else:
        seqs = [task.pack(rng, cfg.seq_len) for _ in range(batch_size)]
    x = torch.tensor([s[0] for s in seqs], dtype=torch.long, device=cfg.device)
    diffs = torch.tensor([s[1] for s in seqs], dtype=torch.long, device=cfg.device)
    ans = torch.tensor([s[2] for s in seqs], dtype=torch.long, device=cfg.device)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = TOKENIZER.pad_id
    return x, y, diffs, ans
