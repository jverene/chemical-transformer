"""Hardened arithmetic regimes for the scale-up (ICLR expansion).

Same contracts as tagged/mixed, extended so large models (150M-1B) stay inside
the difficulty window: hard-bin accuracy neither saturated nor collapsed.
Model-relative difficulty: the probe runs (P1) pick the regime/scale combo.

  tagged-v2: E = 1-2 digit +/-; M = 4-5 digit +/-;
             H = {4x4, 5x4, 6x5} digit multiplication (uniform mix).
  mixed-v2:  operands 1-7 digits, ops {+,-,*}; bin = larger operand's digit
             count (7 bins). Digit-count targets generalize linearly.
"""
import random

from .base import Task
from .mixed import _randint_digits


class TaggedHardArithmetic(Task):
    name = "tagged-v2"
    bin_names = ["easy", "medium", "hard"]
    has_tags = True

    def sample(self, rng: random.Random, d=None):
        if d is None:
            r = rng.random()
            d = 0 if r < 0.30 else (1 if r < 0.70 else 2)
        if d == 0:
            da, db = rng.randint(1, 2), rng.randint(1, 2)
            a, b = _randint_digits(rng, da), _randint_digits(rng, db)
            op = rng.choice(['+', '-'])
            ans = a + b if op == '+' else a - b
            s = f"[E]{a}{op}{b}={ans}"
        elif d == 1:
            da, db = rng.randint(4, 5), rng.randint(4, 5)
            a, b = _randint_digits(rng, da), _randint_digits(rng, db)
            op = rng.choice(['+', '-'])
            ans = a + b if op == '+' else a - b
            s = f"[M]{a}{op}{b}={ans}"
        else:
            variant = rng.choice([(4, 4), (5, 4), (6, 5)])
            a = _randint_digits(rng, variant[0])
            b = _randint_digits(rng, variant[1])
            s = f"[H]{a}*{b}={a * b}"
        toks, amask = self.encode(s)
        return toks, d, amask

    def build_pool(self, rng: random.Random) -> list:
        pool = []
        for d in (0, 1, 2):
            for _ in range(self.pool_per_bin):
                pool.append(self.sample(rng, d))
        return pool


class MixedHardArithmetic(Task):
    name = "mixed-v2"
    bin_names = [f"{k}-digit" for k in range(1, 8)]
    has_tags = False

    def sample(self, rng: random.Random, d=None):
        op = rng.choice(['+', '-', '*'])
        da, db = rng.randint(1, 7), rng.randint(1, 7)
        a = _randint_digits(rng, da)
        b = _randint_digits(rng, db)
        if op == '+':
            ans = a + b
        elif op == '-':
            ans = a - b
        else:
            ans = a * b
        s = f"{a}{op}{b}={ans}"
        bin_idx = max(da, db) - 1
        toks, amask = self.encode(s)
        return toks, bin_idx, amask

    def build_pool(self, rng: random.Random) -> list:
        return [self.sample(rng) for _ in range(self.pool_per_bin * self.n_bins)]
