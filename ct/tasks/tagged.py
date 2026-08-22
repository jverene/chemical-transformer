"""Arithmetic with explicit difficulty tags [E]/[M]/[H].

  Easy:   [E] a op b = ans     a,b in [1,9]        ~162 unique (memorizable)
  Medium: [M] a op b = ans     a,b in [100,999]    ~1.6M unique
  Hard:   [H] a * b  = ans     a in [1000,9999], b in [100,999]  ~9M unique
"""
import random

from .base import Task


class TaggedArithmetic(Task):
    name = "tagged"
    bin_names = ["easy", "medium", "hard"]
    has_tags = True

    def sample(self, rng: random.Random, d=None):
        if d is None:
            r = rng.random()
            d = 0 if r < 0.30 else (1 if r < 0.70 else 2)
        if d == 0:
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            op = rng.choice(['+', '-'])
            ans = a + b if op == '+' else a - b
            s = f"[E]{a}{op}{b}={ans}"
        elif d == 1:
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            op = rng.choice(['+', '-'])
            ans = a + b if op == '+' else a - b
            s = f"[M]{a}{op}{b}={ans}"
        else:
            a = rng.randint(1000, 9999)
            b = rng.randint(100, 999)
            s = f"[H]{a}*{b}={a * b}"
        toks, amask = self.encode(s)
        return toks, d, amask

    def build_pool(self, rng: random.Random) -> list:
        pool = []
        for d in (0, 1, 2):
            for _ in range(self.pool_per_bin):
                pool.append(self.sample(rng, d))
        return pool
