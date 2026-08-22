"""Mixed arithmetic with NO difficulty tags.

Operands have 1-4 digits, op in {+, -, *}, sampled uniformly. The model must
infer difficulty dynamically from the problem itself. Difficulty bins used for
evaluation are defined by the larger operand's digit count (1-4).
"""
import random

from .base import Task


def _randint_digits(rng: random.Random, digits: int) -> int:
    if digits == 1:
        return rng.randint(1, 9)
    return rng.randint(10 ** (digits - 1), 10 ** digits - 1)


class MixedArithmetic(Task):
    name = "mixed"
    bin_names = ["1-digit", "2-digit", "3-digit", "4-digit"]
    has_tags = False

    def sample(self, rng: random.Random, d=None):
        op = rng.choice(['+', '-', '*'])
        da, db = rng.randint(1, 4), rng.randint(1, 4)
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
