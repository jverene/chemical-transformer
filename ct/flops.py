"""FLOPs accounting.

Counts operations actually performed: attention runs for every token, while
FFN ops are scaled by the per-token gate (gated/s skipped tokens contribute
proportionally less). The same accounting is applied to every method so
comparisons are fair.
"""


class FLOPsCounter:
    @staticmethod
    def attention(B: int, S: int, D: int, H: int) -> float:
        Dh = D // H
        qkv = 3 * B * S * D * D
        scores = B * H * S * S * Dh
        apply = B * H * S * S * Dh
        out = B * S * D * D
        return float(qkv + scores + apply + out)

    @staticmethod
    def ffn_per_token(D: int, d_ff: int) -> float:
        # two linear layers, MACs -> 2x
        return float(2 * (D * d_ff + d_ff * D) * 2)
