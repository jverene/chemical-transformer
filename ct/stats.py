"""Aggregation and significance testing across seeds."""
import numpy as np
from scipy import stats as scipy_stats


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def welch_ttest(a, b):
    """Welch's t-test between two seed groups. Returns (t, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)
