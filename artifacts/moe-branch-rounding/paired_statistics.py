"""Balanced paired experiments and bootstrap intervals for median-ratio savings."""

import random
import numpy as np


def balanced_orders(pairs, seed):
    if pairs <= 0 or pairs % 2:
        raise ValueError("An even, positive pair count is required")
    orders = [("base", "candidate")] * (pairs // 2)
    orders += [("candidate", "base")] * (pairs // 2)
    random.Random(seed).shuffle(orders)
    return orders


def statistics(rows, seed=47):
    base = np.asarray([r["base"] for r in rows], dtype=np.float64)
    candidate = np.asarray([r["candidate"] for r in rows], dtype=np.float64)
    if (
        not len(base)
        or not np.all(np.isfinite([base, candidate]))
        or np.any(base <= 0)
        or np.any(candidate <= 0)
    ):
        raise ValueError("Paired times must be positive and finite")
    index = np.random.default_rng(seed).integers(0, len(base), size=(10000, len(base)))
    bootstrap = 1 - np.median(candidate[index], axis=1) / np.median(base[index], axis=1)

    def describe(values):
        median = np.median(values)
        return dict(
            median_ms=float(median),
            p95_ms=float(np.quantile(values, 0.95)),
            iqr_ms=float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            mad_ms=float(np.median(np.abs(values - median))),
        )

    return dict(
        pairs=len(base),
        base=describe(base),
        candidate=describe(candidate),
        improvement=float(1 - np.median(candidate) / np.median(base)),
        improvement_ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        paired_relative_median=float(np.median(1 - candidate / base)),
        paired_delta_median_ms=float(np.median(base - candidate)),
        bootstrap_repetitions=10000,
        bootstrap_seed=seed,
        base_first=sum(r["order"][0] == "base" for r in rows),
    )
