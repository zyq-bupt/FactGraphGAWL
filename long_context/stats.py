"""Pearson 与配对 bootstrap。不把“一组显著另一组不显著”当成差异显著。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def pearson_with_p(human: Sequence[float], model: Sequence[float]) -> tuple[float | None, float | None]:
    xs = np.asarray(human, dtype=float)
    ys = np.asarray(model, dtype=float)
    if xs.size < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None, None
    r, p = pearsonr(xs, ys)
    return float(r), float(p)


def paired_bootstrap_delta_r(
    human: Sequence[float],
    baseline: Sequence[float],
    sliding: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 20260818,
) -> dict:
    h = np.asarray(human, dtype=float)
    b = np.asarray(baseline, dtype=float)
    s = np.asarray(sliding, dtype=float)
    n = h.size
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    skipped = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        hh, bb, ss = h[idx], b[idx], s[idx]
        if len(np.unique(hh)) < 2 or len(np.unique(bb)) < 2 or len(np.unique(ss)) < 2:
            skipped += 1
            continue
        rb, _p = pearsonr(hh, bb)
        rs, _p = pearsonr(hh, ss)
        if not np.isfinite(rb) or not np.isfinite(rs):
            skipped += 1
            continue
        deltas.append(float(rs - rb))
    out = {
        "n_boot_requested": int(n_boot),
        "n_boot_valid": len(deltas),
        "n_boot_skipped": skipped,
        "delta_r_ci95_low": None,
        "delta_r_ci95_high": None,
        "delta_r_boot_mean": None,
    }
    if deltas:
        arr = np.asarray(deltas, dtype=float)
        out["delta_r_ci95_low"] = float(np.percentile(arr, 2.5))
        out["delta_r_ci95_high"] = float(np.percentile(arr, 97.5))
        out["delta_r_boot_mean"] = float(np.mean(arr))
        out["deltas"] = deltas
    else:
        out["deltas"] = []
    return out


def spearman_optional(x: Sequence[float], y: Sequence[float]) -> tuple[float | None, float | None]:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None, None
    r, p = spearmanr(xs, ys)
    return float(r), float(p)


def mean_std(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None, None
    arr = np.asarray(nums, dtype=float)
    if arr.size > 1:
        return float(arr.mean()), float(arr.std(ddof=1))
    return float(arr.mean()), 0.0


def median(values: Sequence[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(np.median(np.asarray(nums, dtype=float)))
