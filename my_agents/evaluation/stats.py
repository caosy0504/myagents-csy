"""Small-sample uncertainty and paired comparisons for fixed evaluation suites."""

import math
import random
from typing import Any, Sequence


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> dict[str, float] | None:
    """Two-sided Wilson binomial interval, approximately 95% for z=1.96."""
    if total == 0:
        return None
    if not 0 <= successes <= total or z <= 0:
        raise ValueError("Invalid successes, total, or z")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def paired_bootstrap_delta(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], *,
    id_key: str, score_key: str, seed: int = 42, resamples: int = 2000,
) -> dict:
    """Estimate left-minus-right difference on common case IDs only."""
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    a = {row[id_key]: row[score_key] for row in left if row.get(score_key) is not None}
    b = {row[id_key]: row[score_key] for row in right if row.get(score_key) is not None}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("Reports have no comparable scored case IDs")
    differences = [float(a[item]) - float(b[item]) for item in ids]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    values = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(resamples)
    )
    return {
        "common_cases": len(ids), "delta": observed,
        "ci95": {"lower": values[int(0.025 * resamples)],
                 "upper": values[min(resamples - 1, int(0.975 * resamples))]},
        "seed": seed, "resamples": resamples,
        "note": "Bootstrap interval describes this fixed suite, not all future tasks.",
    }
