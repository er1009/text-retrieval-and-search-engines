"""Multi-signal fusion strategies."""

from __future__ import annotations

from typing import Sequence


def normalize_run_scores(run: dict[str, float]) -> dict[str, float]:
    """
    Min-max normalize scores in a run.
    """
    if not run:
        return {}
    
    scores = list(run.values())
    min_s = min(scores)
    max_s = max(scores)
    
    if max_s == min_s:
        return {docid: 0.5 for docid in run}
    
    return {
        docid: (score - min_s) / (max_s - min_s)
        for docid, score in run.items()
    }


def run_to_ranks(run: dict[str, float]) -> dict[str, int]:
    """
    Convert a scored run to ranks (1-indexed).
    """
    sorted_docs = sorted(run.items(), key=lambda x: -x[1])
    return {docid: rank + 1 for rank, (docid, _) in enumerate(sorted_docs)}


def rrf(
    runs: Sequence[dict[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion.
    
    RRF formula: score(d) = sum(1 / (k + rank_r(d)))
    
    k=60 is standard (Cormack et al.)
    
    Args:
        runs: List of dicts mapping docid -> score
        k: Smoothing constant (default 60)
    
    Returns:
        Fused run with RRF scores
    """
    # Convert to ranks
    rank_runs = [run_to_ranks(run) for run in runs]
    
    # Compute RRF scores
    fused = {}
    for rank_run in rank_runs:
        for docid, rank in rank_run.items():
            if docid not in fused:
                fused[docid] = 0.0
            fused[docid] += 1.0 / (k + rank)
    
    return dict(sorted(fused.items(), key=lambda x: -x[1]))


def combsum(runs: Sequence[dict[str, float]]) -> dict[str, float]:
    """
    CombSUM: Sum of normalized scores across runs.
    
    Args:
        runs: List of dicts mapping docid -> score
    
    Returns:
        Fused run with summed normalized scores
    """
    # Normalize each run
    normalized_runs = [normalize_run_scores(run) for run in runs]
    
    # Sum scores
    fused = {}
    for norm_run in normalized_runs:
        for docid, score in norm_run.items():
            if docid not in fused:
                fused[docid] = 0.0
            fused[docid] += score
    
    return dict(sorted(fused.items(), key=lambda x: -x[1]))


def combmnz(runs: Sequence[dict[str, float]]) -> dict[str, float]:
    """
    CombMNZ: Sum of scores × count of non-zero appearances.
    
    Rewards documents appearing in multiple runs.
    
    Args:
        runs: List of dicts mapping docid -> score
    
    Returns:
        Fused run with CombMNZ scores
    """
    # Normalize each run
    normalized_runs = [normalize_run_scores(run) for run in runs]
    
    # Sum scores and count appearances
    fused = {}
    counts = {}
    
    for norm_run in normalized_runs:
        for docid, score in norm_run.items():
            if docid not in fused:
                fused[docid] = 0.0
                counts[docid] = 0
            fused[docid] += score
            counts[docid] += 1
    
    # Multiply by count
    result = {docid: fused[docid] * counts[docid] for docid in fused}
    
    return dict(sorted(result.items(), key=lambda x: -x[1]))


def weighted_fusion(
    runs: Sequence[dict[str, float]],
    weights: Sequence[float],
) -> dict[str, float]:
    """
    Weighted linear combination of normalized scores.
    
    Args:
        runs: List of dicts mapping docid -> score
        weights: Weight for each run (should sum to 1 for interpretability)
    
    Returns:
        Fused run with weighted scores
    """
    if len(runs) != len(weights):
        raise ValueError("Number of runs must match number of weights")
    
    # Normalize each run
    normalized_runs = [normalize_run_scores(run) for run in runs]
    
    # Weighted sum
    fused = {}
    for norm_run, weight in zip(normalized_runs, weights):
        for docid, score in norm_run.items():
            if docid not in fused:
                fused[docid] = 0.0
            fused[docid] += weight * score
    
    return dict(sorted(fused.items(), key=lambda x: -x[1]))


# Fusion method registry
FUSION_METHODS = {
    "rrf": rrf,
    "combsum": combsum,
    "combmnz": combmnz,
    "weighted": weighted_fusion,
}


def fuse_runs(
    runs: Sequence[dict[str, float]],
    method: str = "rrf",
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """
    Fuse multiple runs using specified method.
    
    Args:
        runs: List of dicts mapping docid -> score
        method: Fusion method ('rrf', 'combsum', 'combmnz', 'weighted')
        rrf_k: k parameter for RRF
        weights: Weights for weighted fusion
    
    Returns:
        Fused run
    """
    if not runs:
        return {}
    
    # If only one run, return it as-is (sorted by score)
    if len(runs) == 1:
        return dict(sorted(runs[0].items(), key=lambda x: -x[1]))
    
    if method == "rrf":
        return rrf(runs, k=rrf_k)
    elif method == "combsum":
        return combsum(runs)
    elif method == "combmnz":
        return combmnz(runs)
    elif method == "weighted":
        if weights is None:
            # Equal weights if not specified
            weights = [1.0 / len(runs)] * len(runs)
        return weighted_fusion(runs, weights)
    else:
        raise ValueError(
            f"Unknown fusion method: {method}. "
            f"Available: {list(FUSION_METHODS.keys())}"
        )

