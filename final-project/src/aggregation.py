"""Score aggregation strategies for passage-level reranking."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .document_processor import Passage


def maxp(passage_scores: list[float]) -> float:
    """
    MaxP: Document score = max(passage scores).
    
    Best for: Documents where one passage contains the answer.
    """
    if not passage_scores:
        return 0.0
    return max(passage_scores)


def sump(passage_scores: list[float]) -> float:
    """
    SumP: Document score = sum(passage scores).
    
    Best for: Documents with multiple relevant passages.
    """
    if not passage_scores:
        return 0.0
    return sum(passage_scores)


def firstp(passage_scores: list[float]) -> float:
    """
    FirstP: Document score = first passage score.
    
    Best for: News articles (lead paragraph is key).
    """
    if not passage_scores:
        return 0.0
    return passage_scores[0]


def avgp(passage_scores: list[float]) -> float:
    """
    AvgP: Document score = mean(passage scores).
    """
    if not passage_scores:
        return 0.0
    return float(np.mean(passage_scores))


def topkp(passage_scores: list[float], k: int = 3) -> float:
    """
    TopKP: Document score = mean of top-k passage scores.
    """
    if not passage_scores:
        return 0.0
    sorted_scores = sorted(passage_scores, reverse=True)
    top_k = sorted_scores[:k]
    return float(np.mean(top_k))


def max_first_hybrid(passage_scores: list[float], first_weight: float = 0.3) -> float:
    """
    Hybrid: Weighted combination of MaxP and FirstP.
    
    Best for news articles (ROBUST04): the first paragraph is important,
    but the max passage might capture a more relevant section.
    
    Default: 70% MaxP + 30% FirstP
    """
    if not passage_scores:
        return 0.0
    max_score = max(passage_scores)
    first_score = passage_scores[0]
    return (1 - first_weight) * max_score + first_weight * first_score


# Strategy registry
AGGREGATION_STRATEGIES: dict[str, Callable[[list[float]], float]] = {
    "maxp": maxp,
    "sump": sump,
    "firstp": firstp,
    "avgp": avgp,
    "top3p": lambda scores: topkp(scores, k=3),
    "top5p": lambda scores: topkp(scores, k=5),
    "hybrid": max_first_hybrid,  # Best for news articles (ROBUST04)
}


def get_aggregation_function(strategy: str) -> Callable[[list[float]], float]:
    """Get aggregation function by name."""
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}. "
            f"Available: {list(AGGREGATION_STRATEGIES.keys())}"
        )
    return AGGREGATION_STRATEGIES[strategy]


def aggregate_passage_scores(
    passages: list[Passage],
    scores: list[float],
    doc_to_passage_indices: dict[str, list[int]],
    strategy: str = "maxp",
) -> dict[str, float]:
    """
    Aggregate passage scores to document scores.
    
    Args:
        passages: List of all passages
        scores: Scores aligned with passages
        doc_to_passage_indices: Mapping from docid to passage indices
        strategy: Aggregation strategy name
    
    Returns:
        Dict mapping docid -> aggregated score
    """
    agg_func = get_aggregation_function(strategy)
    
    doc_scores = {}
    for docid, indices in doc_to_passage_indices.items():
        passage_scores = [scores[i] for i in indices if i < len(scores)]
        doc_scores[docid] = agg_func(passage_scores)
    
    return doc_scores


def rank_documents_by_score(doc_scores: dict[str, float]) -> list[tuple[str, int, float]]:
    """
    Rank documents by score.
    
    Returns:
        List of (docid, rank, score) tuples sorted by score descending.
    """
    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
    return [(docid, rank + 1, score) for rank, (docid, score) in enumerate(sorted_docs)]

