"""Evaluation metrics using pytrec_eval."""

from __future__ import annotations

from typing import Any

import pytrec_eval

from .data_loader import load_qrels


def convert_results_to_trec_format(
    results: dict[str, list],  # qid -> list of (docid, rank, score) or SearchResult
) -> dict[str, dict[str, float]]:
    """
    Convert results to pytrec_eval format.
    
    pytrec_eval expects: {qid: {docid: score, ...}, ...}
    """
    trec_results = {}
    for qid, docs in results.items():
        trec_results[qid] = {}
        for item in docs:
            if hasattr(item, 'docid'):  # SearchResult object
                trec_results[qid][item.docid] = float(item.score)
            elif isinstance(item, tuple) and len(item) >= 3:
                docid, rank, score = item[0], item[1], item[2]
                trec_results[qid][docid] = float(score)
            elif isinstance(item, tuple) and len(item) == 2:
                docid, score = item
                trec_results[qid][docid] = float(score)
    return trec_results


def convert_qrels_to_trec_format(
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """
    Ensure qrels are in pytrec_eval format.
    
    pytrec_eval expects: {qid: {docid: relevance, ...}, ...}
    """
    return qrels  # Already in correct format


def evaluate_map(
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    Args:
        results: {qid: {docid: score, ...}, ...}
        qrels: {qid: {docid: relevance, ...}, ...}
    
    Returns:
        MAP score
    """
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'map'}
    )
    metrics = evaluator.evaluate(results)
    
    # Average across queries
    map_scores = [m['map'] for m in metrics.values()]
    return sum(map_scores) / len(map_scores) if map_scores else 0.0


def comprehensive_evaluation(
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    # Comprehensive TREC-style evaluation
    metrics_to_compute = {
        'map',                # Primary metric for ROBUST04
        'ndcg',               # Overall NDCG
        'ndcg_cut_10',        # Early precision
        'ndcg_cut_100',       # Mid-range
        'P_10',               # Precision at 10
        'P_20',               # Precision at 20
        'P_100',              # Precision at 100
        'P_1000',             # Precision at 1000
        'recall_100',         # Recall at 100
        'recall_1000',        # Recall at 1000 (should be ~1.0)
    }
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    per_query_metrics = evaluator.evaluate(results)
    
    # Average across queries
    aggregated = {}
    for metric in metrics_to_compute:
        scores = [m.get(metric, 0) for m in per_query_metrics.values()]
        aggregated[metric] = sum(scores) / len(scores) if scores else 0.0
    
    return {
        'map': aggregated.get('map', 0),
        'ndcg': aggregated.get('ndcg', 0),
        'ndcg_10': aggregated.get('ndcg_cut_10', 0),
        'ndcg_100': aggregated.get('ndcg_cut_100', 0),
        'p_10': aggregated.get('P_10', 0),
        'p_20': aggregated.get('P_20', 0),
        'p_100': aggregated.get('P_100', 0),
        'p_1000': aggregated.get('P_1000', 0),
        'recall_100': aggregated.get('recall_100', 0),
        'recall_1000': aggregated.get('recall_1000', 0),
    }


def per_query_evaluation(
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """
    Get per-query metrics.
    
    Returns:
        {qid: {metric: value, ...}, ...}
    """
    metrics_to_compute = {'map', 'ndcg', 'ndcg_cut_10', 'P_10'}
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    return evaluator.evaluate(results)


def identify_weak_queries(
    per_query_results: dict[str, dict[str, float]],
    metric: str = 'map',
    threshold: float = 0.1,
) -> list[str]:
    """
    Find queries where performance is below threshold.
    
    Args:
        per_query_results: Output from per_query_evaluation
        metric: Metric to check
        threshold: Performance threshold
    
    Returns:
        List of weak query IDs
    """
    weak = []
    for qid, metrics in per_query_results.items():
        if metrics.get(metric, 0) < threshold:
            weak.append(qid)
    return weak


def print_evaluation_report(metrics: dict[str, float]) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MAP:          {metrics.get('map', 0):.4f}")
    print(f"  NDCG:         {metrics.get('ndcg', 0):.4f}")
    print(f"  NDCG@10:      {metrics.get('ndcg_10', 0):.4f}")
    print(f"  P@10:         {metrics.get('p_10', 0):.4f}")
    print(f"  P@20:         {metrics.get('p_20', 0):.4f}")
    print(f"  Recall@100:   {metrics.get('recall_100', 0):.4f}")
    print(f"  Recall@1000:  {metrics.get('recall_1000', 0):.4f}")
    print("=" * 50)

