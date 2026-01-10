"""Parameter tuning with grid search and cross-validation."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from .config import (
    Config, BM25Config, RM3Config, FusionConfig, DEFAULT_CONFIG
)
from .data_loader import load_queries, load_expanded_queries, load_qrels, get_train_qids
from .bm25_retrieval import batch_search_bm25, batch_search_bm25_rm3
from .evaluation import evaluate_map, convert_results_to_trec_format


def tune_bm25_params(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k1_range: list[float] | None = None,
    b_range: list[float] | None = None,
) -> tuple[dict[str, float], float]:
    """
    Grid search BM25 parameters.
    
    Returns:
        Tuple of (best_params, best_map)
    """
    if k1_range is None:
        k1_range = DEFAULT_CONFIG.bm25.k1_range
    if b_range is None:
        b_range = DEFAULT_CONFIG.bm25.b_range
    
    best_map = 0.0
    best_params = {'k1': 0.9, 'b': 0.4}
    
    total = len(k1_range) * len(b_range)
    pbar = tqdm(total=total, desc="Tuning BM25")
    
    for k1, b in product(k1_range, b_range):
        results = batch_search_bm25(queries, k=1000, k1=k1, b=b)
        
        # Convert to evaluation format
        eval_results = {}
        for qid, hits in results.items():
            eval_results[qid] = {hit.docid: hit.score for hit in hits}
        
        # Filter to only queries with qrels
        eval_results = {qid: docs for qid, docs in eval_results.items() if qid in qrels}
        
        map_score = evaluate_map(eval_results, qrels)
        
        if map_score > best_map:
            best_map = map_score
            best_params = {'k1': k1, 'b': b}
        
        pbar.update(1)
        pbar.set_postfix({'best_map': f'{best_map:.4f}'})
    
    pbar.close()
    return best_params, best_map


def tune_rm3_params(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    bm25_params: dict[str, float],
    fb_docs_range: list[int] | None = None,
    fb_terms_range: list[int] | None = None,
    orig_weight_range: list[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Grid search RM3 parameters on top of best BM25.
    
    Returns:
        Tuple of (best_params, best_map)
    """
    if fb_docs_range is None:
        fb_docs_range = DEFAULT_CONFIG.rm3.fb_docs_range
    if fb_terms_range is None:
        fb_terms_range = DEFAULT_CONFIG.rm3.fb_terms_range
    if orig_weight_range is None:
        orig_weight_range = DEFAULT_CONFIG.rm3.original_weight_range
    
    best_map = 0.0
    best_params = {
        'fb_docs': 10,
        'fb_terms': 10,
        'original_query_weight': 0.5,
    }
    
    total = len(fb_docs_range) * len(fb_terms_range) * len(orig_weight_range)
    pbar = tqdm(total=total, desc="Tuning RM3")
    
    for fb_docs, fb_terms, orig_weight in product(fb_docs_range, fb_terms_range, orig_weight_range):
        results = batch_search_bm25_rm3(
            queries,
            k=1000,
            k1=bm25_params['k1'],
            b=bm25_params['b'],
            fb_docs=fb_docs,
            fb_terms=fb_terms,
            original_query_weight=orig_weight,
        )
        
        # Convert to evaluation format
        eval_results = {}
        for qid, hits in results.items():
            eval_results[qid] = {hit.docid: hit.score for hit in hits}
        
        eval_results = {qid: docs for qid, docs in eval_results.items() if qid in qrels}
        
        map_score = evaluate_map(eval_results, qrels)
        
        if map_score > best_map:
            best_map = map_score
            best_params = {
                'fb_docs': fb_docs,
                'fb_terms': fb_terms,
                'original_query_weight': orig_weight,
            }
        
        pbar.update(1)
        pbar.set_postfix({'best_map': f'{best_map:.4f}'})
    
    pbar.close()
    return best_params, best_map


def cross_validated_tuning(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Tune parameters using k-fold cross-validation.
    
    This helps avoid overfitting on the 50 training queries.
    
    Returns:
        Best configuration as a dict
    """
    train_qids = list(queries.keys())
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Track best params across folds
    all_bm25_params = []
    all_rm3_params = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_qids)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        train_fold_qids = [train_qids[i] for i in train_idx]
        val_fold_qids = [train_qids[i] for i in val_idx]
        
        train_queries = {qid: queries[qid] for qid in train_fold_qids}
        val_queries = {qid: queries[qid] for qid in val_fold_qids}
        train_qrels = {qid: qrels[qid] for qid in train_fold_qids if qid in qrels}
        val_qrels = {qid: qrels[qid] for qid in val_fold_qids if qid in qrels}
        
        # Tune BM25 on train fold
        bm25_params, _ = tune_bm25_params(train_queries, train_qrels)
        all_bm25_params.append(bm25_params)
        
        # Tune RM3 on train fold
        rm3_params, _ = tune_rm3_params(train_queries, train_qrels, bm25_params)
        all_rm3_params.append(rm3_params)
    
    # Average best params across folds (for numerical params)
    # For simplicity, take most common values
    avg_k1 = np.mean([p['k1'] for p in all_bm25_params])
    avg_b = np.mean([p['b'] for p in all_bm25_params])
    avg_fb_docs = int(np.round(np.mean([p['fb_docs'] for p in all_rm3_params])))
    avg_fb_terms = int(np.round(np.mean([p['fb_terms'] for p in all_rm3_params])))
    avg_orig_weight = np.mean([p['original_query_weight'] for p in all_rm3_params])
    
    return {
        'bm25': {'k1': avg_k1, 'b': avg_b},
        'rm3': {
            'fb_docs': avg_fb_docs,
            'fb_terms': avg_fb_terms,
            'original_query_weight': avg_orig_weight,
        }
    }


def run_full_tuning(
    output_path: Path | None = None,
    use_cv: bool = True,
) -> dict[str, Any]:
    """
    Run complete parameter tuning pipeline.
    
    Args:
        output_path: Path to save best config JSON
        use_cv: Whether to use cross-validation
    
    Returns:
        Best configuration
    """
    print("Loading data...")
    queries = load_queries()
    expanded_queries = load_expanded_queries()
    qrels = load_qrels()
    
    # Use only train queries for tuning
    train_qids = get_train_qids()
    train_queries = {qid: queries[qid] for qid in train_qids if qid in queries}
    train_expanded = {qid: expanded_queries[qid] for qid in train_qids if qid in expanded_queries}
    train_qrels = {qid: qrels[qid] for qid in train_qids if qid in qrels}
    
    print(f"Tuning on {len(train_queries)} training queries...")
    
    if use_cv:
        # Cross-validated tuning
        best_config = cross_validated_tuning(train_expanded, train_qrels)
    else:
        # Simple grid search
        print("\n=== Tuning BM25 ===")
        bm25_params, bm25_map = tune_bm25_params(train_expanded, train_qrels)
        print(f"Best BM25: k1={bm25_params['k1']}, b={bm25_params['b']}, MAP={bm25_map:.4f}")
        
        print("\n=== Tuning RM3 ===")
        rm3_params, rm3_map = tune_rm3_params(train_expanded, train_qrels, bm25_params)
        print(f"Best RM3: fb_docs={rm3_params['fb_docs']}, "
              f"fb_terms={rm3_params['fb_terms']}, "
              f"weight={rm3_params['original_query_weight']}, MAP={rm3_map:.4f}")
        
        best_config = {'bm25': bm25_params, 'rm3': rm3_params}
    
    # Save config
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nSaved best config to {output_path}")
    
    return best_config


def load_tuned_config(path: Path) -> dict[str, Any]:
    """Load tuned configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

