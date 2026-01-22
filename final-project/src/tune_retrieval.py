"""
Fast retrieval-only tuning script.
Tunes BM25, RM3, and RRF parameters without neural reranking.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pytrec_eval

from .bm25_retrieval import batch_search_bm25_rm3
from .data_loader import load_queries, load_qrels, load_expanded_queries, get_train_qids, get_test_qids
from .dense_index import DensePassageIndex
from .fusion import fuse_runs


def evaluate_retrieval(results: dict, qrels: dict) -> dict:
    """Evaluate retrieval results."""
    # Convert to pytrec format
    run = {}
    for qid, docs in results.items():
        if isinstance(docs, dict):
            run[qid] = docs
        else:
            run[qid] = {d.docid: d.score for d in docs}
    
    # Filter to queries with qrels
    run = {qid: run[qid] for qid in run if qid in qrels}
    
    metrics = {'map', 'recall_1000', 'recall_100', 'P_10', 'ndcg_cut_10'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)
    
    return {
        'map': np.mean([r['map'] for r in results.values()]),
        'recall_1000': np.mean([r['recall_1000'] for r in results.values()]),
        'recall_100': np.mean([r['recall_100'] for r in results.values()]),
        'p10': np.mean([r['P_10'] for r in results.values()]),
        'ndcg10': np.mean([r['ndcg_cut_10'] for r in results.values()]),
    }


def run_retrieval(
    queries_orig: dict,
    queries_exp: dict,
    dense_index: DensePassageIndex,
    splade_index,
    k: int,
    k1: float,
    b: float,
    fb_docs: int,
    fb_terms: int,
    orig_weight: float,
    rrf_k: int,
    retrieval_weights: tuple = None,  # (bm25, dense, splade) weights
) -> dict:
    """Run 6-way hybrid retrieval with given parameters."""
    
    # BM25+RM3 original
    bm25_orig = batch_search_bm25_rm3(
        queries_orig, k=k, k1=k1, b=b,
        fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
    )
    
    # BM25+RM3 expanded
    bm25_exp = batch_search_bm25_rm3(
        queries_exp, k=k, k1=k1, b=b,
        fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
    )
    
    # Dense original
    dense_orig = dense_index.batch_search(queries_orig, top_k=k, show_progress=False)
    
    # Dense expanded  
    dense_exp = dense_index.batch_search(queries_exp, top_k=k, show_progress=False)
    
    # SPLADE (if available)
    if splade_index:
        splade_orig = splade_index.batch_search(queries_orig, top_k=k, show_progress=False)
        splade_exp = splade_index.batch_search(queries_exp, top_k=k, show_progress=False)
    
    # Convert BM25 results to dict format
    bm25_orig_dict = {qid: {r.docid: r.score for r in docs} for qid, docs in bm25_orig.items()}
    bm25_exp_dict = {qid: {r.docid: r.score for r in docs} for qid, docs in bm25_exp.items()}
    
    # Build weights list: [bm25_orig, bm25_exp, dense_orig, dense_exp, splade_orig, splade_exp]
    if retrieval_weights:
        w_bm25, w_dense, w_splade = retrieval_weights
        weights = [w_bm25, w_bm25, w_dense, w_dense]  # orig and exp get same weight per method
        if splade_index:
            weights.extend([w_splade, w_splade])
    else:
        weights = None  # Equal weights
    
    # Fuse all retrievers
    fused = {}
    for qid in queries_orig.keys():
        runs = [
            bm25_orig_dict.get(qid, {}),
            bm25_exp_dict.get(qid, {}),
            dense_orig.get(qid, {}),
            dense_exp.get(qid, {}),
        ]
        if splade_index:
            runs.append(splade_orig.get(qid, {}))
            runs.append(splade_exp.get(qid, {}))
        
        # Filter empty runs and adjust weights
        non_empty_runs = []
        non_empty_weights = []
        for i, r in enumerate(runs):
            if r:
                non_empty_runs.append(r)
                if weights:
                    non_empty_weights.append(weights[i])
        
        if non_empty_runs:
            fused[qid] = fuse_runs(
                non_empty_runs, 
                method="rrf", 
                rrf_k=rrf_k,
                weights=non_empty_weights if weights else None
            )
    
    return fused


def main():
    parser = argparse.ArgumentParser(description="Fast retrieval tuning")
    parser.add_argument("--dense-index-path", required=True, help="Path to dense index")
    parser.add_argument("--splade-index-path", default=None, help="Path to SPLADE index")
    parser.add_argument("--output", default="tuning_results.json", help="Output file")
    parser.add_argument("--retrieval-k", type=int, default=2000, help="Retrieval depth")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer configs)")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to tune on (train or test)")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"RETRIEVAL PARAMETER TUNING ({args.split.upper()})")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading data...")
    all_queries = load_queries()
    all_expanded = load_expanded_queries()
    all_qrels = load_qrels()
    
    # Filter by split
    if args.split == "train":
        qids = get_train_qids()
    else:
        qids = get_test_qids()
    
    queries_orig = {qid: all_queries[qid] for qid in qids if qid in all_queries}
    queries_exp = {qid: all_expanded.get(qid, all_queries.get(qid, "")) for qid in qids}
    qrels = {qid: all_qrels[qid] for qid in qids if qid in all_qrels}
    
    print(f"  Split: {args.split}")
    print(f"  Queries: {len(queries_orig)}")
    
    # Load indexes
    print("\n[2/4] Loading indexes...")
    dense_index = DensePassageIndex(args.dense_index_path)
    dense_index.load()
    
    splade_index = None
    if args.splade_index_path:
        from .splade_index import SpladeIndex
        splade_index = SpladeIndex(args.splade_index_path)
        splade_index.load()
    
    # Define parameter grid
    # Retrieval weights: (bm25, dense, splade) - relative weights for each method type
    weight_patterns = [
        (1.0, 1.0, 1.0),  # Equal weights
        (1.0, 1.0, 1.5),  # Higher SPLADE
        (1.0, 1.5, 1.0),  # Higher Dense
        (0.5, 1.0, 1.5),  # Lower BM25, higher SPLADE
    ]
    
    if args.quick:
        param_grid = {
            'k1': [0.9, 1.2],
            'b': [0.4, 0.75],
            'fb_docs': [10, 15],
            'fb_terms': [10, 20],
            'orig_weight': [0.5, 0.7],
            'rrf_k': [45, 60],
            'retrieval_weights': weight_patterns,
        }
    else:
        param_grid = {
            'k1': [0.6, 0.9, 1.2, 1.5],
            'b': [0.3, 0.4, 0.6, 0.75],
            'fb_docs': [5, 10, 15, 20],
            'fb_terms': [10, 15, 20, 25],
            'orig_weight': [0.4, 0.5, 0.6, 0.7, 0.8],
            'rrf_k': [30, 45, 60, 80, 100],
            'retrieval_weights': weight_patterns + [
                (1.5, 1.0, 1.0),  # Higher BM25
                (1.0, 0.5, 1.5),  # Lower Dense, higher SPLADE
                (0.5, 1.5, 1.0),  # Higher Dense, lower BM25
            ],
        }
    
    # Calculate total configs
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"\n[3/4] Testing {total} configurations...")
    
    # Run grid search
    results = []
    best_map = 0
    best_config = None
    
    start_time = time.time()
    
    configs = list(itertools.product(
        param_grid['k1'],
        param_grid['b'],
        param_grid['fb_docs'],
        param_grid['fb_terms'],
        param_grid['orig_weight'],
        param_grid['rrf_k'],
        param_grid['retrieval_weights'],
    ))
    
    for i, (k1, b, fb_docs, fb_terms, orig_weight, rrf_k, ret_weights) in enumerate(configs):
        # Run retrieval
        fused = run_retrieval(
            queries_orig, queries_exp, dense_index, splade_index,
            k=args.retrieval_k, k1=k1, b=b,
            fb_docs=fb_docs, fb_terms=fb_terms, orig_weight=orig_weight,
            rrf_k=rrf_k,
            retrieval_weights=ret_weights,
        )
        
        # Evaluate
        metrics = evaluate_retrieval(fused, qrels)
        
        config = {
            'k1': k1, 'b': b,
            'fb_docs': fb_docs, 'fb_terms': fb_terms,
            'orig_weight': orig_weight, 'rrf_k': rrf_k,
            'ret_weights': ret_weights,  # (bm25, dense, splade)
            **metrics
        }
        results.append(config)
        
        # Track best
        if metrics['map'] > best_map:
            best_map = metrics['map']
            best_config = config
            w_str = f"w={ret_weights[0]}/{ret_weights[1]}/{ret_weights[2]}"
            print(f"  [{i+1}/{total}] NEW BEST: MAP={metrics['map']:.4f} R@1K={metrics['recall_1000']:.4f} "
                  f"(k1={k1}, b={b}, fb={fb_docs}/{fb_terms}, ow={orig_weight}, rrf_k={rrf_k}, {w_str})")
        elif (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] Current best MAP: {best_map:.4f}")
    
    elapsed = time.time() - start_time
    
    # Sort by MAP
    results = sorted(results, key=lambda x: -x['map'])
    
    # Print results
    print("\n" + "=" * 70)
    print("[4/4] RESULTS")
    print("=" * 70)
    
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/total:.2f}s per config)")
    
    print("\n  TOP 10 CONFIGURATIONS:")
    print("  " + "-" * 85)
    print(f"  {'Rank':<5} {'MAP':>7} {'R@1K':>7} {'P@10':>6} | k1    b     fb_d  fb_t  ow    rrf_k  weights(B/D/S)")
    print("  " + "-" * 85)
    
    for i, r in enumerate(results[:10]):
        w = r.get('ret_weights', (1,1,1))
        w_str = f"{w[0]}/{w[1]}/{w[2]}"
        print(f"  {i+1:<5} {r['map']:>7.4f} {r['recall_1000']:>7.4f} {r['p10']:>6.4f} | "
              f"{r['k1']:<5} {r['b']:<5} {r['fb_docs']:<5} {r['fb_terms']:<5} {r['orig_weight']:<5} {r['rrf_k']:<6} {w_str}")
    
    best_w = best_config.get('ret_weights', (1,1,1))
    print("\n  " + "=" * 85)
    print(f"  🏆 BEST CONFIG: MAP={best_config['map']:.4f}, R@1K={best_config['recall_1000']:.4f}")
    print(f"     Weights: BM25={best_w[0]}, Dense={best_w[1]}, SPLADE={best_w[2]}")
    print("  " + "=" * 85)
    print(f"""
  Recommended command:
  
  python -m src.main train \\
      --bm25-k1 {best_config['k1']} \\
      --bm25-b {best_config['b']} \\
      --fb-docs {best_config['fb_docs']} \\
      --fb-terms {best_config['fb_terms']} \\
      --original-weight {best_config['orig_weight']} \\
      --rrf-k {best_config['rrf_k']} \\
      ...
  
  Note: Retrieval weights (BM25={best_w[0]}, Dense={best_w[1]}, SPLADE={best_w[2]}) 
  need to be applied in main.py (currently not exposed as CLI args).
""")
    
    # Save results (convert tuples to lists for JSON)
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(x) for x in obj]
        return obj
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(make_json_safe({
            'split': args.split,
            'num_queries': len(queries_orig),
            'best_config': best_config,
            'all_results': results[:100],  # Top 100
            'param_grid': param_grid,
        }), f, indent=2)
    print(f"  ✓ Saved to {output_path}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
