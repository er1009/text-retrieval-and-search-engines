"""
Fast weight-only tuning script.
Fixes BM25/RM3 parameters and only tunes retrieval weights.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bm25_retrieval import batch_search_bm25_rm3
from .data_loader import load_queries, load_qrels, load_expanded_queries, get_train_qids, get_test_qids
from .dense_index import DensePassageIndex
from .fusion import fuse_runs


def calc_recall(retrieved: set, qrels_dict: dict) -> float:
    relevant = set(d for d, r in qrels_dict.items() if r > 0)
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def calc_map(retrieved_ranked: list, qrels_dict: dict) -> float:
    relevant = set(d for d, r in qrels_dict.items() if r > 0)
    if not relevant:
        return 0.0
    
    precisions = []
    num_rel_found = 0
    for i, docid in enumerate(retrieved_ranked, 1):
        if docid in relevant:
            num_rel_found += 1
            precisions.append(num_rel_found / i)
    
    return sum(precisions) / len(relevant) if precisions else 0.0


def evaluate_fused(fused_results: dict, qrels: dict) -> dict:
    """Evaluate fused results with manual calculations."""
    maps, recalls = [], []
    
    for qid, doc_scores in fused_results.items():
        if qid not in qrels:
            continue
        
        # Sort by score for MAP calculation
        ranked_docs = [d for d, _ in sorted(doc_scores.items(), key=lambda x: -x[1])]
        retrieved_set = set(ranked_docs[:1000])
        
        maps.append(calc_map(ranked_docs[:1000], qrels[qid]))
        recalls.append(calc_recall(retrieved_set, qrels[qid]))
    
    return {
        'map': np.mean(maps),
        'recall_1000': np.mean(recalls),
    }


def main():
    parser = argparse.ArgumentParser(description="Fast weight-only tuning")
    parser.add_argument("--dense-index-path", required=True, help="Path to dense index")
    parser.add_argument("--splade-index-path", default=None, help="Path to SPLADE index")
    parser.add_argument("--output", default="weight_tuning.json", help="Output file")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--retrieval-k", type=int, default=2000)
    
    # Fixed BM25/RM3 parameters (from first tuning)
    parser.add_argument("--bm25-k1", type=float, default=0.9)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--fb-docs", type=int, default=10)
    parser.add_argument("--fb-terms", type=int, default=20)
    parser.add_argument("--original-weight", type=float, default=0.5)
    parser.add_argument("--rrf-k", type=int, default=60)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"WEIGHT-ONLY TUNING ({args.split.upper()})")
    print("=" * 70)
    print(f"\n  Fixed BM25: k1={args.bm25_k1}, b={args.bm25_b}")
    print(f"  Fixed RM3: fb_docs={args.fb_docs}, fb_terms={args.fb_terms}, weight={args.original_weight}")
    print(f"  Fixed RRF k: {args.rrf_k}")
    
    # Load data
    print("\n[1/4] Loading data...")
    all_queries = load_queries()
    all_expanded = load_expanded_queries()
    all_qrels = load_qrels()
    
    if args.split == "train":
        qids = get_train_qids()
    else:
        qids = get_test_qids()
    
    queries_orig = {qid: all_queries[qid] for qid in qids if qid in all_queries}
    queries_exp = {qid: all_expanded.get(qid, all_queries.get(qid, "")) for qid in qids}
    qrels = {qid: all_qrels[qid] for qid in qids if qid in all_qrels}
    
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
    
    # Run retrieval ONCE (fixed parameters)
    print("\n[3/4] Running retrieval (one time)...")
    
    print("  BM25+RM3 (original)...")
    bm25_orig = batch_search_bm25_rm3(
        queries_orig, k=args.retrieval_k,
        k1=args.bm25_k1, b=args.bm25_b,
        fb_docs=args.fb_docs, fb_terms=args.fb_terms,
        original_query_weight=args.original_weight,
    )
    bm25_orig_dict = {qid: {r.docid: r.score for r in docs} for qid, docs in bm25_orig.items()}
    
    print("  BM25+RM3 (Q2D)...")
    bm25_exp = batch_search_bm25_rm3(
        queries_exp, k=args.retrieval_k,
        k1=args.bm25_k1, b=args.bm25_b,
        fb_docs=args.fb_docs, fb_terms=args.fb_terms,
        original_query_weight=args.original_weight,
    )
    bm25_exp_dict = {qid: {r.docid: r.score for r in docs} for qid, docs in bm25_exp.items()}
    
    print("  Dense (original)...")
    dense_orig = dense_index.batch_search(queries_orig, top_k=args.retrieval_k, show_progress=True)
    
    print("  Dense (Q2D)...")
    dense_exp = dense_index.batch_search(queries_exp, top_k=args.retrieval_k, show_progress=True)
    
    if splade_index:
        print("  SPLADE (original)...")
        splade_orig = splade_index.batch_search(queries_orig, top_k=args.retrieval_k, show_progress=True)
        
        print("  SPLADE (Q2D)...")
        splade_exp = splade_index.batch_search(queries_exp, top_k=args.retrieval_k, show_progress=True)
    
    # Define weight grid
    # Weights: (BM25, Dense, SPLADE)
    weight_patterns = [
        # Baseline
        (1.0, 1.0, 1.0),
        
        # Lower Dense (it's hurting)
        (1.0, 0.5, 1.0),
        (1.0, 0.3, 1.0),
        (1.0, 0.0, 1.0),  # No Dense
        
        # Higher SPLADE
        (1.0, 1.0, 1.5),
        (1.0, 0.5, 1.5),
        (1.0, 0.3, 1.5),
        (1.0, 0.0, 1.5),  # No Dense, Higher SPLADE
        
        # Much Higher SPLADE
        (1.0, 0.5, 2.0),
        (1.0, 0.3, 2.0),
        (1.0, 0.0, 2.0),
        
        # Lower BM25
        (0.5, 0.3, 1.5),
        (0.5, 0.0, 2.0),
        
        # Extremes
        (1.0, 0.0, 3.0),  # SPLADE dominant
        (0.0, 0.0, 1.0),  # SPLADE only
        (1.0, 0.0, 0.0),  # BM25 only
    ]
    
    print(f"\n[4/4] Testing {len(weight_patterns)} weight configurations...")
    
    results = []
    best_map = 0
    best_config = None
    
    for i, (w_bm25, w_dense, w_splade) in enumerate(weight_patterns):
        # Build weights list for 6-way fusion
        # Order: bm25_orig, bm25_exp, dense_orig, dense_exp, splade_orig, splade_exp
        weights = [w_bm25, w_bm25, w_dense, w_dense]
        if splade_index:
            weights.extend([w_splade, w_splade])
        
        # Fuse
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
            for j, r in enumerate(runs):
                if r:
                    non_empty_runs.append(r)
                    non_empty_weights.append(weights[j])
            
            if non_empty_runs:
                fused[qid] = fuse_runs(
                    non_empty_runs,
                    method="rrf",
                    rrf_k=args.rrf_k,
                    weights=non_empty_weights,
                )
        
        # Evaluate
        metrics = evaluate_fused(fused, qrels)
        
        config = {
            'w_bm25': w_bm25,
            'w_dense': w_dense,
            'w_splade': w_splade,
            **metrics,
        }
        results.append(config)
        
        if metrics['map'] > best_map:
            best_map = metrics['map']
            best_config = config
            print(f"  [{i+1}/{len(weight_patterns)}] NEW BEST: MAP={metrics['map']:.4f} R@1K={metrics['recall_1000']:.4f} "
                  f"(BM25={w_bm25}, Dense={w_dense}, SPLADE={w_splade})")
    
    # Sort by MAP
    results = sorted(results, key=lambda x: -x['map'])
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n  TOP 10 WEIGHT CONFIGURATIONS:")
    print("  " + "-" * 55)
    print(f"  {'Rank':<5} {'MAP':>8} {'R@1K':>8} | {'BM25':>6} {'Dense':>6} {'SPLADE':>6}")
    print("  " + "-" * 55)
    
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:<5} {r['map']:>8.4f} {r['recall_1000']:>8.4f} | "
              f"{r['w_bm25']:>6.1f} {r['w_dense']:>6.1f} {r['w_splade']:>6.1f}")
    
    print("\n  " + "=" * 55)
    print(f"  🏆 BEST: MAP={best_config['map']:.4f}, R@1K={best_config['recall_1000']:.4f}")
    print(f"     Weights: BM25={best_config['w_bm25']}, Dense={best_config['w_dense']}, SPLADE={best_config['w_splade']}")
    print("  " + "=" * 55)
    
    # Save
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'split': args.split,
            'fixed_params': {
                'k1': args.bm25_k1, 'b': args.bm25_b,
                'fb_docs': args.fb_docs, 'fb_terms': args.fb_terms,
                'original_weight': args.original_weight, 'rrf_k': args.rrf_k,
            },
            'best_config': best_config,
            'all_results': results,
        }, f, indent=2)
    print(f"\n  ✓ Saved to {output_path}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
