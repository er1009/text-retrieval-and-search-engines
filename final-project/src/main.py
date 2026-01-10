"""Main CLI for ROBUST04 ranking competition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from .config import Config, DEFAULT_CONFIG, set_all_seeds
from .data_loader import (
    load_queries, load_expanded_queries, load_qrels,
    get_train_qids, get_test_qids, get_all_qids
)
from .bm25_retrieval import (
    batch_search_bm25, batch_search_bm25_rm3, get_searcher
)
from .document_processor import get_all_passages_for_query
from .neural_reranker import FastCrossEncoder, DualReranker, ThreeStageReranker, normalize_scores
from .aggregation import aggregate_passage_scores, rank_documents_by_score
from .fusion import fuse_runs
from .evaluation import (
    evaluate_map, comprehensive_evaluation, print_evaluation_report
)
from .tuning import run_full_tuning, load_tuned_config
from .trec_io import write_trec_run, write_trec_run_from_scores, read_trec_run_as_scores


def cmd_tune(args):
    """Run parameter tuning."""
    print("=" * 60)
    print("ROBUST04 PARAMETER TUNING")
    print("=" * 60)
    
    set_all_seeds(42)
    
    output_path = Path(args.output) / "best_config.json"
    
    best_config = run_full_tuning(
        output_path=output_path,
        use_cv=not args.no_cv,
    )
    
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(json.dumps(best_config, indent=2))


def cmd_run1(args):
    """
    Run 1: BM25 + RM3 + Query2Doc (lexical baseline with PRF and semantic expansion)
    """
    print("=" * 60)
    print("RUN 1: BM25 + RM3 + Claude Query2Doc")
    print("=" * 60)
    
    set_all_seeds(42)
    
    # Load data
    print("Loading queries and expansions...")
    queries = load_queries()
    expanded_queries = load_expanded_queries()
    
    # Load tuned params if available
    config_path = Path(args.config) if args.config else None
    if config_path and config_path.exists():
        print(f"Loading tuned config from {config_path}")
        config = load_tuned_config(config_path)
        k1 = config['bm25']['k1']
        b = config['bm25']['b']
        fb_docs = config['rm3']['fb_docs']
        fb_terms = config['rm3']['fb_terms']
        orig_weight = config['rm3']['original_query_weight']
    else:
        # Default params
        k1, b = 0.9, 0.4
        fb_docs, fb_terms, orig_weight = 10, 10, 0.5
    
    print(f"BM25 params: k1={k1}, b={b}")
    print(f"RM3 params: fb_docs={fb_docs}, fb_terms={fb_terms}, weight={orig_weight}")
    
    # BEST PRACTICE: Use BOTH original and expanded queries, then fuse!
    all_qids = get_train_qids() + get_test_qids()
    all_original = {qid: queries[qid] for qid in all_qids if qid in queries}
    all_expanded = {}
    for qid in all_qids:
        if qid in expanded_queries:
            all_expanded[qid] = expanded_queries[qid]
        elif qid in queries:
            all_expanded[qid] = queries[qid]  # Fallback to original
    
    print(f"Running BM25+RM3 with ORIGINAL queries ({len(all_original)} queries)...")
    results_orig = batch_search_bm25_rm3(
        queries=all_original,
        k=1000,
        k1=k1, b=b,
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        original_query_weight=orig_weight,
    )
    
    print(f"Running BM25+RM3 with EXPANDED queries ({len(all_expanded)} queries)...")
    results_exp = batch_search_bm25_rm3(
        queries=all_expanded,
        k=1000,
        k1=k1, b=b,
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        original_query_weight=orig_weight,
    )
    
    # Fuse original and expanded results using RRF
    print("Fusing original and expanded query results...")
    from .trec_io import read_trec_run_as_scores, write_trec_run_from_scores
    from .fusion import fuse_runs
    
    # Convert to score dicts for fusion
    orig_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_orig.items()}
    exp_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_exp.items()}
    
    # Fuse per query
    results = {}
    for qid in all_qids:
        runs_to_fuse = []
        if qid in orig_scores:
            runs_to_fuse.append(orig_scores[qid])
        if qid in exp_scores:
            runs_to_fuse.append(exp_scores[qid])
        
        if runs_to_fuse:
            results[qid] = fuse_runs(runs_to_fuse, method="rrf", rrf_k=60)
    
    # Convert back to SearchResult format for write_trec_run
    from .bm25_retrieval import SearchResult
    results_formatted = {}
    for qid, doc_scores in results.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:1000]
        results_formatted[qid] = [
            SearchResult(docid=docid, rank=i+1, score=score)
            for i, (docid, score) in enumerate(sorted_docs)
        ]
    results = results_formatted
    
    # Write FULL output (for evaluation on train queries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run(results, output_path, run_name="BM25_RM3_Q2D")
    print(f"Saved full run to {output_path}")
    
    # Also write SUBMISSION version (only test queries, as required by competition)
    test_qids_set = set(get_test_qids())
    submission_results = {qid: docs for qid, docs in results.items() if qid in test_qids_set}
    submission_path = output_path.parent / f"{output_path.stem}_submission{output_path.suffix}"
    write_trec_run(submission_results, submission_path, run_name="BM25_RM3_Q2D")
    print(f"Saved submission run (199 test queries only) to {submission_path}")


def cmd_run2(args):
    """
    Run 2: Neural MaxP Reranking (cross-encoder on passages)
    """
    print("=" * 60)
    print("RUN 2: Neural MaxP Cross-Encoder Reranking")
    print("=" * 60)
    
    set_all_seeds(42)
    
    # Load data
    print("Loading queries and expansions...")
    queries = load_queries()
    expanded_queries = load_expanded_queries()
    
    # Load tuned BM25 params
    config_path = Path(args.config) if args.config else None
    if config_path and config_path.exists():
        config = load_tuned_config(config_path)
        k1 = config['bm25']['k1']
        b = config['bm25']['b']
    else:
        k1, b = 0.9, 0.4
    
    # Get parameters from command line
    device = "cuda" if args.gpu else "cpu"
    chunk_size = getattr(args, 'chunk_size', 256)
    chunk_overlap = getattr(args, 'chunk_overlap', 64)
    bi_encoder_top_k = getattr(args, 'bi_encoder_top_k', 4000)
    bi_batch_size = getattr(args, 'bi_batch_size', 2048)
    ce_batch_size = getattr(args, 'ce_batch_size', 2048)
    monot5_batch_size = getattr(args, 'monot5_batch_size', 512)
    ce_weight = getattr(args, 'ce_weight', 0.5)
    neural_weight = getattr(args, 'neural_weight', 0.8)
    
    print(f"Rerank depth: {args.rerank_depth} docs")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    print(f"Bi-encoder top_k: {bi_encoder_top_k}")
    
    reranker = ThreeStageReranker(
        bi_encoder_model="BAAI/bge-base-en-v1.5",
        ce_model=None,
        monot5_model="castorini/monot5-base-msmarco",
        ce_weight=ce_weight,
        bi_encoder_top_k=bi_encoder_top_k,
        use_ensemble_ce=True,
        device=device,
        use_bf16=True,
    )
    
    # Get ALL queries (train + test) for evaluation capability
    all_qids = get_train_qids() + get_test_qids()
    all_queries = {qid: queries[qid] for qid in all_qids if qid in queries}
    all_expanded = {}
    for qid in all_qids:
        if qid in expanded_queries:
            all_expanded[qid] = expanded_queries[qid]
        elif qid in queries:
            all_expanded[qid] = queries[qid]
    
    print(f"Processing {len(all_expanded)} queries (50 train + 199 test)...")
    
    # ALWAYS retrieve 1000 docs for recall, rerank top N for precision
    print("Running batch BM25 retrieval...")
    all_bm25_results = batch_search_bm25(
        all_expanded,
        k=1000,
        k1=k1, b=b,
    )
    
    # Build BM25 score lookup for interpolation
    bm25_scores_all = {}
    for qid, results in all_bm25_results.items():
        max_score = results[0].score if results else 1.0
        min_score = results[-1].score if results else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0
        bm25_scores_all[qid] = {
            r.docid: (r.score - min_score) / score_range 
            for r in results
        }
    
    all_results = {}
    
    for qid in tqdm(all_qids, desc="Neural reranking"):
        if qid not in all_bm25_results or not all_bm25_results[qid]:
            continue
        
        bm25_results = all_bm25_results[qid]
        bm25_norm = bm25_scores_all[qid]
        
        passages, doc_to_indices = get_all_passages_for_query(
            bm25_results,
            top_k=args.rerank_depth,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        if not passages:
            all_results[qid] = {r.docid: r.score for r in bm25_results[:1000]}
            continue
        
        original_query = all_queries.get(qid, all_expanded.get(qid, ""))
        
        passage_scores, _ = reranker.rerank_passages(
            original_query,
            passages,
            bi_batch_size=bi_batch_size,
            ce_batch_size=ce_batch_size,
            monot5_batch_size=monot5_batch_size,
        )
        
        doc_scores = aggregate_passage_scores(
            passages,
            passage_scores,
            doc_to_indices,
            strategy="maxp",
        )
        
        # Normalize neural scores to [0, 1]
        if doc_scores:
            neural_max = max(doc_scores.values())
            neural_min = min(doc_scores.values())
            neural_range = neural_max - neural_min if neural_max != neural_min else 1.0
            doc_scores = {
                docid: (score - neural_min) / neural_range 
                for docid, score in doc_scores.items()
            }
        
        # Interpolate neural + BM25 for reranked docs
        # Place unreranked docs below reranked but preserve BM25 ordering
        final_scores = {}
        
        # First pass: compute reranked scores
        for r in bm25_results:
            if r.docid in doc_scores:
                final_scores[r.docid] = (
                    neural_weight * doc_scores[r.docid] + 
                    (1 - neural_weight) * bm25_norm[r.docid]
                )
        
        # Find minimum reranked score to anchor unreranked docs below
        min_reranked = min(final_scores.values()) if final_scores else 0.5
        # Ensure ceiling is positive and creates a gap
        unreranked_ceiling = max(0.01, min_reranked * 0.9)
        
        # Second pass: place unreranked docs below reranked
        for r in bm25_results:
            if r.docid not in doc_scores:
                # Scale BM25 to [0, unreranked_ceiling] to preserve ordering
                final_scores[r.docid] = unreranked_ceiling * bm25_norm.get(r.docid, 0)
        
        all_results[qid] = final_scores
    
    # Write FULL output (for evaluation on train queries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run_from_scores(all_results, output_path, run_name="Neural_MaxP")
    print(f"Saved full run to {output_path}")
    
    # Also write SUBMISSION version (only test queries, as required by competition)
    test_qids_set = set(get_test_qids())
    submission_results = {qid: docs for qid, docs in all_results.items() if qid in test_qids_set}
    submission_path = output_path.parent / f"{output_path.stem}_submission{output_path.suffix}"
    write_trec_run_from_scores(submission_results, submission_path, run_name="Neural_MaxP")
    print(f"Saved submission run (199 test queries only) to {submission_path}")


def cmd_run3(args):
    """
    Run 3: Optimal Multi-Signal Fusion
    """
    print("=" * 60)
    print("RUN 3: Optimal Multi-Signal Fusion")
    print("=" * 60)
    
    set_all_seeds(42)
    
    # Load the other two runs
    run1_path = Path(args.run1)
    run2_path = Path(args.run2)
    
    if not run1_path.exists():
        print(f"Error: run_1 not found at {run1_path}")
        sys.exit(1)
    if not run2_path.exists():
        print(f"Error: run_2 not found at {run2_path}")
        sys.exit(1)
    
    print("Loading run_1 and run_2...")
    run1 = read_trec_run_as_scores(run1_path)
    run2 = read_trec_run_as_scores(run2_path)
    
    # Get ALL query IDs (train + test)
    all_qids = get_train_qids() + get_test_qids()
    
    print(f"Fusing {len(all_qids)} queries (50 train + 199 test) with RRF...")
    
    # Fuse per query
    fused_results = {}
    for qid in tqdm(all_qids, desc="Fusing"):
        runs_to_fuse = []
        if qid in run1:
            runs_to_fuse.append(run1[qid])
        if qid in run2:
            runs_to_fuse.append(run2[qid])
        
        if runs_to_fuse:
            fused_results[qid] = fuse_runs(
                runs_to_fuse,
                method="rrf",
                rrf_k=60,
            )
    
    # Write FULL output (for evaluation on train queries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run_from_scores(fused_results, output_path, run_name="RRF_Fusion")
    print(f"Saved full run to {output_path}")
    
    # Also write SUBMISSION version (only test queries, as required by competition)
    test_qids_set = set(get_test_qids())
    submission_results = {qid: docs for qid, docs in fused_results.items() if qid in test_qids_set}
    submission_path = output_path.parent / f"{output_path.stem}_submission{output_path.suffix}"
    write_trec_run_from_scores(submission_results, submission_path, run_name="RRF_Fusion")
    print(f"Saved submission run (199 test queries only) to {submission_path}")


def cmd_run_all(args):
    """Run all three methods."""
    print("=" * 60)
    print("RUNNING ALL THREE METHODS")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run 1
    run1_args = argparse.Namespace(
        config=args.config,
        output=str(output_dir / "run_1.res"),
    )
    cmd_run1(run1_args)
    
    # Run 2
    run2_args = argparse.Namespace(
        config=args.config,
        output=str(output_dir / "run_2.res"),
        rerank_depth=args.rerank_depth,
        gpu=args.gpu,
    )
    cmd_run2(run2_args)
    
    # Run 3
    run3_args = argparse.Namespace(
        run1=str(output_dir / "run_1.res"),
        run2=str(output_dir / "run_2.res"),
        output=str(output_dir / "run_3.res"),
    )
    cmd_run3(run3_args)
    
    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETE")
    print("=" * 60)
    print(f"Results saved to {output_dir}")


def cmd_evaluate(args):
    """Evaluate runs on training queries."""
    print("=" * 60)
    print("EVALUATION ON TRAINING QUERIES")
    print("=" * 60)
    
    qrels = load_qrels()
    train_qids = get_train_qids()
    train_qrels = {qid: qrels[qid] for qid in train_qids if qid in qrels}
    
    for run_path in args.runs:
        path = Path(run_path)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        
        print(f"\n--- {path.name} ---")
        run = read_trec_run_as_scores(path)
        
        # Filter to train queries only
        train_run = {qid: docs for qid, docs in run.items() if qid in train_qrels}
        
        if not train_run:
            print("No training queries found in run")
            continue
        
        metrics = comprehensive_evaluation(train_run, train_qrels)
        print_evaluation_report(metrics)


def cmd_quick_train(args):
    """
    Run all methods on TRAINING queries only for fast iteration.
    Runs all 3 methods + evaluates in one command.
    """
    import time
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("  🚀 QUICK TRAIN: Fast Iteration Mode")
    print("=" * 70)
    
    set_all_seeds(42)
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== DATA LOADING ====================
    print("\n📂 LOADING DATA")
    print("-" * 50)
    
    queries = load_queries()
    expanded_queries = load_expanded_queries()
    qrels = load_qrels()
    train_qids = get_train_qids()
    
    train_queries = {qid: queries[qid] for qid in train_qids if qid in queries}
    train_expanded = {}
    for qid in train_qids:
        if qid in expanded_queries:
            train_expanded[qid] = expanded_queries[qid]
        elif qid in queries:
            train_expanded[qid] = queries[qid]
    
    train_qrels = {qid: qrels[qid] for qid in train_qids if qid in qrels}
    
    # Count total relevant docs
    total_relevant = sum(sum(1 for rel in rels.values() if rel > 0) for rels in train_qrels.values())
    
    print(f"  Queries:          {len(train_queries)} (training only: 301-350)")
    print(f"  Expanded queries: {len(train_expanded)}")
    print(f"  Qrels:            {len(train_qrels)} queries, {total_relevant} relevant docs total")
    
    # Load tuned params
    config_path = Path(args.config) if args.config else None
    if config_path and config_path.exists():
        config = load_tuned_config(config_path)
        k1 = config['bm25']['k1']
        b = config['bm25']['b']
        fb_docs = config['rm3']['fb_docs']
        fb_terms = config['rm3']['fb_terms']
        orig_weight = config['rm3']['original_query_weight']
        print(f"  Config:           {config_path.name} ✓")
    else:
        k1, b = 0.9, 0.4
        fb_docs, fb_terms, orig_weight = 10, 10, 0.5
        print(f"  Config:           defaults (no tuned config)")
    
    print(f"\n  BM25:  k1={k1}, b={b}")
    print(f"  RM3:   fb_docs={fb_docs}, fb_terms={fb_terms}, weight={orig_weight}")
    
    # ==================== RUN 1: LEXICAL ====================
    run1_start = time.time()
    # Extract RRF parameter early (needed for Run 1)
    rrf_k = getattr(args, 'rrf_k', 60)
    
    print("\n" + "=" * 70)
    print("  📊 RUN 1: BM25 + RM3 + Query2Doc Fusion")
    print("=" * 70)
    
    print("\n  [1/3] BM25+RM3 with ORIGINAL queries...")
    results_orig = batch_search_bm25_rm3(
        queries=train_queries, k=1000, k1=k1, b=b,
        fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
    )
    orig_docs = sum(len(docs) for docs in results_orig.values())
    print(f"        → Retrieved {orig_docs:,} docs ({orig_docs/len(train_queries):.0f}/query avg)")
    
    print("\n  [2/3] BM25+RM3 with EXPANDED queries (Query2Doc)...")
    results_exp = batch_search_bm25_rm3(
        queries=train_expanded, k=1000, k1=k1, b=b,
        fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
    )
    exp_docs = sum(len(docs) for docs in results_exp.values())
    print(f"        → Retrieved {exp_docs:,} docs ({exp_docs/len(train_expanded):.0f}/query avg)")
    
    print(f"\n  [3/3] Fusing with RRF (k={rrf_k})...")
    orig_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_orig.items()}
    exp_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_exp.items()}
    
    run1_results = {}
    for qid in train_qids:
        runs_to_fuse = []
        if qid in orig_scores:
            runs_to_fuse.append(orig_scores[qid])
        if qid in exp_scores:
            runs_to_fuse.append(exp_scores[qid])
        if runs_to_fuse:
            run1_results[qid] = fuse_runs(runs_to_fuse, method="rrf", rrf_k=rrf_k)
    
    fused_docs = sum(len(docs) for docs in run1_results.values())
    print(f"        → Fused: {fused_docs:,} docs ({fused_docs/len(run1_results):.0f}/query avg)")
    
    write_trec_run_from_scores(run1_results, output_dir / "run_1_train.res", run_name="BM25_RM3_Q2D")
    run1_time = time.time() - run1_start
    print(f"\n  ✓ Run 1 complete: {len(run1_results)} queries in {run1_time:.1f}s")
    
    # ==================== RUN 2: NEURAL ====================
    run2_start = time.time()
    print("\n" + "=" * 70)
    print("  🧠 RUN 2: Three-Stage Neural Reranking")
    print("=" * 70)
    
    device = "cuda" if args.gpu else "cpu"
    
    # Calculate pipeline parameters
    # Extract all parameters from args
    bm25_k = getattr(args, 'bm25_k', 1000)
    chunk_size = getattr(args, 'chunk_size', 256)
    chunk_overlap = getattr(args, 'chunk_overlap', 64)
    bi_batch_size = getattr(args, 'bi_batch_size', 2048)
    ce_batch_size = getattr(args, 'ce_batch_size', 2048)
    monot5_batch_size = getattr(args, 'monot5_batch_size', 512)
    ce_weight = getattr(args, 'ce_weight', 0.5)
    neural_weight = getattr(args, 'neural_weight', 0.8)
    # rrf_k already extracted before Run 1
    
    # Calculate bi_encoder_top_k (auto-scale or use provided)
    # ROBUST04 docs average ~50 passages each (256 chars, 64 overlap)
    estimated_passages = args.rerank_depth * 50
    if getattr(args, 'bi_encoder_top_k', None) is not None:
        bi_encoder_top_k = args.bi_encoder_top_k
    else:
        # Keep 40% of passages for high recall (critical for MAP)
        bi_encoder_top_k = max(10000, int(estimated_passages * 0.4))
    
    # Handle --no-gpu flag
    if getattr(args, 'no_gpu', False):
        device = "cpu"
    
    print(f"\n  Pipeline Configuration:")
    print(f"  ┌────────────────────────────────────────────────────────────────┐")
    print(f"  │ RETRIEVAL                                                      │")
    print(f"  │   BM25 k:            {bm25_k:<6} docs/query                       │")
    print(f"  │   Rerank depth:      {args.rerank_depth:<6} docs → ~{estimated_passages:<5} passages         │")
    print(f"  │                                                                │")
    print(f"  │ CHUNKING                                                       │")
    print(f"  │   Chunk size:        {chunk_size:<6} chars                            │")
    print(f"  │   Chunk overlap:     {chunk_overlap:<6} chars                            │")
    print(f"  │                                                                │")
    print(f"  │ NEURAL PIPELINE                                                │")
    print(f"  │   Bi-encoder top_k:  {bi_encoder_top_k:<6} passages                        │")
    print(f"  │   Bi-encoder batch:  {bi_batch_size:<6}                                  │")
    print(f"  │   Cross-enc batch:   {ce_batch_size:<6}                                  │")
    print(f"  │   MonoT5 batch:      {monot5_batch_size:<6}                                  │")
    print(f"  │                                                                │")
    print(f"  │ ENSEMBLE WEIGHTS                                               │")
    print(f"  │   CE/MonoT5:         {ce_weight:.2f} / {1-ce_weight:.2f}                              │")
    print(f"  │   Neural/BM25:       {neural_weight:.2f} / {1-neural_weight:.2f}                              │")
    print(f"  │   RRF k:             {rrf_k:<6}                                  │")
    print(f"  │                                                                │")
    print(f"  │ DEVICE:              {device.upper():<6}                                  │")
    print(f"  └────────────────────────────────────────────────────────────────┘")
    
    print("\n  Loading models...")
    reranker = ThreeStageReranker(
        bi_encoder_model="BAAI/bge-base-en-v1.5",
        ce_model=None,
        monot5_model="castorini/monot5-base-msmarco",
        ce_weight=ce_weight,
        bi_encoder_top_k=bi_encoder_top_k,
        use_ensemble_ce=True,
        device=device,
        use_bf16=True,
    )
    
    print(f"\n  Running BM25 retrieval ({bm25_k} docs/query)...")
    all_bm25_results = batch_search_bm25(
        train_expanded, k=bm25_k, k1=k1, b=b,
    )
    total_bm25_docs = sum(len(docs) for docs in all_bm25_results.values())
    print(f"  → Retrieved {total_bm25_docs:,} total docs")
    
    # Build BM25 score lookup for interpolation
    bm25_scores_all = {}
    for qid, results in all_bm25_results.items():
        max_score = results[0].score if results else 1.0
        min_score = results[-1].score if results else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0
        bm25_scores_all[qid] = {
            r.docid: (r.score - min_score) / score_range 
            for r in results
        }
    
    print(f"\n  Neural reranking {len(train_qids)} queries...")
    print(f"  (chunking top {args.rerank_depth} docs, {chunk_size} chars/chunk, {chunk_overlap} overlap)\n")
    
    run2_results = {}
    total_passages = 0
    total_reranked_docs = 0
    
    for qid in tqdm(train_qids, desc="  Processing"):
        if qid not in all_bm25_results or not all_bm25_results[qid]:
            continue
        
        bm25_results = all_bm25_results[qid]
        bm25_norm = bm25_scores_all[qid]
        
        passages, doc_to_indices = get_all_passages_for_query(
            bm25_results, top_k=args.rerank_depth, 
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        
        if not passages:
            run2_results[qid] = {r.docid: r.score for r in bm25_results[:bm25_k]}
            continue
        
        total_passages += len(passages)
        total_reranked_docs += len(doc_to_indices)
        
        original_query = train_queries.get(qid, train_expanded.get(qid, ""))
        passage_scores, _ = reranker.rerank_passages(
            original_query, passages,
            bi_batch_size=bi_batch_size, 
            ce_batch_size=ce_batch_size, 
            monot5_batch_size=monot5_batch_size,
        )
        
        doc_scores = aggregate_passage_scores(passages, passage_scores, doc_to_indices, strategy="maxp")
        
        # Normalize neural scores to [0, 1]
        if doc_scores:
            neural_max = max(doc_scores.values())
            neural_min = min(doc_scores.values())
            neural_range = neural_max - neural_min if neural_max != neural_min else 1.0
            doc_scores = {
                docid: (score - neural_min) / neural_range 
                for docid, score in doc_scores.items()
            }
        
        # Best practice: Interpolate neural + BM25 for reranked docs
        # For unreranked docs: place them below reranked but preserve BM25 ordering
        final_scores = {}
        
        # First pass: compute reranked scores
        for r in bm25_results:
            if r.docid in doc_scores:
                final_scores[r.docid] = (
                    neural_weight * doc_scores[r.docid] + 
                    (1 - neural_weight) * bm25_norm[r.docid]
                )
        
        # Find minimum reranked score to anchor unreranked docs below
        min_reranked = min(final_scores.values()) if final_scores else 0.5
        # Ensure ceiling is positive and creates a gap
        unreranked_ceiling = max(0.01, min_reranked * 0.9)
        
        # Second pass: assign unreranked docs below the minimum
        for r in bm25_results:
            if r.docid not in doc_scores:
                # Scale BM25 to [0, unreranked_ceiling] to preserve ordering
                final_scores[r.docid] = unreranked_ceiling * bm25_norm.get(r.docid, 0)
        
        run2_results[qid] = final_scores
    
    # Summary statistics
    avg_passages = total_passages / len(train_qids) if train_qids else 0
    avg_reranked = total_reranked_docs / len(train_qids) if train_qids else 0
    
    print(f"\n  Summary:")
    print(f"  ├── Total passages processed:  {total_passages:,} ({avg_passages:.0f}/query)")
    print(f"  ├── Docs neurally reranked:    {total_reranked_docs:,} ({avg_reranked:.0f}/query)")
    print(f"  ├── Docs with BM25 fallback:   {len(train_qids) * bm25_k - total_reranked_docs:,}")
    print(f"  └── Score interpolation:       {neural_weight*100:.0f}% neural + {(1-neural_weight)*100:.0f}% BM25")
    
    write_trec_run_from_scores(run2_results, output_dir / "run_2_train.res", run_name="Neural_MaxP")
    run2_time = time.time() - run2_start
    print(f"\n  ✓ Run 2 complete: {len(run2_results)} queries in {run2_time:.1f}s")
    
    # ==================== RUN 3: FUSION ====================
    run3_start = time.time()
    print("\n" + "=" * 70)
    print("  🔀 RUN 3: Multi-Signal RRF Fusion")
    print("=" * 70)
    
    print(f"\n  Fusing Run 1 (lexical) + Run 2 (neural) with RRF (k={rrf_k})...")
    
    run3_results = {}
    for qid in train_qids:
        runs_to_fuse = []
        if qid in run1_results:
            runs_to_fuse.append(run1_results[qid])
        if qid in run2_results:
            runs_to_fuse.append(run2_results[qid])
        if runs_to_fuse:
            run3_results[qid] = fuse_runs(runs_to_fuse, method="rrf", rrf_k=rrf_k)
    
    total_fused = sum(len(docs) for docs in run3_results.values())
    print(f"  → Fused: {total_fused:,} docs ({total_fused/len(run3_results):.0f}/query)")
    
    write_trec_run_from_scores(run3_results, output_dir / "run_3_train.res", run_name="RRF_Fusion")
    run3_time = time.time() - run3_start
    print(f"\n  ✓ Run 3 complete: {len(run3_results)} queries in {run3_time:.1f}s")
    
    # ==================== EVALUATION ====================
    print("\n" + "=" * 70)
    print("  📈 EVALUATION RESULTS (Training Queries)")
    print("=" * 70)
    
    results_summary = []
    for name, run in [("Run 1 (Lexical)", run1_results), 
                      ("Run 2 (Neural)", run2_results), 
                      ("Run 3 (Fusion)", run3_results)]:
        metrics = comprehensive_evaluation(run, train_qrels)
        results_summary.append((name, metrics))
    
    # Print comparison table
    print(f"\n  {'Metric':<12} {'Run 1':>10} {'Run 2':>10} {'Run 3':>10}  {'Best':>8}")
    print("  " + "-" * 56)
    
    metric_names = ['map', 'ndcg', 'ndcg_10', 'p_10', 'p_20', 'p_100', 'recall_100', 'recall_1000']
    display_names = ['MAP', 'NDCG', 'NDCG@10', 'P@10', 'P@20', 'P@100', 'R@100', 'R@1000']
    
    for metric, display in zip(metric_names, display_names):
        values = [r[1].get(metric, 0) for r in results_summary]
        best_idx = values.index(max(values))
        markers = ['', '', '']
        markers[best_idx] = '← BEST'
        print(f"  {display:<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}  {markers[best_idx]}")
    
    # ==================== SUMMARY ====================
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("  ✅ QUICK TRAIN COMPLETE")
    print("=" * 70)
    print(f"\n  Output files:")
    print(f"    • {output_dir}/run_1_train.res (lexical)")
    print(f"    • {output_dir}/run_2_train.res (neural)")
    print(f"    • {output_dir}/run_3_train.res (fusion)")
    print(f"\n  Timing:")
    print(f"    • Run 1: {run1_time:>6.1f}s")
    print(f"    • Run 2: {run2_time:>6.1f}s")
    print(f"    • Run 3: {run3_time:>6.1f}s")
    print(f"    • Total: {total_time:>6.1f}s ({total_time/60:.1f} min)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="ROBUST04 Ranking Competition Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune parameters")
    tune_parser.add_argument(
        "--output", "-o",
        default="tuning_results",
        help="Output directory for tuning results"
    )
    tune_parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Disable cross-validation (faster but may overfit)"
    )
    
    # Run1 command
    run1_parser = subparsers.add_parser("run1", help="BM25+RM3+Query2Doc")
    run1_parser.add_argument(
        "--config", "-c",
        help="Path to tuned config JSON"
    )
    run1_parser.add_argument(
        "--output", "-o",
        default="results/run_1.res",
        help="Output file path"
    )
    
    # Run2 command
    run2_parser = subparsers.add_parser("run2", help="Neural MaxP Reranking")
    run2_parser.add_argument(
        "--config", "-c",
        help="Path to tuned config JSON"
    )
    run2_parser.add_argument(
        "--output", "-o",
        default="results/run_2.res",
        help="Output file path"
    )
    run2_parser.add_argument(
        "--rerank-depth",
        type=int,
        default=1000,
        help="Number of documents to rerank"
    )
    run2_parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size in characters"
    )
    run2_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Chunk overlap in characters"
    )
    run2_parser.add_argument(
        "--bi-encoder-top-k",
        type=int,
        default=4000,
        help="Passages to keep after bi-encoder filtering"
    )
    run2_parser.add_argument(
        "--bi-batch-size",
        type=int,
        default=2048,
        help="Bi-encoder batch size"
    )
    run2_parser.add_argument(
        "--ce-batch-size",
        type=int,
        default=2048,
        help="Cross-encoder batch size"
    )
    run2_parser.add_argument(
        "--monot5-batch-size",
        type=int,
        default=512,
        help="MonoT5 batch size"
    )
    run2_parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.5,
        help="Cross-encoder weight in CE/MonoT5 ensemble"
    )
    run2_parser.add_argument(
        "--neural-weight",
        type=float,
        default=0.8,
        help="Neural weight in neural/BM25 interpolation"
    )
    run2_parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU for neural model"
    )
    
    # Run3 command
    run3_parser = subparsers.add_parser("run3", help="Multi-Signal Fusion")
    run3_parser.add_argument(
        "--run1",
        default="results/run_1.res",
        help="Path to run_1"
    )
    run3_parser.add_argument(
        "--run2",
        default="results/run_2.res",
        help="Path to run_2"
    )
    run3_parser.add_argument(
        "--output", "-o",
        default="results/run_3.res",
        help="Output file path"
    )
    
    # Run all command
    all_parser = subparsers.add_parser("run_all", help="Run all methods")
    all_parser.add_argument(
        "--config", "-c",
        help="Path to tuned config JSON"
    )
    all_parser.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Output directory"
    )
    all_parser.add_argument(
        "--rerank-depth",
        type=int,
        default=1000,  # MAXIMUM: rerank 1000 docs for best recall!
        help="Number of documents to rerank for neural method"
    )
    all_parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU for neural model"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate runs")
    eval_parser.add_argument(
        "runs",
        nargs="+",
        help="Run files to evaluate"
    )
    
    # Quick train command - for fast iteration with FULL control
    quick_parser = subparsers.add_parser(
        "quick_train", 
        help="Run all methods on training queries only (fast iteration)"
    )
    quick_parser.add_argument(
        "--config", "-c",
        help="Path to tuned config JSON"
    )
    quick_parser.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Output directory"
    )
    
    # Retrieval parameters
    quick_parser.add_argument(
        "--rerank-depth",
        type=int,
        default=500,
        help="Number of docs to neurally rerank (default: 500)"
    )
    quick_parser.add_argument(
        "--bm25-k",
        type=int,
        default=1000,
        help="Number of docs to retrieve from BM25 (default: 1000)"
    )
    
    # Chunking parameters
    quick_parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size in characters (default: 256)"
    )
    quick_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Chunk overlap in characters (default: 64)"
    )
    
    # Bi-encoder parameters
    quick_parser.add_argument(
        "--bi-encoder-top-k",
        type=int,
        default=None,
        help="Passages to keep after bi-encoder (default: auto-scaled with rerank-depth)"
    )
    quick_parser.add_argument(
        "--bi-batch-size",
        type=int,
        default=2048,
        help="Bi-encoder batch size (default: 2048)"
    )
    
    # Cross-encoder parameters
    quick_parser.add_argument(
        "--ce-batch-size",
        type=int,
        default=2048,
        help="Cross-encoder batch size (default: 2048)"
    )
    
    # MonoT5 parameters
    quick_parser.add_argument(
        "--monot5-batch-size",
        type=int,
        default=512,
        help="MonoT5 batch size (default: 512)"
    )
    
    # Ensemble weights
    quick_parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.5,
        help="Cross-encoder weight in CE/MonoT5 ensemble (default: 0.5)"
    )
    quick_parser.add_argument(
        "--neural-weight",
        type=float,
        default=0.8,
        help="Neural weight in neural/BM25 interpolation (default: 0.8)"
    )
    
    # RRF parameter
    quick_parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k parameter (default: 60)"
    )
    
    # Device
    quick_parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU for neural models"
    )
    quick_parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU (disable GPU)"
    )
    
    args = parser.parse_args()
    
    if args.command == "tune":
        cmd_tune(args)
    elif args.command == "run1":
        cmd_run1(args)
    elif args.command == "run2":
        cmd_run2(args)
    elif args.command == "run3":
        cmd_run3(args)
    elif args.command == "run_all":
        cmd_run_all(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "quick_train":
        cmd_quick_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

