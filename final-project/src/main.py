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
from .neural_reranker import FastCrossEncoder, normalize_scores
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
    
    # Get test query IDs - use expanded if available, fallback to original
    test_qids = get_test_qids()
    test_expanded = {}
    for qid in test_qids:
        if qid in expanded_queries:
            test_expanded[qid] = expanded_queries[qid]
        elif qid in queries:
            test_expanded[qid] = queries[qid]  # Fallback to original
    
    print(f"Running BM25+RM3 on {len(test_expanded)} test queries with Query2Doc...")
    
    results = batch_search_bm25_rm3(
        queries=test_expanded,
        k=1000,
        k1=k1, b=b,
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        original_query_weight=orig_weight,
    )
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run(results, output_path, run_name="BM25_RM3_Q2D")
    print(f"Saved run_1 to {output_path}")


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
    
    # Initialize cross-encoder
    print("Loading Cross-Encoder model...")
    cross_encoder = FastCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        device="cuda" if args.gpu else "cpu",
    )
    
    # Get test queries - use expanded if available, fallback to original
    test_qids = get_test_qids()
    test_queries = {qid: queries[qid] for qid in test_qids if qid in queries}
    test_expanded = {}
    for qid in test_qids:
        if qid in expanded_queries:
            test_expanded[qid] = expanded_queries[qid]
        elif qid in queries:
            test_expanded[qid] = queries[qid]  # Fallback to original
    
    print(f"Processing {len(test_expanded)} test queries...")
    
    # OPTIMIZATION: Batch BM25 retrieval for all queries first
    print("Running batch BM25 retrieval...")
    all_bm25_results = batch_search_bm25(
        test_expanded,
        k=max(100, args.rerank_depth),  # Get enough docs for reranking
        k1=k1, b=b,
    )
    
    all_results = {}
    
    for qid in tqdm(test_qids, desc="Neural reranking"):
        if qid not in all_bm25_results or not all_bm25_results[qid]:
            continue
        
        bm25_results = all_bm25_results[qid]
        
        # Get passages for top documents
        passages, doc_to_indices = get_all_passages_for_query(
            bm25_results,
            top_k=args.rerank_depth,
            chunk_size=256,
            chunk_overlap=64,
        )
        
        if not passages:
            # Fallback to BM25 scores
            all_results[qid] = {r.docid: r.score for r in bm25_results[:1000]}
            continue
        
        # Score passages with original query (better precision)
        original_query = test_queries.get(qid, test_expanded.get(qid, ""))
        passage_scores = cross_encoder.rerank_passages(
            original_query,
            passages,
            batch_size=256,
        )
        
        # Normalize scores
        passage_scores = normalize_scores(passage_scores)
        
        # Aggregate to document scores (MaxP)
        doc_scores = aggregate_passage_scores(
            passages,
            passage_scores,
            doc_to_indices,
            strategy="maxp",
        )
        
        # Add back documents not in top-k with lower scores
        min_score = min(doc_scores.values()) if doc_scores else 0
        for r in bm25_results:
            if r.docid not in doc_scores:
                doc_scores[r.docid] = min_score - (r.rank / 10000)
        
        all_results[qid] = doc_scores
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run_from_scores(all_results, output_path, run_name="Neural_MaxP")
    print(f"Saved run_2 to {output_path}")


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
    
    # Get all test query IDs
    test_qids = get_test_qids()
    
    print(f"Fusing {len(test_qids)} queries with RRF...")
    
    # Fuse per query
    fused_results = {}
    for qid in tqdm(test_qids, desc="Fusing"):
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
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run_from_scores(fused_results, output_path, run_name="RRF_Fusion")
    print(f"Saved run_3 to {output_path}")


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
        default=100,
        help="Number of documents to rerank"
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
        default=100,
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

