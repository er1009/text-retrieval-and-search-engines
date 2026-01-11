"""Main CLI for ROBUST04 ranking competition."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from .config import set_all_seeds
from .data_loader import (
    load_queries, load_expanded_queries, load_qrels,
    get_train_qids, get_test_qids
)
from .bm25_retrieval import batch_search_bm25, batch_search_bm25_rm3
from .document_processor import get_all_passages_for_query
from .neural_reranker import ThreeStageReranker
from .aggregation import aggregate_passage_scores
from .fusion import fuse_runs
from .evaluation import comprehensive_evaluation
from .tuning import load_tuned_config
from .trec_io import write_trec_run_from_scores


def run_pipeline(
    qids: list[str],
    queries: dict[str, str],
    expanded_queries: dict[str, str],
    output_dir: Path,
    args,
    evaluate: bool = False,
    qrels: dict = None,
):
    """
    Run the full 3-run pipeline on given queries.
    
    Args:
        qids: Query IDs to process
        queries: Original queries
        expanded_queries: Query2Doc expanded queries
        output_dir: Directory to save results
        args: Command line arguments
        evaluate: Whether to evaluate (only for training)
        qrels: Relevance judgments (only needed if evaluate=True)
    """
    start_time = time.time()
    
    # Load config
    config_path = Path(args.config) if args.config else None
    if config_path and config_path.exists():
        config = load_tuned_config(config_path)
        k1, b = config['bm25']['k1'], config['bm25']['b']
        fb_docs = config['rm3']['fb_docs']
        fb_terms = config['rm3']['fb_terms']
        orig_weight = config['rm3']['original_query_weight']
        print(f"  Config: {config_path.name}")
    else:
        k1, b = 0.9, 0.4
        fb_docs, fb_terms, orig_weight = 10, 10, 0.5
        print("  Config: defaults")
    
    print(f"  BM25: k1={k1}, b={b}")
    print(f"  RM3: fb_docs={fb_docs}, fb_terms={fb_terms}, weight={orig_weight}")
    
    # Filter queries to requested QIDs
    q_orig = {qid: queries[qid] for qid in qids if qid in queries}
    q_exp = {qid: expanded_queries.get(qid, queries.get(qid, "")) for qid in qids}
    
    # ==================== RUN 1: LEXICAL ====================
    run1_start = time.time()
    print("\n" + "=" * 60)
    print("  RUN 1: BM25 + RM3 + Query2Doc Fusion")
    print("=" * 60)
    
    print("\n  [1/3] BM25+RM3 with original queries...")
    results_orig = batch_search_bm25_rm3(
        q_orig, k=1000, k1=k1, b=b,
        fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
    )
    
    print("  [2/3] BM25+RM3 with expanded queries...")
    results_exp = batch_search_bm25_rm3(
        q_exp, k=1000, k1=k1, b=b,
        fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
    )
    
    print(f"  [3/3] Fusing with RRF (k={args.rrf_k})...")
    orig_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_orig.items()}
    exp_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_exp.items()}
    
    run1_results = {}
    for qid in qids:
        runs = [r for r in [orig_scores.get(qid), exp_scores.get(qid)] if r]
        if runs:
            run1_results[qid] = fuse_runs(runs, method="rrf", rrf_k=args.rrf_k)
    
    write_trec_run_from_scores(run1_results, output_dir / "run_1.res", run_name="BM25_RM3_Q2D")
    print(f"  ✓ Run 1: {len(run1_results)} queries in {time.time() - run1_start:.1f}s")
    
    # ==================== RUN 2: NEURAL ====================
    run2_start = time.time()
    print("\n" + "=" * 60)
    print("  RUN 2: Three-Stage Neural Reranking")
    print("=" * 60)
    
    device = "cpu" if args.no_gpu else "cuda"
    
    print(f"\n  Loading models (device={device})...")
    reranker = ThreeStageReranker(
        bi_encoder_model="BAAI/bge-base-en-v1.5",
        monot5_model="castorini/monot5-base-msmarco",
        ce_weight=args.ce_weight,
        bi_encoder_top_k=args.bi_encoder_top_k,
        use_ensemble_ce=True,
        device=device,
        use_bf16=True,
    )
    
    print(f"  BM25 retrieval ({args.bm25_k} docs/query)...")
    all_bm25 = batch_search_bm25(q_exp, k=args.bm25_k, k1=k1, b=b)
    
    # Normalize BM25 scores
    bm25_norm = {}
    for qid, results in all_bm25.items():
        if not results:
            continue
        max_s, min_s = results[0].score, results[-1].score
        rng = max_s - min_s if max_s != min_s else 1.0
        bm25_norm[qid] = {r.docid: (r.score - min_s) / rng for r in results}
    
    print(f"  Neural reranking {len(qids)} queries...")
    run2_results = {}
    
    for qid in tqdm(qids, desc="  Reranking"):
        if qid not in all_bm25 or not all_bm25[qid]:
            continue
        
        bm25_results = all_bm25[qid]
        
        passages, doc_indices = get_all_passages_for_query(
            bm25_results, top_k=args.rerank_depth,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        )
        
        if not passages:
            run2_results[qid] = {r.docid: r.score for r in bm25_results}
            continue
        
        query_text = q_orig.get(qid, q_exp.get(qid, ""))
        passage_scores, _ = reranker.rerank_passages(
            query_text, passages,
            bi_batch_size=args.bi_batch_size,
            ce_batch_size=args.ce_batch_size,
            monot5_batch_size=args.monot5_batch_size,
        )
        
        doc_scores = aggregate_passage_scores(passages, passage_scores, doc_indices, strategy="maxp")
        
        # Normalize neural scores
        if doc_scores:
            max_n, min_n = max(doc_scores.values()), min(doc_scores.values())
            rng = max_n - min_n if max_n != min_n else 1.0
            doc_scores = {d: (s - min_n) / rng for d, s in doc_scores.items()}
        
        # Interpolate neural + BM25 for reranked, scale down BM25 for unreranked
        final = {}
        for r in bm25_results:
            if r.docid in doc_scores:
                final[r.docid] = (
                    args.neural_weight * doc_scores[r.docid] +
                    (1 - args.neural_weight) * bm25_norm[qid][r.docid]
                )
            else:
                # Simple fixed scaling (preserves BM25 ranking for unreranked)
                final[r.docid] = bm25_norm[qid].get(r.docid, 0) * 0.3
        
        run2_results[qid] = final
    
    write_trec_run_from_scores(run2_results, output_dir / "run_2.res", run_name="Neural_MaxP")
    print(f"  ✓ Run 2: {len(run2_results)} queries in {time.time() - run2_start:.1f}s")
    
    # ==================== RUN 3: FUSION ====================
    run3_start = time.time()
    print("\n" + "=" * 60)
    print("  RUN 3: RRF Fusion")
    print("=" * 60)
    
    run3_results = {}
    for qid in qids:
        runs = [r for r in [run1_results.get(qid), run2_results.get(qid)] if r]
        if runs:
            run3_results[qid] = fuse_runs(runs, method="rrf", rrf_k=args.rrf_k)
    
    write_trec_run_from_scores(run3_results, output_dir / "run_3.res", run_name="RRF_Fusion")
    print(f"  ✓ Run 3: {len(run3_results)} queries in {time.time() - run3_start:.1f}s")
    
    # ==================== EVALUATION ====================
    if evaluate and qrels:
        print("\n" + "=" * 60)
        print("  EVALUATION")
        print("=" * 60)
        
        q_rels = {qid: qrels[qid] for qid in qids if qid in qrels}
        
        print(f"\n  {'Metric':<12} {'Run 1':>10} {'Run 2':>10} {'Run 3':>10}")
        print("  " + "-" * 46)
        
        metrics = ['map', 'ndcg_10', 'p_10', 'recall_100', 'recall_1000']
        names = ['MAP', 'NDCG@10', 'P@10', 'R@100', 'R@1000']
        
        all_metrics = []
        for run in [run1_results, run2_results, run3_results]:
            all_metrics.append(comprehensive_evaluation(run, q_rels))
        
        for metric, name in zip(metrics, names):
            vals = [m.get(metric, 0) for m in all_metrics]
            best = '←' if vals.index(max(vals)) == 2 else ''
            print(f"  {name:<12} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {best}")
    
    # ==================== SUMMARY ====================
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ✅ COMPLETE ({total_time:.1f}s)")
    print("=" * 60)
    print(f"  Output: {output_dir}/run_1.res, run_2.res, run_3.res")


def cmd_train(args):
    """Run pipeline on training queries and evaluate."""
    print("\n" + "=" * 60)
    print("  TRAIN: Running on training queries (301-350)")
    print("=" * 60)
    
    set_all_seeds(42)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    queries = load_queries()
    expanded = load_expanded_queries()
    qrels = load_qrels()
    train_qids = get_train_qids()
    
    print(f"\n  Queries: {len(train_qids)} training")
    
    run_pipeline(
        qids=train_qids,
        queries=queries,
        expanded_queries=expanded,
        output_dir=output_dir,
        args=args,
        evaluate=True,
        qrels=qrels,
    )


def cmd_test(args):
    """Run pipeline on test queries and save submission files."""
    print("\n" + "=" * 60)
    print("  TEST: Running on test queries (351-700)")
    print("=" * 60)
    
    set_all_seeds(42)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    queries = load_queries()
    expanded = load_expanded_queries()
    test_qids = get_test_qids()
    
    print(f"\n  Queries: {len(test_qids)} test")
    
    run_pipeline(
        qids=test_qids,
        queries=queries,
        expanded_queries=expanded,
        output_dir=output_dir,
        args=args,
        evaluate=False,
    )


def add_common_args(parser):
    """Add common arguments to a parser."""
    parser.add_argument("--config", "-c", help="Path to tuned config JSON")
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory")
    
    # Retrieval
    parser.add_argument("--bm25-k", type=int, default=1000, help="BM25 retrieval depth")
    parser.add_argument("--rerank-depth", type=int, default=1000, help="Docs to rerank")
    
    # Chunking
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size (chars)")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap (chars)")
    
    # Neural
    parser.add_argument("--bi-encoder-top-k", type=int, default=4000, help="Bi-encoder filter")
    parser.add_argument("--bi-batch-size", type=int, default=2048, help="Bi-encoder batch")
    parser.add_argument("--ce-batch-size", type=int, default=2048, help="Cross-encoder batch")
    parser.add_argument("--monot5-batch-size", type=int, default=512, help="MonoT5 batch")
    
    # Weights
    parser.add_argument("--ce-weight", type=float, default=0.5, help="CE weight in ensemble")
    parser.add_argument("--neural-weight", type=float, default=0.8, help="Neural/BM25 weight")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter")
    
    # Device
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")


def main():
    parser = argparse.ArgumentParser(description="ROBUST04 Ranking Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run on training queries + evaluate")
    add_common_args(train_parser)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run on test queries (submission)")
    add_common_args(test_parser)
    
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
