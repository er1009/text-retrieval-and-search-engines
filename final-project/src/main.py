"""Main CLI for ROBUST04 ranking competition."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from tqdm import tqdm

from .config import set_all_seeds
from .data_loader import (
    load_queries, load_expanded_queries, load_qrels,
    get_train_qids, get_test_qids
)
from .bm25_retrieval import batch_search_bm25_rm3, batch_get_documents
from .document_processor import get_all_passages_for_query, clean_document_text
from .neural_reranker import CascadeReranker
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
    start_time = time.time()
    
    # Load config (CLI args override config file)
    config_path = Path(args.config) if args.config else None
    if config_path and config_path.exists():
        config = load_tuned_config(config_path)
        k1 = args.bm25_k1 if args.bm25_k1 is not None else config['bm25']['k1']
        b = args.bm25_b if args.bm25_b is not None else config['bm25']['b']
        fb_docs = args.fb_docs if args.fb_docs is not None else config['rm3']['fb_docs']
        fb_terms = args.fb_terms if args.fb_terms is not None else config['rm3']['fb_terms']
        orig_weight = args.original_weight if args.original_weight is not None else config['rm3']['original_query_weight']
        print(f"  Config: {config_path.name} (CLI overrides applied)")
    else:
        k1 = args.bm25_k1 if args.bm25_k1 is not None else 0.9
        b = args.bm25_b if args.bm25_b is not None else 0.4
        fb_docs = args.fb_docs if args.fb_docs is not None else 10
        fb_terms = args.fb_terms if args.fb_terms is not None else 10
        orig_weight = args.original_weight if args.original_weight is not None else 0.5
        print("  Config: defaults")
    
    print(f"  BM25: k1={k1}, b={b}")
    print(f"  RM3: fb_docs={fb_docs}, fb_terms={fb_terms}, weight={orig_weight}")
    
    # Filter queries
    q_orig = {qid: queries[qid] for qid in qids if qid in queries}
    q_exp = {qid: expanded_queries.get(qid, queries.get(qid, "")) for qid in qids}
    
    # ==================== SHARED: RETRIEVAL (BM25+RM3 + Dense + SPLADE) ====================
    use_q2d = not args.disable_q2d
    use_splade = args.splade_index_path is not None
    
    # Calculate number of retrieval methods
    base_methods = 2  # BM25 + Dense
    if use_splade:
        base_methods += 1  # + SPLADE
    num_methods = base_methods * 2 if use_q2d else base_methods
    
    print("\n" + "=" * 60)
    methods_str = "BM25+RM3 + Dense"
    if use_splade:
        methods_str += " + SPLADE"
    print(f"  RETRIEVAL: {methods_str} ({num_methods}-way {'Hybrid' if use_q2d else 'Original Only'})")
    print("=" * 60)
    print(f"  Retrieval k={args.retrieval_k} per method")
    if args.disable_q2d:
        print("  ⚠️  Q2D DISABLED (ablation mode)")
    
    step = 1
    
    print(f"\n  [{step}/{num_methods}] BM25+RM3 with original queries...")
    results_orig = batch_search_bm25_rm3(
        q_orig, k=args.retrieval_k, k1=k1, b=b,
        fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
    )
    step += 1
    
    # Load dense index
    from .dense_index import DensePassageIndex
    dense_index = DensePassageIndex(args.dense_index_path)
    dense_index.load()
    
    print(f"  [{step}/{num_methods}] Dense retrieval (original queries)...")
    dense_orig = dense_index.batch_search(q_orig, top_k=args.retrieval_k, show_progress=True)
    step += 1
    
    # Load SPLADE index (optional)
    splade_orig = {}
    splade_exp = {}
    if use_splade:
        from .splade_index import SpladeIndex
        splade_index = SpladeIndex(args.splade_index_path)
        splade_index.load()
        
        print(f"  [{step}/{num_methods}] SPLADE retrieval (original queries)...")
        splade_orig = splade_index.batch_search(q_orig, top_k=args.retrieval_k, show_progress=True)
        step += 1
    
    # Q2D retrieval (optional)
    if use_q2d:
        print(f"  [{step}/{num_methods}] BM25+RM3 with expanded queries (Q2D)...")
        results_exp = batch_search_bm25_rm3(
            q_exp, k=args.retrieval_k, k1=k1, b=b,
            fb_docs=fb_docs, fb_terms=fb_terms, original_query_weight=orig_weight,
        )
        step += 1
        
        print(f"  [{step}/{num_methods}] Dense retrieval (Q2D queries)...")
        dense_exp = dense_index.batch_search(q_exp, top_k=args.retrieval_k, show_progress=True)
        step += 1
        
        if use_splade:
            print(f"  [{step}/{num_methods}] SPLADE retrieval (Q2D queries)...")
            splade_exp = splade_index.batch_search(q_exp, top_k=args.retrieval_k, show_progress=True)
            step += 1
        
        exp_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_exp.items()}
    else:
        exp_scores = {}
        dense_exp = {}
    
    orig_scores = {qid: {r.docid: r.score for r in docs} for qid, docs in results_orig.items()}
    
    # Parse retrieval weights
    ret_weights = [float(w) for w in args.retrieval_weights.split(",")]
    w_bm25, w_dense, w_splade = ret_weights[0], ret_weights[1], ret_weights[2] if len(ret_weights) > 2 else 1.0
    
    # ==================== RUN 1: Hybrid Fusion ====================
    run1_start = time.time()
    print("\n" + "=" * 60)
    
    # Build description of fusion (only include methods with weight > 0)
    sources = []
    if w_bm25 > 0:
        sources.append(f"BM25(orig)×{w_bm25}")
    if w_dense > 0:
        sources.append(f"Dense(orig)×{w_dense}")
    if use_splade and w_splade > 0:
        sources.append(f"SPLADE(orig)×{w_splade}")
    if use_q2d:
        if w_bm25 > 0:
            sources.append(f"BM25(Q2D)×{w_bm25}")
        if w_dense > 0:
            sources.append(f"Dense(Q2D)×{w_dense}")
        if use_splade and w_splade > 0:
            sources.append(f"SPLADE(Q2D)×{w_splade}")
    
    print(f"  RUN 1: {' + '.join(sources)} → RRF")
    print("=" * 60)
    print(f"  Retrieval weights: BM25={w_bm25}, Dense={w_dense}, SPLADE={w_splade}")
    
    # Fuse retrieval methods with weights
    print(f"  Fusing retrievers with RRF (k={args.rrf_k})...")
    run1_results = {}
    for qid in qids:
        # Build runs and weights lists (only non-zero weights)
        runs = []
        weights = []
        
        if w_bm25 > 0 and orig_scores.get(qid):
            runs.append(orig_scores[qid])
            weights.append(w_bm25)
        if w_dense > 0 and dense_orig.get(qid):
            runs.append(dense_orig[qid])
            weights.append(w_dense)
        if use_splade and w_splade > 0 and splade_orig.get(qid):
            runs.append(splade_orig[qid])
            weights.append(w_splade)
        
        if use_q2d:
            if w_bm25 > 0 and exp_scores.get(qid):
                runs.append(exp_scores[qid])
                weights.append(w_bm25)
            if w_dense > 0 and dense_exp.get(qid):
                runs.append(dense_exp[qid])
                weights.append(w_dense)
            if use_splade and w_splade > 0 and splade_exp.get(qid):
                runs.append(splade_exp[qid])
                weights.append(w_splade)
        
        if runs:
            run1_results[qid] = fuse_runs(runs, method="rrf", rrf_k=args.rrf_k, weights=weights)
    
    write_trec_run_from_scores(run1_results, output_dir / "run_1.res", run_name="Hybrid_RRF")
    print(f"  ✓ Run 1: {len(run1_results)} queries in {time.time() - run1_start:.1f}s")
    
    # ==================== RUN 2: Cascade Neural Reranking ====================
    run2_start = time.time()
    print("\n" + "=" * 60)
    print("  RUN 2: Cascade Neural Reranking")
    print("=" * 60)
    print(f"  Input: {args.rerank_depth} docs from Run 1")
    print(f"  Stage 1: Bi-Encoder → {args.bi_top_k} docs")
    print(f"  Stage 2: Cross-Encoder → {args.ce_top_k} docs")
    print(f"  Stage 3: MonoT5 → final ranking")
    
    # Use Run 1 results as candidates
    run2_candidates = run1_results
    
    # Initialize cascade reranker
    device = "cpu" if args.no_gpu else "cuda"
    ce_models = [m.strip() for m in args.ce_model.split(",")]
    
    reranker = CascadeReranker(
        bi_top_k=args.bi_top_k,
        ce_top_k=args.ce_top_k,
        bi_encoder_model=args.bi_encoder_model,
        ce_model=ce_models,
        monot5_model=args.monot5_model,
        device=device,
        use_bf16=True,
    )
    
    print(f"\n  Reranking {len(qids)} queries...")
    run2_results = {}
    
    from dataclasses import dataclass
    @dataclass
    class MockResult:
        docid: str
        score: float
    
    for qid in tqdm(qids, desc="  Reranking"):
        if qid not in run2_candidates:
            continue
        
        # Get top docs from fused candidates
        sorted_docs = sorted(run2_candidates[qid].items(), key=lambda x: -x[1])[:args.rerank_depth]
        top_docids = [d[0] for d in sorted_docs]
        
        # Fetch documents and create passages
        docs = batch_get_documents(top_docids)
        mock_results = [MockResult(docid=d, score=run2_candidates[qid][d]) for d in top_docids if d in docs]
        
        passages, doc_indices = get_all_passages_for_query(
            mock_results, top_k=args.rerank_depth,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        )
        
        if not passages:
            run2_results[qid] = run2_candidates[qid]
            continue
        
        query_text = q_orig.get(qid, q_exp.get(qid, ""))
        
        # Run cascade reranking (returns scores from all stages)
        cascade_scores = reranker.rerank(
            query_text, passages,
            bi_batch_size=args.bi_batch_size,
            ce_batch_size=args.ce_batch_size,
            monot5_batch_size=args.monot5_batch_size,
        )
        
        neural_scores = cascade_scores['final']
        
        # Normalize neural scores
        if neural_scores:
            max_n, min_n = max(neural_scores.values()), min(neural_scores.values())
            rng = max_n - min_n if max_n != min_n else 1.0
            neural_norm = {d: (s - min_n) / rng for d, s in neural_scores.items()}
        else:
            neural_norm = {}
        
        # Normalize candidate scores (from Run 1 hybrid retrieval)
        cand_scores = run2_candidates[qid]
        max_c, min_c = max(cand_scores.values()), min(cand_scores.values())
        rng_c = max_c - min_c if max_c != min_c else 1.0
        cand_norm = {d: (s - min_c) / rng_c for d, s in cand_scores.items()}
        
        # Interpolate: neural (cascade) + retrieval (Run 1)
        final = {}
        for docid, cand_score in cand_scores.items():
            if docid in neural_norm:
                final[docid] = args.neural_weight * neural_norm[docid] + (1 - args.neural_weight) * cand_norm[docid]
            else:
                # Doc not in rerank candidates - use scaled retrieval score
                final[docid] = cand_norm[docid] * 0.3
        
        run2_results[qid] = final
    
    write_trec_run_from_scores(run2_results, output_dir / "run_2.res", run_name="Cascade_Neural")
    print(f"  ✓ Run 2: {len(run2_results)} queries in {time.time() - run2_start:.1f}s")
    
    # ==================== RUN 3: RRF(Run1, Run2, LLM) or just Run2 ====================
    run3_start = time.time()
    print("\n" + "=" * 60)
    
    if args.disable_llm:
        print("  RUN 3: RRF(Run1, Run2) - No LLM")
        print("=" * 60)
        print("  ⚠️  LLM DISABLED (ablation mode)")
        
        # Two-way RRF fusion (Run1 + Run2)
        weights = [args.rrf_weight_run1, args.rrf_weight_run2]
        print(f"  Fusing Run1 + Run2 with RRF (k={args.rrf_k}, weights={weights})...")
        run3_results = {}
        for qid in qids:
            runs_to_fuse = []
            run_weights = []
            if qid in run1_results:
                runs_to_fuse.append(run1_results[qid])
                run_weights.append(args.rrf_weight_run1)
            if qid in run2_results:
                runs_to_fuse.append(run2_results[qid])
                run_weights.append(args.rrf_weight_run2)
            if runs_to_fuse:
                run3_results[qid] = fuse_runs(runs_to_fuse, method="rrf", rrf_k=args.rrf_k, weights=run_weights)
        
        write_trec_run_from_scores(run3_results, output_dir / "run_3.res", run_name="RRF_No_LLM")
        print(f"  ✓ Run 3: {len(run3_results)} queries in {time.time() - run3_start:.1f}s")
    else:
        print("  RUN 3: RRF(Run1, Run2, LLM) - Robust Fusion")
        print("=" * 60)
        
        # Step 1: LLM reranks Run2 top-k (best neural candidates)
        from .llm_reranker import (
            LLMReranker, LLMRerankerConfig, batch_rerank_with_llm, estimate_cost
        )
        
        cost_est = estimate_cost(
            num_queries=len(qids),
            passages_per_query=args.llm_top_k,
            window_size=args.llm_window_size,
            step_size=args.llm_step_size,
            model=args.llm_model,
        )
        print(f"\n  [1/2] LLM Reranking Run2 top-{args.llm_top_k}:")
        print(f"    Model: {args.llm_model}")
        print(f"    Est. requests: {cost_est['total_requests']}, Est. cost: ${cost_est['estimated_cost_usd']:.2f}")
        
        # Prepare LLM input from Run2 results (neural reranked)
        llm_input = {}
        for qid in qids:
            if qid not in run2_results:
                continue
            sorted_docs = sorted(run2_results[qid].items(), key=lambda x: -x[1])
            top_docids = [d[0] for d in sorted_docs[:args.llm_top_k]]
            docs = batch_get_documents(top_docids)
            llm_input[qid] = [
                {"docid": d, "text": clean_document_text(docs.get(d, ""))} 
                for d in top_docids if docs.get(d)
            ]
        
        llm_config = LLMRerankerConfig(
            model=args.llm_model,
            window_size=args.llm_window_size,
            step_size=args.llm_step_size,
            max_passage_length=args.llm_max_passage_length,
            max_concurrent_requests=args.llm_concurrency,
            use_dynamic_few_shot=args.llm_dynamic_few_shot,
        )
        llm_reranker = LLMReranker(llm_config)
        
        llm_rankings = asyncio.run(batch_rerank_with_llm(
            llm_reranker, q_orig, llm_input,
            top_k=args.llm_top_k,
            show_progress=True,
        ))
        
        print(f"\n  [LLM Stage 1] Requests: {llm_reranker.stats.total_requests}, Errors: {llm_reranker.stats.errors}")
        
        # Step 2: Strong LLM refines top-k (cascade)
        if args.llm_strong_model and args.llm_strong_top_k > 0:
            print(f"\n  [1b/2] Strong LLM refining top-{args.llm_strong_top_k}:")
            print(f"    Model: {args.llm_strong_model}")
            
            strong_input = {}
            for qid in qids:
                if qid not in llm_rankings or not llm_rankings[qid]:
                    continue
                sorted_docs = sorted(llm_rankings[qid].items(), key=lambda x: -x[1])
                top_docids = [d[0] for d in sorted_docs[:args.llm_strong_top_k]]
                docs = batch_get_documents(top_docids)
                strong_input[qid] = [
                    {"docid": d, "text": clean_document_text(docs.get(d, ""))} 
                    for d in top_docids if docs.get(d)
                ]
            
            strong_config = LLMRerankerConfig(
                model=args.llm_strong_model,
                window_size=args.llm_strong_top_k,
                step_size=args.llm_strong_top_k,
                max_passage_length=args.llm_max_passage_length,
                max_concurrent_requests=args.llm_concurrency,
                use_dynamic_few_shot=args.llm_dynamic_few_shot,
            )
            strong_reranker = LLMReranker(strong_config)
            
            strong_rankings = asyncio.run(batch_rerank_with_llm(
                strong_reranker, q_orig, strong_input,
                top_k=args.llm_strong_top_k,
                show_progress=True,
            ))
            
            print(f"  [LLM Stage 2] Requests: {strong_reranker.stats.total_requests}, Errors: {strong_reranker.stats.errors}")
            
            # Merge: strong LLM top-k replaces first LLM top-k
            for qid in qids:
                if qid in strong_rankings and strong_rankings[qid]:
                    merged = strong_rankings[qid].copy()
                    if qid in llm_rankings:
                        for docid, score in llm_rankings[qid].items():
                            if docid not in merged:
                                merged[docid] = score * 0.5
                    llm_rankings[qid] = merged
        
        # Step 3: Three-way Weighted RRF fusion (Run1 + Run2 + LLM)
        weights = [args.rrf_weight_run1, args.rrf_weight_run2, args.rrf_weight_llm]
        print(f"\n  [2/2] Weighted RRF fusion (k={args.rrf_k}, weights={weights})...")
        run3_results = {}
        for qid in qids:
            runs_to_fuse = []
            run_weights = []
            if qid in run1_results:
                runs_to_fuse.append(run1_results[qid])
                run_weights.append(args.rrf_weight_run1)
            if qid in run2_results:
                runs_to_fuse.append(run2_results[qid])
                run_weights.append(args.rrf_weight_run2)
            if qid in llm_rankings and llm_rankings[qid]:
                runs_to_fuse.append(llm_rankings[qid])
                run_weights.append(args.rrf_weight_llm)
            if runs_to_fuse:
                run3_results[qid] = fuse_runs(runs_to_fuse, method="rrf", rrf_k=args.rrf_k, weights=run_weights)
        
        write_trec_run_from_scores(run3_results, output_dir / "run_3.res", run_name="LLM_Cascade")
        print(f"  ✓ Run 3: {len(run3_results)} queries in {time.time() - run3_start:.1f}s")
    
    # ==================== EVALUATION ====================
    if evaluate and qrels:
        print("\n" + "=" * 60)
        print("  EVALUATION")
        print("=" * 60)
        
        q_rels = {qid: qrels[qid] for qid in qids if qid in qrels}
        all_metrics = [comprehensive_evaluation(run, q_rels) for run in [run1_results, run2_results, run3_results]]
        
        print(f"\n  {'Metric':<12} {'Run 1':>10} {'Run 2':>10} {'Run 3':>10}")
        print("  " + "-" * 46)
        
        for metric, name in [('map', 'MAP'), ('ndcg', 'NDCG'), ('ndcg_10', 'NDCG@10'), ('ndcg_100', 'NDCG@100')]:
            vals = [m.get(metric, 0) for m in all_metrics]
            best = '←' if vals.index(max(vals)) == 2 else ''
            print(f"  {name:<12} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {best}")
        
        print("\n  Precision:")
        print("  " + "-" * 46)
        for k in [10, 100, 500, 1000]:
            vals = [m.get(f'p_{k}', 0) for m in all_metrics]
            best = '←' if vals.index(max(vals)) == 2 else ''
            print(f"  P@{k:<9} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {best}")
        
        print("\n  Recall:")
        print("  " + "-" * 46)
        for k in [10, 100, 500, 1000]:
            vals = [m.get(f'recall_{k}', 0) for m in all_metrics]
            best = '←' if vals.index(max(vals)) == 2 else ''
            print(f"  R@{k:<9} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {best}")
        
        metrics_output = {'run_1': all_metrics[0], 'run_2': all_metrics[1], 'run_3': all_metrics[2]}
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics_output, f, indent=2)
        print(f"\n  ✓ Metrics saved to {output_dir}/metrics.json")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ✅ COMPLETE ({total_time:.1f}s)")
    print("=" * 60)
    print(f"  Output: {output_dir}/run_1.res, run_2.res, run_3.res")


def cmd_train(args):
    print("\n" + "=" * 60)
    print("  TRAIN: Training queries (301-350)")
    print("=" * 60)
    
    set_all_seeds(42)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    queries = load_queries()
    expanded = load_expanded_queries()
    qrels = load_qrels()
    train_qids = get_train_qids()
    
    if args.limit_queries:
        train_qids = train_qids[:args.limit_queries]
        print(f"\n  ⚠️  LIMIT: {args.limit_queries} queries")
    
    print(f"\n  Queries: {len(train_qids)}")
    
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
    print("\n" + "=" * 60)
    print("  TEST: Test queries (351-700)")
    print("=" * 60)
    
    set_all_seeds(42)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    queries = load_queries()
    expanded = load_expanded_queries()
    test_qids = get_test_qids()
    
    print(f"\n  Queries: {len(test_qids)}")
    
    run_pipeline(
        qids=test_qids,
        queries=queries,
        expanded_queries=expanded,
        output_dir=output_dir,
        args=args,
        evaluate=False,
    )


def add_common_args(parser):
    parser.add_argument("--config", "-c", help="Tuned config JSON")
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory")
    parser.add_argument("--limit-queries", type=int, default=None, help="Limit queries")
    
    # Ablation flags
    parser.add_argument("--disable-q2d", action="store_true",
                        help="Disable Q2D expansion (use only original queries)")
    parser.add_argument("--disable-llm", action="store_true",
                        help="Disable LLM reranking (Run3 = Run2)")
    
    # Dense index (required)
    parser.add_argument("--dense-index-path", type=str, required=True, help="Path to dense index")
    
    # SPLADE index (optional - significantly improves recall)
    parser.add_argument("--splade-index-path", type=str, default=None,
                        help="Path to SPLADE index (optional, but recommended for better recall)")
    
    # Retrieval weights (BM25, Dense, SPLADE) - tune these!
    parser.add_argument("--retrieval-weights", type=str, default="1.0,0.0,1.0",
                        help="Comma-separated weights for BM25,Dense,SPLADE (default: 1.0,0.0,1.0 - no Dense)")
    
    # BM25 parameters
    parser.add_argument("--bm25-k1", type=float, default=None, help="BM25 k1 (default: from config or 0.9)")
    parser.add_argument("--bm25-b", type=float, default=None, help="BM25 b (default: from config or 0.4)")
    
    # RM3 parameters
    parser.add_argument("--fb-docs", type=int, default=None, help="RM3 feedback docs (default: from config or 10)")
    parser.add_argument("--fb-terms", type=int, default=None, help="RM3 feedback terms (default: from config or 10)")
    parser.add_argument("--original-weight", type=float, default=None, help="RM3 original query weight (default: from config or 0.5)")
    
    # Retrieval
    parser.add_argument("--retrieval-k", type=int, default=2000, help="Retrieval depth per method (BM25, Dense)")
    parser.add_argument("--rerank-depth", type=int, default=1000, help="Docs to rerank")
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap")
    
    # Cascade reranking stage cutoffs (number of DOCUMENTS to keep)
    # Less aggressive = better for strong retrieval baselines
    parser.add_argument("--bi-top-k", type=int, default=400,
                        help="Docs to keep after bi-encoder stage (default: 400)")
    parser.add_argument("--ce-top-k", type=int, default=200,
                        help="Docs to keep after cross-encoder stage (default: 200)")
    
    # Neural models
    parser.add_argument("--bi-encoder-model", type=str, default="BAAI/bge-large-en-v1.5",
                        help="Bi-encoder model for stage 1")
    parser.add_argument("--ce-model", type=str, 
                        default="BAAI/bge-reranker-v2-m3,cross-encoder/ms-marco-MiniLM-L-12-v2",
                        help="Cross-encoder model(s) for stage 2, comma-separated for ensemble")
    parser.add_argument("--monot5-model", type=str, default="castorini/monot5-large-msmarco",
                        help="MonoT5 model for stage 3")
    
    # Batch sizes for each stage
    parser.add_argument("--bi-batch-size", type=int, default=512, help="Bi-encoder batch size")
    parser.add_argument("--ce-batch-size", type=int, default=256, help="Cross-encoder batch size")
    parser.add_argument("--monot5-batch-size", type=int, default=64, help="MonoT5 batch size")
    
    # Interpolation weight (neural vs retrieval)
    parser.add_argument("--neural-weight", type=float, default=0.8,
                        help="Weight for neural scores in final interpolation (default: 0.8)")
    parser.add_argument("--rrf-k", type=int, default=60)
    
    # Weighted RRF for Run 3 (Run1 + Run2 + LLM)
    parser.add_argument("--rrf-weight-run1", type=float, default=0.3, 
                        help="RRF weight for Run1 (hybrid retrieval)")
    parser.add_argument("--rrf-weight-run2", type=float, default=1.0,
                        help="RRF weight for Run2 (neural reranking) - strongest")
    parser.add_argument("--rrf-weight-llm", type=float, default=0.5,
                        help="RRF weight for LLM reranking")
    
    # Device
    parser.add_argument("--no-gpu", action="store_true")
    
    # LLM (required)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm-top-k", type=int, default=100)
    parser.add_argument("--llm-window-size", type=int, default=20)
    parser.add_argument("--llm-step-size", type=int, default=10)
    parser.add_argument("--llm-max-passage-length", type=int, default=300)
    parser.add_argument("--llm-concurrency", type=int, default=10)
    parser.add_argument("--llm-dynamic-few-shot", action="store_true",
                        help="Select few-shot examples by query similarity (recommended)")
    
    # Strong LLM for cascade (refines top-k before RRF fusion)
    parser.add_argument("--llm-strong-model", type=str, default="gpt-5",
                        help="Strong LLM for top-k refinement (default: gpt-5)")
    parser.add_argument("--llm-strong-top-k", type=int, default=10,
                        help="Docs for strong LLM to refine (default: 10)")


def main():
    parser = argparse.ArgumentParser(description="ROBUST04 Ranking Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    train_parser = subparsers.add_parser("train", help="Train + evaluate")
    add_common_args(train_parser)
    
    test_parser = subparsers.add_parser("test", help="Test submission")
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
