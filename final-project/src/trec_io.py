"""TREC format I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .bm25_retrieval import SearchResult


def write_trec_run(
    results: dict[str, list],
    output_path: Path,
    run_name: str = "run",
    max_docs: int = 1000,
) -> None:
    """
    Write results in standard TREC 6-column format.
    
    Format: qid Q0 docid rank score run_name
    
    Args:
        results: Dict mapping qid -> list of (docid, rank, score) or SearchResult
        output_path: Path to write the run file
        run_name: Name of the run (6th column)
        max_docs: Maximum documents per query
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for qid in sorted(results.keys(), key=lambda x: (len(x), x)):
            docs = results[qid][:max_docs]
            
            for i, item in enumerate(docs):
                rank = i + 1
                
                if hasattr(item, 'docid'):  # SearchResult
                    docid = item.docid
                    score = item.score
                elif isinstance(item, tuple):
                    if len(item) >= 3:
                        docid, _, score = item[0], item[1], item[2]
                    else:
                        docid, score = item
                else:
                    continue
                
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")


def write_trec_run_from_scores(
    results: dict[str, dict[str, float]],
    output_path: Path,
    run_name: str = "run",
    max_docs: int = 1000,
) -> None:
    """
    Write results from score dicts in TREC format.
    
    Args:
        results: Dict mapping qid -> {docid: score, ...}
        output_path: Path to write the run file
        run_name: Name of the run
        max_docs: Maximum documents per query
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for qid in sorted(results.keys(), key=lambda x: (len(x), x)):
            # Sort by score descending
            sorted_docs = sorted(
                results[qid].items(),
                key=lambda x: -x[1]
            )[:max_docs]
            
            for rank, (docid, score) in enumerate(sorted_docs, 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")


def read_trec_run(path: Path) -> dict[str, list[tuple[str, int, float]]]:
    """
    Read a TREC run file.
    
    Returns:
        Dict mapping qid -> list of (docid, rank, score)
    """
    results = {}
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid = parts[0]
                docid = parts[2]
                rank = int(parts[3])
                score = float(parts[4])
                
                if qid not in results:
                    results[qid] = []
                results[qid].append((docid, rank, score))
    
    # Sort by rank
    for qid in results:
        results[qid].sort(key=lambda x: x[1])
    
    return results


def read_trec_run_as_scores(path: Path) -> dict[str, dict[str, float]]:
    """
    Read a TREC run file as score dicts.
    
    Returns:
        Dict mapping qid -> {docid: score, ...}
    """
    run = read_trec_run(path)
    return {
        qid: {docid: score for docid, _, score in docs}
        for qid, docs in run.items()
    }


def merge_runs_to_submission(
    run_paths: list[Path],
    output_dir: Path,
) -> None:
    """
    Prepare runs for submission.
    
    Ensures proper naming: run_1.res, run_2.res, run_3.res
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, path in enumerate(run_paths, 1):
        run = read_trec_run(path)
        output_path = output_dir / f"run_{i}.res"
        
        with open(output_path, 'w') as f:
            for qid in sorted(run.keys(), key=lambda x: (len(x), x)):
                for docid, rank, score in run[qid]:
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} run_{i}\n")

