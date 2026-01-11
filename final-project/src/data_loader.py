"""Data loading utilities for queries, qrels, and expanded queries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .config import PathConfig, DEFAULT_CONFIG


def load_queries(path: Path | None = None) -> dict[str, str]:
    """
    Load queries from ROBUST format file.
    
    Format: qid<TAB>query_text
    
    Returns:
        dict mapping qid -> query text
    """
    if path is None:
        path = DEFAULT_CONFIG.paths.queries_file
    
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                qid, text = parts
                queries[qid] = text
    
    return queries


def load_expanded_queries(path: Path | None = None) -> dict[str, str]:
    """
    Load Claude's pre-generated query expansions.
    
    Format: CSV with columns qid, original_query, expanded_query
    
    Returns:
        dict mapping qid -> expanded query text
    """
    if path is None:
        path = DEFAULT_CONFIG.paths.expanded_queries_file
    
    expanded = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"]
            expanded[qid] = row["expanded_query"]
    
    return expanded


def load_qrels(path: Path | None = None) -> dict[str, dict[str, int]]:
    """
    Load relevance judgments in TREC qrels format.
    
    Format: qid 0 docid relevance
    
    Returns:
        Nested dict: qrels[qid][docid] = relevance (0, 1, or 2)
    """
    if path is None:
        path = DEFAULT_CONFIG.paths.qrels_file
    
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = rel
    
    return qrels


def iter_queries(path: Path | None = None) -> Iterator[tuple[str, str]]:
    """
    Iterate over queries yielding (qid, query_text) tuples.
    """
    queries = load_queries(path)
    for qid, text in queries.items():
        yield qid, text


def get_train_qids() -> list[str]:
    """Get training query IDs (301-350)."""
    return [str(i) for i in range(301, 351)]


def get_test_qids() -> list[str]:
    """Get test query IDs (351-450, 601-700)."""
    test_qids = [str(i) for i in range(351, 451)]
    test_qids.extend([str(i) for i in range(601, 701)])
    # Remove 672 which doesn't exist in the query file
    if "672" in test_qids:
        test_qids.remove("672")
    return test_qids


def get_all_qids() -> list[str]:
    """Get all query IDs in order."""
    return get_train_qids() + get_test_qids()


def filter_qrels_by_qids(qrels: dict[str, dict[str, int]], 
                         qids: list[str]) -> dict[str, dict[str, int]]:
    """Filter qrels to only include specified query IDs."""
    return {qid: rels for qid, rels in qrels.items() if qid in qids}

