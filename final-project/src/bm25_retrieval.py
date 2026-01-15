"""BM25 and BM25+RM3 retrieval using pyserini."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

from pyserini.search.lucene import LuceneSearcher

from .config import BM25Config, RM3Config, DEFAULT_CONFIG


# Thread-safe singleton for the index (avoid race conditions)
_index_lock = threading.Lock()
_cached_index_path: str | None = None


@dataclass
class SearchResult:
    """A single search result."""
    docid: str
    rank: int
    score: float


def _ensure_index_cached(index_name: str = "robust04") -> None:
    """Ensure the index is downloaded and cached (thread-safe, called once)."""
    global _cached_index_path
    with _index_lock:
        if _cached_index_path is None:
            # This triggers download/validation ONCE
            searcher = LuceneSearcher.from_prebuilt_index(index_name)
            _cached_index_path = index_name
            searcher.close()


def get_searcher(index_name: str = "robust04") -> LuceneSearcher:
    """Get a LuceneSearcher for the ROBUST04 index."""
    # After first call to _ensure_index_cached, this is safe to call from threads
    return LuceneSearcher.from_prebuilt_index(index_name)


def search_bm25(
    query: str,
    searcher: LuceneSearcher | None = None,
    k: int = 1000,
    k1: float = 0.9,
    b: float = 0.4,
) -> list[SearchResult]:
    """
    Search using BM25 with specified parameters.
    
    Args:
        query: Query text
        searcher: Optional pre-initialized searcher
        k: Number of results to retrieve
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (document length normalization)
    
    Returns:
        List of SearchResult objects
    """
    close_searcher = False
    if searcher is None:
        searcher = get_searcher()
        close_searcher = True
    
    searcher.set_bm25(k1=k1, b=b)
    
    # Disable RM3 for pure BM25
    searcher.unset_rm3()
    
    hits = searcher.search(query, k=k)
    
    results = [
        SearchResult(docid=hit.docid, rank=i + 1, score=hit.score)
        for i, hit in enumerate(hits)
    ]
    
    if close_searcher:
        searcher.close()
    
    return results


def search_bm25_rm3(
    query: str,
    searcher: LuceneSearcher | None = None,
    k: int = 1000,
    k1: float = 0.9,
    b: float = 0.4,
    fb_docs: int = 10,
    fb_terms: int = 10,
    original_query_weight: float = 0.5,
) -> list[SearchResult]:
    """
    Search using BM25 + RM3 pseudo-relevance feedback.
    
    Args:
        query: Query text
        searcher: Optional pre-initialized searcher
        k: Number of results to retrieve
        k1, b: BM25 parameters
        fb_docs: Number of feedback documents for RM3
        fb_terms: Number of expansion terms for RM3
        original_query_weight: Weight of original query vs expansion
    
    Returns:
        List of SearchResult objects
    """
    close_searcher = False
    if searcher is None:
        searcher = get_searcher()
        close_searcher = True
    
    searcher.set_bm25(k1=k1, b=b)
    searcher.set_rm3(
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        original_query_weight=original_query_weight,
    )
    
    hits = searcher.search(query, k=k)
    
    results = [
        SearchResult(docid=hit.docid, rank=i + 1, score=hit.score)
        for i, hit in enumerate(hits)
    ]
    
    if close_searcher:
        searcher.close()
    
    return results


def batch_search_bm25(
    queries: dict[str, str],
    k: int = 1000,
    k1: float = 0.9,
    b: float = 0.4,
    num_threads: int = 8,
) -> dict[str, list[SearchResult]]:
    """
    Batch search using BM25 with multi-threading.
    
    Args:
        queries: Dict mapping qid -> query text
        k: Number of results per query
        k1, b: BM25 parameters
        num_threads: Number of parallel threads
    
    Returns:
        Dict mapping qid -> list of SearchResult
    """
    # CRITICAL: Ensure index is cached BEFORE spawning threads
    _ensure_index_cached()
    
    # Create a searcher per thread to avoid contention
    def search_single(qid_query):
        qid, query = qid_query
        searcher = get_searcher()
        searcher.set_bm25(k1=k1, b=b)
        searcher.unset_rm3()
        hits = searcher.search(query, k=k)
        searcher.close()
        return qid, [
            SearchResult(docid=hit.docid, rank=i + 1, score=hit.score)
            for i, hit in enumerate(hits)
        ]
    
    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for qid, search_results in executor.map(search_single, queries.items()):
            results[qid] = search_results
    
    return results


def batch_search_bm25_rm3(
    queries: dict[str, str],
    k: int = 1000,
    k1: float = 0.9,
    b: float = 0.4,
    fb_docs: int = 10,
    fb_terms: int = 10,
    original_query_weight: float = 0.5,
    num_threads: int = 8,
) -> dict[str, list[SearchResult]]:
    """
    Batch search using BM25+RM3 with multi-threading.
    """
    # CRITICAL: Ensure index is cached BEFORE spawning threads
    _ensure_index_cached()
    
    def search_single(qid_query):
        qid, query = qid_query
        searcher = get_searcher()
        searcher.set_bm25(k1=k1, b=b)
        searcher.set_rm3(
            fb_docs=fb_docs,
            fb_terms=fb_terms,
            original_query_weight=original_query_weight,
        )
        hits = searcher.search(query, k=k)
        searcher.close()
        return qid, [
            SearchResult(docid=hit.docid, rank=i + 1, score=hit.score)
            for i, hit in enumerate(hits)
        ]
    
    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for qid, search_results in executor.map(search_single, queries.items()):
            results[qid] = search_results
    
    return results


def get_document_text(docid: str, searcher: LuceneSearcher | None = None) -> str:
    """
    Retrieve document text by docid.
    
    Returns the raw document content.
    """
    close_searcher = False
    if searcher is None:
        searcher = get_searcher()
        close_searcher = True
    
    doc = searcher.doc(docid)
    text = ""
    if doc:
        raw = doc.raw()
        if raw:
            text = raw
    
    if close_searcher:
        searcher.close()
    
    return text


def batch_get_documents(
    docids: list[str],
    searcher: LuceneSearcher | None = None,
) -> dict[str, str]:
    """
    Retrieve multiple documents by docid.
    
    Returns dict mapping docid -> document text.
    """
    close_searcher = False
    if searcher is None:
        searcher = get_searcher()
        close_searcher = True
    
    docs = {}
    for docid in docids:
        doc = searcher.doc(docid)
        if doc:
            raw = doc.raw()
            if raw:
                docs[docid] = raw
    
    if close_searcher:
        searcher.close()
    
    return docs

