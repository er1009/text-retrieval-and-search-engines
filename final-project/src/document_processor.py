"""Document processing: fetching and contextual chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import ChunkingConfig, DEFAULT_CONFIG
from .bm25_retrieval import get_searcher, batch_get_documents


@dataclass
class Passage:
    """A passage (chunk) from a document."""
    docid: str
    passage_idx: int
    text: str
    context: str  # Document title/context prepended


def extract_document_context(doc_text: str, max_length: int = 150) -> str:
    """
    Extract document context (title or first meaningful sentence).
    
    This is based on Anthropic's Contextual Retrieval approach:
    prepending context improves BM25 matching by 35%+.
    """
    if not doc_text:
        return ""
    
    # Try to extract from common document structures
    lines = doc_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and very short lines
        if not line or len(line) < 10:
            continue
        
        # Skip XML/HTML tags
        if line.startswith('<') and line.endswith('>'):
            continue
        
        # Found a meaningful line - use as context
        if len(line) > max_length:
            # Find first sentence end
            for end_char in ['. ', '? ', '! ']:
                idx = line.find(end_char)
                if 20 < idx < max_length:
                    return line[:idx + 1]
            return line[:max_length]
        
        return line
    
    return ""


def chunk_document(
    doc_text: str,
    docid: str,
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
    prepend_context: bool = True,
) -> list[Passage]:
    """
    Split document into passages with contextual chunking.
    
    Implements best practices:
    1. Use RecursiveCharacterTextSplitter with semantic separators
    2. Prepend document context to each chunk (Anthropic's approach)
    
    Args:
        doc_text: Raw document text
        docid: Document ID
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        separators: Priority list of separators
        prepend_context: Whether to prepend document context
    
    Returns:
        List of Passage objects
    """
    if not doc_text or not doc_text.strip():
        return []
    
    if separators is None:
        separators = DEFAULT_CONFIG.chunking.separators
    
    # Extract context for contextual chunking
    context = extract_document_context(doc_text) if prepend_context else ""
    
    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )
    
    # Split text
    chunks = splitter.split_text(doc_text)
    
    # Create passages with context
    passages = []
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Prepend context if available and chunk doesn't already contain it
        if context and prepend_context:
            if not chunk.startswith(context[:30]):
                chunk = f"[{context}] {chunk}"
        
        passages.append(Passage(
            docid=docid,
            passage_idx=idx,
            text=chunk,
            context=context,
        ))
    
    return passages


def batch_chunk_documents(
    docids: list[str],
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    prepend_context: bool = True,
) -> dict[str, list[Passage]]:
    """
    Fetch and chunk multiple documents.
    
    Args:
        docids: List of document IDs to process
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        prepend_context: Whether to prepend context
    
    Returns:
        Dict mapping docid -> list of Passage objects
    """
    # Fetch all documents
    docs = batch_get_documents(docids)
    
    # Chunk each document
    result = {}
    for docid, doc_text in docs.items():
        passages = chunk_document(
            doc_text=doc_text,
            docid=docid,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            prepend_context=prepend_context,
        )
        result[docid] = passages
    
    return result


def get_all_passages_for_query(
    search_results: list,  # List of SearchResult
    top_k: int = 100,
    chunk_size: int = 256,
    chunk_overlap: int = 64,
) -> tuple[list[Passage], dict[str, list[int]]]:
    """
    Get all passages for top-k documents from search results.
    
    Returns:
        Tuple of:
        - List of all passages
        - Dict mapping docid -> list of passage indices in the passages list
    """
    # Get top-k docids
    docids = [r.docid for r in search_results[:top_k]]
    
    # Chunk all documents
    doc_passages = batch_chunk_documents(
        docids=docids,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # Flatten to list and track indices
    all_passages = []
    doc_to_passage_indices = {}
    
    for docid in docids:
        if docid not in doc_passages:
            continue
        
        passages = doc_passages[docid]
        indices = []
        for passage in passages:
            indices.append(len(all_passages))
            all_passages.append(passage)
        
        doc_to_passage_indices[docid] = indices
    
    return all_passages, doc_to_passage_indices

