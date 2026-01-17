"""
Dense Passage Index for Hybrid Retrieval.

Creates and manages a FAISS index over document passages for semantic search.
Supports persistence to Google Drive for reuse across sessions.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Reuse chunking logic from document_processor (DRY principle)
from .document_processor import chunk_document, Passage

logger = logging.getLogger(__name__)

# Index configuration for validation
INDEX_CONFIG = {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "chunk_size": 256,
    "chunk_overlap": 64,
    "embedding_dim": 768,
    "index_type": "IVF4096,Flat",
    "version": "2.0",
}


class DensePassageIndex:
    def __init__(
        self,
        index_path: str | Path,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        device: str | None = None,
    ):
        self.index_path = Path(index_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load config from existing index, otherwise use defaults
        if self.config_path.exists():
            with open(self.config_path) as f:
                saved = json.load(f)
            self.embedding_model_name = saved.get("embedding_model", "BAAI/bge-base-en-v1.5")
            self.chunk_size = saved.get("chunk_size", 256)
            self.chunk_overlap = saved.get("chunk_overlap", 64)
        else:
            self.embedding_model_name = embedding_model or "BAAI/bge-small-en-v1.5"
            self.chunk_size = chunk_size or 1500
            self.chunk_overlap = chunk_overlap or 200
        
        # Lazy-loaded components
        self._encoder = None
        self._index = None
        self._passages = None  # {idx: {"docid": str, "text": str}}
        self._docid_to_indices = None  # {docid: [idx1, idx2, ...]}
    
    @property
    def config_path(self) -> Path:
        return self.index_path / "config.json"
    
    @property
    def index_file_path(self) -> Path:
        return self.index_path / "index.faiss"
    
    @property
    def passages_path(self) -> Path:
        return self.index_path / "passages.pkl"
    
    def exists(self) -> bool:
        """Check if a valid index exists at the path."""
        if not self.index_path.exists():
            return False
        
        required_files = [self.config_path, self.index_file_path, self.passages_path]
        if not all(f.exists() for f in required_files):
            return False
        
        # Only check version compatibility
        try:
            with open(self.config_path) as f:
                saved_config = json.load(f)
            
            if saved_config.get("version") != INDEX_CONFIG["version"]:
                print(f"  [Index] Version mismatch: {saved_config.get('version')} != {INDEX_CONFIG['version']} (rebuild required)")
                return False
            
            return True
        except Exception as e:
            print(f"  [Index] Config validation failed: {e}")
            return False
    
    def _get_encoder(self):
        """Lazy load the sentence transformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print(f"  [Index] Loading encoder: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._encoder
    
    def _chunk_document(self, docid: str, text: str) -> list[dict]:
        """
        Chunk document using shared chunking logic from document_processor.
        
        Reuses the best-practice contextual chunking:
        - RecursiveCharacterTextSplitter with semantic separators
        - Context prepending (Anthropic's approach - 35%+ improvement)
        """
        # Use shared chunking function (DRY principle)
        passages = chunk_document(
            doc_text=text,
            docid=docid,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            prepend_context=True,
        )
        
        # Convert Passage objects to dicts for indexing
        return [
            {"docid": p.docid, "chunk_idx": p.passage_idx, "text": p.text}
            for p in passages
        ]
    
    def _iter_all_documents(self) -> Iterator[tuple[str, str]]:
        """Iterate over all documents in the ROBUST04 corpus."""
        from pyserini.search.lucene import LuceneSearcher
        
        print("  [Index] Loading document corpus...")
        searcher = LuceneSearcher.from_prebuilt_index("robust04")
        
        # Get all document IDs
        reader = searcher.object.reader
        num_docs = reader.numDocs()
        print(f"  [Index] Found {num_docs:,} documents")
        
        for i in range(num_docs):
            doc = reader.storedFields().document(i)
            docid = doc.get("id")
            raw = doc.get("raw")
            if docid and raw:
                yield docid, raw
        
        searcher.close()
    
    def build(self, batch_size: int = 2048, show_progress: bool = True):
        """Build the dense index from scratch."""
        import faiss
        
        print("\n" + "=" * 60)
        print("BUILDING DENSE PASSAGE INDEX")
        print("=" * 60)
        print(f"  Model: {self.embedding_model_name}")
        print(f"  Chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.index_path}")
        print("=" * 60)
        
        start_time = time.time()
        encoder = self._get_encoder()
        
        # Phase 1: Chunk all documents
        print("\n[1/3] Chunking documents...")
        all_passages = []
        doc_count = 0
        
        for docid, text in tqdm(self._iter_all_documents(), desc="Chunking", disable=not show_progress):
            chunks = self._chunk_document(docid, text)
            for chunk in chunks:
                chunk["idx"] = len(all_passages)
                all_passages.append(chunk)
            doc_count += 1
        
        num_passages = len(all_passages)
        print(f"  ✓ {doc_count:,} documents → {num_passages:,} passages")
        print(f"  ✓ Avg passages/doc: {num_passages / doc_count:.1f}")
        
        # Phase 2: Encode all passages
        print("\n[2/3] Encoding passages...")
        texts = [p["text"] for p in all_passages]
        
        # Use SentenceTransformer's built-in batching (more efficient)
        print(f"  Encoding {len(texts):,} passages (batch_size={batch_size})...")
        
        all_embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        embeddings = all_embeddings.astype(np.float32)
        print(f"  ✓ Embeddings shape: {embeddings.shape}")
        
        # Phase 3: Build FAISS index
        print("\n[3/3] Building FAISS index...")
        dim = embeddings.shape[1]
        
        # Use IVF for faster search
        # Rule: need ~40 points per cluster for good quality
        nlist = min(4096, num_passages // 40)
        quantizer = faiss.IndexFlatIP(dim)  # Inner product (for normalized vectors = cosine)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train on enough samples (40 * nlist recommended)
        train_size = min(nlist * 40, num_passages)
        if train_size < num_passages:
            train_indices = np.random.choice(num_passages, train_size, replace=False)
            train_embeddings = embeddings[train_indices]
        else:
            train_embeddings = embeddings
        print(f"  Training on {train_size:,} samples ({nlist} clusters)...")
        index.train(train_embeddings)
        
        # Add all vectors
        index.add(embeddings)
        print(f"  ✓ Index built: {index.ntotal:,} vectors, {nlist} clusters")
        
        # Save everything
        self._save(index, all_passages)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Index built in {elapsed / 60:.1f} minutes")
        print("=" * 60)
        
        self._index = index
        self._passages = {p["idx"]: p for p in all_passages}
        self._build_docid_mapping()
    
    def _save(self, index, passages: list[dict]):
        """Save index and metadata to disk."""
        import faiss
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_passages": len(passages),
            "embedding_dim": INDEX_CONFIG["embedding_dim"],
            "index_type": INDEX_CONFIG["index_type"],
            "version": INDEX_CONFIG["version"],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Saved config: {self.config_path}")
        
        # Save FAISS index
        faiss.write_index(index, str(self.index_file_path))
        print(f"  ✓ Saved index: {self.index_file_path}")
        
        # Save passages metadata
        passages_dict = {p["idx"]: {"docid": p["docid"], "text": p["text"]} for p in passages}
        with open(self.passages_path, "wb") as f:
            pickle.dump(passages_dict, f)
        print(f"  ✓ Saved passages: {self.passages_path}")
    
    def load(self):
        """Load existing index from disk."""
        import faiss
        
        if not self.exists():
            raise FileNotFoundError(f"No valid index found at {self.index_path}")
        
        print(f"  [Index] Loading from {self.index_path}...")
        start = time.time()
        
        # Load FAISS index
        self._index = faiss.read_index(str(self.index_file_path))
        
        # Load passages
        with open(self.passages_path, "rb") as f:
            self._passages = pickle.load(f)
        
        self._build_docid_mapping()
        
        elapsed = time.time() - start
        print(f"  ✓ Loaded {self._index.ntotal:,} passages in {elapsed:.1f}s")
    
    def _build_docid_mapping(self):
        """Build mapping from docid to passage indices."""
        self._docid_to_indices = {}
        for idx, passage in self._passages.items():
            docid = passage["docid"]
            if docid not in self._docid_to_indices:
                self._docid_to_indices[docid] = []
            self._docid_to_indices[docid].append(idx)
    
    def search(
        self,
        query: str,
        top_k: int = 1000,
        nprobe: int = 128,  # Higher = better recall
    ) -> dict[str, float]:
        """
        Search for relevant documents.
        
        Returns document scores aggregated from passage scores using MaxP.
        """
        if self._index is None:
            self.load()
        
        encoder = self._get_encoder()
        
        # Encode query
        query_embedding = encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Search
        self._index.nprobe = nprobe
        scores, indices = self._index.search(query_embedding, top_k * 4)  # Get more passages for MaxP
        
        # Aggregate to document scores (MaxP)
        doc_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            passage = self._passages.get(idx)
            if passage:
                docid = passage["docid"]
                doc_scores[docid] = max(doc_scores.get(docid, -float("inf")), float(score))
        
        # Sort and return top-k documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
        return dict(sorted_docs)
    
    def batch_search(
        self,
        queries: dict[str, str],
        top_k: int = 1000,
        nprobe: int = 128,  # Higher = better recall
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> dict[str, dict[str, float]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: {qid: query_text}
            top_k: Number of documents to return per query
            
        Returns:
            {qid: {docid: score}}
        """
        if self._index is None:
            self.load()
        
        encoder = self._get_encoder()
        qids = list(queries.keys())
        query_texts = [queries[qid] for qid in qids]
        
        # Encode all queries
        query_embeddings = encoder.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Search
        self._index.nprobe = nprobe
        all_scores, all_indices = self._index.search(query_embeddings, top_k * 4)
        
        # Process results
        results = {}
        iterator = enumerate(zip(qids, all_scores, all_indices))
        if show_progress:
            iterator = tqdm(list(iterator), desc="Aggregating")
        
        for i, (qid, scores, indices) in iterator:
            doc_scores = {}
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                passage = self._passages.get(idx)
                if passage:
                    docid = passage["docid"]
                    doc_scores[docid] = max(doc_scores.get(docid, -float("inf")), float(score))
            
            sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
            results[qid] = dict(sorted_docs)
        
        return results


def get_or_create_index(
    index_path: str | Path,
    force_rebuild: bool = False,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    chunk_size: int = 256,
    chunk_overlap: int = 64,
    batch_size: int = 2048,
) -> DensePassageIndex:
    """
    Get existing index or create new one.
    
    This is the main entry point for the notebook.
    
    Args:
        index_path: Path to store/load index (e.g., /content/drive/MyDrive/robust04_index)
        force_rebuild: If True, rebuild even if exists
        embedding_model: Sentence transformer model name
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        batch_size: Batch size for encoding
        
    Returns:
        Loaded DensePassageIndex ready for search
    """
    index = DensePassageIndex(
        index_path=index_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    if index.exists() and not force_rebuild:
        print(f"\n✓ Found existing index at {index_path}")
        index.load()
    else:
        if force_rebuild:
            print(f"\n⚠️  Force rebuild requested")
        else:
            print(f"\n⚠️  No index found at {index_path}")
        print("Building new index (this takes ~45-60 minutes on A100)...")
        index.build(batch_size=batch_size)
    
    return index


def main():
    """CLI for building the dense index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ROBUST04 dense passage index")
    parser.add_argument("--index-path", type=str, required=True, help="Path to store index")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild even if exists")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    
    args = parser.parse_args()
    
    get_or_create_index(
        index_path=args.index_path,
        force_rebuild=args.force_rebuild,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    
    print("\n✅ Done! Memory released.")


if __name__ == "__main__":
    main()
