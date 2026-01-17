"""
SPLADE Index for Learned Sparse Retrieval.

Creates and manages a SPLADE index over document passages.
SPLADE learns term expansion and importance, fixing specificity issues
that BM25 and dense retrieval miss (e.g., "new" in "new hydroelectric projects").
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from .document_processor import chunk_document

# Index configuration
SPLADE_CONFIG = {
    "model_name": "naver/splade-cocondenser-ensembledistil",
    "max_length": 256,
    "version": "1.0",
}


class SpladeEncoder:
    """Encodes text to SPLADE sparse vectors."""
    
    def __init__(
        self,
        model_name: str = SPLADE_CONFIG["model_name"],
        device: str = None,
        max_length: int = SPLADE_CONFIG["max_length"],
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        print(f"  [SPLADE] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.vocab_size = self.tokenizer.vocab_size
        print(f"  [SPLADE] ✓ Loaded (vocab_size={self.vocab_size}, device={self.device})")
    
    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> sparse.csr_matrix:
        """Encode texts to sparse vectors (batch)."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        output = self.model(**inputs)
        
        # SPLADE: log(1 + ReLU(logits)) aggregated over sequence
        # Shape: [batch, seq_len, vocab_size] -> [batch, vocab_size]
        logits = output.logits
        splade_vecs = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        )[0]
        
        # Convert to scipy sparse (much more memory efficient)
        vecs_np = splade_vecs.cpu().numpy()
        return sparse.csr_matrix(vecs_np)
    
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> sparse.csr_matrix:
        """Encode texts to sparse vectors with batching."""
        all_vecs = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding SPLADE", total=len(texts) // batch_size + 1)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            vecs = self.encode_batch(batch)
            all_vecs.append(vecs)
        
        return sparse.vstack(all_vecs)


class SpladeIndex:
    """SPLADE sparse index for ROBUST04."""
    
    def __init__(
        self,
        index_path: str | Path,
        model_name: str = SPLADE_CONFIG["model_name"],
        chunk_size: int = 1500,  # Larger chunks for SPLADE (256 token limit)
        chunk_overlap: int = 200,
        device: str = None,
    ):
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lazy-loaded components
        self._encoder = None
        self._index = None  # sparse.csr_matrix
        self._passages = None  # {idx: {"docid": str, "text": str}}
        self._docid_to_indices = None
    
    @property
    def config_path(self) -> Path:
        return self.index_path / "config.json"
    
    @property
    def index_file_path(self) -> Path:
        return self.index_path / "splade_index.npz"
    
    @property
    def passages_path(self) -> Path:
        return self.index_path / "passages.pkl"
    
    def exists(self) -> bool:
        """Check if a valid index exists."""
        if not self.index_path.exists():
            return False
        
        required = [self.config_path, self.index_file_path, self.passages_path]
        if not all(f.exists() for f in required):
            return False
        
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            if config.get("version") != SPLADE_CONFIG["version"]:
                print(f"  [SPLADE] Version mismatch, rebuild required")
                return False
            return True
        except Exception as e:
            print(f"  [SPLADE] Config error: {e}")
            return False
    
    def _get_encoder(self) -> SpladeEncoder:
        if self._encoder is None:
            self._encoder = SpladeEncoder(
                model_name=self.model_name,
                device=self.device,
            )
        return self._encoder
    
    def _iter_all_documents(self) -> Iterator[tuple[str, str]]:
        """Iterate over all ROBUST04 documents."""
        from pyserini.search.lucene import LuceneSearcher
        
        print("  [SPLADE] Loading ROBUST04 corpus...")
        searcher = LuceneSearcher.from_prebuilt_index("robust04")
        
        reader = searcher.object.reader
        num_docs = reader.numDocs()
        print(f"  [SPLADE] Found {num_docs:,} documents")
        
        for i in range(num_docs):
            doc = reader.storedFields().document(i)
            docid = doc.get("id")
            raw = doc.get("raw")
            if docid and raw:
                yield docid, raw
        
        searcher.close()
    
    def _chunk_document(self, docid: str, text: str) -> list[dict]:
        """Chunk document using shared chunking logic."""
        passages = chunk_document(
            doc_text=text,
            docid=docid,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            prepend_context=True,
        )
        return [
            {"docid": p.docid, "chunk_idx": p.passage_idx, "text": p.text}
            for p in passages
        ]
    
    def build(self, batch_size: int = 64, show_progress: bool = True):
        """Build the SPLADE index from scratch."""
        print("\n" + "=" * 60)
        print("BUILDING SPLADE INDEX")
        print("=" * 60)
        print(f"  Model: {self.model_name}")
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
        
        # Phase 2: Encode with SPLADE
        print("\n[2/3] Encoding with SPLADE...")
        texts = [p["text"] for p in all_passages]
        
        sparse_matrix = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        
        print(f"  ✓ Sparse matrix: {sparse_matrix.shape}, nnz={sparse_matrix.nnz:,}")
        print(f"  ✓ Sparsity: {1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4f}")
        
        # Phase 3: Save
        print("\n[3/3] Saving index...")
        self._save(sparse_matrix, all_passages)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SPLADE index built in {elapsed / 60:.1f} minutes")
        print("=" * 60)
        
        self._index = sparse_matrix
        self._passages = {p["idx"]: p for p in all_passages}
        self._build_docid_mapping()
    
    def _save(self, index: sparse.csr_matrix, passages: list[dict]):
        """Save index and metadata."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_passages": len(passages),
            "vocab_size": index.shape[1],
            "nnz": index.nnz,
            "version": SPLADE_CONFIG["version"],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Saved config: {self.config_path}")
        
        # Save sparse matrix
        sparse.save_npz(self.index_file_path, index)
        print(f"  ✓ Saved index: {self.index_file_path}")
        
        # Save passages
        passages_dict = {p["idx"]: {"docid": p["docid"], "text": p["text"]} for p in passages}
        with open(self.passages_path, "wb") as f:
            pickle.dump(passages_dict, f)
        print(f"  ✓ Saved passages: {self.passages_path}")
    
    def load(self):
        """Load existing index."""
        if not self.exists():
            raise FileNotFoundError(f"No valid SPLADE index at {self.index_path}")
        
        print(f"  [SPLADE] Loading from {self.index_path}...")
        start = time.time()
        
        # Load config
        with open(self.config_path) as f:
            config = json.load(f)
        self.model_name = config.get("model_name", self.model_name)
        self.chunk_size = config.get("chunk_size", self.chunk_size)
        self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
        
        # Load sparse matrix
        self._index = sparse.load_npz(self.index_file_path)
        
        # Load passages
        with open(self.passages_path, "rb") as f:
            self._passages = pickle.load(f)
        
        self._build_docid_mapping()
        
        elapsed = time.time() - start
        print(f"  ✓ Loaded {self._index.shape[0]:,} passages in {elapsed:.1f}s")
    
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
    ) -> dict[str, float]:
        """Search for relevant documents."""
        if self._index is None:
            self.load()
        
        encoder = self._get_encoder()
        
        # Encode query
        query_vec = encoder.encode_batch([query])  # [1, vocab_size]
        
        # Sparse dot product: [1, vocab] x [num_passages, vocab].T = [1, num_passages]
        scores = (query_vec @ self._index.T).toarray()[0]
        
        # Get top passage indices
        top_indices = np.argsort(scores)[::-1][:top_k * 3]  # Get more for MaxP
        
        # Aggregate to document scores (MaxP)
        doc_scores = {}
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            passage = self._passages.get(idx)
            if passage:
                docid = passage["docid"]
                doc_scores[docid] = max(doc_scores.get(docid, -float("inf")), float(scores[idx]))
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
        return dict(sorted_docs)
    
    def batch_search(
        self,
        queries: dict[str, str],
        top_k: int = 1000,
        show_progress: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Batch search for multiple queries."""
        if self._index is None:
            self.load()
        
        encoder = self._get_encoder()
        qids = list(queries.keys())
        query_texts = [queries[qid] for qid in qids]
        
        # Encode all queries
        query_vecs = encoder.encode(query_texts, batch_size=32, show_progress=False)
        
        # Batch sparse dot product
        all_scores = (query_vecs @ self._index.T).toarray()  # [num_queries, num_passages]
        
        # Process results
        results = {}
        iterator = enumerate(zip(qids, all_scores))
        if show_progress:
            iterator = tqdm(list(iterator), desc="SPLADE Aggregating")
        
        for i, (qid, scores) in iterator:
            top_indices = np.argsort(scores)[::-1][:top_k * 3]
            
            doc_scores = {}
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                passage = self._passages.get(idx)
                if passage:
                    docid = passage["docid"]
                    doc_scores[docid] = max(doc_scores.get(docid, -float("inf")), float(scores[idx]))
            
            sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
            results[qid] = dict(sorted_docs)
        
        return results


def get_or_create_splade_index(
    index_path: str | Path,
    force_rebuild: bool = False,
    model_name: str = SPLADE_CONFIG["model_name"],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    batch_size: int = 64,
) -> SpladeIndex:
    """Get existing SPLADE index or create new one."""
    index = SpladeIndex(
        index_path=index_path,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    if index.exists() and not force_rebuild:
        print(f"\n✓ Found existing SPLADE index at {index_path}")
        index.load()
    else:
        if force_rebuild:
            print(f"\n⚠️  Force rebuild requested")
        else:
            print(f"\n⚠️  No SPLADE index found at {index_path}")
        print("Building new SPLADE index (this takes ~1.5-2 hours on A100)...")
        index.build(batch_size=batch_size)
    
    return index


def main():
    """CLI for building the SPLADE index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ROBUST04 SPLADE index")
    parser.add_argument("--index-path", type=str, required=True, help="Path to store index")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild even if exists")
    parser.add_argument("--model-name", type=str, default=SPLADE_CONFIG["model_name"])
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    
    args = parser.parse_args()
    
    get_or_create_splade_index(
        index_path=args.index_path,
        force_rebuild=args.force_rebuild,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
