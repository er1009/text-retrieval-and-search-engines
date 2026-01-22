"""Neural reranking with clean 3-stage cascade: Bi-Encoder → Cross-Encoder → MonoT5."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import T5ForConditionalGeneration, AutoTokenizer

from .document_processor import Passage


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalization per query."""
    if not scores:
        return []
    
    scores = list(scores)
    min_s = min(scores)
    max_s = max(scores)
    
    if max_s == min_s:
        return [0.5] * len(scores)
    
    return [(s - min_s) / (max_s - min_s) for s in scores]


def maxp_aggregate(passages: list[Passage], scores: list[float]) -> dict[str, float]:
    """Aggregate passage scores to document scores using MaxP strategy."""
    doc_scores = {}
    for passage, score in zip(passages, scores):
        docid = passage.docid
        doc_scores[docid] = max(doc_scores.get(docid, -float('inf')), score)
    return doc_scores


class FastBiEncoder:
    """SOTA bi-encoder for fast passage scoring."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        print(f"    Loading Bi-Encoder: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        if use_fp16 and device == "cuda":
            self.model.half()
        
        self.device = device
        self.model_name = model_name
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"    ✓ Bi-Encoder loaded ({params / 1e6:.0f}M params)")
    
    def score_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 512,
    ) -> list[float]:
        """Score all passages with bi-encoder."""
        if not passages:
            return []
        
        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        
        # Encode passages in batches
        passage_texts = [p.text for p in passages]
        passage_embs = self.model.encode(
            passage_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        # Compute similarities
        similarities = torch.matmul(passage_embs, query_emb).cpu().numpy()
        return similarities.tolist()


class FastCrossEncoder:
    """Optimized Cross-Encoder for A100 GPU."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        print(f"    Loading Cross-Encoder: {model_name}")
        
        if "bge-reranker" in model_name.lower():
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device,
                automodel_args={"torch_dtype": torch.float16 if use_fp16 else torch.float32},
            )
        else:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device,
            )
        
        self.device = device
        self.model_name = model_name
        
        params = sum(p.numel() for p in self.model.model.parameters())
        print(f"    ✓ Cross-Encoder loaded ({params / 1e6:.0f}M params)")
    
    def score_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 256,
    ) -> list[float]:
        """Score passages for a single query."""
        if not passages:
            return []
        
        pairs = [(query, p.text) for p in passages]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return scores.tolist()


class MultiCrossEncoder:
    """Ensemble of multiple cross-encoders for better accuracy."""
    
    def __init__(
        self,
        models: list[str],
        weights: list[float] | None = None,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        assert len(models) == len(weights), "Models and weights must match"
        
        print(f"    Loading {len(models)} cross-encoders for ensemble...")
        self.encoders = []
        for model_name in models:
            encoder = FastCrossEncoder(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
            )
            self.encoders.append(encoder)
        
        self.weights = weights
        self.device = device
    
    def score_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 256,
    ) -> list[float]:
        """Score passages using ensemble of cross-encoders."""
        if not passages:
            return []
        
        all_scores = []
        for encoder in self.encoders:
            scores = encoder.score_passages(query, passages, batch_size=batch_size)
            all_scores.append(scores)
        
        # Normalize and ensemble
        normalized_scores = [normalize_scores(scores) for scores in all_scores]
        
        ensemble_scores = []
        for i in range(len(passages)):
            score = sum(
                weight * norm_scores[i]
                for weight, norm_scores in zip(self.weights, normalized_scores)
            )
            ensemble_scores.append(score)
        
        return ensemble_scores


class FastMonoT5:
    """Optimized MonoT5 reranker for A100 GPU."""
    
    def __init__(
        self,
        model_name: str = "castorini/monot5-large-msmarco",
        max_length: int = 512,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        print(f"    Loading MonoT5: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        
        self.device = device
        self.max_length = max_length
        
        self.true_token_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
        self.decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(device)
        
        print(f"    ✓ MonoT5 loaded ({sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B params)")
    
    @torch.no_grad()
    def _score_batch(self, input_texts: list[str]) -> list[float]:
        """Score a batch efficiently using single decoder step."""
        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        batch_size = len(input_texts)
        decoder_inputs = self.decoder_input_ids.expand(batch_size, -1)
        
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_inputs,
        )
        
        logits = outputs.logits[:, 0, :]
        probs = torch.softmax(logits, dim=-1)
        true_probs = probs[:, self.true_token_id]
        false_probs = probs[:, self.false_token_id]
        
        scores = true_probs / (true_probs + false_probs + 1e-10)
        return scores.cpu().tolist()
    
    def score_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 64,
    ) -> list[float]:
        """Score passages for a single query."""
        if not passages:
            return []
        
        input_texts = [
            f"Query: {query} Document: {p.text} Relevant:"
            for p in passages
        ]
        
        scores = []
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i + batch_size]
            batch_scores = self._score_batch(batch_texts)
            scores.extend(batch_scores)
        
        return scores


class CascadeReranker:
    """
    Clean 3-stage cascade reranking with MaxP aggregation BEFORE document filtering.
    
    Stage 1: Bi-Encoder (all passages → MaxP → top bi_top_k docs)
    Stage 2: Cross-Encoder (top bi_top_k docs → MaxP → top ce_top_k docs)
    Stage 3: MonoT5 (top ce_top_k docs → MaxP → final ranking)
    
    Key principle: Never lose a document due to passage-level filtering.
    Each stage filters DOCUMENTS, not passages.
    """
    
    def __init__(
        self,
        # Stage cutoffs (number of DOCUMENTS to keep)
        bi_top_k: int = 500,
        ce_top_k: int = 200,
        # Models
        bi_encoder_model: str = "BAAI/bge-large-en-v1.5",
        ce_model: str | list[str] = None,
        monot5_model: str = "castorini/monot5-large-msmarco",
        # Ensemble weights
        ce_ensemble_weights: list[float] | None = None,
        # Device
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        self.bi_top_k = bi_top_k
        self.ce_top_k = ce_top_k
        
        print("=" * 60)
        print("  LOADING CASCADE RERANKER")
        print("=" * 60)
        print(f"  Stage 1: Bi-Encoder → top {bi_top_k} docs")
        print(f"  Stage 2: Cross-Encoder → top {ce_top_k} docs")
        print(f"  Stage 3: MonoT5 → final ranking")
        print("=" * 60)
        
        # Stage 1: Bi-encoder
        self.bi_encoder = FastBiEncoder(
            model_name=bi_encoder_model,
            device=device,
            use_fp16=True,
        )
        
        # Stage 2: Cross-encoder (single or ensemble)
        if ce_model is None:
            ce_model = ["BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L-12-v2"]
        
        if isinstance(ce_model, list) and len(ce_model) > 1:
            self.cross_encoder = MultiCrossEncoder(
                models=ce_model,
                weights=ce_ensemble_weights or [0.7, 0.3],
                device=device,
                use_fp16=True,
            )
        else:
            model_name = ce_model[0] if isinstance(ce_model, list) else ce_model
            self.cross_encoder = FastCrossEncoder(
                model_name=model_name,
                device=device,
                use_fp16=True,
            )
        
        # Stage 3: MonoT5
        self.monot5 = FastMonoT5(
            model_name=monot5_model,
            device=device,
            use_bf16=use_bf16,
        )
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n  Total GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        print("=" * 60)
    
    def rerank(
        self,
        query: str,
        passages: list[Passage],
        bi_batch_size: int = 512,
        ce_batch_size: int = 256,
        monot5_batch_size: int = 64,
    ) -> dict[str, dict[str, float]]:
        """
        Run 3-stage cascade reranking.
        
        Returns dict with scores from each stage:
        {
            'bi': {docid: score, ...},      # All docs
            'ce': {docid: score, ...},      # Top bi_top_k docs
            't5': {docid: score, ...},      # Top ce_top_k docs
            'final': {docid: score, ...},   # All docs, interpolated
        }
        """
        if not passages:
            return {'bi': {}, 'ce': {}, 't5': {}, 'final': {}}
        
        # Build mapping: docid → passages
        doc_passages = defaultdict(list)
        for i, p in enumerate(passages):
            doc_passages[p.docid].append((i, p))
        
        all_docids = list(doc_passages.keys())
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 1: Bi-Encoder (score ALL passages, aggregate to docs)
        # ═══════════════════════════════════════════════════════════════════
        bi_passage_scores = self.bi_encoder.score_passages(query, passages, batch_size=bi_batch_size)
        bi_doc_scores = maxp_aggregate(passages, bi_passage_scores)
        
        # Keep top bi_top_k documents
        sorted_bi = sorted(bi_doc_scores.items(), key=lambda x: -x[1])
        top_bi_docids = set(d[0] for d in sorted_bi[:self.bi_top_k])
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 2: Cross-Encoder (score passages from top bi_top_k docs)
        # ═══════════════════════════════════════════════════════════════════
        ce_passages = [p for p in passages if p.docid in top_bi_docids]
        
        if ce_passages:
            ce_passage_scores = self.cross_encoder.score_passages(query, ce_passages, batch_size=ce_batch_size)
            ce_doc_scores = maxp_aggregate(ce_passages, ce_passage_scores)
        else:
            ce_doc_scores = {}
        
        # Keep top ce_top_k documents
        sorted_ce = sorted(ce_doc_scores.items(), key=lambda x: -x[1])
        top_ce_docids = set(d[0] for d in sorted_ce[:self.ce_top_k])
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 3: MonoT5 (score passages from top ce_top_k docs)
        # ═══════════════════════════════════════════════════════════════════
        t5_passages = [p for p in passages if p.docid in top_ce_docids]
        
        if t5_passages:
            t5_passage_scores = self.monot5.score_passages(query, t5_passages, batch_size=monot5_batch_size)
            t5_doc_scores = maxp_aggregate(t5_passages, t5_passage_scores)
        else:
            t5_doc_scores = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # FINAL: Tiered ranking (use best available stage's rank)
        # ═══════════════════════════════════════════════════════════════════
        # T5 docs get ranks 1 to ce_top_k (best tier)
        # CE-only docs get ranks ce_top_k+1 to bi_top_k (middle tier)
        # Bi-only docs get ranks bi_top_k+1 onwards (lowest tier)
        
        # Get rankings within each stage
        bi_ranking = self._scores_to_ranks(bi_doc_scores)
        ce_ranking = self._scores_to_ranks(ce_doc_scores)
        t5_ranking = self._scores_to_ranks(t5_doc_scores)
        
        # Tier offsets
        t5_offset = 0                    # T5 docs: rank 1+
        ce_offset = self.ce_top_k        # CE-only: rank ce_top_k+1+
        bi_offset = self.bi_top_k        # Bi-only: rank bi_top_k+1+
        
        # Assign final ranks based on best stage reached
        final_scores = {}
        for docid in all_docids:
            if docid in t5_ranking:
                # Best tier: use T5 rank directly
                final_rank = t5_offset + t5_ranking[docid]
            elif docid in ce_ranking:
                # Middle tier: CE rank + offset
                final_rank = ce_offset + ce_ranking[docid]
            else:
                # Lowest tier: Bi rank + offset
                final_rank = bi_offset + bi_ranking.get(docid, len(bi_ranking))
            
            # Convert rank to score (higher rank = lower score)
            # Score = 1/rank so rank 1 → 1.0, rank 100 → 0.01
            final_scores[docid] = 1.0 / final_rank
        
        return {
            'bi': bi_doc_scores,
            'ce': ce_doc_scores,
            't5': t5_doc_scores,
            'final': final_scores,
        }
    
    def _scores_to_ranks(self, scores: dict[str, float]) -> dict[str, int]:
        """Convert score dict to rank dict (rank 1 = highest score)."""
        if not scores:
            return {}
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
        return {docid: rank + 1 for rank, (docid, _) in enumerate(sorted_docs)}


# Backward compatibility alias
NeuralReranker = CascadeReranker
