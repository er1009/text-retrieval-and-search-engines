"""Neural reranking with Bi-Encoder, Cross-Encoder, and MonoT5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer

from .config import NeuralConfig, DEFAULT_CONFIG
from .document_processor import Passage


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """
    Min-max normalization per query.
    
    CRITICAL: Different models have different score ranges.
    Must normalize before ensembling!
    """
    if not scores:
        return []
    
    scores = list(scores)
    min_s = min(scores)
    max_s = max(scores)
    
    if max_s == min_s:
        return [0.5] * len(scores)
    
    return [(s - min_s) / (max_s - min_s) for s in scores]


class FastBiEncoder:
    """
    Fast Bi-Encoder for initial reranking stage.
    
    Bi-encoders encode query and documents SEPARATELY, then compute similarity.
    This is MUCH faster than cross-encoders for large candidate sets.
    
    Use this to filter 1000 → 300-500 docs before expensive cross-encoder.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",  # Best English bi-encoder
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        print(f"  Loading Bi-Encoder: {model_name}")
        
        self.model = SentenceTransformer(model_name, device=device)
        if use_fp16 and device == "cuda":
            self.model.half()  # FP16 for speed
        
        self.device = device
        self.model_name = model_name
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  ✓ Bi-Encoder loaded ({params / 1e6:.0f}M params)")
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 1024,  # Bi-encoder is FAST, max it out!
        top_k: int | None = None,
    ) -> tuple[list[float], list[int]]:
        """
        Rerank passages using bi-encoder similarity.
        
        Returns:
            Tuple of (scores, indices_sorted_by_score)
            If top_k is set, only returns top_k passages.
        """
        if not passages:
            return [], []
        
        # Encode query once
        query_emb = self.model.encode(
            query, 
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        # Encode all passages in batches
        passage_texts = [p.text for p in passages]
        passage_embs = self.model.encode(
            passage_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        # Compute cosine similarity (dot product since normalized)
        scores = torch.matmul(passage_embs, query_emb).cpu().numpy()
        
        # Get sorted indices
        sorted_indices = np.argsort(-scores)  # Descending
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        return scores.tolist(), sorted_indices.tolist()
    
    def filter_passages(
        self,
        query: str,
        passages: list[Passage],
        top_k: int = 500,
        batch_size: int = 1024,  # Bi-encoder is FAST!
    ) -> tuple[list[Passage], list[float]]:
        """
        Filter passages to top_k using bi-encoder.
        
        Returns:
            Tuple of (filtered_passages, their_scores)
        """
        if len(passages) <= top_k:
            # No filtering needed
            scores, _ = self.rerank_passages(query, passages, batch_size)
            return passages, scores
        
        scores, sorted_indices = self.rerank_passages(
            query, passages, batch_size, top_k=top_k
        )
        
        filtered_passages = [passages[i] for i in sorted_indices]
        filtered_scores = [scores[i] for i in sorted_indices]
        
        return filtered_passages, filtered_scores


class FastCrossEncoder:
    """
    Optimized Cross-Encoder for A100 GPU.
    Supports both sentence-transformers models and BAAI/BGE rerankers.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",  # State-of-the-art!
        max_length: int = 512,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        print(f"  Loading Cross-Encoder: {model_name}")
        
        # BGE rerankers need special handling for FP16
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
        
        # Count parameters
        params = sum(p.numel() for p in self.model.model.parameters())
        print(f"  ✓ Cross-Encoder loaded ({params / 1e6:.0f}M params)")
    
    def predict(
        self,
        query_passage_pairs: list[tuple[str, str]],
        batch_size: int = 1024,  # A100 80GB: huge batches!
        show_progress: bool = True,
    ) -> np.ndarray:
        """Score query-passage pairs in batches."""
        return self.model.predict(
            query_passage_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 1024,  # A100 80GB: huge batches!
    ) -> list[float]:
        """Rerank passages for a single query."""
        if not passages:
            return []
        
        pairs = [(query, p.text) for p in passages]
        scores = self.predict(pairs, batch_size=batch_size, show_progress=False)
        return scores.tolist()


class FastMonoT5:
    """
    Optimized MonoT5 reranker for A100 GPU.
    Uses TRUE batch processing for maximum GPU utilization!
    """
    
    def __init__(
        self,
        model_name: str = "castorini/monot5-3b-msmarco",  # 3B for quality
        max_length: int = 512,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        print(f"  Loading MonoT5: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",  # Auto device placement for large models
        )
        self.model.eval()
        
        self.device = device
        self.max_length = max_length
        
        # Token IDs for "true" and "false"
        self.true_token_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
        
        # Pre-compute decoder input (just the start token)
        self.decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(device)
        
        print(f"  ✓ MonoT5 loaded ({sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B params)")
    
    @torch.no_grad()
    def _score_batch(self, input_texts: list[str]) -> list[float]:
        """
        Score a batch of query-passage pairs efficiently.
        Uses encoder-only forward pass + single decoder step for speed!
        """
        # Tokenize all inputs at once
        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        batch_size = len(input_texts)
        
        # Expand decoder input for batch
        decoder_inputs = self.decoder_input_ids.expand(batch_size, -1)
        
        # Single forward pass for entire batch!
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_inputs,
        )
        
        # Get logits for first generated position
        logits = outputs.logits[:, 0, :]  # [batch_size, vocab_size]
        
        # Get probabilities for true/false tokens
        probs = torch.softmax(logits, dim=-1)
        true_probs = probs[:, self.true_token_id]
        false_probs = probs[:, self.false_token_id]
        
        # Score is P(true) / (P(true) + P(false))
        scores = true_probs / (true_probs + false_probs + 1e-10)
        
        return scores.cpu().tolist()
    
    def predict(
        self,
        query_passage_pairs: list[tuple[str, str]],
        batch_size: int = 256,  # A100 80GB can do 256 with MonoT5-3B!
        show_progress: bool = True,
    ) -> np.ndarray:
        """Score query-passage pairs in TRUE batches - fully utilizing GPU!"""
        scores = []
        
        # Format inputs for MonoT5
        input_texts = [
            f"Query: {query} Document: {passage} Relevant:"
            for query, passage in query_passage_pairs
        ]
        
        iterator = range(0, len(input_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="MonoT5 scoring", total=len(input_texts) // batch_size + 1)
        
        for i in iterator:
            batch_texts = input_texts[i:i + batch_size]
            batch_scores = self._score_batch(batch_texts)
            scores.extend(batch_scores)
        
        return np.array(scores)
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 256,  # A100 80GB can do 256!
    ) -> list[float]:
        """Rerank passages for a single query."""
        if not passages:
            return []
        
        pairs = [(query, p.text) for p in passages]
        scores = self.predict(pairs, batch_size=batch_size, show_progress=False)
        return scores.tolist()


class DualReranker:
    """
    Ensemble of BGE-Reranker-v2-m3 + MonoT5-3B.
    
    Optimized for A100 80GB - uses state-of-the-art models!
    BGE-Reranker: ~2.3GB VRAM
    MonoT5-3B: ~12GB VRAM
    Total: ~15GB VRAM (plenty of headroom on 80GB!)
    """
    
    def __init__(
        self,
        ce_model: str = "BAAI/bge-reranker-v2-m3",  # State-of-the-art!
        monot5_model: str = "castorini/monot5-3b-msmarco",  # 3B for quality!
        ce_weight: float = 0.5,  # Both are strong, balanced weight
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        print("=" * 60)
        print("Loading DUAL RERANKER (BGE-v2-m3 + MonoT5-3B)")
        print("  State-of-the-art ensemble for maximum quality!")
        print("=" * 60)
        
        self.cross_encoder = FastCrossEncoder(
            model_name=ce_model,
            device=device,
            use_fp16=True,  # FP16 for BGE
        )
        
        self.monot5 = FastMonoT5(
            model_name=monot5_model,
            device=device,
            use_bf16=use_bf16,
        )
        
        self.ce_weight = ce_weight
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n  Total GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            print("=" * 60)
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        ce_batch_size: int = 1024,   # A100 80GB: huge batches!
        monot5_batch_size: int = 256,  # A100 80GB can do 256!
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Rerank passages using both models.
        
        Returns:
            Tuple of (ce_scores, monot5_scores, ensemble_scores)
            All scores are normalized to [0, 1].
        """
        if not passages:
            return [], [], []
        
        # Get scores from both models (run in sequence, both use GPU)
        ce_raw = self.cross_encoder.rerank_passages(
            query, passages, batch_size=ce_batch_size
        )
        monot5_raw = self.monot5.rerank_passages(
            query, passages, batch_size=monot5_batch_size
        )
        
        # Normalize scores
        ce_norm = normalize_scores(ce_raw)
        monot5_norm = normalize_scores(monot5_raw)
        
        # Ensemble: weighted combination
        ensemble = [
            self.ce_weight * ce + (1 - self.ce_weight) * mt5
            for ce, mt5 in zip(ce_norm, monot5_norm)
        ]
        
        return ce_norm, monot5_norm, ensemble


class ThreeStageReranker:
    """
    Three-stage reranking pipeline for MAXIMUM quality:
    
    Stage 1: Bi-Encoder (fast) - filter 1000 → 500 passages
    Stage 2: Cross-Encoder (medium) - rerank 500 passages  
    Stage 3: MonoT5-3B (slow but best) - ensemble with cross-encoder
    
    This is the gold standard for neural IR!
    
    GPU Memory: ~17GB total (BGE-large + BGE-reranker + MonoT5-3B)
    """
    
    def __init__(
        self,
        bi_encoder_model: str = "BAAI/bge-large-en-v1.5",
        ce_model: str = "BAAI/bge-reranker-v2-m3",
        monot5_model: str = "castorini/monot5-3b-msmarco",
        ce_weight: float = 0.5,
        bi_encoder_top_k: int = 500,  # Filter to top-k before cross-encoder
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        print("=" * 60)
        print("Loading THREE-STAGE RERANKER")
        print("  Stage 1: BGE Bi-Encoder (fast filtering)")
        print("  Stage 2: BGE Cross-Encoder (precision)")
        print("  Stage 3: MonoT5-3B (ensemble)")
        print("=" * 60)
        
        self.bi_encoder = FastBiEncoder(
            model_name=bi_encoder_model,
            device=device,
            use_fp16=True,
        )
        
        self.cross_encoder = FastCrossEncoder(
            model_name=ce_model,
            device=device,
            use_fp16=True,
        )
        
        self.monot5 = FastMonoT5(
            model_name=monot5_model,
            device=device,
            use_bf16=use_bf16,
        )
        
        self.ce_weight = ce_weight
        self.bi_encoder_top_k = bi_encoder_top_k
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n  Total GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            print("=" * 60)
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        bi_batch_size: int = 1024,    # Bi-encoder is VERY fast, max it out!
        ce_batch_size: int = 1024,    # A100 80GB can handle this easily
        monot5_batch_size: int = 256,  # MonoT5-3B with 80GB can do 256
    ) -> tuple[list[float], list[int]]:
        """
        Three-stage reranking.
        
        Returns:
            Tuple of (final_scores, original_indices)
            Scores are for ALL input passages (not just top-k).
            Passages not in top-k get score of 0.
        """
        if not passages:
            return [], []
        
        n_passages = len(passages)
        
        # Stage 1: Bi-encoder filtering (fast)
        if n_passages > self.bi_encoder_top_k:
            _, bi_sorted_indices = self.bi_encoder.rerank_passages(
                query, passages, batch_size=bi_batch_size, top_k=self.bi_encoder_top_k
            )
            filtered_passages = [passages[i] for i in bi_sorted_indices]
            original_indices = bi_sorted_indices
        else:
            filtered_passages = passages
            original_indices = list(range(n_passages))
        
        # Stage 2 & 3: Cross-encoder + MonoT5 ensemble
        ce_raw = self.cross_encoder.rerank_passages(
            query, filtered_passages, batch_size=ce_batch_size
        )
        monot5_raw = self.monot5.rerank_passages(
            query, filtered_passages, batch_size=monot5_batch_size
        )
        
        # Normalize and ensemble
        ce_norm = normalize_scores(ce_raw)
        monot5_norm = normalize_scores(monot5_raw)
        
        ensemble_scores = [
            self.ce_weight * ce + (1 - self.ce_weight) * mt5
            for ce, mt5 in zip(ce_norm, monot5_norm)
        ]
        
        # Map back to original indices
        # Passages not in filtered set get score of 0
        final_scores = [0.0] * n_passages
        for i, orig_idx in enumerate(original_indices):
            final_scores[orig_idx] = ensemble_scores[i]
        
        return final_scores, original_indices


def create_reranker(config: NeuralConfig | None = None) -> DualReranker:
    """Factory function to create a reranker with config."""
    if config is None:
        config = DEFAULT_CONFIG.neural
    
    return DualReranker(
        ce_model=config.cross_encoder_model,
        monot5_model=config.monot5_model,
        ce_weight=config.ce_weight,
        device=config.device,
        use_bf16=config.use_bf16,
    )

