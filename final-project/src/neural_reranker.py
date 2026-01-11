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
    
    Use this to filter 1000 → 500-1000 docs before cross-encoder.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",  # Smaller, faster (~0.4GB)
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


class MultiCrossEncoder:
    """
    Ensemble of multiple cross-encoders for better accuracy.
    
    BEST PRACTICE: Different architectures complement each other!
    """
    
    def __init__(
        self,
        models: list[str] | None = None,
        weights: list[float] | None = None,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        if models is None:
            # Default: 3 different cross-encoders
            models = [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast 6-layer
                "cross-encoder/ms-marco-MiniLM-L-12-v2",  # 12-layer (more accurate)
                "BAAI/bge-reranker-base-en-v1",  # Different architecture
            ]
        
        if weights is None:
            # Equal weights by default
            weights = [1.0 / len(models)] * len(models)
        
        assert len(models) == len(weights), "Models and weights must match"
        assert abs(sum(weights) - 1.0) < 0.01, "Weights should sum to 1.0"
        
        print(f"  Loading {len(models)} cross-encoders for ensemble...")
        self.encoders = []
        for i, model_name in enumerate(models):
            print(f"    [{i+1}/{len(models)}] {model_name}")
            encoder = FastCrossEncoder(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
            )
            self.encoders.append(encoder)
        
        self.weights = weights
        self.device = device
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 2048,
    ) -> list[float]:
        """Rerank passages using ensemble of cross-encoders."""
        if not passages:
            return []
        
        # Get scores from all encoders
        all_scores = []
        for encoder in self.encoders:
            scores = encoder.rerank_passages(query, passages, batch_size=batch_size)
            all_scores.append(scores)
        
        # Normalize each encoder's scores
        normalized_scores = [normalize_scores(scores) for scores in all_scores]
        
        # Weighted ensemble
        ensemble_scores = []
        for i in range(len(passages)):
            score = sum(
                weight * norm_scores[i]
                for weight, norm_scores in zip(self.weights, normalized_scores)
            )
            ensemble_scores.append(score)
        
        return ensemble_scores


class FastCrossEncoder:
    """
    Optimized Cross-Encoder for A100 GPU.
    Supports both sentence-transformers models and BAAI/BGE rerankers.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast 6-layer!
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
        model_name: str = "castorini/monot5-base-msmarco",  # Base model, faster!
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
        batch_size: int = 512,  # Base model = bigger batches!
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
        batch_size: int = 512,  # Base model = bigger batches!
    ) -> list[float]:
        """Rerank passages for a single query."""
        if not passages:
            return []
        
        pairs = [(query, p.text) for p in passages]
        scores = self.predict(pairs, batch_size=batch_size, show_progress=False)
        return scores.tolist()


class DualReranker:
    """
    Ensemble of Cross-Encoder + MonoT5.
    
    SPEED-OPTIMIZED with smaller models for faster processing!
    """
    
    def __init__(
        self,
        ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast 6-layer!
        monot5_model: str = "castorini/monot5-base-msmarco",  # Base model
        ce_weight: float = 0.5,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        print("=" * 60)
        print("Loading DUAL RERANKER (MiniLM-L6 + MonoT5-base)")
        print("  SPEED-OPTIMIZED ensemble!")
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
        ce_batch_size: int = 2048,   # Small model = huge batches!
        monot5_batch_size: int = 512,  # Base model = bigger batches!
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
    Three-stage reranking pipeline - SPEED OPTIMIZED:
    
    Stage 1: Bi-Encoder (fast) - filter passages
    Stage 2: Cross-Encoder (fast 6-layer) - rerank passages  
    Stage 3: MonoT5-base - ensemble with cross-encoder
    
    Smaller models = FASTER = can process MORE docs = better recall!
    
    GPU Memory: ~3GB total (much lighter!)
    """
    
    def __init__(
        self,
        bi_encoder_model: str = "BAAI/bge-base-en-v1.5",  # Smaller, fast
        ce_model: str | list[str] = None,  # Single model or list for ensemble
        monot5_model: str = "castorini/monot5-base-msmarco",  # Base model
        ce_weight: float = 0.5,
        bi_encoder_top_k: int = 2000,  # MAXIMUM: Keep 2000 passages!
        use_ensemble_ce: bool = True,  # BEST PRACTICE: Use ensemble!
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        print("=" * 60)
        print("Loading THREE-STAGE RERANKER (BEST PRACTICES)")
        print("  Stage 1: BGE-base Bi-Encoder (fast)")
        if use_ensemble_ce:
            print("  Stage 2: Multi Cross-Encoder Ensemble (L6 + L12)")
        else:
            print("  Stage 2: Cross-Encoder")
        print("  Stage 3: MonoT5-base (ensemble)")
        print("=" * 60)
        
        self.bi_encoder = FastBiEncoder(
            model_name=bi_encoder_model,
            device=device,
            use_fp16=True,
        )
        
        # BEST PRACTICE: Use ensemble of 2 cross-encoders
        if use_ensemble_ce:
            if ce_model is None:
                ce_models = [
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast
                    "cross-encoder/ms-marco-MiniLM-L-12-v2",  # More accurate
                ]
            else:
                ce_models = ce_model if isinstance(ce_model, list) else [ce_model]
            
            self.cross_encoder = MultiCrossEncoder(
                models=ce_models,
                weights=[0.4, 0.6],  # Slightly favor L12
                device=device,
                use_fp16=True,
            )
        else:
            if ce_model is None:
                ce_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            elif isinstance(ce_model, list):
                ce_model = ce_model[0]
            
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
        bi_batch_size: int = 2048,    # Small model = HUGE batches!
        ce_batch_size: int = 2048,    # 6-layer model = HUGE batches!
        monot5_batch_size: int = 512,  # Base model = bigger batches!
        bi_weight: float = 0.3,       # Weight for bi-encoder in final score
    ) -> tuple[list[float], list[int]]:
        """
        Three-stage reranking with bi-encoder score preservation.
        
        Returns:
            Tuple of (final_scores, top_k_indices)
            Scores are for ALL input passages:
            - Top-k passages: ensemble of CE/MonoT5 + bi-encoder
            - Other passages: bi-encoder score (scaled below top-k)
        """
        if not passages:
            return [], []
        
        n_passages = len(passages)
        
        # Stage 1: Bi-encoder scores ALL passages
        bi_scores_raw, bi_sorted_indices = self.bi_encoder.rerank_passages(
            query, passages, batch_size=bi_batch_size, top_k=None  # Get ALL scores
        )
        bi_scores_norm = normalize_scores(bi_scores_raw)
        
        # Filter top-k for cross-encoder
        if n_passages > self.bi_encoder_top_k:
            top_k_indices = bi_sorted_indices[:self.bi_encoder_top_k]
            filtered_passages = [passages[i] for i in top_k_indices]
        else:
            top_k_indices = list(range(n_passages))
            filtered_passages = passages
        
        top_k_set = set(top_k_indices)
        
        # Stage 2 & 3: Cross-encoder + MonoT5 on top-k passages
        ce_raw = self.cross_encoder.rerank_passages(
            query, filtered_passages, batch_size=ce_batch_size
        )
        monot5_raw = self.monot5.rerank_passages(
            query, filtered_passages, batch_size=monot5_batch_size
        )
        
        # Normalize CE/MonoT5 scores
        ce_norm = normalize_scores(ce_raw)
        monot5_norm = normalize_scores(monot5_raw)
        
        # Ensemble CE + MonoT5
        ensemble_scores = [
            self.ce_weight * ce + (1 - self.ce_weight) * mt5
            for ce, mt5 in zip(ce_norm, monot5_norm)
        ]
        
        # Build final scores for top-k passages: ensemble + bi-encoder
        top_k_final_scores = {}
        for i, orig_idx in enumerate(top_k_indices):
            top_k_final_scores[orig_idx] = (
                (1 - bi_weight) * ensemble_scores[i] + 
                bi_weight * bi_scores_norm[orig_idx]
            )
        
        # Find minimum top-k score to anchor non-top-k passages below
        min_top_k_score = min(top_k_final_scores.values()) if top_k_final_scores else 0.5
        
        # Ensure a gap between top-k and non-top-k passages
        # Non-top-k scores will be in range [0, min_top_k_score * 0.9]
        non_top_k_ceiling = max(0.0, min_top_k_score * 0.9)
        
        # Assemble final scores for ALL passages
        final_scores = []
        for i in range(n_passages):
            if i in top_k_set:
                # Top-k: use ensemble + bi-encoder score
                final_scores.append(top_k_final_scores[i])
            else:
                # Not in top-k: scale bi-encoder to [0, non_top_k_ceiling]
                # This preserves bi-encoder ranking while keeping them below top-k
                final_scores.append(non_top_k_ceiling * bi_scores_norm[i])
        
        return final_scores, top_k_indices


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

