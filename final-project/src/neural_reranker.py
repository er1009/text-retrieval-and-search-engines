"""Neural reranking with Cross-Encoder and MonoT5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder
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


class FastCrossEncoder:
    """Optimized Cross-Encoder for A100 GPU with large batch sizes."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        max_length: int = 512,
        device: str = "cuda",
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        self.device = device
    
    def predict(
        self,
        query_passage_pairs: list[tuple[str, str]],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Score query-passage pairs.
        
        A100 80GB can handle batch_size=256 easily.
        """
        return self.model.predict(
            query_passage_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 256,
    ) -> list[float]:
        """
        Rerank passages for a single query.
        
        Returns list of scores aligned with passage indices.
        """
        if not passages:
            return []
        
        pairs = [(query, p.text) for p in passages]
        scores = self.predict(pairs, batch_size=batch_size, show_progress=False)
        return scores.tolist()


class FastMonoT5:
    """Optimized MonoT5 reranker for A100 GPU."""
    
    def __init__(
        self,
        model_name: str = "castorini/monot5-base-msmarco",
        max_length: int = 512,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        
        self.device = device
        self.max_length = max_length
        
        # Token IDs for "true" and "false"
        self.true_token_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
    
    def _score_single(self, query: str, passage: str) -> float:
        """Score a single query-passage pair."""
        input_text = f"Query: {query} Document: {passage} Relevant:"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
            
            # Get logits for first generated token
            logits = outputs.scores[0][0]
            
            # Get probabilities for true/false tokens
            true_prob = torch.softmax(logits, dim=-1)[self.true_token_id].item()
            false_prob = torch.softmax(logits, dim=-1)[self.false_token_id].item()
            
            # Score is the probability of "true"
            score = true_prob / (true_prob + false_prob + 1e-10)
        
        return score
    
    def predict(
        self,
        query_passage_pairs: list[tuple[str, str]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Score query-passage pairs in batches.
        """
        scores = []
        
        iterator = range(0, len(query_passage_pairs), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="MonoT5 scoring")
        
        for i in iterator:
            batch = query_passage_pairs[i:i + batch_size]
            batch_scores = []
            
            # Process batch
            for query, passage in batch:
                score = self._score_single(query, passage)
                batch_scores.append(score)
            
            scores.extend(batch_scores)
        
        return np.array(scores)
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 64,
    ) -> list[float]:
        """
        Rerank passages for a single query.
        """
        if not passages:
            return []
        
        pairs = [(query, p.text) for p in passages]
        scores = self.predict(pairs, batch_size=batch_size, show_progress=False)
        return scores.tolist()


class DualReranker:
    """
    Ensemble of Cross-Encoder and MonoT5.
    
    Combines scores from both models with normalization.
    """
    
    def __init__(
        self,
        ce_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        monot5_model: str = "castorini/monot5-base-msmarco",
        ce_weight: float = 0.5,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        self.cross_encoder = FastCrossEncoder(
            model_name=ce_model,
            device=device,
        )
        self.monot5 = FastMonoT5(
            model_name=monot5_model,
            device=device,
            use_bf16=use_bf16,
        )
        self.ce_weight = ce_weight
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        ce_batch_size: int = 256,
        monot5_batch_size: int = 64,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Rerank passages using both models.
        
        Returns:
            Tuple of (ce_scores, monot5_scores, ensemble_scores)
            All scores are normalized to [0, 1].
        """
        if not passages:
            return [], [], []
        
        # Get scores from both models
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

