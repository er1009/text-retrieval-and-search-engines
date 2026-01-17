"""Neural reranking with Cross-Encoder and MonoT5 ensemble."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder
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


class MultiCrossEncoder:
    """Ensemble of multiple cross-encoders for better accuracy."""
    
    def __init__(
        self,
        models: list[str] | None = None,
        weights: list[float] | None = None,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        if models is None:
            models = [
                "BAAI/bge-reranker-v2-m3",
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
            ]
        
        if weights is None:
            weights = [0.7, 0.3]  # Favor SOTA BGE-reranker-v2-m3
        
        assert len(models) == len(weights), "Models and weights must match"
        
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
        batch_size: int = 256,
    ) -> list[float]:
        """Rerank passages using ensemble of cross-encoders."""
        if not passages:
            return []
        
        all_scores = []
        for encoder in self.encoders:
            scores = encoder.rerank_passages(query, passages, batch_size=batch_size)
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


class FastCrossEncoder:
    """Optimized Cross-Encoder for A100 GPU."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        print(f"  Loading Cross-Encoder: {model_name}")
        
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
        print(f"  ✓ Cross-Encoder loaded ({params / 1e6:.0f}M params)")
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 256,
    ) -> list[float]:
        """Rerank passages for a single query."""
        if not passages:
            return []
        
        pairs = [(query, p.text) for p in passages]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return scores.tolist()


class FastMonoT5:
    """Optimized MonoT5 reranker for A100 GPU."""
    
    def __init__(
        self,
        model_name: str = "castorini/monot5-3b-msmarco",
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
            device_map="auto",
        )
        self.model.eval()
        
        self.device = device
        self.max_length = max_length
        
        self.true_token_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
        self.decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(device)
        
        print(f"  ✓ MonoT5 loaded ({sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B params)")
    
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
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        batch_size: int = 64,
    ) -> list[float]:
        """Rerank passages for a single query."""
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


class NeuralReranker:
    """
    Two-stage neural reranking: Cross-Encoder + MonoT5 ensemble.
    
    No bi-encoder - assumes semantic filtering is done upstream
    (either by dense index in hybrid mode, or we process all passages).
    """
    
    def __init__(
        self,
        ce_model: str | list[str] = None,
        monot5_model: str = "castorini/monot5-3b-msmarco",
        ce_weight: float = 0.5,
        use_ensemble_ce: bool = True,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        # Determine CE model names for logging
        if ce_model is None:
            ce_display = "bge-reranker-v2-m3 + MiniLM-L12" if use_ensemble_ce else "bge-reranker-v2-m3"
        elif isinstance(ce_model, list):
            ce_display = " + ".join(m.split("/")[-1] for m in ce_model)
        else:
            ce_display = ce_model.split("/")[-1]
        
        print("=" * 60)
        print("Loading NEURAL RERANKER")
        print(f"  Stage 1: Cross-Encoder ({ce_display})")
        print(f"  Stage 2: {monot5_model.split('/')[-1]}")
        print("=" * 60)
        
        if use_ensemble_ce:
            if ce_model is None:
                ce_models = [
                    "BAAI/bge-reranker-v2-m3",
                    "cross-encoder/ms-marco-MiniLM-L-12-v2",
                ]
            else:
                ce_models = ce_model if isinstance(ce_model, list) else [ce_model]
            
            self.cross_encoder = MultiCrossEncoder(
                models=ce_models,
                weights=[0.7, 0.3],
                device=device,
                use_fp16=True,
            )
        else:
            if ce_model is None:
                ce_model = "BAAI/bge-reranker-v2-m3"
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
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n  Total GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            print("=" * 60)
    
    def rerank_passages(
        self,
        query: str,
        passages: list[Passage],
        ce_batch_size: int = 256,
        monot5_batch_size: int = 64,
    ) -> list[float]:
        """
        Rerank passages using CE + MonoT5 ensemble.
        
        Returns scores for ALL input passages.
        """
        if not passages:
            return []
        
        # Cross-encoder scoring
        ce_raw = self.cross_encoder.rerank_passages(query, passages, batch_size=ce_batch_size)
        
        # MonoT5 scoring
        monot5_raw = self.monot5.rerank_passages(query, passages, batch_size=monot5_batch_size)
        
        # Normalize and ensemble
        ce_norm = normalize_scores(ce_raw)
        monot5_norm = normalize_scores(monot5_raw)
        
        ensemble_scores = [
            self.ce_weight * ce + (1 - self.ce_weight) * mt5
            for ce, mt5 in zip(ce_norm, monot5_norm)
        ]
        
        return ensemble_scores
