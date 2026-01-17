"""
LLM-based document reranking using sliding window listwise approach.

Clean implementation with:
- Pydantic models for config and validation
- OpenAI structured outputs for reliable JSON parsing
- Async API calls with rate limiting for maximum throughput
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models
# ============================================================================

class RankingOutput(BaseModel):
    """Structured output for document ranking."""
    ranking: list[int] = Field(
        description="Document indices in descending order of relevance (1-indexed)"
    )


class LLMRerankerConfig(BaseModel):
    """Configuration for LLM reranker."""
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    window_size: int = Field(default=20, ge=2, le=100, description="Docs per ranking window")
    step_size: int = Field(default=10, ge=1, description="Window step size")
    max_passage_length: int = Field(default=300, ge=50, description="Max chars per passage")
    max_concurrent_requests: int = Field(default=10, ge=1, description="Concurrent API calls (lower = less rate limiting)")
    requests_per_minute: int = Field(default=450, ge=1, description="Rate limit")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=50)

    class Config:
        frozen = True


# ============================================================================
# Prompt Template
# ============================================================================

SYSTEM_PROMPT = """You are a relevance ranking expert. Your task is to rank documents by their relevance to a search query.

Instructions:
1. Read the query carefully to understand the information need
2. Evaluate each document for topical relevance, informativeness, and query coverage
3. Return a JSON object with the ranking

Output format: {"ranking": [best_doc_index, second_best, ..., worst_doc_index]}
- Use 1-indexed document numbers
- Include ALL document indices exactly once
- Order from most relevant to least relevant"""


def build_user_prompt(query: str, passages: list[str]) -> str:
    """Build the user prompt with query and passages."""
    formatted_passages = "\n\n".join(
        f"[{i+1}] {text}" for i, text in enumerate(passages)
    )
    
    return f"""Query: {query}

Documents to rank:
{formatted_passages}

Rank all {len(passages)} documents by relevance to the query. Return JSON: {{"ranking": [...]}}"""


# ============================================================================
# Rate Limiter
# ============================================================================

class AsyncRateLimiter:
    """Token bucket rate limiter for async operations."""
    
    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute / 60.0
        self.tokens = self.rate
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                await asyncio.sleep((1 - self.tokens) / self.rate)
                self.tokens = 0
            else:
                self.tokens -= 1


# ============================================================================
# LLM Reranker
# ============================================================================

@dataclass
class RerankerStats:
    """Statistics for reranking operations."""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: int = 0


class LLMReranker:
    """
    Sliding window listwise reranker using OpenAI API.
    
    Uses structured JSON output for reliable parsing.
    """
    
    def __init__(self, config: Optional[LLMRerankerConfig] = None):
        self.config = config or LLMRerankerConfig()
        self._client = None
        self._rate_limiter = AsyncRateLimiter(self.config.requests_per_minute)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.stats = RerankerStats()
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("Install openai: pip install openai>=1.0.0")
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client
    
    def _truncate_passage(self, text: str) -> str:
        """Truncate and clean passage text."""
        text = " ".join(text.split())  # Normalize whitespace
        return text[:self.config.max_passage_length]
    
    def _parse_ranking(self, content: str, window_size: int) -> list[int]:
        """Parse ranking from JSON response, with fallback."""
        import json
        
        try:
            data = json.loads(content)
            ranking = data.get("ranking", [])
            
            # Validate and convert to 0-indexed
            result = []
            seen = set()
            for idx in ranking:
                idx_0 = int(idx) - 1
                if 0 <= idx_0 < window_size and idx_0 not in seen:
                    result.append(idx_0)
                    seen.add(idx_0)
            
            # Add missing indices
            for i in range(window_size):
                if i not in seen:
                    result.append(i)
            
            return result
            
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: return original order
            return list(range(window_size))
    
    async def _rank_window(
        self, query: str, passages: list[dict], indices: list[int]
    ) -> list[int]:
        """Rank a single window of passages."""
        passage_texts = [
            self._truncate_passage(passages[i]["text"]) 
            for i in indices
        ]
        
        user_prompt = build_user_prompt(query, passage_texts)
        
        async with self._semaphore:
            await self._rate_limiter.acquire()
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
                
                self.stats.total_requests += 1
                if response.usage:
                    self.stats.total_input_tokens += response.usage.prompt_tokens
                    self.stats.total_output_tokens += response.usage.completion_tokens
                
                content = response.choices[0].message.content or "{}"
                local_ranking = self._parse_ranking(content, len(indices))
                
                return [indices[i] for i in local_ranking]
                
            except Exception as e:
                self.stats.errors += 1
                print(f"  [LLM] API error: {e}")
                return indices  # Return original order on error
    
    async def rerank(
        self, query: str, passages: list[dict], top_k: int = 20
    ) -> list[tuple[str, float]]:
        """
        Rerank passages using sliding window approach.
        
        Args:
            query: Search query
            passages: List of {"docid": str, "text": str}
            top_k: Number of top results to return
            
        Returns:
            List of (docid, score) in reranked order
        """
        n = len(passages)
        
        if n == 0:
            return []
        
        if n <= self.config.window_size:
            # Single pass for small lists
            indices = list(range(n))
            ranked = await self._rank_window(query, passages, indices)
            return [(passages[i]["docid"], 1.0 - j / n) for j, i in enumerate(ranked[:top_k])]
        
        # Sliding window: bubble sort from bottom to top
        current_order = list(range(n))
        
        end = n
        while end > self.config.window_size:
            start = end - self.config.window_size
            window_indices = current_order[start:end]
            ranked_window = await self._rank_window(query, passages, window_indices)
            current_order[start:end] = ranked_window
            end -= self.config.step_size
        
        # Final pass on top window
        window_indices = current_order[:self.config.window_size]
        ranked_window = await self._rank_window(query, passages, window_indices)
        current_order[:self.config.window_size] = ranked_window
        
        # Assign scores based on final rank
        return [(passages[i]["docid"], 1.0 - j / top_k) for j, i in enumerate(current_order[:top_k])]


# ============================================================================
# Batch Processing
# ============================================================================

async def batch_rerank_with_llm(
    reranker: LLMReranker,
    queries: dict[str, str],
    results: dict[str, list[dict]],
    top_k: int = 20,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Batch rerank multiple queries.
    
    Args:
        reranker: LLMReranker instance
        queries: qid -> query text
        results: qid -> [{"docid": str, "text": str}]
        top_k: Results per query
        show_progress: Show progress bar
        
    Returns:
        qid -> {docid: score}
    """
    async def rerank_one(qid: str):
        query = queries.get(qid, "")
        passages = results.get(qid, [])
        
        if not passages or not query:
            return qid, {}
        
        ranked = await reranker.rerank(query, passages, top_k)
        return qid, dict(ranked)
    
    qids = list(queries.keys())
    
    if show_progress:
        try:
            from tqdm.asyncio import tqdm_asyncio
            tasks = [rerank_one(qid) for qid in qids]
            results_list = await tqdm_asyncio.gather(*tasks, desc="  [LLM] Reranking")
        except ImportError:
            results_list = await asyncio.gather(*[rerank_one(qid) for qid in qids])
    else:
        results_list = await asyncio.gather(*[rerank_one(qid) for qid in qids])
    
    return dict(results_list)


# ============================================================================
# Score Merging
# ============================================================================

def merge_llm_with_neural(
    llm_results: dict[str, dict[str, float]],
    neural_results: dict[str, dict[str, float]],
    llm_top_k: int = 20,
    llm_weight: float = 0.7,
) -> dict[str, dict[str, float]]:
    """
    Merge LLM reranked top-k with neural results.
    
    Strategy:
    - LLM-ranked docs: interpolate LLM + neural scores
    - Other docs: use penalized neural scores
    """
    merged = {}
    
    for qid, neural_scores in neural_results.items():
        llm_scores = llm_results.get(qid, {})
        final_scores = {}
        
        llm_docids = set(llm_scores.keys())
        
        for docid, llm_score in llm_scores.items():
            neural_score = neural_scores.get(docid, 0)
            # Boost LLM scores to keep top-k on top
            final_scores[docid] = llm_weight * (llm_score + 1.0) + (1 - llm_weight) * neural_score
        
        for docid, neural_score in neural_scores.items():
            if docid not in llm_docids:
                # Penalty for not being in LLM top-k
                final_scores[docid] = neural_score * 0.5
        
        merged[qid] = final_scores
    
    return merged


# ============================================================================
# Cost Estimation
# ============================================================================

def estimate_cost(
    num_queries: int,
    passages_per_query: int = 100,
    window_size: int = 20,
    step_size: int = 10,
    chars_per_passage: int = 300,
    model: str = "gpt-4o-mini",
) -> dict:
    """Estimate API cost for LLM reranking."""
    
    if passages_per_query <= window_size:
        windows_per_query = 1
    else:
        windows_per_query = 1 + (passages_per_query - window_size) // step_size + 1
    
    total_requests = num_queries * windows_per_query
    
    # Token estimates
    tokens_per_passage = chars_per_passage / 4
    input_tokens = total_requests * (window_size * tokens_per_passage + 200)
    output_tokens = total_requests * 100
    
    # Pricing (per million tokens)
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.0),
        "gpt-4.1-mini": (0.40, 1.60),
    }
    
    in_price, out_price = pricing.get(model, (0.15, 0.60))
    cost = input_tokens * in_price / 1_000_000 + output_tokens * out_price / 1_000_000
    
    return {
        "num_queries": num_queries,
        "windows_per_query": windows_per_query,
        "total_requests": total_requests,
        "total_input_tokens": int(input_tokens),
        "total_output_tokens": int(output_tokens),
        "estimated_cost_usd": round(cost, 2),
        "estimated_time_minutes": round(total_requests / 450, 1),
    }
