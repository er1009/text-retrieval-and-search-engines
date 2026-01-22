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
    """Reference model for expected LLM output format (used for documentation, not enforced)."""
    ranking: list[int] = Field(
        description="Document indices in descending order of relevance (1-indexed)"
    )
    reasoning: str | None = Field(
        default=None,
        description="Optional chain-of-thought reasoning (improves quality but not parsed)"
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
    max_tokens: int = Field(default=1024, ge=50)  # More tokens for reasoning
    use_dynamic_few_shot: bool = Field(default=False, description="Select few-shot examples by query similarity")

    class Config:
        frozen = True


# ============================================================================
# Prompt Template
# ============================================================================

SYSTEM_PROMPT = """You are an expert information retrieval system for the TREC ROBUST04 collection.

DATASET CONTEXT:
- Documents are NEWS ARTICLES from the 1990s (Financial Times, LA Times, FBIS foreign broadcasts)
- Queries are TREC-style information needs (e.g., "international organized crime", "wildlife extinction")
- Relevance means: would this article help a researcher studying this topic?

RELEVANCE CRITERIA (in order of importance):
1. TOPICAL MATCH: Does the document directly address the query's main topic?
2. INFORMATION NEED: Does it provide substantive information (facts, events, analysis)?
3. SPECIFICITY: Does it give specific details, not just passing mentions?
4. COVERAGE: Does it cover key aspects the query implies?

SCORING GUIDANCE:
- HIGHLY relevant: Directly about the topic, provides substantial information
- MEDIUM relevant: Related to topic but peripheral or brief coverage
- LOW relevant: Only mentions keywords, not actually about the topic
- NOT relevant: Off-topic, even if shares some words

Output a single JSON object with this exact format:
{"reasoning": "Brief 1-sentence analysis per doc", "ranking": [best_doc_num, ..., worst_doc_num]}
- Use 1-indexed numbers, include ALL documents exactly once
- Reasoning helps you think but keep it brief (1 sentence per doc max)"""

# Few-shot examples using ACTUAL ROBUST04 documents (not synthetic!)
# These are real documents with known relevance judgments
FEW_SHOT_EXAMPLES = """
EXAMPLE 1 (Easy query - broad topic):
Query: "international organized crime"

[1] Colombian Prosecutor General Gustavo de Greiff said the U.S. Government is not interested in supplying evidence to condemn the chiefs of the drug trafficking mafia because it does not trust Colombian justice. De Greiff reaffirmed difficulties collecting evidence against the main "capos" of the Cali cartel.

[2] State of the Nation Address by President Guillermo Endara at the Legislative Palace in Panama City: Your Excellencies, president of the Legislative Assembly; president of the Supreme Court of Justice; Vice President Guillermo Ford Boyd; Mr. Ministers and Deputy Ministers...

[3] The International Drugs Control Board has charged drug traffickers are continuously infiltrating the Colombian Government and Congress. The board has information about visits attorneys of the Cali Cartel have made to Congress during discussions of the Penal Procedures Code.

{"reasoning": "Doc 1: HIGHLY - drug trafficking mafia, Cali cartel, international cooperation. Doc 3: HIGHLY - Cali Cartel infiltrating government, international board. Doc 2: NOT - government address, no crime content.", "ranking": [1, 3, 2]}

EXAMPLE 2 (Hard query - nuanced topic, few relevant):
Query: "rap crime"

[1] Two young men did not kill themselves because they heard alleged subliminal messages in the heavy metal music of Judas Priest, a judge in Reno ruled. The case examined whether rock music could cause violent behavior.

[2] Speech by President Wasmosy before Congress giving account of executive branch performance for the past 200 days of administration. Discussion of economic policy and governmental reforms.

[3] Rock musicians like Ozzy Osbourne are no more responsible than Goethe or Shakespeare for encouraging suicides. The artist was obsessed with hell, death and Satan. One of the works that made him famous inspired suicides.

{"reasoning": "Doc 1: RELEVANT - music causing crime/violence, legal case. Doc 3: RELEVANT - music and suicide/violence connection. Doc 2: NOT - government speech, completely off-topic.", "ranking": [1, 3, 2]}

EXAMPLE 3 (Medium query - specific topic):
Query: "pope beatifications"

[1] The arcane process of beatification has become the subject of controversy. Before an expected crowd of 200,000 pilgrims at St Peter's in Rome, Pope John Paul II will beatify the founder of Opus Dei, moving him one step closer to sainthood.

[2] The members of the Haitian Episcopal Conference will go to Rome to meet with Pope John Paul II. Relations between the Aristide government and the Vatican have always been strained.

[3] Fifty years after Auschwitz, the camp where Nazis killed a million people serves as memorial, museum, and tourist attraction. Some victims have been considered for beatification as martyrs.

{"reasoning": "Doc 1: HIGHLY - beatification ceremony, sainthood process, Pope John Paul II. Doc 3: MEDIUM - mentions beatification of martyrs but focus is Auschwitz tourism. Doc 2: NOT - Pope meeting but no beatification.", "ranking": [1, 3, 2]}
"""


def build_user_prompt(
    query: str, 
    passages: list[str],
    qid: str = None,
    use_dynamic_few_shot: bool = False,
) -> str:
    """
    Build the user prompt with query and passages.
    
    Args:
        query: Query text
        passages: List of passage texts
        qid: Query ID (used to exclude from few-shot selection)
        use_dynamic_few_shot: If True, select examples similar to query
    """
    formatted_passages = "\n\n".join(
        f"[{i+1}] {text}" for i, text in enumerate(passages)
    )
    
    # Use dynamic or static few-shot examples
    if use_dynamic_few_shot:
        from .few_shot_selector import DynamicFewShotSelector
        selector = DynamicFewShotSelector()
        few_shot_text = selector.get_few_shot_prompt(query, qid)
    else:
        few_shot_text = FEW_SHOT_EXAMPLES
    
    return f"""{few_shot_text}

Now rank these documents:

Query: "{query}"

Documents:
{formatted_passages}

Output the JSON with reasoning and ranking."""


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
        self, query: str, passages: list[dict], indices: list[int], qid: str = None
    ) -> list[int]:
        """Rank a single window of passages."""
        passage_texts = [
            self._truncate_passage(passages[i]["text"]) 
            for i in indices
        ]
        
        user_prompt = build_user_prompt(
            query, passage_texts, 
            qid=qid, 
            use_dynamic_few_shot=self.config.use_dynamic_few_shot
        )
        
        async with self._semaphore:
            await self._rate_limiter.acquire()
            
            try:
                # Newer models (gpt-5, o3, etc.) have different API requirements
                uses_new_api = any(x in self.config.model for x in ["gpt-5", "o3", "o4"])
                
                # Build request parameters
                request_params = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                }
                
                # Token parameter differs by model
                if uses_new_api:
                    request_params["max_completion_tokens"] = self.config.max_tokens
                    # gpt-5/o3/o4 only support temperature=1
                else:
                    request_params["max_tokens"] = self.config.max_tokens
                    request_params["temperature"] = self.config.temperature
                
                response = await self.client.chat.completions.create(**request_params)
                
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
        self, query: str, passages: list[dict], top_k: int = 20, qid: str = None
    ) -> list[tuple[str, float]]:
        """
        Rerank passages using sliding window approach.
        
        Args:
            query: Search query
            passages: List of {"docid": str, "text": str}
            top_k: Number of top results to return
            qid: Query ID (for dynamic few-shot selection)
            
        Returns:
            List of (docid, score) in reranked order
        """
        n = len(passages)
        
        if n == 0:
            return []
        
        if n <= self.config.window_size:
            # Single pass for small lists
            indices = list(range(n))
            ranked = await self._rank_window(query, passages, indices, qid=qid)
            return [(passages[i]["docid"], 1.0 - j / n) for j, i in enumerate(ranked[:top_k])]
        
        # Sliding window: bubble sort from bottom to top
        current_order = list(range(n))
        
        end = n
        while end > self.config.window_size:
            start = end - self.config.window_size
            window_indices = current_order[start:end]
            ranked_window = await self._rank_window(query, passages, window_indices, qid=qid)
            current_order[start:end] = ranked_window
            end -= self.config.step_size
        
        # Final pass on top window
        window_indices = current_order[:self.config.window_size]
        ranked_window = await self._rank_window(query, passages, window_indices, qid=qid)
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
        
        ranked = await reranker.rerank(query, passages, top_k, qid=qid)
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
    
    # Pricing (per million tokens) - input, output
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.0),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-5": (10.0, 30.0),  # Estimate based on typical pricing
        "gpt-5-mini": (2.00, 8.00),
        "gpt-5-nano": (0.50, 2.00),
        "o3": (10.0, 40.0),
        "o4-mini": (1.10, 4.40),
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
