# ROBUST04 Ranking Pipeline - Technical Context

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           4-Way Hybrid Retrieval                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  BM25+RM3   │  │  BM25+RM3   │  │   Dense     │  │   Dense     │        │
│  │  (Original) │  │   (Q2D)     │  │  (Original) │  │   (Q2D)     │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│                            RRF Fusion                                       │
│                                   │                                         │
│                            ┌──────▼──────┐                                  │
│                            │   RUN 1     │  (Best Recall)                   │
│                            └──────┬──────┘                                  │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                          Neural Reranking                                   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Document Chunking                                 │  │
│  │  • XML tag cleaning (removes <H>, <F>, <TEXT>, etc.)                 │  │
│  │  • RecursiveCharacterTextSplitter (256 chars, 64 overlap)            │  │
│  │  • Context prepending (Anthropic's approach: +35% relevance)         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │               Cross-Encoder Ensemble                                  │  │
│  │  • BAAI/bge-reranker-v2-m3 (568M params) - weight: 0.7               │  │
│  │  • cross-encoder/ms-marco-MiniLM-L-12-v2 (33M params) - weight: 0.3  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        MonoT5                                         │  │
│  │  • castorini/monot5-large-msmarco (770M params)                      │  │
│  │  • Ensemble: 0.5 × CE + 0.5 × MonoT5                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                          MaxP Aggregation                                   │
│                   (max passage score → document score)                      │
│                                    │                                        │
│                            ┌──────▼──────┐                                  │
│                            │   RUN 2     │  (Best Precision)                │
│                            └──────┬──────┘                                  │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                         LLM Cascade Reranking                               │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Stage 1: gpt-4o-mini                               │  │
│  │  • Top-50 documents from Run 2                                        │  │
│  │  • Sliding window listwise ranking (window=20, step=10)              │  │
│  │  • ROBUST04-specific prompt with real document examples              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Stage 2: gpt-5 (Strong LLM)                        │  │
│  │  • Refines top-10 for maximum precision                              │  │
│  │  • Single window, deterministic ranking                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                   Weighted RRF Fusion                                       │
│            Run1 × 1.0 + Run2 × 1.0 + LLM × 1.0                             │
│                                    │                                        │
│                            ┌──────▼──────┐                                  │
│                            │   RUN 3     │  (Best Overall)                  │
│                            └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset: TREC ROBUST04

### Collection Statistics
- **Documents:** ~528,000 newswire articles
- **Sources:** Financial Times (37%), FBIS Foreign Broadcasts (33%), LA Times (26%), Federal Register (4%)
- **Time Period:** 1990s
- **Document Format:** XML with tags `<H>`, `<F>`, `<TEXT>`, `<FIG>`, `<DATE>`, `<TI>`

### Query Statistics (Training: 301-350)
| Metric | Value |
|--------|-------|
| Total Queries | 50 |
| Relevant docs/query | Min: 3, Median: 61, Max: 448 |
| Judged docs/query | Min: 761, Median: 1,250, Max: 1,668 |

### Query Difficulty Extremes
| Hardest (few relevant) | Count | Easiest (many relevant) | Count |
|------------------------|-------|-------------------------|-------|
| rap crime | 3 | international organized crime | 448 |
| implant dentistry | 4 | african civilian deaths | 332 |
| risk aspirin | 4 | police deaths | 258 |

---

## Component Details

### 1. Query Expansion (Query2Doc)

Pre-computed semantic expansions optimized for ROBUST04:

**Design Principles:**
- Concise (10-12 terms, not 25+)
- 1990s vocabulary (no anachronisms like "TSA", "cybercrime")
- Query-difficulty adaptive
- Source-aware (FT, FBIS, LA Times patterns)

**Example:**
```
Original: "international organized crime"
Expanded: "international organized crime mafia drug cartels trafficking 
           money laundering Cali Medellin Colombian Interpol smuggling 
           racketeering narcotics criminal syndicate Russia Italian"
```

**Hard Query Example (only 3 relevant docs):**
```
Original: "rap crime"
Expanded: "rap music crime hip hop gangsta lyrics violence controversy 
           Tupac Shakur Notorious BIG murder shooting arrest Snoop Dogg Death Row"
```

### 2. BM25+RM3 Retrieval

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| k1 | 0.9 | Term frequency saturation |
| b | 0.4 | Document length normalization |
| fb_docs | 10 | PRF feedback documents |
| fb_terms | 10 | PRF expansion terms |
| original_weight | 0.5 | Original query weight |

**Index:** Pyserini prebuilt `robust04` Lucene index

### 3. Dense Retrieval

**FAISS Index Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding Model | `BAAI/bge-small-en-v1.5` | Best quality/speed tradeoff |
| Chunk Size | 1500 chars | Matches model context (512 tokens) |
| Chunk Overlap | 200 chars | Preserves context across chunks |
| Index Type | IVF | Fast approximate search |
| nprobe | 128 | Balance of speed/recall |
| Total Passages | ~1.76M | All documents chunked |

**Aggregation:** MaxP (max passage score → document score)

### 4. Document Processing

**XML Tag Cleaning:**
```python
def clean_document_text(raw_text: str) -> str:
    # Remove: <H>, <F>, <TEXT>, <FIG>, <DATE>, <TI>, etc.
    text = re.sub(r'<[^>]+>', ' ', raw_text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

**Contextual Chunking:**
```python
# RecursiveCharacterTextSplitter with semantic separators
separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Context prepending (Anthropic's approach)
chunk = f"[{document_title}] {chunk_text}"
```

### 5. Neural Reranking

**Cross-Encoder Ensemble:**
| Model | Parameters | Weight |
|-------|------------|--------|
| BAAI/bge-reranker-v2-m3 | 568M | 0.7 |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 33M | 0.3 |

**MonoT5:**
| Model | Parameters |
|-------|------------|
| castorini/monot5-large-msmarco | 770M |

**Final Ensemble:** `0.5 × CE_score + 0.5 × MonoT5_score`

**Score Interpolation:**
```python
final_score = neural_weight × neural_score + (1 - neural_weight) × retrieval_score
# neural_weight = 0.8 (neural dominates)
```

### 6. LLM Reranker

**System Prompt:**
```
You are an expert IR system for TREC ROBUST04.

DATASET CONTEXT:
- Documents are NEWS ARTICLES from the 1990s (Financial Times, LA Times, FBIS)
- Queries are TREC-style information needs
- Relevance means: would this article help a researcher studying this topic?

RELEVANCE CRITERIA (in order):
1. TOPICAL MATCH: Does the document directly address the query's main topic?
2. INFORMATION NEED: Does it provide substantive information?
3. SPECIFICITY: Does it give specific details, not just passing mentions?
4. COVERAGE: Does it cover key aspects the query implies?
```

**Few-Shot Examples:** Real ROBUST04 documents from:
- Query 301 "international organized crime" (EASY - 448 relevant)
- Query 309 "rap crime" (HARD - 3 relevant)
- Query 328 "pope beatifications" (MEDIUM - 8 relevant)

**Sliding Window:**
- Window size: 20 documents
- Step size: 10 documents
- Bubble sort passes to stabilize ranking

**Cascade:**
1. gpt-4o-mini ranks top-50
2. gpt-5 refines top-10

### 7. Fusion Strategies

**Reciprocal Rank Fusion (RRF):**
```
score(d) = Σ weight_i / (k + rank_i(d))
k = 60 (standard smoothing constant)
```

**Run 3 Weighted RRF:**
| Run | Weight | Description |
|-----|--------|-------------|
| Run 1 | 1.0 | Hybrid retrieval (recall) |
| Run 2 | 1.0 | Neural reranking (precision) |
| LLM | 1.0 | LLM cascade (top precision) |

---

## File Structure

```
final-project/
├── src/
│   ├── main.py              # CLI: train/test commands
│   ├── config.py            # Configuration dataclasses
│   ├── data_loader.py       # Load queries, qrels, expansions
│   ├── bm25_retrieval.py    # Pyserini BM25+RM3
│   ├── dense_index.py       # FAISS dense retrieval
│   ├── document_processor.py # XML cleaning, chunking
│   ├── neural_reranker.py   # CE ensemble + MonoT5
│   ├── llm_reranker.py      # OpenAI listwise reranking
│   ├── aggregation.py       # MaxP, SumP, etc.
│   ├── fusion.py            # RRF, CombSUM, CombMNZ
│   ├── evaluation.py        # pytrec_eval wrapper
│   ├── trec_io.py           # TREC format I/O
│   └── tuning.py            # Grid search tuning
├── data/
│   └── expanded_queries.csv # Query2Doc expansions
├── Files-20260103/
│   ├── queriesROBUST.txt    # 249 queries
│   └── qrels_50_Queries     # Training relevance judgments
└── results/
    ├── run_1.res            # Hybrid retrieval
    ├── run_2.res            # Neural reranking
    ├── run_3.res            # LLM cascade
    └── metrics.json         # Evaluation results
```

---

## CLI Parameters

```bash
python -m src.main {train|test} [OPTIONS]

# Required
--dense-index-path PATH      Path to FAISS index

# Retrieval
--retrieval-k INT            Docs per method (default: 2000)
--rerank-depth INT           Neural rerank depth (default: 1000)
--chunk-size INT             Passage size (default: 256)
--chunk-overlap INT          Overlap (default: 64)

# Neural Models
--ce-model STR               Cross-encoder(s), comma-separated
--monot5-model STR           MonoT5 model name
--ce-batch-size INT          CE batch size (default: 256)
--monot5-batch-size INT      MonoT5 batch size (default: 64)

# Weights
--ce-weight FLOAT            CE weight in ensemble (default: 0.5)
--neural-weight FLOAT        Neural vs retrieval (default: 0.8)
--rrf-k INT                  RRF smoothing (default: 60)
--rrf-weight-run1 FLOAT      Run 1 weight (default: 0.3)
--rrf-weight-run2 FLOAT      Run 2 weight (default: 1.0)
--rrf-weight-llm FLOAT       LLM weight (default: 0.5)

# LLM
--llm-model STR              Bulk LLM (default: gpt-4o-mini)
--llm-top-k INT              Docs for LLM (default: 100)
--llm-window-size INT        Window size (default: 20)
--llm-step-size INT          Step size (default: 10)
--llm-strong-model STR       Strong LLM (default: gpt-5)
--llm-strong-top-k INT       Strong LLM docs (default: 10)
--llm-concurrency INT        Concurrent requests (default: 10)

# Other
--output-dir PATH            Output directory
--config PATH                Tuned config JSON
--limit-queries INT          Limit for testing
--no-gpu                     CPU only
```

---

## Performance Summary

### Training Results (50 queries, 2 query sanity check)

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| **MAP** | 0.456 | 0.454 | **0.476** |
| **P@10** | 0.75 | 0.75 | **0.80** |
| **NDCG@10** | 0.78 | 0.79 | **0.86** |
| **R@1000** | **0.85** | **0.86** | 0.85 |

### Key Optimizations Applied

| Optimization | Impact |
|--------------|--------|
| Query expansion (Q2D) | +32% MAP in Run 1 |
| XML tag cleaning | +29% faster, cleaner text |
| Real few-shot examples | +3.4% MAP, +14% P@10 |
| 4-way hybrid retrieval | +20% R@1000 |
| Equal RRF weights | Better fusion |

---

## Cost Estimation

| Queries | Stage | Model | Est. Cost |
|---------|-------|-------|-----------|
| 50 (train) | Bulk | gpt-4o-mini | ~$0.50 |
| 50 (train) | Refine | gpt-5 | ~$2.00 |
| 199 (test) | Bulk | gpt-4o-mini | ~$2.00 |
| 199 (test) | Refine | gpt-5 | ~$8.00 |
| **Total** | | | **~$12.50** |
