# ROBUST04 Ranking Pipeline

A high-performance document ranking system for the TREC ROBUST04 collection, optimized for Mean Average Precision (MAP).

## Architecture Overview

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

## Key Components

### 1. Query Expansion (Query2Doc)

Pre-computed semantic query expansions using Claude, optimized for 1990s news vocabulary:

```
Original: "international organized crime"
Expanded: "international organized crime mafia drug cartels trafficking 
           money laundering Cali Medellin Colombian Interpol smuggling 
           racketeering narcotics criminal syndicate Russia Italian"
```

**Optimization Details:**
- Concise expansions (10-12 terms, not 25+)
- 1990s-appropriate vocabulary (no anachronisms like "TSA")
- Query-difficulty adaptive (hard queries get specific terms)
- Source-aware (FT, FBIS, LA Times vocabulary patterns)

### 2. Dense Index

FAISS IVF index for semantic retrieval:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding Model | `BAAI/bge-small-en-v1.5` | Best quality/speed tradeoff |
| Chunk Size | 1500 chars | Matches model context window |
| Chunk Overlap | 200 chars | Preserves context |
| Index Type | IVF (nprobe=128) | Fast approximate search |
| Aggregation | MaxP | Max passage score → doc score |

### 3. Document Processing

XML tag cleaning for ROBUST04 documents:

```python
def clean_document_text(raw_text: str) -> str:
    # Remove XML tags: <H>, <F>, <TEXT>, <FIG>, <DATE>, <TI>, etc.
    text = re.sub(r'<[^>]+>', ' ', raw_text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

**Impact:** +29% faster processing, cleaner text for neural models.

### 4. LLM Reranker

Sliding window listwise reranking with ROBUST04-specific prompting:

```python
SYSTEM_PROMPT = """You are an expert IR system for TREC ROBUST04.

DATASET CONTEXT:
- Documents are NEWS ARTICLES from the 1990s (Financial Times, LA Times, FBIS)
- Queries are TREC-style information needs
- Relevance means: would this article help a researcher studying this topic?

RELEVANCE CRITERIA:
1. TOPICAL MATCH: Does the document directly address the query's main topic?
2. INFORMATION NEED: Does it provide substantive information?
3. SPECIFICITY: Does it give specific details, not just passing mentions?
4. COVERAGE: Does it cover key aspects the query implies?
"""
```

**Few-Shot Examples:** Uses REAL ROBUST04 documents from different difficulty levels:
- Easy: Query 301 "international organized crime" (448 relevant docs)
- Hard: Query 309 "rap crime" (only 3 relevant docs)
- Medium: Query 328 "pope beatifications" (8 relevant docs)

## Installation

```bash
# Clone repository
git clone https://github.com/er1009/text-retrieval-and-search-engines.git
cd text-retrieval-and-search-engines/final-project

# Install dependencies
pip install pyserini faiss-cpu torch transformers sentence-transformers \
    pytrec_eval langchain-text-splitters tqdm accelerate openai pydantic

# Java required for Pyserini
apt-get install openjdk-21-jdk-headless
```

## Usage

### Quick Start (Colab)

```python
# Set OpenAI API key
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'

# Build dense index (first time only, ~45 min)
!python -m src.dense_index \
    --index-path "/content/drive/MyDrive/robust04_dense_index" \
    --embedding-model "BAAI/bge-small-en-v1.5" \
    --chunk-size 1500 \
    --chunk-overlap 200 \
    --batch-size 1024

# Train (50 queries, evaluate)
!python -m src.main train \
    --output-dir results \
    --dense-index-path "/content/drive/MyDrive/robust04_dense_index" \
    --retrieval-k 2000 \
    --rerank-depth 1000 \
    --rrf-weight-run1 1.0 \
    --rrf-weight-run2 1.0 \
    --rrf-weight-llm 1.0

# Test (submission)
!python -m src.main test \
    --output-dir submission \
    --dense-index-path "/content/drive/MyDrive/robust04_dense_index" \
    --retrieval-k 2000 \
    --rerank-depth 1000
```

### CLI Reference

```bash
python -m src.main {train|test} [OPTIONS]

Commands:
  train    Run on training queries (301-350), evaluate against qrels
  test     Run on test queries (351-700), generate submission files

Required:
  --dense-index-path PATH    Path to FAISS dense index

Retrieval:
  --retrieval-k INT          Docs per retrieval method (default: 2000)
  --rerank-depth INT         Docs for neural reranking (default: 1000)

Neural Models:
  --ce-model STR             Cross-encoder model(s), comma-separated
  --monot5-model STR         MonoT5 model (default: monot5-large-msmarco)
  --ce-batch-size INT        Cross-encoder batch size (default: 256)
  --monot5-batch-size INT    MonoT5 batch size (default: 64)

Fusion Weights:
  --rrf-k INT                RRF smoothing constant (default: 60)
  --rrf-weight-run1 FLOAT    Weight for Run 1 (default: 0.3)
  --rrf-weight-run2 FLOAT    Weight for Run 2 (default: 1.0)
  --rrf-weight-llm FLOAT     Weight for LLM (default: 0.5)

LLM Reranking:
  --llm-model STR            LLM for bulk reranking (default: gpt-4o-mini)
  --llm-top-k INT            Docs for LLM reranking (default: 100)
  --llm-window-size INT      Sliding window size (default: 20)
  --llm-step-size INT        Window step (default: 10)
  --llm-strong-model STR     Strong LLM for top-k (default: gpt-5)
  --llm-strong-top-k INT     Docs for strong LLM (default: 10)

Other:
  --output-dir PATH          Output directory (default: results)
  --config PATH              Tuned config JSON file
  --limit-queries INT        Limit queries for testing
  --no-gpu                   Disable GPU
```

## Project Structure

```
final-project/
├── src/
│   ├── main.py              # CLI and pipeline orchestration
│   ├── config.py            # Configuration dataclasses
│   ├── data_loader.py       # Query and qrels loading
│   ├── bm25_retrieval.py    # BM25+RM3 search
│   ├── dense_index.py       # FAISS dense retrieval
│   ├── document_processor.py # XML cleaning & chunking
│   ├── neural_reranker.py   # CE + MonoT5 reranking
│   ├── llm_reranker.py      # LLM listwise reranking
│   ├── aggregation.py       # MaxP passage aggregation
│   ├── fusion.py            # RRF and other fusion methods
│   ├── evaluation.py        # pytrec_eval metrics
│   ├── trec_io.py           # TREC format I/O
│   └── tuning.py            # Hyperparameter tuning
├── data/
│   └── expanded_queries.csv # Query2Doc expansions
├── Files-20260103/
│   ├── queriesROBUST.txt    # Original queries
│   └── qrels_50_Queries     # Relevance judgments
├── results/
│   ├── run_1.res            # Hybrid retrieval results
│   ├── run_2.res            # Neural reranking results
│   ├── run_3.res            # LLM cascade results
│   └── metrics.json         # Evaluation metrics
├── run_pipeline.ipynb       # Colab notebook
├── requirements.txt
└── README.md
```

## Performance

### Training Set Results (50 queries)

| Run | MAP | P@10 | R@1000 | Description |
|-----|-----|------|--------|-------------|
| Run 1 | 0.456 | 0.75 | 0.85 | 4-way hybrid retrieval |
| Run 2 | 0.454 | 0.75 | 0.86 | + Neural reranking |
| Run 3 | **0.476** | **0.80** | 0.85 | + LLM cascade |

### Key Optimizations

1. **Query Expansion** (+32% MAP in Run 1)
   - Concise, 1990s-appropriate expansions
   - Query-difficulty adaptive

2. **XML Tag Cleaning** (+29% faster)
   - Removes noisy document markup
   - Cleaner input for neural models

3. **Real Few-Shot Examples** (+3.4% MAP, +14% P@10)
   - Uses actual ROBUST04 documents
   - Covers different difficulty levels

4. **4-Way Hybrid Retrieval** (+20% R@1000)
   - BM25+RM3 (original + expanded)
   - Dense (original + expanded)
   - RRF fusion maximizes recall

## Evaluation Metrics

The pipeline reports comprehensive metrics:

```
MAP          Mean Average Precision (primary metric)
NDCG         Normalized Discounted Cumulative Gain
NDCG@10/100  NDCG at rank cutoffs
P@10/100/500/1000    Precision at rank cutoffs
R@10/100/500/1000    Recall at rank cutoffs
```

## Configuration

### BM25 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| k1 | 0.9 | Term frequency saturation |
| b | 0.4 | Document length normalization |

### RM3 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| fb_docs | 10 | Feedback documents |
| fb_terms | 10 | Expansion terms |
| original_weight | 0.5 | Original query weight |

### Neural Reranking

| Parameter | Value | Description |
|-----------|-------|-------------|
| chunk_size | 256 | Characters per passage |
| chunk_overlap | 64 | Overlap between passages |
| ce_weight | 0.5 | Cross-encoder weight in ensemble |
| neural_weight | 0.8 | Neural vs retrieval score interpolation |

## Cost Estimation

LLM reranking costs (approximate):

| Queries | Model | Est. Cost |
|---------|-------|-----------|
| 50 (train) | gpt-4o-mini | ~$0.50 |
| 50 (train) | gpt-5 (top-10) | ~$2.00 |
| 199 (test) | gpt-4o-mini | ~$2.00 |
| 199 (test) | gpt-5 (top-10) | ~$8.00 |

## License

MIT License
