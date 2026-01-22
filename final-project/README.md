# ROBUST04 Document Ranking Pipeline

A high-performance document ranking system for the TREC ROBUST04 collection, achieving **0.41 MAP** on the test set through 6-way hybrid retrieval, neural reranking, and LLM cascade refinement.

## Results

### Test Set (199 queries)

| Run | MAP | NDCG | P@10 | R@1000 |
|-----|-----|------|------|--------|
| Run 1 (Hybrid) | 0.3920 | 0.6907 | 0.5930 | 0.8837 |
| Run 2 (Neural) | 0.3966 | 0.6900 | 0.5683 | 0.8837 |
| **Run 3 (LLM)** | **0.4139** | **0.7044** | **0.6015** | 0.8837 |

### Training Set (50 queries)

| Run | MAP | NDCG | P@10 | R@1000 |
|-----|-----|------|------|--------|
| Run 1 (Hybrid) | 0.3460 | 0.6895 | 0.5040 | 0.8034 |
| Run 2 (Neural) | 0.3268 | 0.6724 | 0.4980 | 0.8034 |
| **Run 3 (LLM)** | **0.3577** | **0.6981** | **0.5260** | 0.8034 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          6-Way Hybrid Retrieval                                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│  │  BM25+RM3 │ │  BM25+RM3 │ │   Dense   │ │   Dense   │ │  SPLADE   │ │  SPLADE   │
│  │ (Original)│ │   (Q2D)   │ │ (Original)│ │   (Q2D)   │ │ (Original)│ │   (Q2D)   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
│        └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
│                                        │                                        
│                                  RRF Fusion                                     
│                                        │                                        
│                                  ┌─────▼─────┐                                  
│                                  │   RUN 1   │                                  
│                                  └─────┬─────┘                                  
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│                             Neural Reranking                                    │
│                                                                                 │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│   │   Bi-Encoder    │ →  │  Cross-Encoder  │ →  │     MonoT5      │            │
│   │   (Filtering)   │    │   (Ensemble)    │    │   (Reranking)   │            │
│   │ bge-large-en    │    │ bge-reranker +  │    │ monot5-large    │            │
│   │                 │    │ ms-marco-MiniLM │    │                 │            │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                        │                                        │
│                               MaxP Aggregation                                  │
│                                        │                                        │
│                                  ┌─────▼─────┐                                  │
│                                  │   RUN 2   │                                  │
│                                  └─────┬─────┘                                  │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│                            LLM Cascade Reranking                                │
│                                                                                 │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐           │
│   │     Stage 1: gpt-4o-mini    │ →  │     Stage 2: gpt-5          │           │
│   │  Top-30 docs, sliding window│    │  Top-20 docs, final refine  │           │
│   └─────────────────────────────┘    └─────────────────────────────┘           │
│                                        │                                        │
│                        Weighted RRF: Run1 + Run2 + LLM                          │
│                                        │                                        │
│                                  ┌─────▼─────┐                                  │
│                                  │   RUN 3   │                                  │
│                                  └───────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Retrieval Methods

| Method | Model | Purpose |
|--------|-------|---------|
| BM25+RM3 | Pyserini (robust04) | Lexical matching with pseudo-relevance feedback |
| Dense | BAAI/bge-small-en-v1.5 | Semantic similarity search |
| SPLADE | naver/splade-cocondenser-ensembledistil | Learned sparse retrieval |
| Query2Doc | Claude (pre-computed) | Semantic query expansion |

### Neural Reranking

| Stage | Model | Parameters |
|-------|-------|------------|
| Bi-Encoder | BAAI/bge-large-en-v1.5 | 335M |
| Cross-Encoder | BAAI/bge-reranker-v2-m3 | 568M |
| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L-12-v2 | 33M |
| MonoT5 | castorini/monot5-large-msmarco | 770M |

### LLM Reranking

| Stage | Model | Documents | Strategy |
|-------|-------|-----------|----------|
| Bulk | gpt-4o-mini | Top-30 | Sliding window (size=10, step=5) |
| Refine | gpt-5 | Top-20 | Single pass with reasoning |

---

## Installation

```bash
git clone https://github.com/er1009/text-retrieval-and-search-engines.git
cd text-retrieval-and-search-engines/final-project

pip install pyserini faiss-cpu torch transformers sentence-transformers \
    pytrec_eval langchain-text-splitters tqdm accelerate openai pydantic

# Java required for Pyserini
apt-get install openjdk-21-jdk-headless
```

---

## Usage

### Build Indexes (First Time)

```bash
# Dense Index (~45 min)
python -m src.dense_index \
    --index-path "/path/to/dense_index" \
    --embedding-model "BAAI/bge-small-en-v1.5" \
    --chunk-size 1500 --chunk-overlap 200

# SPLADE Index (~60 min)
python -m src.splade_index \
    --index-path "/path/to/splade_index" \
    --chunk-size 1500 --chunk-overlap 200
```

### Run Pipeline

```bash
# Training (50 queries, with evaluation)
python -m src.main train \
    --output-dir results \
    --dense-index-path "/path/to/dense_index" \
    --splade-index-path "/path/to/splade_index" \
    --retrieval-k 2000 \
    --rerank-depth 500

# Test (199 queries, submission)
python -m src.main test \
    --output-dir submission \
    --dense-index-path "/path/to/dense_index" \
    --splade-index-path "/path/to/splade_index" \
    --retrieval-k 2000 \
    --rerank-depth 500
```

---

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--retrieval-k` | 2000 | Documents per retrieval method |
| `--rerank-depth` | 500 | Documents for neural reranking |
| `--rrf-k` | 60 | RRF smoothing constant |
| `--ce-weight` | 0.5 | Cross-encoder weight in ensemble |
| `--neural-weight` | 0.8 | Neural vs BM25 interpolation |

### BM25 Parameters

| Parameter | Value |
|-----------|-------|
| k1 | 0.9 |
| b | 0.4 |
| RM3 fb_docs | 10 |
| RM3 fb_terms | 10 |

---

## Project Structure

```
final-project/
├── src/
│   ├── main.py              # CLI and pipeline orchestration
│   ├── bm25_retrieval.py    # BM25+RM3 retrieval
│   ├── dense_index.py       # FAISS dense index
│   ├── splade_index.py      # SPLADE sparse index
│   ├── document_processor.py # Chunking and cleaning
│   ├── neural_reranker.py   # CE + MonoT5 reranking
│   ├── llm_reranker.py      # LLM cascade reranking
│   ├── fusion.py            # RRF fusion
│   └── evaluation.py        # Metrics computation
├── data/
│   └── expanded_queries.csv # Query2Doc expansions
├── run_pipeline.ipynb       # Colab notebook
└── README.md
```

---

## Cost Estimation

| Queries | gpt-4o-mini | gpt-5 | Total |
|---------|-------------|-------|-------|
| 50 (train) | ~$0.10 | ~$2.00 | ~$2.10 |
| 199 (test) | ~$0.40 | ~$8.00 | ~$8.40 |

---

## License

MIT License
