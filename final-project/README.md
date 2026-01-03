# ROBUST04 Ranking Competition - Ultimate Best Practices

A state-of-the-art information retrieval pipeline for the ROBUST04 ranking competition, optimized for MAP (Mean Average Precision).

## Overview

This project implements three complementary retrieval methods:

| Run | Method | Techniques | Expected MAP |
|-----|--------|------------|--------------|
| 1 | BM25+RM3+Query2Doc | Claude expansions, tuned BM25/RM3, PRF | 0.33-0.36 |
| 2 | Neural MaxP | Cross-Encoder reranking, contextual chunking | 0.37-0.40 |
| 3 | RRF Fusion | Reciprocal Rank Fusion of Run 1 & 2 | 0.40-0.44 |

## Key Features

### Query2Doc Expansion (Claude Opus 4.5)
Pre-generated semantic query expansions including:
- Synonyms and paraphrases
- Key entities (organizations, people, places)
- Domain terminology
- Related concepts

### Contextual Chunking (Anthropic's Approach)
Based on Anthropic's Contextual Retrieval research:
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Semantic separators: `["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]`
- Prepends document context to each chunk (+35% fewer retrieval failures)

### Neural Reranking
- Cross-Encoder: `ms-marco-MiniLM-L-12-v2`
- MaxP aggregation: Document score = max(passage scores)
- Score normalization for proper ensembling

### Multi-Signal Fusion
- Reciprocal Rank Fusion (RRF) with k=60
- Also supports: CombSUM, CombMNZ, weighted linear

## Project Structure

```
final-project/
├── src/
│   ├── config.py               # Hyperparameters, paths, seeds
│   ├── data_loader.py          # Load queries, qrels, expansions
│   ├── bm25_retrieval.py       # BM25, RM3, batch search
│   ├── document_processor.py   # Contextual chunking (LangChain)
│   ├── neural_reranker.py      # Cross-Encoder + MonoT5
│   ├── aggregation.py          # MaxP, SumP, FirstP, TopKP
│   ├── fusion.py               # RRF, CombSUM, CombMNZ
│   ├── evaluation.py           # MAP, NDCG, P@K
│   ├── tuning.py               # Grid search, 5-fold CV
│   ├── trec_io.py              # TREC format I/O
│   └── main.py                 # CLI orchestration
├── data/
│   └── expanded_queries.csv    # 249 Claude Query2Doc expansions
├── Files-20260103/
│   ├── queriesROBUST.txt       # 249 queries
│   └── qrels_50_Queries        # Training relevance judgments
├── results/                    # Output run files
├── tuning_results/             # Parameter tuning logs
└── run_pipeline.ipynb          # Colab notebook
```

## Installation

```bash
pip install pyserini faiss-cpu torch transformers sentence-transformers \
    pytrec_eval langchain langchain-text-splitters tqdm scikit-learn numpy
```

## Usage

### Quick Start (Colab)

1. Clone repo and open `run_pipeline.ipynb` in Google Colab
2. Run cells sequentially to generate all three runs
3. Download `Final_Project_Part_A.zip` for submission

### CLI Commands

```bash
# Parameter tuning (optional, ~30 min)
python -m src.main tune --output tuning_results/

# Generate Run 1: BM25 + RM3 + Query2Doc
python -m src.main run1 \
    --config tuning_results/best_config.json \
    --output results/run_1.res

# Generate Run 2: Neural MaxP Reranking
python -m src.main run2 \
    --config tuning_results/best_config.json \
    --output results/run_2.res \
    --rerank-depth 100 \
    --gpu

# Generate Run 3: RRF Fusion
python -m src.main run3 \
    --run1 results/run_1.res \
    --run2 results/run_2.res \
    --output results/run_3.res

# Evaluate on training queries
python -m src.main evaluate results/run_1.res results/run_2.res results/run_3.res
```

### Run All at Once

```bash
python -m src.main run_all \
    --config tuning_results/best_config.json \
    --output-dir results/ \
    --rerank-depth 100 \
    --gpu
```

## Parameters

### BM25 (Tunable)
| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| k1 | Term frequency saturation | [0.4-1.6] | 0.9 |
| b | Document length normalization | [0.2-0.8] | 0.4 |

### RM3 (Tunable)
| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| fb_docs | Feedback documents | [3-20] | 10 |
| fb_terms | Expansion terms | [5-25] | 10 |
| original_query_weight | Original vs expansion | [0.3-0.8] | 0.5 |

### Chunking
| Parameter | Value |
|-----------|-------|
| chunk_size | 256 chars |
| chunk_overlap | 64 chars |
| context_prepend | True |

## Output Format

Standard TREC 6-column format:
```
qid Q0 docid rank score run_name
351 Q0 FBIS3-10082 1 25.432100 BM25_RM3_Q2D
351 Q0 FBIS3-10169 2 24.891200 BM25_RM3_Q2D
...
```

## Evaluation Metrics

- **MAP**: Mean Average Precision (primary metric)
- **NDCG**: Normalized Discounted Cumulative Gain
- **P@10/P@20**: Precision at cutoffs
- **Recall@100/1000**: Recall at cutoffs

## Hardware Requirements

- **Minimum**: CPU with 16GB RAM
- **Recommended**: GPU (A100 80GB for fastest neural reranking)
- Neural reranking on CPU takes ~2-3 hours; GPU ~30-60 minutes

## References

- Pyserini: https://github.com/castorini/pyserini
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Cross-Encoder: https://www.sbert.net/docs/pretrained_cross-encoders.html
- RRF: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"

