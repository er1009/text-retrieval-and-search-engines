# ROBUST04 Ranking Pipeline

A high-performance information retrieval system for the TREC ROBUST04 collection, achieving **MAP 0.33+** through multi-stage neural reranking and fusion.

## Architecture

```
Query → BM25+RM3 → Chunking → Bi-Encoder → Cross-Encoder → MonoT5 → MaxP → Fusion
```

| Run | Method | Description |
|-----|--------|-------------|
| 1 | **Lexical** | BM25 + RM3 + Query2Doc expansion (RRF fusion) |
| 2 | **Neural** | Three-stage reranking with CE+MonoT5 ensemble |
| 3 | **Fusion** | RRF combination of Run 1 + Run 2 |

## Results (Training Set)

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| MAP | 0.2765 | 0.3287 | **0.3341** |
| P@10 | 0.4420 | 0.5140 | **0.5180** |
| R@1000 | 0.7334 | 0.7115 | **0.7703** |

## Quick Start

### Google Colab (Recommended)

1. Open `run_pipeline.ipynb` in Colab with GPU runtime
2. Run setup cells to install dependencies
3. Execute `train` command to validate on training queries
4. Execute `test` command to generate submission files

### Local

```bash
pip install pyserini faiss-cpu torch transformers sentence-transformers \
    pytrec_eval langchain-text-splitters tqdm accelerate
```

## Usage

### Train (Evaluate on 50 training queries)

```bash
python -m src.main train \
    --output-dir results \
    --bm25-k 1500 \
    --rerank-depth 1500 \
    --chunk-size 256 \
    --chunk-overlap 64 \
    --bi-batch-size 4096 \
    --ce-batch-size 4096 \
    --monot5-batch-size 1024 \
    --ce-weight 0.5 \
    --neural-weight 0.8 \
    --rrf-k 60
```

### Test (Generate submission for 199 queries)

```bash
python -m src.main test \
    --output-dir submission \
    --bm25-k 1500 \
    --rerank-depth 1500 \
    --chunk-size 256 \
    --chunk-overlap 64 \
    --bi-batch-size 4096 \
    --ce-batch-size 4096 \
    --monot5-batch-size 1024 \
    --ce-weight 0.5 \
    --neural-weight 0.8 \
    --rrf-k 60
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bm25-k` | 1000 | Documents retrieved per query |
| `--rerank-depth` | 1000 | Documents to rerank |
| `--chunk-size` | 256 | Passage chunk size (chars) |
| `--chunk-overlap` | 64 | Overlap between chunks |
| `--bi-encoder-top-k` | auto | Passages kept after bi-encoder filtering |
| `--bi-batch-size` | 2048 | Bi-encoder batch size |
| `--ce-batch-size` | 2048 | Cross-encoder batch size |
| `--monot5-batch-size` | 512 | MonoT5 batch size |
| `--ce-weight` | 0.5 | Cross-encoder weight in CE/MonoT5 ensemble |
| `--neural-weight` | 0.8 | Neural vs BM25 interpolation weight |
| `--rrf-k` | 60 | RRF fusion parameter |
| `--no-gpu` | flag | Disable GPU |

## Project Structure

```
final-project/
├── src/
│   ├── main.py                 # CLI (train/test commands)
│   ├── config.py               # Hyperparameters
│   ├── data_loader.py          # Query/qrel loading
│   ├── bm25_retrieval.py       # BM25 + RM3 retrieval
│   ├── document_processor.py   # Contextual chunking
│   ├── neural_reranker.py      # Bi-encoder, Cross-encoder, MonoT5
│   ├── aggregation.py          # MaxP passage-to-document
│   ├── fusion.py               # RRF fusion
│   ├── evaluation.py           # MAP, NDCG, P@K metrics
│   └── trec_io.py              # TREC format I/O
├── data/
│   └── expanded_queries.csv    # Query2Doc expansions (Claude)
├── Files-20260103/
│   ├── queriesROBUST.txt       # 249 queries
│   └── qrels_50_Queries        # Training relevance judgments
└── run_pipeline.ipynb          # Colab notebook
```

## Neural Pipeline

### Stage 1: Bi-Encoder (Fast Filtering)
- Model: `BAAI/bge-base-en-v1.5`
- Filters ~50K passages → 12K passages
- Encodes query and passages independently

### Stage 2: Cross-Encoder Ensemble (Precision)
- Models: `ms-marco-MiniLM-L-6-v2` + `ms-marco-MiniLM-L-12-v2`
- Joint query-passage encoding
- Averaged ensemble scores

### Stage 3: MonoT5 (Final Reranking)
- Model: `castorini/monot5-base-msmarco`
- Sequence-to-sequence relevance prediction
- Combined with cross-encoder via weighted average

### Aggregation
- **MaxP**: Document score = max(passage scores)
- Score interpolation: `0.8 × neural + 0.2 × BM25`

## Output Format

TREC 6-column format:
```
qid Q0 docid rank score run_name
351 Q0 FBIS3-10082 1 0.9523 neural_maxp
```

## Hardware

| Setup | Run 2 Time (50 queries) |
|-------|-------------------------|
| A100 80GB | ~35 min |
| T4 16GB | ~2 hours |
| CPU | ~6 hours |

## References

- [Pyserini](https://github.com/castorini/pyserini)
- [Sentence-Transformers](https://www.sbert.net/)
- [MonoT5](https://github.com/castorini/pygaggle)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
