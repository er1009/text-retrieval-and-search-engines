# ROBUST04 Ranking Competition - Full Context for Coding Agent

## Project Overview

This is a **Text Retrieval competition** for the "Text Retrieval and Search Engines" course. The goal is to achieve the **highest MAP (Mean Average Precision)** on the ROBUST04 dataset.

### Key Requirements
- Submit **3 different retrieval runs** (run_1.res, run_2.res, run_3.res)
- Each run: top 1000 documents for **199 test queries** (351-450, 601-700 minus 672)
- Training data: 50 queries (301-350) with relevance judgments for tuning
- Format: Standard 6-column TREC format
- Tools: Pyserini for index access, GPU resources via Colab (A100 80GB)

### Competition Deadline
- Due: 22/01/26
- Presentation: 27/01/26

---

## Three Runs Implementation

### Run 1: BM25 + RM3 + Query2Doc
**Lexical retrieval with pseudo-relevance feedback and semantic query expansion**

- **BM25**: Traditional lexical matching (tunable k1, b parameters)
- **RM3**: Pseudo-relevance feedback (expands query with terms from top docs)
- **Query2Doc**: Pre-generated query expansions by Claude Opus 4.5 (stored in `data/expanded_queries.csv`)

### Run 2: Three-Stage Neural Reranking
**State-of-the-art neural pipeline**

```
BM25 (1000 docs) → Chunking → Bi-Encoder (500) → Cross-Encoder + MonoT5-3B → MaxP → Final
```

Pipeline stages:
1. **BM25**: Retrieve 1000 documents per query
2. **Chunking**: Split top 200 docs into 256-char passages with contextual prefixes
3. **Bi-Encoder** (BGE-large-en-v1.5): Fast filtering to top 500 passages
4. **Cross-Encoder** (BGE-reranker-v2-m3): Precise scoring of 500 passages
5. **MonoT5-3B**: Seq2seq relevance scoring, ensembled with cross-encoder
6. **MaxP Aggregation**: doc_score = max(passage_scores)

### Run 3: RRF Fusion
**Reciprocal Rank Fusion of Run 1 and Run 2**

- Combines lexical and neural signals
- RRF formula: score(d) = Σ(1 / (k + rank_r(d))), k=60

---

## File Structure

```
final-project/
├── Files-20260103/
│   ├── queriesROBUST.txt      # 249 queries (TAB-separated: qid\tquery)
│   └── qrels_50_Queries       # Relevance judgments for queries 301-350
├── data/
│   └── expanded_queries.csv   # Claude's Query2Doc expansions (249 queries)
├── results/
│   ├── run_1.res              # Full run (249 queries) for evaluation
│   ├── run_1_submission.res   # Submission run (199 test queries only)
│   ├── run_2.res / run_2_submission.res
│   └── run_3.res / run_3_submission.res
├── tuning_results/
│   └── best_config.json       # Tuned BM25/RM3 parameters
├── src/
│   ├── __init__.py
│   ├── config.py              # All configuration and hyperparameters
│   ├── data_loader.py         # Load queries, qrels, expanded queries
│   ├── bm25_retrieval.py      # BM25 and BM25+RM3 search functions
│   ├── document_processor.py  # Contextual chunking with LangChain
│   ├── neural_reranker.py     # Bi-Encoder, Cross-Encoder, MonoT5, ThreeStageReranker
│   ├── aggregation.py         # MaxP, SumP, etc. passage-to-doc aggregation
│   ├── fusion.py              # RRF, CombSUM, CombMNZ fusion methods
│   ├── evaluation.py          # pytrec_eval wrapper for MAP, NDCG, etc.
│   ├── tuning.py              # Grid search for BM25/RM3 parameters
│   ├── trec_io.py             # Read/write TREC format files
│   └── main.py                # CLI: tune, run1, run2, run3, run_all, evaluate
├── run_pipeline.ipynb         # Colab notebook to run everything
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Key Modules Detail

### config.py
Central configuration with dataclasses:
- `PathConfig`: File paths
- `BM25Config`: k1, b parameters and tuning ranges
- `RM3Config`: fb_docs, fb_terms, original_query_weight
- `ChunkingConfig`: chunk_size=256, overlap=64, separators
- `NeuralConfig`: Model names, batch sizes, ensemble weights

**Current optimized settings for A100 80GB:**
```python
bi_encoder_batch_size = 1024
cross_encoder_batch_size = 1024
monot5_batch_size = 256
bi_encoder_top_k = 500
rerank_depth = 200
```

### neural_reranker.py
Four main classes:

1. **FastBiEncoder**: Wraps SentenceTransformer for fast passage filtering
   - Model: `BAAI/bge-large-en-v1.5` (~1.3GB)
   - Encodes query once, docs once, dot product similarity

2. **FastCrossEncoder**: Wraps sentence_transformers.CrossEncoder
   - Model: `BAAI/bge-reranker-v2-m3` (~2.3GB, 568M params)
   - Joint query-doc encoding, very accurate

3. **FastMonoT5**: T5-based seq2seq reranker with TRUE batch processing
   - Model: `castorini/monot5-3b-msmarco` (~12GB, 3B params)
   - Scores P(true) / (P(true) + P(false)) for relevance

4. **ThreeStageReranker**: Combines all three for the full pipeline
   - Bi-encoder filters 1000→500 passages
   - Cross-encoder + MonoT5 ensemble on 500 passages
   - Returns scores mapped back to original passage indices

### document_processor.py
Implements **Contextual Chunking** (Anthropic's best practice):

1. Extract document context (title or first meaningful sentence)
2. Split with `RecursiveCharacterTextSplitter` (256 chars, 64 overlap)
3. Prepend context to each chunk: `"[Context] chunk_text"`

This improves BM25 matching by ~35% according to Anthropic's research.

### data_loader.py
Key functions:
- `get_train_qids()`: Returns ['301', '302', ..., '350']
- `get_test_qids()`: Returns ['351', ..., '450', '601', ..., '700'] minus '672'
- `load_expanded_queries()`: Reads Claude's Query2Doc expansions from CSV

---

## Important Design Decisions

### 1. Two Output Files per Run
- `run_X.res`: Contains ALL 249 queries (for evaluation on training set)
- `run_X_submission.res`: Contains only 199 TEST queries (for competition submission)

### 2. Query2Doc by Claude (not runtime LLM)
The user requested Claude Opus 4.5 generate query expansions offline. These are stored in `data/expanded_queries.csv` and loaded at runtime. This:
- Ensures high quality expansions
- Frees GPU for neural models
- Makes results reproducible

### 3. Three-Stage vs Two-Stage Reranking
Originally used Cross-Encoder + MonoT5 only. Added Bi-Encoder as first stage to:
- Process more passages (500 instead of 200)
- Maintain speed (bi-encoder is O(1) per doc after encoding)
- Better recall with same latency

### 4. Contextual Chunking
Each chunk gets document context prepended. Without this, chunks lose context and BM25/neural models perform worse.

### 5. MaxP Aggregation
For passage-to-document scoring: `doc_score = max(passage_scores)`
This works best when relevance is localized (one passage contains the answer).

---

## How to Run

### In Colab (recommended)
```python
# Clone repo
!git clone https://github.com/YOUR_USERNAME/text_retrieval.git
%cd text_retrieval/final-project

# Install dependencies
!apt-get install -qq openjdk-21-jdk-headless
%pip install pyserini transformers sentence-transformers pytrec_eval langchain-text-splitters

# Run all
!python -m src.main run_all --output-dir results --gpu --rerank-depth 200

# Evaluate
!python -m src.main evaluate results/run_1.res results/run_2.res results/run_3.res
```

### CLI Commands
```bash
# Tune parameters (optional, ~30 min)
python -m src.main tune --output tuning_results/

# Individual runs
python -m src.main run1 --output results/run_1.res
python -m src.main run2 --output results/run_2.res --rerank-depth 200 --gpu
python -m src.main run3 --output results/run_3.res

# Evaluate on training queries
python -m src.main evaluate results/run_1.res results/run_2.res results/run_3.res
```

---

## GPU Memory Usage (A100 80GB)

| Model | VRAM | Batch Size |
|-------|------|------------|
| BGE-large-en-v1.5 (bi-encoder) | ~1.3GB | 1024 |
| BGE-reranker-v2-m3 (cross-encoder) | ~2.3GB | 1024 |
| MonoT5-3B | ~12GB | 256 |
| **Total** | **~17GB** | - |

Peak usage with activations: ~40-50GB (plenty of headroom on 80GB)

---

## Expected Performance

| Run | Method | Expected MAP |
|-----|--------|--------------|
| 1 | BM25+RM3+Q2D | 0.33-0.36 |
| 2 | Three-Stage Neural | 0.44-0.50 |
| 3 | RRF Fusion | 0.47-0.52 |

---

## Submission Format

```
# TREC 6-column format
qid Q0 docid rank score run_name

# Example
351 Q0 LA042389-0012 1 0.95 run_1
351 Q0 FT931-3456 2 0.92 run_1
...
```

### Creating Submission Zip
```python
import shutil
shutil.copy('results/run_1_submission.res', 'submission/run_1.res')
shutil.copy('results/run_2_submission.res', 'submission/run_2.res')
shutil.copy('results/run_3_submission.res', 'submission/run_3.res')
!cd submission && zip ../Final_Project_Part_A.zip run_1.res run_2.res run_3.res
```

---

## Known Issues & Solutions

### 1. Pyserini Thread Contention
**Issue**: Too many threads accessing index causes NullPointerException
**Solution**: Pre-validate index once before parallel processing, use 8 threads

### 2. MonoT5 Slow Processing
**Issue**: Original implementation processed one sample at a time
**Solution**: Implemented TRUE batch processing in `FastMonoT5._score_batch()`

### 3. Query 672 Missing
**Issue**: Query 672 doesn't exist in queriesROBUST.txt
**Solution**: `get_test_qids()` explicitly removes '672'

---

## Future Improvements (if needed)

1. **ColBERT**: Could replace bi-encoder for better quality late interaction
2. **Ensemble Tuning**: Tune ce_weight on training queries
3. **Doc2Query**: Generate pseudo-queries for documents (index-time expansion)
4. **Cross-Validation**: Use k-fold CV for more robust parameter tuning
5. **Query-Specific Fusion**: Different RRF k per query type

---

## Key Files to Read First

1. `src/main.py` - Entry point, see `cmd_run1`, `cmd_run2`, `cmd_run3`
2. `src/neural_reranker.py` - The neural pipeline implementation
3. `src/document_processor.py` - Chunking logic
4. `src/config.py` - All hyperparameters
5. `run_pipeline.ipynb` - Colab execution notebook

---

## Contact / References

- **Pyserini Docs**: https://github.com/castorini/pyserini
- **BGE Models**: https://huggingface.co/BAAI
- **MonoT5**: https://huggingface.co/castorini/monot5-3b-msmarco
- **Contextual Retrieval**: Anthropic's blog post on contextual chunking

