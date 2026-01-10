# Best Practices for Maximum MAP on ROBUST04

Based on state-of-the-art research and competition-winning strategies.

## 🎯 Target: MAP > 0.35

## 1. Multiple BM25 Variants + Fusion (Run 1 Improvement)

**Strategy**: Run BM25 with different parameters and fuse them

```python
# Run 1a: BM25 with tuned params + RM3
run1a = bm25_rm3(k1=0.9, b=0.4, fb_docs=10, fb_terms=10)

# Run 1b: BM25 with different params (more aggressive)
run1b = bm25_rm3(k1=1.2, b=0.75, fb_docs=15, fb_terms=15)

# Run 1c: BM25 without RM3 (pure lexical)
run1c = bm25(k1=0.9, b=0.4)

# Fuse all three
run1_final = rrf([run1a, run1b, run1c], k=60)
```

**Expected gain**: +0.02-0.03 MAP

## 2. Query Expansion: Original + Expanded Fusion

**Strategy**: Use BOTH original and expanded queries, fuse results

```python
# Run with original queries
results_orig = bm25_rm3(queries=original_queries, ...)

# Run with expanded queries  
results_exp = bm25_rm3(queries=expanded_queries, ...)

# Fuse
results = rrf([results_orig, results_exp], k=60)
```

**Expected gain**: +0.01-0.02 MAP

## 3. Ensemble Multiple Cross-Encoders (Run 2 Improvement)

**Strategy**: Use 2-3 different cross-encoders and ensemble

```python
# Cross-encoder 1: Fast 6-layer
ce1 = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores1 = ce1.predict(pairs)

# Cross-encoder 2: 12-layer (more accurate)
ce2 = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
scores2 = ce2.predict(pairs)

# Cross-encoder 3: BGE (different architecture)
ce3 = CrossEncoder('BAAI/bge-reranker-base-en-v1')
scores3 = ce3.predict(pairs)

# Ensemble (normalize first!)
scores1_norm = normalize(scores1)
scores2_norm = normalize(scores2)
scores3_norm = normalize(scores3)

final_scores = 0.4 * scores1_norm + 0.3 * scores2_norm + 0.3 * scores3_norm
```

**Expected gain**: +0.02-0.03 MAP

## 4. Learning-to-Rank Features

**Strategy**: Combine multiple signals with learned weights

```python
features = {
    'bm25_score': bm25_scores,
    'neural_score': neural_scores,
    'doc_length': doc_lengths,
    'query_length': query_lengths,
    'term_overlap': term_overlaps,
    'rm3_score': rm3_scores,
}

# Train simple linear model on training queries
weights = train_ltr(features, qrels_train)

# Apply to test
final_scores = sum(w * f for w, f in zip(weights, features))
```

**Expected gain**: +0.02-0.04 MAP

## 5. Adaptive Rerank Depth

**Strategy**: Rerank more docs for difficult queries

```python
# Short queries are harder - rerank more
if len(query.split()) < 5:
    rerank_depth = 1000
elif len(query.split()) < 10:
    rerank_depth = 700
else:
    rerank_depth = 500
```

**Expected gain**: +0.01-0.02 MAP

## 6. Multiple Passage Aggregation Strategies

**Strategy**: Try different aggregation methods and ensemble

```python
# MaxP (best single passage)
maxp_scores = [max(passage_scores) for doc in docs]

# SumP (multiple relevant passages)
sump_scores = [sum(passage_scores) for doc in docs]

# TopKP (top-k passages)
topkp_scores = [mean(topk(passage_scores, k=3)) for doc in docs]

# Ensemble
doc_scores = 0.5 * maxp + 0.3 * sump + 0.2 * topkp
```

**Expected gain**: +0.01-0.02 MAP

## 7. Increase Coverage Dramatically

**Strategy**: Rerank MANY more documents

```python
# Current: 500 docs → ~15K passages → 1000 passages
# Better: 1000 docs → ~30K passages → 2000 passages

rerank_depth = 1000  # Rerank top 1000 docs!
bi_encoder_top_k = 2000  # Keep 2000 passages
```

**Expected gain**: +0.02-0.03 MAP (especially recall)

## 8. Document-Level + Passage-Level Fusion

**Strategy**: Score documents at both levels and fuse

```python
# Document-level scoring (fast, good for long docs)
doc_scores = bi_encoder.score_documents(query, docs)

# Passage-level scoring (precise, good for specific info)
passage_scores = cross_encoder.score_passages(query, passages)
passage_agg = maxp(passage_scores)

# Fuse
final_scores = 0.3 * doc_scores + 0.7 * passage_agg
```

**Expected gain**: +0.01-0.02 MAP

## 9. Query Type Detection

**Strategy**: Different strategies for different query types

```python
if is_factual_query(query):  # "What is X?"
    # Use more aggressive expansion
    use_expanded = True
    rerank_depth = 700
elif is_exploratory_query(query):  # "Tell me about X"
    # Use original query, more docs
    use_expanded = False
    rerank_depth = 1000
```

**Expected gain**: +0.01-0.02 MAP

## 10. Post-Processing: Score Calibration

**Strategy**: Calibrate scores across queries

```python
# Normalize scores per query to [0, 1]
# Then apply query-specific boost for high-confidence queries

if max_score > 0.8:  # High confidence query
    scores = scores * 1.1  # Boost top results
```

**Expected gain**: +0.005-0.01 MAP

---

## Implementation Priority

### Phase 1 (Quick Wins - Do First):
1. ✅ **Multiple BM25 variants** - Easy, high impact
2. ✅ **Original + Expanded fusion** - Easy, good gain
3. ✅ **Increase rerank depth to 1000** - Easy, big recall gain

### Phase 2 (Medium Effort):
4. ✅ **Ensemble 2-3 cross-encoders** - Medium, high impact
5. ✅ **Multiple passage aggregation** - Medium, good gain
6. ✅ **Adaptive rerank depth** - Medium, good gain

### Phase 3 (Advanced):
7. Learning-to-rank features
8. Document-level + passage fusion
9. Query type detection

---

## Expected Final MAP

| Run | Current | After Phase 1 | After Phase 2 | Target |
|-----|---------|---------------|---------------|--------|
| Run 1 | 0.234 | 0.28-0.30 | 0.30-0.32 | 0.32+ |
| Run 2 | 0.279 | 0.32-0.34 | 0.35-0.38 | 0.38+ |
| Run 3 | 0.288 | 0.33-0.35 | 0.38-0.42 | 0.42+ |

---

## Code Changes Needed

1. **cmd_run1**: Add multiple BM25 variants, fuse original+expanded
2. **cmd_run2**: Ensemble multiple cross-encoders, increase depth to 1000
3. **neural_reranker.py**: Add MultiCrossEncoder class
4. **aggregation.py**: Add ensemble aggregation function
5. **main.py**: Add adaptive rerank depth logic

