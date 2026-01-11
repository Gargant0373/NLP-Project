# Methodology Documentation

## Overview

This pipeline implements two complementary NLP techniques to analyze partisan rhetoric in parliamentary debates: semantic framing analysis and discrete emotion classification. Both use efficient, CPU-compatible transformer models optimized for local execution.

## 1. Comparative Framing Analysis

### Approach
BERTopic-based semantic topic modeling to identify how government and opposition frame identical policy issues using different rhetoric.

### Technical Implementation

**Sentence Embeddings:**
- Model: `all-MiniLM-L6-v2` (SentenceTransformers)
- Rationale: 384-dimensional embeddings, CPU-efficient, preserves semantic similarity
- Output: Dense vector representations of speech texts

**Clustering:**
- UMAP: Dimensionality reduction (5 components, cosine distance)
- HDBSCAN: Density-based clustering with minimum cluster size = 15
- Rationale: Identifies semantic topics without requiring predefined topic count

**Class-based TF-IDF (c-TF-IDF):**
- Aggregates speeches by class (Government/Opposition) within each topic
- Computes term importance scores per class
- Identifies distinctive vocabulary for each partisan group

### Algorithm Flow
```
Speeches → Embeddings → UMAP → HDBSCAN → Topics
                                            ↓
                    Government speeches ← Topic N → Opposition speeches
                            ↓                              ↓
                      c-TF-IDF terms              c-TF-IDF terms
                            ↓                              ↓
                          Distinctive framing comparison
```

### Parameters
- `min_topic_size`: 15 (minimum speeches per topic)
- `n_neighbors`: 15 (UMAP local structure)
- `ngram_range`: (1, 2) (unigrams and bigrams)
- `min_df`: 2 (minimum document frequency)

## 2. Emotion Analysis

### Approach
Discrete emotion classification using a distilled transformer model fine-tuned on emotional text.

### Technical Implementation

**Model:**
- `j-hartmann/emotion-english-distilroberta-base`
- Architecture: DistilRoBERTa (66M parameters vs 125M for full RoBERTa)
- Output: 7-class probability distribution
  - anger, disgust, fear, joy, neutral, sadness, surprise

**Inference:**
- Batch processing (batch_size=16)
- CPU/GPU agnostic
- Max sequence length: 512 tokens
- Softmax over logits for probability scores

### Aggregation Strategy
For each group (Government/Opposition) and topic:
1. Mean probability per emotion category
2. Predicted emotion distribution (argmax)
3. Dominant emotion identification
4. Comparative difference metrics (Opposition - Government)

### Algorithm Flow
```
Speech text → Tokenization → DistilRoBERTa → Logits → Softmax → Probabilities
                                                                       ↓
                                               Group by Party + Topic
                                                                       ↓
                                        Aggregate emotions, compute differences
```

### Parameters
- `batch_size`: 16 (memory/speed tradeoff)
- `max_length`: 512 tokens
- `truncation`: True (for long speeches)
- `min_speeches_per_group`: 30 (reliable statistics)

## 3. Data Pipeline

### Input Processing
1. Parse TEI-XML (ParlaMint format)
2. Extract metadata: speaker, party, date, CAP topics
3. Reconstruct speech text from tokenized annotations
4. Map parties to government/opposition status by date

### Filtering
- Minimum speech length: 20 characters
- Speaker types: Government, Opposition only
- Topics: Must have CAP topic label
- Temporal: 2018-2022

### Quality Controls
- Remove extremely short speeches (noise)
- Verify sufficient samples per group (statistical validity)
- Handle missing metadata (fallback to 'Unknown')

## 4. Comparative Analysis Structure

### By Topic
For each CAP policy topic (Health, Environment, etc.):
1. Filter speeches to topic
2. Verify minimum sample size per group
3. Run framing analysis: identify semantic sub-topics
4. Run emotion analysis: quantify emotional patterns
5. Compare Government vs Opposition on both dimensions

### Output Metrics

**Framing:**
- Topic count and distribution per group
- Distinctive terms (top 15 per group per topic)
- Semantic topic labels (top 5 words)

**Emotion:**
- Mean probability per emotion (7 categories)
- Dominant emotion percentage
- Difference scores (Opposition - Government)
- Predicted emotion distribution

## 5. Implementation Notes

### Efficiency Optimizations
- Sentence embeddings computed once, cached for clustering
- Batch processing for emotion classification
- Parquet caching for parsed data
- No GPU required (but utilized if available)

### Reproducibility
- Fixed random seeds (UMAP: 42)
- Deterministic clustering (HDBSCAN EOM)
- Versioned model checkpoints
- Full parameter documentation

### Extensibility
- Modular design: `data.py`, `framing_analysis.py`, `emotion_analysis.py`
- Simple API: `fit()`, `transform()`, `compare()`
- Configuration via `config.ini`
- Add new topics/countries without code changes

## 6. Validation Considerations

### Framing Analysis
- Manual inspection of topic coherence
- Verify distinctive terms are semantically meaningful
- Check topic size distribution (avoid fragmentation)

### Emotion Analysis
- Cross-validate predictions on sample speeches
- Check for model bias on political text
- Compare to existing sentiment scores in ParlaMint
- Assess confidence scores distribution

### Comparative Validity
- Ensure balanced samples across groups
- Control for speech length effects
- Consider temporal trends
- Account for country-specific patterns

## References

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. EMNLP.
- Hartmann, J. (2022). Emotion English DistilRoBERTa-base. HuggingFace Model Hub.
