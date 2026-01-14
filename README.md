# Partisan Framing and Emotional Appeals in Parliamentary Debate

NLP analysis of how government and opposition parties use different rhetorical strategies in parliamentary debates.

**Methods:** BERTopic semantic framing + DistilRoBERTa emotion detection + ParlaSent institutional sentiment
**Data:** ParlaMint corpus (26 countries, 1,900+ speeches, 2004-2024)

## Quick Start

**Download Data**:
Run the included script to clone the ParlaMint repository.
```bash
chmod +x download_data.sh
./download_data.sh
```

```bash
pip install -r requirements.txt
python example_run.py                           # Validate data extraction
python main.py --topics "Health"                # Analyze Health topic
```

## Usage

```bash
python main.py                                  # All analyses, all topics
python main.py --topics "Health,Environment"   # Specific topics
python main.py --analysis framing              # Framing only
python main.py --analysis emotion              # Emotion only
python main.py --analysis sentiment            # Sentiment only
python main.py --analysis all                  # All three analyses
```

## Output

Results in `results/`:
- `framing_<topic>.csv` - Semantic topics and distinctive party vocabulary
- `emotion_<topic>.csv` - Emotion probabilities by party
- `sentiment_<topic>.csv` - Sentiment toward institutional targets (ORG/LOC/MISC)

Example sentiment output:
```
group,n_speeches,overall_sentiment_mean,ORG_sentiment_mean,LOC_sentiment_mean,MISC_sentiment_mean
Government,58,0.987,0.985,0.922,0.917
Opposition,27,0.959,0.925,0.945,1.000
```

## Example

```python
# Framing analysis
from data import get_full_dataframe
from framing_analysis import FramingAnalyzer

df = get_full_dataframe()
analyzer = FramingAnalyzer()
analyzer.fit(df['text'].tolist())
comparison = analyzer.compare_framing(df['text'].tolist(), df['speaker_type'].tolist())

# Sentiment analysis
from sentiment_analysis import SentimentAnalyzer, aggregate_by_entity_type, compare_sentiments

sentiment_analyzer = SentimentAnalyzer()
df = sentiment_analyzer.analyze_dataframe(df)       # Add sentiment scores
df_agg = aggregate_by_entity_type(df)               # Aggregate by entity type
comparison = compare_sentiments(df_agg)             # Compare gov vs opp
```

## Files

- `data.py` - ParlaMint XML parsing, metadata enrichment
- `framing_analysis.py` - BERTopic topic modeling
- `emotion_analysis.py` - Emotion classification
- `sentiment_analysis.py` - ParlaSent institutional sentiment analysis
- `main.py` - Pipeline orchestration (all analyses)
- `example_run.py` - Data validation (explains why it's needed below)
- `visualize_sentiment.py` - Generate sentiment visualization plots

## Why example_run.py?

`example_run.py` is a **fast data validation script** (runs in 2-3 minutes, no NLP models):
- Verifies data.py correctly parses 1,900+ XML files
- Confirms Government/Opposition classification works
- Validates sufficient samples for statistical analysis
- Ensures pipeline foundation is sound before running models

## Requirements

Python 3.9+, 8GB RAM. All dependencies in `requirements.txt`.

See [METHODOLOGY.md](METHODOLOGY.md) for technical details.
