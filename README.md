# Partisan Framing and Emotional Appeals in Parliamentary Debate

NLP analysis of how government and opposition parties use different rhetorical strategies in parliamentary debates.

**Methods:** BERTopic semantic framing + DistilRoBERTa emotion detection  
**Data:** ParlaMint corpus (26 countries, 1,900+ speeches, 2004-2024)

## Quick Start

```bash
pip install -r requirements.txt
python example_run.py                           # Validate data extraction
python main.py --topics "Health"                # Analyze Health topic
```

## Usage

```bash
python main.py                                  # All topics
python main.py --topics "Health,Environment"   # Specific topics  
python main.py --analysis framing              # Framing only
python main.py --analysis emotion              # Emotion only
```

## Output

Results in `results/`:
- `framing_<topic>.csv` - Semantic topics and distinctive party vocabulary
- `emotion_<topic>.csv` - Emotion probabilities by party

## Example

```python
from data import get_full_dataframe
from framing_analysis import FramingAnalyzer

df = get_full_dataframe()
analyzer = FramingAnalyzer()
analyzer.fit(df['text'].tolist())
comparison = analyzer.compare_framing(df['text'].tolist(), df['speaker_type'].tolist())
```

## Files

- `data.py` - ParlaMint XML parsing, metadata enrichment
- `framing_analysis.py` - BERTopic topic modeling
- `emotion_analysis.py` - Emotion classification
- `main.py` - Pipeline orchestration
- `example_run.py` - Data validation (explains why it's needed below)

## Why example_run.py?

`example_run.py` is a **fast data validation script** (runs in 2-3 minutes, no NLP models):
- Verifies data.py correctly parses 1,900+ XML files
- Confirms Government/Opposition classification works
- Shows what CAP topics are available in the data
- Validates sufficient samples for statistical analysis
- Ensures pipeline foundation is sound before running expensive ML models

**Use it to:**
1. Quick sanity check that everything works on your system
2. Understand what data is available before committing to full analysis
3. Catch data issues early (missing files, parsing errors)

## Requirements

Python 3.9+, 8GB RAM. All dependencies in `requirements.txt`.

See [METHODOLOGY.md](METHODOLOGY.md) for technical details.
