"""
Institutional targeting sentiment analysis using ParlaSent model.

Analyzes how government vs opposition direct sentiment toward institutional targets:
- ORG (organizations): political parties, EU, corporations, agencies
- LOC (locations): countries, regions, cities
- MISC (miscellaneous): policies, treaties, events, initiatives

Research Question: Do government and opposition parties differ in which
institutions, countries, and policies they direct criticism toward?

This complements framing analysis (what topics) and emotion analysis (what tone)
by adding a 'who gets blamed' dimension focused on institutional rather than
interpersonal targets (excluding PER entities per ParlaSent paper's focus).
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Sentiment analysis using ParlaSent XLM-R model."""

    def __init__(self, model_name="classla/xlm-r-parlasent"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading sentiment model on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts, batch_size=16):
        """Predict sentiment scores for texts.

        Returns list of dicts with sentiment scores (-1 to 1).
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # ParlaSent outputs logits, convert to sentiment scores
                # The model is trained as regression with single output
                scores = outputs.logits.squeeze(-1)

            for score in scores:
                sentiment_score = score.item()
                # Clip to [-1, 1] range for safety
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                results.append({
                    'sentiment_score': sentiment_score
                })

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

        return results

    def analyze_dataframe(self, df, text_col='text', batch_size=16):
        """Add sentiment predictions to dataframe.

        Input: sentence-level dataframe with text and entities columns
        Output: same dataframe with sentiment_score column added
        """
        texts = df[text_col].fillna("").tolist()
        results = self.predict(texts, batch_size)
        sentiment_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)


def aggregate_by_entity_type(df, speech_id_col='u_id', entity_col='entities',
                             sentiment_col='sentiment_score',
                             group_cols=['speaker_type', 'topic_labels']):
    """Aggregate sentence-level sentiment to speech-level by entity type.

    For each speech, compute mean sentiment for sentences containing each entity type.

    Note: Focuses on institutional targets (ORG, LOC, MISC) excluding interpersonal
    targets (PER) as those are covered in the ParlaSent paper's affective polarization study.

    Args:
        df: Sentence-level dataframe with sentiment scores and entity annotations
        speech_id_col: Column identifying unique speeches
        entity_col: Column containing list of entity types per sentence
        sentiment_col: Column with sentiment scores
        group_cols: Additional columns to group by (speaker_type, topics, etc.)

    Returns:
        Speech-level dataframe with sentiment scores per entity type
    """
    # Ensure entities column is list type
    if df[entity_col].dtype == 'object':
        # Handle string representations of lists
        df = df.copy()
        df[entity_col] = df[entity_col].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else (x if isinstance(x, list) else [])
        )

    # Focus on institutional targets (exclude PER for interpersonal sentiment)
    entity_types = ['ORG', 'LOC', 'MISC']

    # Create aggregation results
    results = []

    # Group by speech
    for speech_id, speech_df in df.groupby(speech_id_col):
        result = {speech_id_col: speech_id}

        # Add group columns (speaker_type, topics, etc.)
        for col in group_cols:
            if col in speech_df.columns:
                # Take first value (all should be same within speech)
                result[col] = speech_df[col].iloc[0]

        # Aggregate sentiment by entity type
        for entity_type in entity_types:
            # Get sentences containing this entity type
            mask = speech_df[entity_col].apply(lambda x: entity_type in x if isinstance(x, list) else False)
            entity_sentences = speech_df[mask]

            if len(entity_sentences) > 0:
                result[f'{entity_type}_sentiment'] = entity_sentences[sentiment_col].mean()
                result[f'{entity_type}_count'] = len(entity_sentences)
            else:
                result[f'{entity_type}_sentiment'] = None
                result[f'{entity_type}_count'] = 0

        # Overall speech sentiment (all sentences)
        result['overall_sentiment'] = speech_df[sentiment_col].mean()
        result['n_sentences'] = len(speech_df)

        results.append(result)

    return pd.DataFrame(results)


def compare_sentiments(df, group_col='speaker_type', groups=['Government', 'Opposition']):
    """Compare sentiment toward institutional targets between groups.

    Analyzes how government vs opposition direct sentiment toward:
    - ORG: institutions, parties, corporations
    - LOC: countries, regions
    - MISC: policies, events, initiatives

    Args:
        df: Speech-level dataframe with entity-specific sentiment columns
        group_col: Column to group by (e.g., 'speaker_type')
        groups: List of group values to compare

    Returns:
        Dataframe with comparative statistics per entity type per group
    """
    entity_types = ['ORG', 'LOC', 'MISC']

    results = []
    for group in groups:
        group_df = df[df[group_col] == group]
        if len(group_df) == 0:
            continue

        result = {
            'group': group,
            'n_speeches': len(group_df),
            'overall_sentiment_mean': group_df['overall_sentiment'].mean(),
            'overall_sentiment_std': group_df['overall_sentiment'].std()
        }

        # Stats per entity type
        for entity_type in entity_types:
            sentiment_col = f'{entity_type}_sentiment'
            count_col = f'{entity_type}_count'

            # Filter to speeches that mention this entity type
            entity_df = group_df[group_df[count_col] > 0]

            if len(entity_df) > 0:
                result[f'{entity_type}_sentiment_mean'] = entity_df[sentiment_col].mean()
                result[f'{entity_type}_sentiment_std'] = entity_df[sentiment_col].std()
                result[f'{entity_type}_n_speeches'] = len(entity_df)
                result[f'{entity_type}_total_mentions'] = entity_df[count_col].sum()
            else:
                result[f'{entity_type}_sentiment_mean'] = None
                result[f'{entity_type}_sentiment_std'] = None
                result[f'{entity_type}_n_speeches'] = 0
                result[f'{entity_type}_total_mentions'] = 0

        results.append(result)

    return pd.DataFrame(results)


def compare_by_topic(df, topic_col='topic_labels', group_col='speaker_type',
                     groups=['Government', 'Opposition']):
    """Compare sentiment across topics and groups.

    Returns detailed breakdown per topic showing gov vs opp sentiment
    toward different entity types.
    """
    results = []

    for topic in df[topic_col].unique():
        topic_df = df[df[topic_col] == topic]

        # Compare within this topic
        comparison = compare_sentiments(topic_df, group_col, groups)

        # Add topic info
        comparison['topic'] = topic
        results.append(comparison)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
