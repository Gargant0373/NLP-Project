"""
Main pipeline orchestrating framing, emotion, and sentiment analysis.

Usage:
    python main.py                                      # All analyses, all topics
    python main.py --topics "Health" --analysis emotion # Single topic, emotion only
    python main.py --analysis sentiment                 # Sentiment only
"""

import argparse
import pandas as pd
from pathlib import Path
from data_retrieval.data import get_full_dataframe
from framing_analysis.framing_analysis import FramingAnalyzer
from emotion_analysis.emotion_analysis import EmotionAnalyzer, compare_emotions
from sentiment_analysis.sentiment_analysis import SentimentAnalyzer, aggregate_by_entity_type, compare_sentiments


def load_data(cache_file='speeches_data.csv', force_reload=False):
    """Load or create cached dataset."""
    if Path(cache_file).exists() and not force_reload:
        print(f"Loading from cache: {cache_file}")
        df = pd.read_csv(cache_file)
        # Reconstruct list columns from CSV
        df['topics'] = df['topics'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        df['topic_labels'] = df['topic_labels'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        df['entities'] = df['entities'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        df['tokens'] = df['tokens'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        return df
    
    print("Loading and parsing XML files...")
    df = get_full_dataframe()
    df.to_csv(cache_file, index=False)
    print(f"Cached to {cache_file}")
    return df


def prepare_data(df, topics=None, min_per_group=20):
    """Filter and prepare data for analysis."""
    # Explode topics
    df_exp = df.explode('topic_labels')
    df_exp = df_exp[df_exp['topic_labels'].notna()]
    
    # Filter to specific topics
    if topics:
        df_exp = df_exp[df_exp['topic_labels'].isin(topics)]
        print(f"\nFiltered to topics: {topics}")
    
    # Get valid topics (enough gov and opp)
    valid_topics = []
    for topic in df_exp['topic_labels'].unique():
        topic_df = df_exp[df_exp['topic_labels'] == topic]
        gov_count = (topic_df['speaker_type'] == 'Government').sum()
        opp_count = (topic_df['speaker_type'] == 'Opposition').sum()
        
        if gov_count >= min_per_group and opp_count >= min_per_group:
            valid_topics.append((topic, gov_count, opp_count))
    
    if not valid_topics:
        print(f"⚠ No topics found with >={min_per_group} speeches per group")
        return None, valid_topics
    
    print(f"\nValid topics ({len(valid_topics)}):")
    for topic, gov, opp in valid_topics:
        print(f"  {topic}: {gov} gov + {opp} opp = {gov+opp} total")
    
    return df_exp, valid_topics


def run_framing(df_exp, valid_topics, output_dir='results'):
    """Run framing analysis on selected topics."""
    print("\n" + "="*70)
    print("FRAMING ANALYSIS")
    print("="*70)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    for topic, gov_count, opp_count in valid_topics:
        print(f"\n{topic} ({gov_count+opp_count} speeches)...")
        
        topic_df = df_exp[df_exp['topic_labels'] == topic]
        
        try:
            analyzer = FramingAnalyzer()
            analyzer.fit(topic_df['text'].tolist(), verbose=False)
            comparison = analyzer.compare_framing(
                topic_df['text'].tolist(),
                topic_df['speaker_type'].tolist()
            )
            
            outfile = Path(output_dir) / f"framing_{topic.replace(' ', '_')}.csv"
            comparison.to_csv(outfile, index=False)
            print(f"  OK - {len(comparison)} semantic topics found")
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")


def run_emotion(df_exp, valid_topics, output_dir='results'):
    """Run emotion analysis on selected topics."""
    print("\n" + "="*70)
    print("EMOTION ANALYSIS")
    print("="*70)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    for topic, gov_count, opp_count in valid_topics:
        print(f"\n{topic} ({gov_count+opp_count} speeches)...")
        
        topic_df = df_exp[df_exp['topic_labels'] == topic].copy()
        
        try:
            analyzer = EmotionAnalyzer()
            df_emotions = analyzer.analyze_dataframe(topic_df, batch_size=16)
            
            # Compare emotions
            comp = compare_emotions(df_emotions)
            outfile = Path(output_dir) / f"emotion_{topic.replace(' ', '_')}.csv"
            comp.to_csv(outfile, index=False)
            
            # Show results
            print("\n  Government vs Opposition emotions:")
            emotion_cols = ['anger', 'fear', 'joy', 'neutral']
            for _, row in comp.iterrows():
                if all(col in row.index for col in emotion_cols):
                    emotions = ', '.join([f"{e}:{row[f'{e}_mean']:.2f}" for e in emotion_cols])
                    print(f"    {row['group']}: {emotions}")
                else:
                    print(f"    {row['group']}: {row.to_dict()}")
        except Exception as e:
            print(f"  Error: {str(e)[:100]}")


def run_sentiment(df_exp, valid_topics, output_dir='results'):
    """Run sentiment analysis on selected topics."""
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS (Institutional Targets)")
    print("="*70)

    Path(output_dir).mkdir(exist_ok=True)

    # Initialize model once
    print("\nLoading ParlaSent model...")
    analyzer = SentimentAnalyzer()

    for topic, gov_count, opp_count in valid_topics:
        print(f"\n{topic} ({gov_count+opp_count} sentences)...")

        topic_df = df_exp[df_exp['topic_labels'] == topic].copy()

        try:
            # Apply sentiment model
            print("  [1/3] Applying ParlaSent model...")
            topic_df = analyzer.analyze_dataframe(topic_df, batch_size=16)

            # Aggregate by entity type
            print("  [2/3] Aggregating by entity type...")
            df_agg = aggregate_by_entity_type(topic_df)

            if df_agg.empty:
                print("  ⚠ No entity data for aggregation")
                continue

            # Compare gov vs opp
            print("  [3/3] Comparing Gov vs Opp...")
            comparison = compare_sentiments(df_agg)

            # Save results
            outfile = Path(output_dir) / f"sentiment_{topic.replace(' ', '_')}.csv"
            comparison.to_csv(outfile, index=False)

            # Show summary
            for _, row in comparison.iterrows():
                org = f"ORG={row.get('ORG_sentiment_mean', 'N/A'):.2f}" if pd.notna(row.get('ORG_sentiment_mean')) else "ORG=N/A"
                loc = f"LOC={row.get('LOC_sentiment_mean', 'N/A'):.2f}" if pd.notna(row.get('LOC_sentiment_mean')) else "LOC=N/A"
                misc = f"MISC={row.get('MISC_sentiment_mean', 'N/A'):.2f}" if pd.notna(row.get('MISC_sentiment_mean')) else "MISC=N/A"
                print(f"    {row['group']}: {org}, {loc}, {misc}")

        except Exception as e:
            print(f"  Error: {str(e)[:100]}")


def main():
    parser = argparse.ArgumentParser(description="Parliamentary debate analysis")
    parser.add_argument('--topics', type=str, default=None,
                       help='Comma-separated topics (e.g., "Health,Environment")')
    parser.add_argument('--analysis', choices=['framing', 'emotion', 'sentiment', 'all'],
                       default='all', help='Analysis type (default: all)')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--output', type=str, default='results')

    args = parser.parse_args()

    # Parse topics
    topics = None
    if args.topics:
        topics = [t.strip() for t in args.topics.split(',')]

    # Load and prepare
    print("Loading data...")
    df = load_data(force_reload=args.reload)
    df_exp, valid_topics = prepare_data(df, topics=topics)

    if df_exp is None:
        print("Cannot proceed without valid topics.")
        return

    # Run analyses
    if args.analysis in ['framing', 'all']:
        run_framing(df_exp, valid_topics, args.output)

    if args.analysis in ['emotion', 'all']:
        run_emotion(df_exp, valid_topics, args.output)

    if args.analysis in ['sentiment', 'all']:
        run_sentiment(df_exp, valid_topics, args.output)

    print("\n" + "="*70)
    print("Complete - Results in", args.output)
    print("="*70)


if __name__ == "__main__":
    main()

