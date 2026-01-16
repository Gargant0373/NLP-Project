"""
Full dataset test for institutional targeting sentiment analysis.

Runs sentiment analysis on all available topics to validate patterns across
different policy areas (Mix, Government Operations, Environment, etc.).

This produces comprehensive results showing whether institutional targeting
patterns are consistent across topics.
"""

import pandas as pd
from pathlib import Path
from sentiment_analysis import SentimentAnalyzer, aggregate_by_entity_type, compare_sentiments
import warnings
warnings.filterwarnings('ignore')


def load_full_data(cache_file='./speeches_data.csv', min_sentences_per_group=20):
    """Load full dataset and identify topics with sufficient data.

    Matches methodology of main.py framing/emotion analyses by filtering on
    sentence counts (not speech counts) to ensure consistent thresholds.
    """
    if not Path(cache_file).exists():
        print(f"Error: {cache_file} not found. Run main.py first to generate cache.")
        return None, None

    print(f"Loading full dataset from {cache_file}...")
    df = pd.read_csv(cache_file)

    # Reconstruct list columns
    df['topics'] = df['topics'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df['topic_labels'] = df['topic_labels'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df['entities'] = df['entities'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    print(f"Total sentences: {len(df)}")

    # Filter to government/opposition only
    df = df[df['speaker_type'].isin(['Government', 'Opposition'])]
    print(f"After filtering to Gov/Opp: {len(df)} sentences")

    # Get topic statistics
    all_topics = []
    for topics in df['topic_labels']:
        if isinstance(topics, list):
            all_topics.extend(topics)

    topic_counts = pd.Series(all_topics).value_counts()
    print(f"\nFound {len(topic_counts)} unique topics")

    # Check entity coverage across all data
    entity_counts = {}
    for entities in df['entities']:
        if isinstance(entities, list):
            for e in entities:
                entity_counts[e] = entity_counts.get(e, 0) + 1

    print(f"\nOverall entity coverage (institutional targets):")
    for entity_type in ['ORG', 'LOC', 'MISC']:
        count = entity_counts.get(entity_type, 0)
        pct = 100 * count / len(df)
        print(f"  {entity_type}: {count:,} mentions ({pct:.1f}% of sentences)")

    per_count = entity_counts.get('PER', 0)
    print(f"  (PER: {per_count:,} mentions - excluded from analysis)")

    # Filter to topics with sufficient sentences per group (matching main.py methodology)
    valid_topics = []
    for topic in topic_counts.head(15).index:  # Check top 15 topics
        topic_df = df[df['topic_labels'].apply(lambda x: topic in x if isinstance(x, list) else False)]

        # Count sentences per group (not unique speeches - matches framing/emotion)
        gov_sentences = (topic_df['speaker_type'] == 'Government').sum()
        opp_sentences = (topic_df['speaker_type'] == 'Opposition').sum()

        # Also track unique speeches for reporting
        gov_speeches = topic_df[topic_df['speaker_type'] == 'Government']['u_id'].nunique()
        opp_speeches = topic_df[topic_df['speaker_type'] == 'Opposition']['u_id'].nunique()

        if gov_sentences >= min_sentences_per_group and opp_sentences >= min_sentences_per_group:
            valid_topics.append((topic, gov_speeches, opp_speeches, len(topic_df)))

    print(f"\n{'='*70}")
    print(f"Valid topics with >={min_sentences_per_group} sentences per group:")
    print(f"{'='*70}")
    for topic, gov_speeches, opp_speeches, total in valid_topics:
        print(f"  {topic:30s}: {gov_speeches:3d} gov speeches + {opp_speeches:3d} opp speeches = {total:5d} sentences")

    return df, valid_topics


def run_sentiment_for_topic(df, topic, analyzer):
    """Run sentiment analysis for a single topic."""
    # Filter to topic
    topic_df = df[df['topic_labels'].apply(lambda x: topic in x if isinstance(x, list) else False)].copy()

    print(f"\n{topic} ({len(topic_df)} sentences)...")

    # Apply sentiment model
    print(f"  [1/3] Applying ParlaSent model...")
    df_with_sentiment = analyzer.analyze_dataframe(topic_df, text_col='text', batch_size=16)

    # Aggregate by entity type
    print(f"  [2/3] Aggregating to speech-level...")
    df_aggregated = aggregate_by_entity_type(
        df_with_sentiment,
        speech_id_col='u_id',
        entity_col='entities',
        group_cols=['speaker_type', 'topic_labels']
    )

    # Compare government vs opposition
    print(f"  [3/3] Comparing Gov vs Opp...")
    comparison = compare_sentiments(df_aggregated, group_col='speaker_type')

    # Add topic label to results
    comparison['topic'] = topic

    # Print summary
    for _, row in comparison.iterrows():
        gov_opp = "Gov" if row['group'] == 'Government' else "Opp"
        org = row.get('ORG_sentiment_mean', float('nan'))
        loc = row.get('LOC_sentiment_mean', float('nan'))
        misc = row.get('MISC_sentiment_mean', float('nan'))

        print(f"    {gov_opp}: Overall={row['overall_sentiment_mean']:+.3f}, "
              f"ORG={org:+.3f}, LOC={loc:+.3f}, MISC={misc:+.3f}")

    return comparison


def main():
    print("="*70)
    print("FULL DATASET SENTIMENT ANALYSIS TEST")
    print("="*70)

    # Load data (using sentence-level filtering to match framing/emotion methodology)
    df, valid_topics = load_full_data(min_sentences_per_group=20)

    if df is None or not valid_topics:
        print("\nCannot proceed without valid data.")
        return

    # Initialize sentiment analyzer once
    print("\n" + "="*70)
    print("LOADING SENTIMENT MODEL")
    print("="*70)
    analyzer = SentimentAnalyzer()

    # Run analysis for each topic
    print("\n" + "="*70)
    print("RUNNING SENTIMENT ANALYSIS PER TOPIC")
    print("="*70)

    all_results = []
    for topic, gov_speeches, opp_speeches, total_sentences in valid_topics:
        try:
            comparison = run_sentiment_for_topic(df, topic, analyzer)
            all_results.append(comparison)
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            continue

    # Combine all results
    if not all_results:
        print("\nNo results generated.")
        return

    combined_results = pd.concat(all_results, ignore_index=True)

    # Save comprehensive results
    output_file = './results/sentiment_results_full_dataset.csv'
    combined_results.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to {output_file}")

    # Print summary analysis
    print("\n" + "="*70)
    print("SUMMARY: Institutional Targeting Patterns Across Topics")
    print("="*70)

    # Compute average sentiment differences (Opp - Gov) per entity type
    entity_types = ['ORG', 'LOC', 'MISC']

    print("\nAverage sentiment differences (Opposition - Government):")
    print(f"{'Topic':30s} | {'Overall':8s} | {'ORG':8s} | {'LOC':8s} | {'MISC':8s}")
    print("-" * 70)

    for topic in combined_results['topic'].unique():
        topic_data = combined_results[combined_results['topic'] == topic]

        if len(topic_data) >= 2:
            gov = topic_data[topic_data['group'] == 'Government'].iloc[0]
            opp = topic_data[topic_data['group'] == 'Opposition'].iloc[0]

            overall_diff = opp['overall_sentiment_mean'] - gov['overall_sentiment_mean']

            diffs = [overall_diff]
            for et in entity_types:
                gov_sent = gov.get(f'{et}_sentiment_mean')
                opp_sent = opp.get(f'{et}_sentiment_mean')
                if pd.notna(gov_sent) and pd.notna(opp_sent):
                    diffs.append(opp_sent - gov_sent)
                else:
                    diffs.append(float('nan'))

            print(f"{topic:30s} | {diffs[0]:+7.3f} | {diffs[1]:+7.3f} | {diffs[2]:+7.3f} | {diffs[3]:+7.3f}")

    # Cross-topic patterns
    print("\n" + "="*70)
    print("CROSS-TOPIC PATTERNS")
    print("="*70)

    gov_data = combined_results[combined_results['group'] == 'Government']
    opp_data = combined_results[combined_results['group'] == 'Opposition']

    print("\nGovernment average sentiment toward institutional targets:")
    for et in entity_types:
        col = f'{et}_sentiment_mean'
        avg = gov_data[col].mean()
        std = gov_data[col].std()
        n = gov_data[col].notna().sum()
        print(f"  {et}: {avg:+.3f} (±{std:.3f}) across {n} topics")

    print("\nOpposition average sentiment toward institutional targets:")
    for et in entity_types:
        col = f'{et}_sentiment_mean'
        avg = opp_data[col].mean()
        std = opp_data[col].std()
        n = opp_data[col].notna().sum()
        print(f"  {et}: {avg:+.3f} (±{std:.3f}) across {n} topics")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Check if opposition is consistently more negative
    org_diffs = []
    loc_diffs = []
    misc_diffs = []

    for topic in combined_results['topic'].unique():
        topic_data = combined_results[combined_results['topic'] == topic]
        if len(topic_data) >= 2:
            gov = topic_data[topic_data['group'] == 'Government'].iloc[0]
            opp = topic_data[topic_data['group'] == 'Opposition'].iloc[0]

            for et, diff_list in [('ORG', org_diffs), ('LOC', loc_diffs), ('MISC', misc_diffs)]:
                gov_sent = gov.get(f'{et}_sentiment_mean')
                opp_sent = opp.get(f'{et}_sentiment_mean')
                if pd.notna(gov_sent) and pd.notna(opp_sent):
                    diff_list.append(opp_sent - gov_sent)

    print("\n1. Consistency of Opposition negativity:")
    for et, diffs, name in [('ORG', org_diffs, 'Organizations'),
                             ('LOC', loc_diffs, 'Locations'),
                             ('MISC', misc_diffs, 'Policies/Events')]:
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            n_negative = sum(1 for d in diffs if d < 0)
            pct_negative = 100 * n_negative / len(diffs)
            print(f"   {et} ({name}): Avg diff = {avg_diff:+.3f}, "
                  f"{n_negative}/{len(diffs)} topics ({pct_negative:.0f}%) show opp more negative")

    print("\n2. Entity type usage patterns:")
    for et in entity_types:
        gov_mentions = gov_data[f'{et}_total_mentions'].sum()
        opp_mentions = opp_data[f'{et}_total_mentions'].sum()
        total = gov_mentions + opp_mentions
        if total > 0:
            gov_pct = 100 * gov_mentions / total
            opp_pct = 100 * opp_mentions / total
            print(f"   {et}: Gov {gov_pct:.1f}% vs Opp {opp_pct:.1f}% of total mentions")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    print("\nReview the CSV file for detailed per-topic comparisons.")
    print("If patterns are consistent, you can integrate sentiment analysis into main.py")


if __name__ == "__main__":
    main()
