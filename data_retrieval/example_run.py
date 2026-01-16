"""
Small test run to understand data.py and test the pipeline on a subset.

This script:
1. Loads a small sample of parliamentary data
2. Shows what fields are extracted
3. Displays sample speeches
4. Tests basic filtering and grouping
"""

import pandas as pd
import sys
from data import get_full_dataframe

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def inspect_dataframe(df, n_samples=5):
    """Show detailed information about the dataframe structure."""
    print("\n" + "="*70)
    print("DATAFRAME STRUCTURE")
    print("="*70)
    
    print(f"\nShape: {df.shape[0]:,} speeches × {df.shape[1]} columns")
    
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  - {col:20s} | {str(dtype):15s} | {non_null:,} non-null")
    
    print("\nMemory usage:")
    print(f"  {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def show_sample_speeches(df, n=3):
    """Display sample speeches with metadata."""
    print("\n" + "="*70)
    print(f"SAMPLE SPEECHES (showing {n})")
    print("="*70)
    
    sample = df.sample(n=min(n, len(df)), random_state=42)
    
    for idx, row in sample.iterrows():
        print(f"\n--- Speech {idx} ---")
        print(f"Country: {row['country_code']}")
        print(f"Date: {row['date']}")
        print(f"Speaker: {row['speaker_name']} ({row['speaker_sex']})")
        print(f"Party: {row['party_name']} ({row['party_id']})")
        print(f"Type: {row['speaker_type']}")
        print(f"Topics: {row['topic_labels']}")
        print(f"Sentiment: {row['sentiment']}")
        print(f"Length: {len(row['text'])} chars, {row['sentence_length']} tokens")
        print(f"Text preview: {row['text'][:200]}...")


def analyze_distribution(df):
    """Show data distribution across key dimensions."""
    print("\n" + "="*70)
    print("DATA DISTRIBUTION")
    print("="*70)
    
    # Countries
    print("\nCountries:")
    country_counts = df['country_code'].value_counts()
    for country, count in country_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"  {country:10s}: {count:6,} speeches ({pct:5.2f}%)")
    
    # Speaker types
    print("\nSpeaker Types:")
    speaker_counts = df['speaker_type'].value_counts()
    for stype, count in speaker_counts.items():
        pct = count / len(df) * 100
        print(f"  {stype:15s}: {count:6,} speeches ({pct:5.2f}%)")
    
    # Date range
    print(f"\nDate Range:")
    print(f"  From: {df['date'].min()}")
    print(f"  To:   {df['date'].max()}")
    
    # Topics
    print("\nTop 15 CAP Topics:")
    all_topics = []
    for topic_list in df['topic_labels']:
        if isinstance(topic_list, list):
            all_topics.extend(topic_list)
    
    topic_counts = pd.Series(all_topics).value_counts()
    for topic, count in topic_counts.head(15).items():
        pct = count / len(all_topics) * 100
        print(f"  {topic:30s}: {count:5,} ({pct:5.2f}%)")
    
    # Text length distribution
    print("\nText Length Statistics:")
    lengths = df['text'].str.len()
    print(f"  Mean:   {lengths.mean():8.1f} characters")
    print(f"  Median: {lengths.median():8.1f} characters")
    print(f"  Min:    {lengths.min():8.1f} characters")
    print(f"  Max:    {lengths.max():8.1f} characters")


def create_test_subset(df, n_speeches=100):
    """Create a small balanced subset for testing."""
    print("\n" + "="*70)
    print("CREATING TEST SUBSET")
    print("="*70)
    
    # Get speeches with topics
    df_with_topics = df[df['topic_labels'].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False
    )]
    
    # Balance government and opposition
    gov_speeches = df_with_topics[df_with_topics['speaker_type'] == 'Government']
    opp_speeches = df_with_topics[df_with_topics['speaker_type'] == 'Opposition']
    
    n_per_group = n_speeches // 2
    
    subset_gov = gov_speeches.sample(n=min(n_per_group, len(gov_speeches)), random_state=42)
    subset_opp = opp_speeches.sample(n=min(n_per_group, len(opp_speeches)), random_state=42)
    
    subset = pd.concat([subset_gov, subset_opp], ignore_index=True)
    
    print(f"\nSubset created: {len(subset)} speeches")
    print(f"  Government: {len(subset_gov)}")
    print(f"  Opposition: {len(subset_opp)}")
    
    # Show topic distribution in subset
    print("\nTopics in subset:")
    subset_topics = []
    for topic_list in subset['topic_labels']:
        if isinstance(topic_list, list):
            subset_topics.extend(topic_list)
    
    topic_counts = pd.Series(subset_topics).value_counts()
    for topic, count in topic_counts.head(10).items():
        print(f"  {topic:30s}: {count:3,}")
    
    return subset


def test_topic_filtering(df):
    """Test filtering by specific topic."""
    print("\n" + "="*70)
    print("TOPIC FILTERING TEST")
    print("="*70)
    
    # Pick a common topic
    all_topics = []
    for topic_list in df['topic_labels']:
        if isinstance(topic_list, list):
            all_topics.extend(topic_list)
    
    if not all_topics:
        print("No topics found in data")
        return
    
    # Get most common topic
    top_topic = pd.Series(all_topics).value_counts().index[0]
    
    print(f"\nFiltering to topic: '{top_topic}'")
    
    # Explode and filter
    df_exploded = df.explode('topic_labels')
    df_topic = df_exploded[df_exploded['topic_labels'] == top_topic]
    
    print(f"Found {len(df_topic)} speeches about '{top_topic}'")
    
    # Show distribution
    print("\nSpeaker type distribution:")
    for stype, count in df_topic['speaker_type'].value_counts().items():
        pct = count / len(df_topic) * 100
        print(f"  {stype:15s}: {count:4,} ({pct:5.2f}%)")
    
    print("\nCountry distribution:")
    for country, count in df_topic['country_code'].value_counts().head(5).items():
        print(f"  {country:10s}: {count:4,}")
    
    # Show sample
    print(f"\nSample speeches about '{top_topic}':")
    sample = df_topic.sample(n=min(2, len(df_topic)), random_state=42)
    for idx, row in sample.iterrows():
        print(f"\n  [{row['speaker_type']}] {row['speaker_name']} ({row['party_name']}):")
        print(f"  {row['text'][:150]}...")


def main():
    """Run the full test pipeline."""
    print("\n" + "="*70)
    print("TESTING DATA.PY - SMALL RUN")
    print("="*70)
    print("\nThis script tests data loading and shows what fields are extracted.")
    print("Loading data from ParlaMint XML files...")
    
    # Load full dataset
    print("\n[1/6] Loading data...")
    df = get_full_dataframe()
    
    # Check if dataframe is empty
    if len(df) == 0:
        print("\n⚠ WARNING: DataFrame is empty after loading!")
        print("This might be because all speeches were filtered out as 'Other' speaker type.")
        print("\nTrying to load without filtering...")
        
        # Import and call without the clean step
        from data import get_countries, parse_parlamint_xml, enrich_dataframe, load_speaker_type, extract_country_code
        from pathlib import Path
        
        data_path = Path("ParlaMint")
        xml_files = get_countries("ParlaMint")
        print(f"Found {len(xml_files)} XML files. Parsing...")
        
        all_records = []
        for f in xml_files:
            country_code = extract_country_code(f)
            file_data = parse_parlamint_xml(f)
            for record in file_data:
                record['country_code'] = country_code
            all_records.extend(file_data)
        
        df = pd.DataFrame(all_records)
        print(f"Parsed {len(df)} speeches before filtering")
        
        df = enrich_dataframe(df, data_path)
        df = load_speaker_type(Path("ParlaMint/Samples"), df)
        
        print(f"\nSpeaker type distribution BEFORE filtering:")
        print(df['speaker_type'].value_counts())
        print(f"\nKeeping all speeches for inspection...")
        
    if len(df) == 0:
        print("\n✗ No data available. Cannot continue.")
        return
    
    # Inspect structure
    print("\n[2/6] Inspecting dataframe structure...")
    inspect_dataframe(df)
    
    # Show samples
    print("\n[3/6] Showing sample speeches...")
    show_sample_speeches(df, n=3)
    
    # Analyze distribution
    print("\n[4/6] Analyzing data distribution...")
    analyze_distribution(df)
    
    # Create test subset
    print("\n[5/6] Creating balanced test subset...")
    subset = create_test_subset(df, n_speeches=100)
    
    # Test topic filtering
    print("\n[6/6] Testing topic filtering...")
    test_topic_filtering(df)
    
    # Save subset for quick testing
    print("\n" + "="*70)
    print("SAVING TEST DATA")
    print("="*70)
    
    subset_file = "test_subset.csv"
    subset.to_csv(subset_file, index=False)
    print(f"\nSaved {len(subset)} speeches to '{subset_file}'")
    
    import os
    file_size_kb = os.path.getsize(subset_file) / 1024
    print(f"File size: {file_size_kb:.2f} KB")
    
    print("\n" + "="*70)
    print("TEST RUN COMPLETE")
    print("="*70)
    print("\nKey findings:")
    print(f"  • Total speeches: {len(df):,}")
    print(f"  • Countries: {df['country_code'].nunique()}")
    print(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  • Government/Opposition balance: {df['speaker_type'].value_counts().to_dict()}")
    print(f"  • Test subset saved: {subset_file} ({len(subset)} speeches)")
    
    print("\nNext steps:")
    print("  1. Review the output above to understand data structure")
    print(f"  2. Use {subset_file} for quick pipeline testing")
    print("  3. Run 'python main.py' for full analysis")
    
if __name__ == "__main__":
    main()
