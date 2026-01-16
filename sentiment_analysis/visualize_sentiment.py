"""
Simple visualizations for institutional targeting sentiment analysis.

Creates:
1. Bar chart comparing Gov vs Opp sentiment by entity type
2. Heatmap of sentiment differences (Opp - Gov) across topics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load results
df = pd.read_csv('sentiment_results_full_dataset.csv')

# Set up clean plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Check if there is a results directory, create if not
if not os.path.exists('results'):
    os.makedirs('results')

# ============================================================================
# Figure 1: Bar Chart - Overall Sentiment by Entity Type
# ============================================================================

entity_types = ['ORG', 'LOC', 'MISC']
gov_data = df[df['group'] == 'Government']
opp_data = df[df['group'] == 'Opposition']

# Calculate average sentiment per entity type
gov_means = [gov_data[f'{et}_sentiment_mean'].mean() for et in entity_types]
opp_means = [opp_data[f'{et}_sentiment_mean'].mean() for et in entity_types]

# Create bar chart
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(entity_types))
width = 0.35

bars1 = ax.bar(x - width/2, gov_means, width, label='Government', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, opp_means, width, label='Opposition', color='#A23B72', alpha=0.8)

# Formatting
ax.set_xlabel('Entity Type (Institutional Target)', fontweight='bold')
ax.set_ylabel('Mean Sentiment Score', fontweight='bold')
ax.set_title('Government vs Opposition Sentiment Toward Institutional Targets',
             fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['Organizations\n(ORG)', 'Locations\n(LOC)', 'Policies/Events\n(MISC)'])
ax.legend(loc='lower left')
ax.set_ylim([0, 1.1])
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/sentiment_by_entity_type.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/sentiment_by_entity_type.png")
plt.close()


# ============================================================================
# Figure 2: Heatmap - Sentiment Differences Across Topics
# ============================================================================

# Calculate differences (Opposition - Government) per topic per entity type
topics = df['topic'].unique()
diff_matrix = []

for topic in topics:
    topic_data = df[df['topic'] == topic]
    if len(topic_data) >= 2:
        gov = topic_data[topic_data['group'] == 'Government'].iloc[0]
        opp = topic_data[topic_data['group'] == 'Opposition'].iloc[0]

        row = []
        for et in entity_types:
            gov_sent = gov.get(f'{et}_sentiment_mean')
            opp_sent = opp.get(f'{et}_sentiment_mean')
            if pd.notna(gov_sent) and pd.notna(opp_sent):
                row.append(opp_sent - gov_sent)
            else:
                row.append(np.nan)
        diff_matrix.append(row)
    else:
        diff_matrix.append([np.nan, np.nan, np.nan])

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 6))

# Create masked array to handle NaN values
masked_data = np.ma.masked_invalid(diff_matrix)

# Plot heatmap
im = ax.imshow(masked_data, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

# Set ticks and labels
ax.set_xticks(np.arange(len(entity_types)))
ax.set_yticks(np.arange(len(topics)))
ax.set_xticklabels(['Organizations\n(ORG)', 'Locations\n(LOC)', 'Policies/Events\n(MISC)'])
ax.set_yticklabels(topics)

# Add title
ax.set_title('Sentiment Differences: Opposition - Government\n(Negative = Opp more negative)',
             fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Sentiment Difference', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations
for i in range(len(topics)):
    for j in range(len(entity_types)):
        if not np.isnan(diff_matrix[i][j]):
            text_color = 'white' if abs(diff_matrix[i][j]) > 0.25 else 'black'
            ax.text(j, i, f'{diff_matrix[i][j]:.2f}',
                   ha="center", va="center", color=text_color, fontsize=9)

plt.tight_layout()
plt.savefig('results/sentiment_differences_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/sentiment_differences_heatmap.png")
plt.close()


# ============================================================================
# Figure 3: Topic Comparison - Overall Sentiment
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 5))

# Prepare data
gov_overall = []
opp_overall = []
topic_labels = []

for topic in topics:
    topic_data = df[df['topic'] == topic]
    if len(topic_data) >= 2:
        gov = topic_data[topic_data['group'] == 'Government'].iloc[0]
        opp = topic_data[topic_data['group'] == 'Opposition'].iloc[0]

        gov_overall.append(gov['overall_sentiment_mean'])
        opp_overall.append(opp['overall_sentiment_mean'])
        topic_labels.append(topic)

x = np.arange(len(topic_labels))
width = 0.35

bars1 = ax.bar(x - width/2, gov_overall, width, label='Government', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, opp_overall, width, label='Opposition', color='#A23B72', alpha=0.8)

# Formatting
ax.set_xlabel('Policy Topic', fontweight='bold')
ax.set_ylabel('Mean Overall Sentiment', fontweight='bold')
ax.set_title('Overall Sentiment by Topic: Government vs Opposition',
             fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(topic_labels, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1.1])
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/sentiment_by_topic.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/sentiment_by_topic.png")
plt.close()

print("\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)
print("\n1. sentiment_by_entity_type.png:")
print("   Bar chart comparing gov vs opp sentiment toward ORG/LOC/MISC")
print("\n2. sentiment_differences_heatmap.png:")
print("   Heatmap showing sentiment differences (Opp - Gov) across topics")
print("\n3. sentiment_by_topic.png:")
print("   Overall sentiment comparison across policy topics")
print("\nAll figures saved to results/ directory")
