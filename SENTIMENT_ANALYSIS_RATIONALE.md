# Sentiment Analysis Rationale

## Research Focus: Institutional Targeting

This third analysis complements **framing** and **emotion** analyses by examining **which institutions, countries, and policies** government and opposition parties direct criticism toward.

## Three-Part Analytical Framework

| Analysis | Research Question | Focus | Output |
|----------|------------------|-------|--------|
| **1. Framing** | "What policy issues do they discuss?" | **Semantic topics** | Government vs Opposition use different vocabulary to frame identical issues (BERTopic + c-TF-IDF) |
| **2. Emotion** | "What affective tone do they use?" | **Emotional expression** | Government vs Opposition express different emotions (anger, joy, fear, sadness) |
| **3. Sentiment** | "What institutions/policies do they blame?" | **Institutional targeting** | Government vs Opposition direct negativity toward different types of targets (ORG/LOC/MISC) |

## Why Exclude PER (Persons)?

**Interpersonal sentiment is already well-studied:**
- The ParlaSent paper (Evkoski et al., 2025) comprehensively covers **affective polarization** between MPs
- Research question: Do MPs show more negativity toward opposing party MPs vs their own?
- Finding: Yes, significant affective polarization exists across all 6 European parliaments
- Methodology: Named entity disambiguation to match PER mentions to specific MPs

**Our novel contribution:**
- Extend ParlaSent methodology to **institutional targets** (ORG, LOC, MISC)
- Research question: Do gov/opp differ in **which institutions and policies** they criticize?
- Complements policy-focused framing analysis by adding "who gets blamed" dimension

## Entity Types Analyzed

### ORG (Organizations)
- Political parties (e.g., "Labour Party", "CDU")
- International bodies (e.g., "European Union", "WHO", "NATO")
- Corporations (e.g., "Pfizer", "Shell")
- Government agencies (e.g., "NHS", "Environmental Agency")

### LOC (Locations)
- Countries (e.g., "Russia", "China", "United States")
- Regions (e.g., "Brussels", "Middle East")
- Cities (e.g., "London", "Paris")

### MISC (Miscellaneous - Policies & Events)
- Policies (e.g., "Brexit", "Green Deal", "austerity measures")
- Treaties (e.g., "Lisbon Treaty", "Paris Agreement")
- Events (e.g., "COVID-19 pandemic", "financial crisis")
- Initiatives (e.g., "vaccination program", "climate plan")

## Hypotheses

### H1: Institutional Focus Differs
- **Government**: More positive toward national/international institutions they work with (ORG)
- **Opposition**: More critical of same institutions, seeking to expose failures

### H2: Policy Criticism Patterns
- **Government**: Positive/neutral toward their own policies (MISC), may criticize opposition's proposed policies
- **Opposition**: More negative toward government policies (MISC), positioning as alternative

### H3: Geographical Targeting
- **Government**: May criticize external actors (other countries via LOC) to deflect blame
- **Opposition**: May criticize government's handling of international relations

## Example Research Insights

### Topic: Healthcare
- **Gov sentiment**: Positive toward WHO (ORG: +0.8), neutral toward their health reforms (MISC: +0.2)
- **Opp sentiment**: Critical of national health agency (ORG: -0.3), very negative toward reforms (MISC: -0.6)
- **Interpretation**: Opposition criticizes institutional failures and policy choices

### Topic: Environment
- **Gov sentiment**: Positive toward international agreements (MISC: +0.5), positive toward EU (ORG: +0.7)
- **Opp sentiment**: Critical of corporations (ORG: -0.4), negative toward government's climate plan (MISC: -0.5)
- **Interpretation**: Opposition focuses blame on polluters and insufficient policies

## Methodological Alignment

### With Framing Analysis
- Framing identifies **semantic topics** (what they discuss)
- Sentiment identifies **target criticism** within those topics (who/what gets blamed)
- Combined insight: "Opposition frames healthcare as 'system failure' and directs negativity toward NHS"

### With Emotion Analysis
- Emotion measures **affective tone** (anger, fear, joy expressed)
- Sentiment measures **directed negativity** (specific targets of criticism)
- Combined insight: "Opposition uses angry tone (emotion) specifically when mentioning EU (sentiment)"

### Distinct from Both
- Not interpersonal (PER excluded → see ParlaSent paper for that)
- Not overall tone (emotion analysis covers that)
- Specifically about **institutional accountability narratives**

## Methodology

### Approach
Target-directed sentiment analysis using ParlaSent to measure how government and opposition direct sentiment toward institutional targets (organizations, countries, policies) rather than individuals.

### Technical Implementation

**Model:**
- `classla/xlm-r-parlasent` (XLM-RoBERTa fine-tuned on parliamentary speech)
- Architecture: Multilingual transformer (270M parameters)
- Output: Continuous sentiment score -1 (negative) to +1 (positive)
- Originally trained for interpersonal sentiment, extended to institutional targets

**Inference:**
- Sentence-level sentiment scoring (batch_size=16)
- CPU/GPU agnostic
- Max sequence length: 512 tokens

**Entity Types (Institutional Targets):**
- **ORG**: Organizations (parties, EU, corporations, agencies)
- **LOC**: Locations (countries, regions, cities)
- **MISC**: Policies & Events (treaties, reforms, initiatives)
- **PER excluded**: Interpersonal sentiment covered in ParlaSent paper

### Aggregation Strategy
For each speech and entity type:
1. Extract sentences containing entity type (ORG/LOC/MISC)
2. Apply ParlaSent model to get sentiment scores
3. Compute mean sentiment per entity type per speech
4. Compare government vs opposition at speech level
5. Report mean, std, speech counts, mention counts

### Algorithm Flow
```
Sentence text → ParlaSent model → Sentiment score
                                        ↓
                          Filter by entity type (ORG/LOC/MISC)
                                        ↓
                          Group by speech + speaker_type
                                        ↓
                          Mean sentiment per entity type
                                        ↓
                    Compare Government vs Opposition
```

### Parameters
- `batch_size`: 16 (memory/speed tradeoff)
- `max_length`: 512 tokens
- `min_sentences_per_group`: 20 (matches framing/emotion methodology)
- `entity_types`: ['ORG', 'LOC', 'MISC'] (excludes PER)

### Quality Controls
- Minimum sentence threshold per topic (20 per group)
- Sufficient entity mentions for reliable aggregation
- Standard deviation reporting for variability assessment
- Speech-level aggregation reduces sentence-level noise

## References

Evkoski, B., Mozetič, I., Ljubešić, N., & Novak, P. K. (2025).
*Affective Polarization across European Parliaments*.
arXiv preprint arXiv:2508.18916v2.

Mochtak, M., Rupnik, P., & Ljubešić, N. (2023).
*The ParlaSent multilingual training dataset for sentiment identification in parliamentary proceedings*.
arXiv preprint arXiv:2309.09783.
