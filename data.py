from lxml import etree
from pathlib import Path
import pandas as pd 

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def get_exploration_path():
    return "ParlaMint/Samples/ParlaMint-BA/"

def get_samples_path():
    return "ParlaMint/Samples/"

def load_taxonomy(taxonomy_path):
    tree = etree.parse(str(taxonomy_path))
    categories = tree.xpath("//tei:category", namespaces=NS)
    mapping = {}
    for cat in categories:
        cat_id = cat.get("{http://www.w3.org/XML/1998/namespace}id")
        full_desc = cat.xpath("string(./tei:catDesc)", namespaces=NS)
        if cat_id and full_desc:
            if ":" in full_desc:
                mapping[cat_id] = full_desc.split(":", 1)[-1].strip()
            else:
                mapping[cat_id] = full_desc.strip()
    return mapping

def get_topic_taxonomy(country_code, data_path="ParlaMint"):
    path = Path(get_samples_path()) / f"ParlaMint-{country_code}"
    topic_tax_path = path / "ParlaMint-taxonomy-topic.xml"
    return load_taxonomy(topic_tax_path)

def get_speaker_taxonomy(country_code, data_path="ParlaMint"):
    path = Path(get_samples_path()) / f"ParlaMint-{country_code}"
    topic_tax_path = path / "ParlaMint-taxonomy-politicalOrientation.xml"
    return load_taxonomy(topic_tax_path)

def get_ches_taxonomy(country_code, data_path="ParlaMint"):
    path = Path(get_samples_path()) / f"ParlaMint-{country_code}"
    topic_tax_path = path / "ParlaMint-taxonomy-CHES.xml"
    return load_taxonomy(topic_tax_path)

def get_named_entity_taxonomy(country_code, data_path="ParlaMint"):
    path = Path(get_samples_path()) / f"ParlaMint-{country_code}"
    topic_tax_path = path / "ParlaMint-taxonomy-NER.ana.xml"
    return load_taxonomy(topic_tax_path)

def get_sentiment_taxonomy(country_code, data_path="ParlaMint"):
    path = Path(get_samples_path()) / f"ParlaMint-{country_code}"
    topic_tax_path = path / "ParlaMint-taxonomy-sentiment.ana.xml"
    return load_taxonomy(topic_tax_path)

def get_speaker_type_taxonomy(country_code, data_path="ParlaMint"):
    path = Path(get_samples_path()) / f"ParlaMint-{country_code}"
    topic_tax_path = path / "ParlaMint-taxonomy-speaker_types.xml"
    return load_taxonomy(topic_tax_path)

def get_party_at_time(row, speaker_info):
    speaker = row['speaker']
    speech_date = row['date']
    
    speaker_data = speaker_info.get(speaker, {})
    affiliations = speaker_data.get('party_info', [])
    
    if not affiliations or affiliations == 'Unknown':
        return 'Unknown'
    
    for party_name, date_from, date_to in affiliations:
        start = pd.to_datetime(date_from) if date_from else pd.to_datetime('1950-01-01')
        end = pd.to_datetime(date_to) if date_to else pd.to_datetime('2030-12-31')
        
        if start <= speech_date <= end:
            return party_name
            
    return speaker_data.get('party', 'Unknown')

def extract_country_code(file_path):
    """(Private) Extracts country code from file path structure."""
    parts = str(file_path).split('/')
    return next((p.replace('ParlaMint-', '') for p in parts if p.startswith('ParlaMint-')), 'Unknown')

def get_min_max_dates(country_code, df):
    """
    Looks into the data for the min and max dates of speeches of a country.
    Returns: (min_date, max_date) as pd.to_datetime objects.
    """
    max_date = pd.to_datetime('1900-12-31')
    min_date = pd.to_datetime('2030-01-01')

    country_df = df[df['country_code'] == country_code]
    if country_df.empty:
        return None, None
    for date in country_df['date'].dropna().unique():
        if date < min_date:
            min_date = date
        if date > max_date:
            max_date = date
                
    return min_date, max_date

def load_country_resources(country_code, data_path, df):
    """
    (Private) Loads all taxonomy and metadata resources for a specific country.
    Returns: (topic_map, speaker_info, org_info) or (None, None, None) if missing.
    """
    country_dir = Path(get_samples_path()) / f"ParlaMint-{country_code}"
    person_path = country_dir / f"ParlaMint-{country_code}-listPerson.xml"
    org_path = country_dir / f"ParlaMint-{country_code}-listOrg.xml"
    
    if not person_path.exists() or not org_path.exists():
        return None, None, None

    # Load Resources
    s_info = load_person_list(person_path)
    o_info = load_org_list(org_path)
    
    # Eliminate from s_info any parties that have the role 'parliament'
    parties_to_remove = {org_id for org_id, info in o_info.items() if info.get('role') == 'parliament'}
    for speaker_id, info in s_info.items():
        filtered_party_info = [
            (party_ref, start_date, end_date)
            for party_ref, start_date, end_date in info.get('party_info', [])
            if party_ref not in parties_to_remove
        ]
        s_info[speaker_id]['party_info'] = filtered_party_info

    # If in s_info a speaker start_date or/and end_date is missing, set to extreme values of the country data
    min_date, max_date = get_min_max_dates(country_code, df)
    for speaker_id, info in s_info.items():
        updated_party_info = []
        for party_ref, start_date, end_date in info.get('party_info', []):
            start = start_date if start_date else str(min_date.date())
            end = end_date if end_date else str(max_date.date())
            updated_party_info.append((party_ref, start, end))
        s_info[speaker_id]['party_info'] = updated_party_info

    try:
        # data_path here is the root "ParlaMint" Folder
        t_map = get_topic_taxonomy(country_code, data_path) 
    except Exception:
        t_map = {}
        
    return t_map, s_info, o_info

def load_person_list(person_path):
    tree = etree.parse(str(person_path))
    root = tree.getroot()
    
    persons = root.xpath("//tei:person", namespaces=NS)
    person_mapping = {}
    
    for p in persons:
        p_id = p.get("{http://www.w3.org/XML/1998/namespace}id")
        
        # Get full name
        forenames = p.xpath(".//tei:persName/tei:forename/text()", namespaces=NS)
        surname = p.xpath(".//tei:persName/tei:surname/text()", namespaces=NS)
        full_name = f"{' '.join(forenames)} {''.join(surname)}".strip()
        
        # Get sex
        sex = p.xpath("./tei:sex/@value", namespaces=NS)
        sex = sex[0] if sex else None
        
        # Get list of party affiliation
        affiliations = p.xpath(".//tei:affiliation[contains(@ref, '#')]", namespaces=NS)

        # Create a list of (party_ref, start_date, end_date) tuples
        party_info = []
        for aff in affiliations:
            party_ref = aff.get("ref").replace("#", "")
            start_date = aff.get("from")
            end_date = aff.get("to")
            party_info.append((party_ref, start_date, end_date))

        person_mapping[f"#{p_id}"] = {
            "name": full_name,
            "sex": sex,
            "party_info": party_info
        }
        
    return person_mapping


def load_org_list(org_path):
    tree = etree.parse(str(org_path))
    root = tree.getroot()
    
    orgs = root.xpath("//tei:org", namespaces=NS)
    org_mapping = {}
    
    for org in orgs:
        org_id = org.get("{http://www.w3.org/XML/1998/namespace}id")
        role = org.get("role")
        
        full_name = org.xpath("./tei:orgName[@full='yes']/text()", namespaces=NS)
        full_name = full_name[0] if full_name else None
        
        if not full_name:
            any_name = org.xpath("./tei:orgName/text()", namespaces=NS)
            full_name = any_name[0] if any_name else None
            
        org_mapping[org_id] = {
            "name": full_name.strip() if full_name else None,
            "role": role
        }
        
    return org_mapping

def load_org_relations(org_path):
    if not Path(org_path).exists():
        return []

    tree = etree.parse(str(org_path))
    root = tree.getroot()
    
    relations = []
    for rel in root.xpath("//tei:listRelation/tei:relation", namespaces=NS):
        name = rel.get("name")
        
        start = rel.get("from", "0001-01-01")
        end = rel.get("to", "9999-12-31")
        
        party_ids = set()
        
        # 1. Coalition is usually 'mutual'
        if rel.get("mutual"):
            party_ids.update(rel.get("mutual").strip().split())
            
        # 2. Opposition is usually 'active' (the ones opposing)
        if rel.get("active"):
             party_ids.update(rel.get("active").strip().split())
        
        clean_ids = {pid.replace("#", "") for pid in party_ids if pid}
        
        if clean_ids:
            relations.append({
                "type": name,
                "start": start,
                "end": end,
                "parties": clean_ids
            })
            
    return relations

def get_gov_opp_status(party_id, date, relations):
    """
    Returns 'Government', 'Opposition', or 'Other' based on the party and date.
    Helper function to query the loaded relations.
    """
    if not party_id or party_id == 'Unknown' or not relations:
        return 'Other'
        
    date_str = str(date.date()) if hasattr(date, 'date') else str(date).split()[0]
    
    for r in relations:
        # check date overlap
        if r['start'] <= date_str <= r['end']:
            if party_id in r['parties']:
                if r['type'] == 'coalition':
                    return 'Government'
                elif r['type'] == 'opposition':
                    return 'Opposition'
                    
    return 'Other'

def load_speaker_type(data_path, df):
    countries = df['country_code'].unique()

    relations = []

    for country_code in countries:
        country_dir = data_path / f'ParlaMint-{country_code}'
        org_path = country_dir / f'ParlaMint-{country_code}-listOrg.xml'
        relation = load_org_relations(org_path)
        relations = relations + relation

    df['speaker_type'] = df.apply(
        lambda row: get_gov_opp_status(row['party_id'], row['date'], relations), axis=1
    )

    return df

def clean_speaker_types(df):
    return df[df['speaker_type'] != 'Other']

def get_countries(data_folder="ParlaMint"):
    data_path = Path(data_folder)
    xml_en_files = []
    for country_dir in (Path(get_samples_path())).iterdir():
        if country_dir.is_dir():
            xml_en_files.extend(list(country_dir.rglob("*-en_*.ana.xml")))
    return xml_en_files

def parse_parlamint_xml(xml_path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    date_str = root.xpath(".//tei:settingDesc/tei:setting/tei:date/@when", namespaces=NS)
    speech_date = pd.to_datetime(date_str[0]) if date_str else None

    data = []
    utterances = root.xpath("//tei:u", namespaces=NS)

    for u in utterances:
        u_id = u.get("{http://www.w3.org/XML/1998/namespace}id")
        speaker = u.get("who")
        ana = u.get("ana", "")
        topics = [a.replace("topic:", "") for a in ana.split() if "topic:" in a]

        sentences = u.xpath(".//tei:s", namespaces=NS)
        for s in sentences:
            s_id = s.get("{http://www.w3.org/XML/1998/namespace}id")
            
            sentiment_node = s.xpath(".//tei:measure[@type='sentiment']", namespaces=NS)
            sentiment_score = float(sentiment_node[0].get("quantity")) if sentiment_node else None
            
            named_entities = s.xpath(".//tei:name/@type", namespaces=NS)
            
            tokens = []
            reconstructed_text = ""
            
            token_elements = s.xpath(".//tei:w | .//tei:pc", namespaces=NS)
            
            for i, token_el in enumerate(token_elements):
                token_text = token_el.text or ""
                
                token_data = {
                    "text": token_text,
                    "type": "word" if token_el.tag.endswith("w") else "punct",
                    "lemma": token_el.get("lemma"),
                    "pos": token_el.get("pos"),
                    "msd": token_el.get("msd"),
                    "sem": token_el.get("function")
                }
                tokens.append(token_data)
                
                reconstructed_text += token_text
                if token_el.get("join") != "right" and i < len(token_elements) - 1:
                    reconstructed_text += " "

            data.append({
                "u_id": u_id,
                "s_id": s_id,
                "speaker": speaker,
                "topics": topics,
                "sentiment": sentiment_score,
                "entities": list(set(named_entities)),
                "text": reconstructed_text,
                "tokens": tokens, 
                "date": speech_date,
                "sentence_length": len(tokens)
            })
            
    return data

def enrich_dataframe(df, data_path):
    """(Private) Enriches the dataframe with metadata by country."""
    unique_countries = df['country_code'].dropna().unique()
    
    # Pre-initialize columns
    new_cols = ['speaker_name', 'speaker_sex', 'party_id', 'party_name', 'topic_labels']
    for col in new_cols:
        df[col] = 'Unknown'
    df['topic_labels'] = df['topic_labels'].apply(lambda x: [])

    for country in unique_countries:
        if country == 'Unknown': continue

        # Load resources once per country
        topic_map, s_info, o_info = load_country_resources(country, data_path, df)
        if not s_info: continue

        mask = df['country_code'] == country
        subset = df[mask]

        # 1. Speaker Types (Dict Lookups are fast)
        df.loc[mask, 'speaker_name'] = subset['speaker'].map(
            lambda x: s_info.get(x, {}).get('name', 'Unknown')
        )
        df.loc[mask, 'speaker_sex'] = subset['speaker'].map(
            lambda x: s_info.get(x, {}).get('sex', 'Unknown')
        )

        # 2. Temporal Party Logic
        df.loc[mask, 'party_id'] = subset.apply(
            lambda row: get_party_at_time(row, s_info), axis=1
        )

        # 3. Party Names
        df.loc[mask, 'party_name'] = df.loc[mask, 'party_id'].map(
            lambda x: o_info.get(x, {}).get('name', 'Unknown')
        )

        # 4. Topics
        if topic_map:
            df.loc[mask, 'topic_labels'] = subset['topics'].map(
                lambda t_list: [topic_map.get(t, t) for t in t_list] if t_list else []
            )
        else:
             df.loc[mask, 'topic_labels'] = subset['topics']
             
    return df

def get_full_dataframe(data_folder="ParlaMint"):
    """
    Main pipeline function to load, parse, and enrich ParlaMint data.
    """
    data_path = Path(data_folder)
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder {data_folder} not found.")

    # 1. Discovery
    xml_files = get_countries(data_folder)
    print(f"Found {len(xml_files)} XML files. Parsing...")

    # 2. Parsing loop (Optimized: Inject country code immediately)
    all_records = []
    for f in xml_files:
        country_code = extract_country_code(f)
        file_data = parse_parlamint_xml(f)
        
        # Inject country code into records before creating DF
        # This avoids complex mapping logic later
        for record in file_data:
            record['country_code'] = country_code
            
        all_records.extend(file_data)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # 3. Enrichment
    print("Enriching data with metadata...")
    df = enrich_dataframe(df, data_path)
    print(df.shape)

    # 4. Load speaker types and clean
    print("Loading speaker types...")
    df = load_speaker_type(Path(get_samples_path()), df)
    df = clean_speaker_types(df)

    print("Dataframe construction complete.")
    return df

if __name__ == "__main__":
    df = get_full_dataframe()
    print(df.shape)
    print(df.head(10))