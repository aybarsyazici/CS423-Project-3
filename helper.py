from rapidfuzz import process, fuzz
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from collections import Counter
import json
import re
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

MIN_VIEWS = 5
MIN_ANCHOR_TARGET_COUNT = 2
NUM_KLAT_LINES = 5_343_564
NUM_PAGE_LINES = 5_362_174

class KdwdLinkAnnotatedText:
    def __init__(self, file_path):
        self.file_path = file_path
    def __iter__(self):
        with open(self.file_path) as fp:
            for line in fp:
                yield json.loads(line)

class AnchorTargetStats:
    
    def __init__(
        self,
        at_count_df,
        text_normalizer,
    ):
        """Anchor-target statistics 
        
        Args:
            at_count_df: (normalized_anchor_text, target_page) counts and metadata
            text_normalizer: text cleaning function for anchor texts
        """
        self._at_count_df = at_count_df
        self.text_normalizer = text_normalizer

    def get_aliases_from_page_id(self, page_id):
        """Return anchor strings used to refer to entity"""
        bool_mask = self._at_count_df['target_page_id'] == page_id
        return (
            self._at_count_df.
            loc[bool_mask].copy().
            sort_values('p_anchor_given_target', ascending=False)
        )
    
    def get_disambiguation_candidates_from_text(self, text):
        """Return candidate entities for input text"""
        normalized_text = self.text_normalizer(text)
        bool_mask = self._at_count_df.index == normalized_text
        return (
            self._at_count_df.loc[bool_mask]
        )

def text_normalizer(text):                              
    """Return text after stripping external whitespace and lower casing."""   
    return text.strip().lower()

def generate_candidates_with_fuzzy_matching(entity_name, wiki_items, redirects, item_aliases, threshold=85):
    candidates = []

    # Add Fuzzy Matching for Aliases
    all_aliases = item_aliases[item_aliases['en_alias'].str.len() >= len(entity_name)/2]['en_alias'].tolist()
    fuzzy_matches = process.extract(entity_name, all_aliases, score_cutoff=threshold, limit=10)

    for match in fuzzy_matches:
        matched_alias = match[0]
        matched_item_ids = item_aliases[item_aliases['en_alias'] == matched_alias]['item_id']
        for matched_item_id in matched_item_ids:
            matched_item = wiki_items[wiki_items['item_id'] == matched_item_id]

            if matched_item.empty:
                continue

            title = matched_item['wikipedia_title'].iloc[0].replace(' ', '_')
            # Handle redirects
            if title in redirects['source'].values:
                print(f"Redirect found for {title}")
                title = redirects[redirects['source'] == title]['target'].iloc[0]
            url = f"http://en.wikipedia.org/wiki/{title}"
            candidates.append((matched_item_id, matched_alias, match[1], url))

    return entity_name, list(set(candidates))  # Remove duplicates


def generate_candidates_with_page_rank(document, statements, item_aliases):
    entities_in_first_doc = document[document['entity_tag'].notnull()]['full_mention'].unique()
    # map these entity mentions to their corresponding item_id
    # entities_in_first_doc = item_aliases[item_aliases['en_alias'].isin(entities_in_first_doc)]
    # The code above doesn't work due to case sensitivity
    # so we have to lower case both sides
    entities_in_first_doc = [entity.lower() for entity in entities_in_first_doc]
    entities_in_first_doc = item_aliases[item_aliases['en_alias'].str.lower().isin(entities_in_first_doc)]
    # Filter statements to include only those involving the identified item IDs
    relevant_statements = statements[statements['source_item_id'].isin(entities_in_first_doc['item_id']) | 
                                 statements['target_item_id'].isin(entities_in_first_doc['item_id'])]
    relevant_statements.drop_duplicates(subset=['source_item_id', 'target_item_id', 'edge_property_id'], inplace=True)

    # Build a graph from the filtered statements
    G = nx.DiGraph()
    for _, row in relevant_statements.iterrows():
        G.add_edge(row['source_item_id'], row['target_item_id'], label=row['edge_property_id'])

    # Personalized PageRank for each entity in the first document
    entity_candidates = {}
    for entity in entities_in_first_doc.en_alias.unique():
        # Use aliases to find potential item IDs for the entity
        potential_item_ids = item_aliases[item_aliases['en_alias'].str.lower() == entity.lower()]['item_id']

        # Personalized PageRank
        personalization = {node: 1 if node in potential_item_ids.values else 0 for node in G.nodes()}
        pagerank_scores = pagerank(G, personalization=personalization)

        # Extract top candidates based on scores
        top_candidates = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        entity_candidates[entity] = top_candidates

    return entity_candidates


def get_anchor_info(file_path, wiki_items_joined=None):

    # check if anchor_target_counts.csv exists
    try:
        at_count_df = pd.read_csv("data/pkl/anchor_target_counts.csv")
        return at_count_df
    except:
        pass

    klat = KdwdLinkAnnotatedText(file_path)
    anchor_target_counts = Counter()
    for page in tqdm(
        klat, 
        total=NUM_KLAT_LINES, 
        desc='calculating anchor-target counts'
    ):
        for section in page['sections']:
            spans = [
                (offset, offset + length) for offset, length in 
                zip(section['link_offsets'], section['link_lengths'])]
            anchor_texts = [section['text'][ii:ff] for ii,ff in spans]
            keys = [
                (anchor_text, target_page_id) for anchor_text, target_page_id in 
                zip(anchor_texts, section['target_page_ids'])]
            anchor_target_counts.update(keys)

    at_count_df = pd.DataFrame([
        (row[0][0], row[0][1], row[1]) for row in anchor_target_counts.most_common()],
        columns=['anchor_text', 'target_page_id', 'anchor_target_count'])

    at_count_df["normalized_anchor_text"] = at_count_df["anchor_text"].apply(text_normalizer)
    at_count_df = at_count_df.loc[at_count_df['normalized_anchor_text'].str.len() > 0, :]

    at_count_df = (                                               
        at_count_df.                                              
        groupby(["normalized_anchor_text", "target_page_id"])["anchor_target_count"].   
        sum().                                                               
        to_frame("anchor_target_count").
        sort_values('anchor_target_count', ascending=False).
        reset_index()                                                        
    )

    at_count_df = pd.merge(
        at_count_df,
        wiki_items_joined,
        how="inner",
        left_on="target_page_id",
        right_on="page_id"
    )

    at_count_df = at_count_df.rename(columns={
        'wikipedia_title': 'target_page_title',
        'item_id': 'target_item_id',
        'views': 'target_page_views'})
    
    at_count_df = at_count_df[[
    "en_label",
    "en_description",
    "normalized_anchor_text",
    "target_page_id",
    "target_item_id",
    "target_page_title",
    "target_page_views",
    "anchor_target_count"]]

    bool_mask_1 = at_count_df["anchor_target_count"] >= MIN_ANCHOR_TARGET_COUNT
    bool_mask_2 = at_count_df["target_page_views"] >= MIN_VIEWS
    bool_mask = bool_mask_1 & bool_mask_2
    at_count_df = at_count_df.loc[bool_mask, :].copy()

    norm = at_count_df.groupby("target_page_id")["anchor_target_count"].transform("sum")
    at_count_df["p_anchor_given_target"] = at_count_df["anchor_target_count"] / norm
    norm = at_count_df.groupby("normalized_anchor_text")["anchor_target_count"].transform("sum")
    at_count_df["p_target_given_anchor"] = at_count_df["anchor_target_count"] / norm

    # save to .csv file
    at_count_df.to_csv("anchor_target_counts.csv", index=False)

    return at_count_df



