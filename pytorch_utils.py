# Torch Dataset definition
from torch.utils.data import Dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from termcolor import colored
import helper
from IPython.display import display
import pickle
import torch


EXTRA_DATA_DIR = './data/extra/'

class EntityDataSample:
    def __init__(self, mention, candidate_entities, label):
        self.mention = mention
        self.candidate_entities = candidate_entities
        self.label = label


class Logger:
    def __init__(self, output=True):
        self.output = output
    
    def log(self, message):
        if self.output:
            color = colored("[PyTorch-Utils]", 'green')
            str = f'{color}: {message}'
            print(str)

class EntityDataset(Dataset):
    def __init__(self, train=True, model_name = 'all-MiniLM-L6-v2', device='cuda', DATA_DIR = './data/'):
        self.device = device
        self.logger = Logger()
        set_type = 'Train' if train else 'Test'


        self.logger.log(f'Loading {model_name} model & Generating sentence embeddings of {set_type} set...')
        self.model = SentenceTransformer(model_name, device=device)
        self.train_data = pd.read_csv(DATA_DIR + 'train_data_preprocessed.csv')
        self.train_data['sentence_id'] = (self.train_data['token'] == '.').cumsum()
        self.not_nan = self.train_data['wiki_url'].notna()
        self.not_nme = self.train_data['wiki_url'] != '--NME--'
        self.entity_df = self.train_data[self.not_nan & self.not_nme]
        train_data_no_doc_start = self.train_data[self.train_data['token'].str.startswith('-DOCSTART-') == False]
        # create sentence-id column by cumsumming the number of tokens that are == '.'
        all_tokens = []
        # Iterate through each document
        for sentence_id in tqdm(train_data_no_doc_start['sentence_id'].unique()):
            # Filter the DataFrame for the current document
            sentence_data = train_data_no_doc_start[train_data_no_doc_start['sentence_id'] == sentence_id]
            
            # remove tokens with NaN values
            sentence_data = sentence_data[sentence_data['token'].notna()]

            # Extract the token column and convert it to a list
            sentence_tokens = sentence_data['token'].tolist()
            
            # Append this list to the all_tokens list
            all_tokens.append(sentence_tokens)
        self.all_tokens = all_tokens
        all_tokens = [' '.join(tokens) for tokens in all_tokens]
        self.sentence_embeddings = self.model.encode(all_tokens, show_progress_bar=True)


        self.logger.log(f'Now generating entity embeddings...')
        not_nan = self.train_data['wiki_url'].notna()
        not_nme = self.train_data['wiki_url'] != '--NME--'
        self.entity_embeddings = self.model.encode(self.train_data[not_nan & not_nme]['full_mention'].to_list(), show_progress_bar=True)
        wiki_items = pd.read_csv(DATA_DIR + 'wiki_lite/wiki_items.csv')

        if False:
            self.logger.log(f'Now loading statements and properties.')
            statements = pd.read_csv(DATA_DIR + 'wiki_lite/statements.csv')
            properties = pd.read_csv(DATA_DIR + 'wiki_lite/property.csv')
            relevant_statements = pd.merge(
                statements,
                properties,
                how="left",
                left_on="edge_property_id",
                right_on="property_id"
            )
            # rename en_description to property_description
            relevant_statements = relevant_statements.rename(columns={'en_description': 'property_description'})
            relevant_statements = relevant_statements[['source_item_id', 'property_id', 'property_description', 'target_item_id']]
            # join on wiki_items on source_item_id and target_item_id to get source_item_description and target_item_description
            relevant_statements = pd.merge(
                relevant_statements,
                wiki_items,
                how="left",
                left_on="source_item_id",
                right_on="item_id"
            )
            # rename description to source_item_description
            relevant_statements.rename(columns={'en_description': 'source_item_description', 'en_label': 'source_label'}, inplace=True)
            relevant_statements = relevant_statements[['source_item_id', 'source_label', 'source_item_description', 'property_id', 'property_description', 'target_item_id']]
            relevant_statements = pd.merge(
                relevant_statements,
                wiki_items,
                how="left",
                left_on="target_item_id",
                right_on="item_id"
            )
            # rename description to target_item_description
            relevant_statements.rename(columns={'en_description': 'target_item_description', 'en_label': 'target_label'}, inplace=True)
            relevant_statements = relevant_statements[['source_item_id', 'source_label', 'source_item_description', 'property_id', 'property_description', 'target_item_id', 'target_label', 'target_item_description']]
            # reset_index to create a new column called row_num
            relevant_statements = relevant_statements.reset_index()
            # rename index column to row_num
            self.relevant_statements = relevant_statements.rename(columns={'index': 'row_num'}, inplace=False)
            # drop NaN rows
            self.relevant_statements = self.relevant_statements.dropna()
            #self.relevant_statements['statement'] = relevant_statements['source_label'] + '('+relevant_statements['source_item_description']+')' + ' ' + relevant_statements['property_description'] + ' ' + relevant_statements['target_label'] + '('+relevant_statements['target_item_description']+')'
        
        
        self.logger.log(f'Generating DataFrame for candidate generation...')
        pages = pd.read_csv(EXTRA_DATA_DIR + 'page.csv')
        wiki_items_joined = wiki_items.merge(pages[['page_id', 'item_id', 'views']], on='item_id', how='left')
        self.at_count_df = helper.get_anchor_info(f'{EXTRA_DATA_DIR}link_annotated_text.jsonl', wiki_items_joined)
        self.at_count_df.set_index('normalized_anchor_text',inplace=True)
        # now sort by p_target_given_anchor
        self.at_count_df.sort_values(by=['p_target_given_anchor'], ascending=False, inplace=True)
        self.at_count_df = self.at_count_df.groupby(level=0).head(5)
        self.anchor_target_counts = helper.AnchorTargetStats(
            self.at_count_df, helper.text_normalizer
        )


        self.logger.log(f'Generating Embedding for candidate descriptions...')
        # load from pickle
        with open(DATA_DIR + 'pkl/item_id_to_description_embedding.pkl', 'rb') as handle:
            self.item_id_to_description_embedding = pickle.load(handle)


    def __len__(self):
        # Find the number of rows that have wiki_url not NaN or --NME--
        return len(self.entity_df)

    def __getitem__(self, index):
        try:
            row = self.entity_df.iloc[index]
            # get sentence_id
            sentence_id = row['sentence_id']
            # get sentence embedding
            sentence_embed = self.sentence_embeddings[sentence_id]
            # get entity embedding
            entity_embed = self.entity_embeddings[index]
            full_mention = row['full_mention']
            # get candidate entities
            candidates = self.at_count_df.loc[[full_mention.strip().lower()]].head(5)
            candidate_ids = candidates['target_item_id'].to_list()
            candidate_description_embeddings = [self.item_id_to_description_embedding[candidate_id][1] for candidate_id in candidate_ids]
            # pad the candidate_description_embeddings and candidate_ids to length 5
            candidate_description_embeddings = candidate_description_embeddings + [torch.zeros(384)] * (5 - len(candidate_description_embeddings))
            # pad candidate_ids with 0s
            candidate_ids = candidate_ids + [0] * (5 - len(candidate_ids))
            # get label
            label = row['item_id']
            return sentence_embed, entity_embed, candidate_ids, candidate_description_embeddings, label
        except Exception as e:
            # return all zeros
            return torch.zeros(384), torch.zeros(384), [0] * 5, [torch.zeros(384)] * 5, 0
    
    # collate_fn is optional, but allows us to do custom batching
    @staticmethod
    def collate_fn(batch, device='cuda'):
        # A data tuple has the form:
        # sentence_embedding, entity_embedding, candidate_ids, candidate_description_embeddings, label
        # where the lists have length batch_size

        # Zip our batch
        sentence_embeddings, entity_embeddings, candidate_ids, candidate_description_embeddings, labels = zip(*batch)


        # their shapes are:
        # sentence_embeddings: (batch_size, 384)
        # entity_embeddings: (batch_size, 384)
        # candidate_ids: (batch_size, 5)
        # candidate_description_embeddings: (batch_size, 5, 384)
        # labels: (batch_size,)
        # convert to torch tensors and move to device
        sentence_embeddings = torch.tensor(sentence_embeddings, dtype=torch.float32, device=device)
        entity_embeddings = torch.tensor(entity_embeddings, dtype=torch.float32, device=device)
        candidate_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        candidate_description_embeddings = torch.tensor(candidate_description_embeddings, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        return sentence_embeddings, entity_embeddings, candidate_ids, candidate_description_embeddings, labels
        