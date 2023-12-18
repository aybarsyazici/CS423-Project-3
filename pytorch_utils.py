# Torch Dataset definition
from ast import List
from calendar import c
from pydoc import doc
import stat
import string
import tokenize
from typing import Type
from torch.utils.data import Dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from termcolor import colored
import helper
from IPython.display import display
import pickle
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import os
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = './data/'
EXTRA_DATA_DIR =  DATA_DIR + 'extra/'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
SPECIAL_TOKENS = {
    "candidate_token": "<candidate>", 
    "context_token": "<context>", 
    "mention_token": "<mention>"
}
# Add special tokens to tokenizer
# tokenizer.add_special_tokens({'additional_special_tokens': list(SPECIAL_TOKENS.values())})

def update_full_mentions(df):
    # Define a function to get the longest full_mention
    def get_longest_mention(group):
        return group.loc[group['full_mention'].str.len().idxmax()]

    # Group by 'doc_id' and 'token', then apply the function
    longest_mentions = df.groupby(['doc_id', 'token']).apply(get_longest_mention)

    # Reset index to turn the group identifiers into columns
    longest_mentions.reset_index(drop=True, inplace=True)

    # Merge the longest mentions back into the original DataFrame
    # This step will replace the full_mention in the original DataFrame
    df = df.drop('full_mention', axis=1).merge(longest_mentions[['doc_id', 'token', 'full_mention']], on=['doc_id', 'token'], how='left')

    return df

class Logger:
    def __init__(self, output=True):
        self.output = output
    
    def log(self, message):
        if self.output:
            color = colored("[PyTorch-Utils]", 'green')
            str = f'{color}: {message}'
            print(str)

logger = Logger()


logger.log('Loading anchor candidate data...')
with open(DATA_DIR + 'pkl/anchor_to_candidate_new.pkl', 'rb') as handle:
    anchor_to_candidate = pickle.load(handle)

logger.log('Loading wikipedia title embedings...')
with open(DATA_DIR + 'pkl/corpus_embeddings.pt', 'rb') as handle:
    corpus_embeddings = torch.load(handle, map_location=torch.device('cuda'))

logger.log('Loading KB explanations...')
with open(DATA_DIR + 'pkl/id_to_kb_explanation_joined.pkl', 'rb') as handle:
    kb_explanations = pickle.load(handle)

logger.log('Loading wikipedia items...')
wiki_items = pd.read_csv(DATA_DIR + 'wiki_lite/wiki_items.csv')
# Fill all the NaNs with empty strings
wiki_items = wiki_items.fillna('')
# create column called description that is the concatenation of the title and the description
wiki_items['description'] = wiki_items['wikipedia_title'] + ' ' + wiki_items['en_description']
# only keep item_id and description
wiki_items_temp = wiki_items[['item_id', 'description']]
# find the max_length of the description
description_max_length = wiki_items_temp['description'].str.len().max()
# set item_id as the index
wiki_items_temp = wiki_items_temp.set_index('item_id')
# convert to dictionary
id_to_description = wiki_items_temp.to_dict('index')
del wiki_items_temp

class EntityDataSample:
    def __init__(self, mention, candidate_entities, label):
        self.mention = mention
        self.candidate_entities = candidate_entities
        self.label = label



class EntityDataset(Dataset):
    def __init__(self, train=True, model_name = 'all-MiniLM-L6-v2', device='cuda', DATA_DIR = './data/'):
        global corpus_embeddings
        self.device = device
        self.train = train
        set_type = 'train' if train else 'test'
        logger.log(f'Loading {set_type} set...')
        self.model = SentenceTransformer(model_name, device='cuda')
        if train:
            self.train_data = pd.read_csv(DATA_DIR + 'train_data_preprocessed.csv')
        else:
            self.train_data = pd.read_csv(DATA_DIR + 'test.csv')
        self.train_data['sentence_id'] = (self.train_data['token'] == '.').cumsum()
        self.train_data['doc_id'] = self.train_data['token'].str.startswith('-DOCSTART-').cumsum()
        self.train_data['full_mention'] = self.train_data['full_mention'].fillna('')
        self.train_data['old_full_mention'] = self.train_data['full_mention']
        self.train_data = update_full_mentions(self.train_data)
        # Create a column called entity_loc that contains the location of the entity in the DOCUMENT
        # This is basically the word offset from the start of the document, to do this we can create a column that has
        # the row_id which resets for each document
        self.train_data_no_doc_start = self.train_data[self.train_data['token'].str.startswith('-DOCSTART-') == False]
        
        file_name_to_load = 'train_context_text_150.pkl' if train else 'test_context_text_150.pkl'
        exists = False
        # does it exist under data/pkl ?
        if os.path.exists(DATA_DIR + 'pkl/' + file_name_to_load):
            logger.log(f'Loading {file_name_to_load} from data/pkl...')
            with open(DATA_DIR + 'pkl/' + file_name_to_load, 'rb') as handle:
                self.context_text = pickle.load(handle)
            exists = True
        if not exists:
            self.train_data_no_doc_start['entity_loc'] = self.train_data_no_doc_start.groupby(['doc_id']).cumcount()
        self.not_nan = self.train_data_no_doc_start['wiki_url'].notna()
        self.not_nme = self.train_data_no_doc_start['wiki_url'] != '--NME--'
        self.entity_df = self.train_data_no_doc_start[self.not_nan & self.not_nme]
        # now for each entity in entity_df, create a context window, of total length 300, from both sides
        # save this to an array so that we retrieve it in __getitem__
        if not exists:
            self.context_text = []
            for i, row in tqdm(self.entity_df.iterrows(), total=self.entity_df.shape[0]):
                doc_id = int(row['doc_id'])
                # get the whole document
                document = self.train_data_no_doc_start[self.train_data_no_doc_start['doc_id'] == doc_id]['token'].tolist()
                # get the entity location
                entity_loc = int(row['entity_loc'])
                # get the context window
                # First try to get 150 tokens before the entity
                context = document[max(0, entity_loc - 75):entity_loc]
                # Now try to get until 300 from the right, note the document might be shorter than 300
                context += document[entity_loc:entity_loc + 150 - len(context)]
                self.context_text.append(' '.join(context))
            # save to filename
            logger.log(f'Saving {file_name_to_load} to data/pkl...')
            with open(DATA_DIR + 'pkl/' + file_name_to_load, 'wb') as handle:
                pickle.dump(self.context_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.log(f'Now generating entity embeddings...')
        not_nan = self.train_data['wiki_url'].notna()
        not_nme = self.train_data['wiki_url'] != '--NME--'
        self.entity_embeddings = self.model.encode(self.train_data[not_nan & not_nme]['full_mention'].to_list(), show_progress_bar=True, convert_to_tensor=True)
        print(f'Entity Length: {len(self.entity_embeddings)}')
        print(f'Entity shape: {self.entity_embeddings.shape}')
        self.old_entity_embeddings = self.model.encode(self.train_data[not_nan & not_nme]['old_full_mention'].to_list(), show_progress_bar=True, convert_to_tensor=True)
        # move entity_embeddings to device
        logger.log(f'Now computing syntax candidates for each entity...')
        self.entity_embeddings = self.entity_embeddings.to(device)
        self.syntax_candidates = util.semantic_search(self.entity_embeddings, corpus_embeddings, top_k=3, score_function=util.dot_score)
        logger.log(f'Now computing OLD syntax candidates for each entity...')
        self.old_entity_embeddings = self.old_entity_embeddings.to(device)
        self.old_syntax_candidates = util.semantic_search(self.old_entity_embeddings, corpus_embeddings, top_k=3, score_function=util.dot_score)
        # move entity_embeddings to cpu
        self.entity_embeddings = self.entity_embeddings.to('cpu')
        self.old_entity_embeddings = self.old_entity_embeddings.to('cpu')
        # delete unnecessary variables
        del self.entity_embeddings
        del self.old_entity_embeddings
        del self.not_nan
        del self.not_nme
        del self.model
        del self.train_data
        del self.train_data_no_doc_start
        self.entity_df_indexes = []
        self.candidate_ids = []
        self.inputs = []
        self.labels = []
        self.skip_count = 0
        # Now generating inputs and labels
        logger.log(f'Now generating inputs and labels...')
        exists = False
        if train:
            # does the already tokenized inputs exist?
            if os.path.exists(DATA_DIR + f'{set_type}_batch_inputs_input_ids.pt') and \
            os.path.exists(DATA_DIR + f'{set_type}_batch_inputs_attention_mask.pt') and \
            os.path.exists(DATA_DIR + f'{set_type}_batch_entity_df_indexes.pkl') and \
            os.path.exists(DATA_DIR + f'{set_type}_batch_labels.pkl') and \
            os.path.exists(DATA_DIR + f'{set_type}_batch_candidate_ids.pkl'):
                logger.log(f'Loading {set_type}_batch_inputs_input_ids.pt, {set_type}_batch_inputs_attention_mask.pt, {set_type}_batch_entity_df_indexes.pkl, {set_type}_batch_labels.pkl, {set_type}_batch_candidate_ids.pkl from data/pkl...')
                self.tokenized_inputs_input_ids = torch.load(DATA_DIR + f'{set_type}_batch_inputs_input_ids.pt', map_location=torch.device('cpu'))
                self.tokenized_inputs_attention_mask = torch.load(DATA_DIR + f'{set_type}_batch_inputs_attention_mask.pt', map_location=torch.device('cpu'))
                with open(DATA_DIR + f'{set_type}_batch_entity_df_indexes.pkl', 'rb') as handle:
                    self.entity_df_indexes = pickle.load(handle)
                with open(DATA_DIR + f'{set_type}_batch_labels.pkl', 'rb') as handle:
                    self.labels = pickle.load(handle)
                with open(DATA_DIR + f'{set_type}_batch_candidate_ids.pkl', 'rb') as handle:
                    self.candidate_ids = pickle.load(handle)
                exists = True
            else:
                for i in range(len(self.entity_df)):
                    candidate_ids_batch = []
                    inputs_batch = []
                    row = self.entity_df.iloc[i]
                    full_mention = row['full_mention'].strip().lower()
                    old_full_mention = row['old_full_mention'].strip().lower()
                    candidate_ids = anchor_to_candidate.get(full_mention, []) + \
                                    [wiki_items.iloc[candidate['corpus_id']]['item_id'] for candidate in self.syntax_candidates[i] if candidate['score'] > 0.95]
                    
                    old_candidate_ids = anchor_to_candidate.get(old_full_mention, []) + \
                                    [wiki_items.iloc[candidate['corpus_id']]['item_id'] for candidate in self.old_syntax_candidates[i] if candidate['score'] > 0.95]
                    
                    candidate_ids = list(set(candidate_ids + old_candidate_ids))
                    if row['item_id'] not in candidate_ids:
                        self.skip_count += 1
                        continue  # Skip this sample if ground truth not in candidates

                    item_id = row['item_id']
                    # Find the index of the correct candidate
                    label = candidate_ids.index(item_id)
                    
                    for candidate_id in candidate_ids:
                        kb_texts = kb_explanations.get(candidate_id, id_to_description[candidate_id]['description'])
                        candidate_text = f"{self.context_text[i]} [SEP] {full_mention} [SEP] {kb_texts}"
                        inputs_batch.append(candidate_text)
                        candidate_ids_batch.append(candidate_id)

                    # are candidate_ids less than 16?
                    if len(candidate_ids_batch) < 16:
                        # pad with -1
                        candidate_ids_batch += [-1] * (16 - len(candidate_ids_batch))
                        inputs_batch += [''] * (16 - len(inputs_batch))
                    
                    self.inputs.append(inputs_batch)
                    self.entity_df_indexes.append(i)
                    self.candidate_ids.append(candidate_ids_batch)
                    self.labels.append(label)

        else:
            # does the already tokenized inputs exist?
            if os.path.exists(DATA_DIR + f'{set_type}_batch_inputs_input_ids.pt') and \
            os.path.exists(DATA_DIR + f'{set_type}_batch_inputs_attention_mask.pt') and \
            os.path.exists(DATA_DIR + f'{set_type}_batch_entity_df_indexes.pkl') and \
            os.path.exists(DATA_DIR + f'{set_type}_batch_candidate_ids.pkl'):
                logger.log(f'Loading {set_type}_batch_inputs_input_ids.pt, {set_type}_batch_inputs_attention_mask.pt, {set_type}_batch_entity_df_indexes.pkl, {set_type}_batch_candidate_ids.pkl from data/pkl...')
                self.tokenized_inputs_input_ids = torch.load(DATA_DIR + f'{set_type}_batch_inputs_input_ids.pt', map_location=torch.device('cpu'))
                self.tokenized_inputs_attention_mask = torch.load(DATA_DIR + f'{set_type}_batch_inputs_attention_mask.pt', map_location=torch.device('cpu'))
                with open(DATA_DIR + f'{set_type}_batch_entity_df_indexes.pkl', 'rb') as handle:
                    self.entity_df_indexes = pickle.load(handle)
                with open(DATA_DIR + f'{set_type}_batch_candidate_ids.pkl', 'rb') as handle:
                    self.candidate_ids = pickle.load(handle)
                exists = True
            else:
                for i in range(len(self.entity_df)):
                    candidate_ids_batch = []
                    inputs_batch = []
                    row = self.entity_df.iloc[i]
                    full_mention = row['full_mention'].strip().lower()
                    old_full_mention = row['old_full_mention'].strip().lower()
                    candidate_ids = anchor_to_candidate.get(full_mention, []) + \
                                    [wiki_items.iloc[candidate['corpus_id']]['item_id'] for candidate in self.syntax_candidates[i] if candidate['score'] > 0.95]
                    
                    old_candidate_ids = anchor_to_candidate.get(old_full_mention, []) + \
                                    [wiki_items.iloc[candidate['corpus_id']]['item_id'] for candidate in self.old_syntax_candidates[i] if candidate['score'] > 0.95]
                    
                    candidate_ids = list(set(candidate_ids + old_candidate_ids))
                    
                    for candidate_id in candidate_ids:
                        kb_texts = kb_explanations.get(candidate_id, id_to_description[candidate_id]['description'])
                        candidate_text = f"{self.context_text[i]} [SEP] {full_mention} [SEP] {kb_texts}"
                        inputs_batch.append(candidate_text)
                        candidate_ids_batch.append(candidate_id)

                    # are candidate_ids less than 16?
                    if len(candidate_ids_batch) < 16:
                        # pad with -1
                        candidate_ids_batch += [-1] * (16 - len(candidate_ids_batch))
                        inputs_batch += [''] * (16 - len(inputs_batch))
                    
                    self.inputs.append(inputs_batch)
                    self.entity_df_indexes.append(i)
                    self.candidate_ids.append(candidate_ids_batch)

        del self.context_text
        del self.syntax_candidates
        del self.old_syntax_candidates
        
        # Tokenize
        if not exists:
            logger.log(f'Now tokenizing...')
            # Flatten inputs
            flat_inputs = [item for sublist in self.inputs for item in sublist]
            tokenized_inputs = tokenizer(flat_inputs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            # Now unflatten (each mention has 16 candidates)
            self.tokenized_inputs_input_ids = torch.split(tokenized_inputs['input_ids'], 16)
            self.tokenized_inputs_attention_mask = torch.split(tokenized_inputs['attention_mask'], 16)
            # Their shape is (num_mentions, 16, 512)

            # save tokenized_inputs_input_ids and tokenized_inputs_attention_mask and entity_df_indexes
            torch.save(self.tokenized_inputs_input_ids, DATA_DIR + f'{set_type}_batch_inputs_input_ids.pt')
            torch.save(self.tokenized_inputs_attention_mask, DATA_DIR + f'{set_type}_batch_inputs_attention_mask.pt')
            with open(DATA_DIR + f'{set_type}_batch_entity_df_indexes.pkl', 'wb') as handle:
                pickle.dump(self.entity_df_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(DATA_DIR + f'{set_type}_batch_candidate_ids.pkl', 'wb') as handle:
                pickle.dump(self.candidate_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if train:
                with open(DATA_DIR + f'{set_type}_batch_labels.pkl', 'wb') as handle:
                    pickle.dump(self.labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del self.inputs




    def __len__(self):
        # Find the number of rows that have wiki_url not NaN or --NME--
        return len(self.entity_df_indexes)
    
    def __getitem__(self, index):
        tokenized_inputs_input_ids = self.tokenized_inputs_input_ids[index]
        tokenized_inputs_attention_mask = self.tokenized_inputs_attention_mask[index]
        entity_df_index = self.entity_df_indexes[index]
        candidate_ids = self.candidate_ids[index]
        # Shapes: (16, 512), (16, 512), (), (16)
        if self.train:
            label = self.labels[index] # Shape: (), value is an int
            return tokenized_inputs_input_ids, tokenized_inputs_attention_mask, entity_df_index, label
        else:
            return tokenized_inputs_input_ids, tokenized_inputs_attention_mask, entity_df_index, candidate_ids

        
    @staticmethod
    def collate_fn_train(batch, device=torch.device('cuda')):
        tokenized_input_id_batch = [] # Shape: (batch_size, 16, 512)
        tokenized_attention_mask_batch = [] # Shape: (batch_size, 16, 512)
        indexes = [] # Shape: (batch_size)
        labels = [] # Shape: (batch_size)
        for i, data in enumerate(batch):
            tokenized_inputs_input_ids, tokenized_inputs_attention_mask, index, label = data
            # Shapes: (16, 512), (16, 512), (), ()
            tokenized_input_id_batch.append(tokenized_inputs_input_ids)
            tokenized_attention_mask_batch.append(tokenized_inputs_attention_mask)
            indexes.append(index)
            labels.append(label)
        
        # No need to pad
        tokenized_input_id_batch = torch.stack(tokenized_input_id_batch).to(device)
        tokenized_attention_mask_batch = torch.stack(tokenized_attention_mask_batch).to(device)
        indexes = torch.tensor(indexes, dtype=torch.long, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        return tokenized_input_id_batch, tokenized_attention_mask_batch, indexes, labels

        

    
    @staticmethod
    def collate_fn_test(batch, device=torch.device('cuda')):
        tokenized_input_id_batch = []
        tokenized_attention_mask_batch = []
        indexes = []
        candidate_ids_batch = []
        for i, data in enumerate(batch):
            tokenized_inputs_input_ids, tokenized_inputs_attention_mask, index, candidate_ids = data
            tokenized_input_id_batch.append(tokenized_inputs_input_ids)
            tokenized_attention_mask_batch.append(tokenized_inputs_attention_mask)
            indexes.append(index)
            candidate_ids_batch.append(candidate_ids)

        # No need to pad
        tokenized_input_id_batch = torch.stack(tokenized_input_id_batch).to(device)
        tokenized_attention_mask_batch = torch.stack(tokenized_attention_mask_batch).to(device)
        indexes = torch.tensor(indexes, dtype=torch.long, device=device)
        candidate_ids_batch = torch.tensor(candidate_ids_batch, dtype=torch.long, device=device)

        return tokenized_input_id_batch, tokenized_attention_mask_batch, indexes, candidate_ids_batch


class EntityClassifier(torch.nn.Module):
    def __init__(self, transformer_model, hidden_size, num_candidates=16, device=torch.device('cuda')):
        super().__init__()
        self.transformer = DistilBertModel.from_pretrained(transformer_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.transformer.config.hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)  # Output is a score for each candidate
        )
        self.num_candidates = num_candidates
        self.device = device
        self.to(device)

    def forward(self, tokenized_inputs_input_ids, tokenized_inputs_attention_mask):
        # input shape: (batch_size, 16, 512)
        # attention mask shape: (batch_size, 16, 512)
        # We want to output a score for each candidate
        # output shape: (batch_size, 16)
        out_probs = torch.zeros((tokenized_inputs_input_ids.shape[0], self.num_candidates), device=self.device)
        for i in range(self.num_candidates):
            out = self.transformer(tokenized_inputs_input_ids[:, i, :], tokenized_inputs_attention_mask[:, i, :])
            out = out.last_hidden_state.mean(dim=1)  # Shape: (batch_size, hidden_size)
            out = self.classifier(out) # Shape: (batch_size, 1)
            out_probs[:, i] = out.squeeze()
        return out_probs # Shape: (batch_size, 16)
        
def delete_corpus_embeds():
    global corpus_embeddings
    logger.log('Deleting corpus embeddings...')
    # Delete it
    del corpus_embeddings
    torch.cuda.empty_cache()