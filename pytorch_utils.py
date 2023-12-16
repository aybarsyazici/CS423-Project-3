# Torch Dataset definition
from ast import List
from pydoc import doc
import stat
import string
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
with open(DATA_DIR + 'pkl/id_to_kb_explanation.pkl', 'rb') as handle:
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
        # move entity_embeddings to device
        logger.log(f'Now computing syntax candidates for each entity...')
        self.entity_embeddings = self.entity_embeddings.to(device)
        self.syntax_candidates = util.semantic_search(self.entity_embeddings, corpus_embeddings, top_k=3, score_function=util.dot_score)
        # move entity_embeddings to cpu
        self.entity_embeddings = self.entity_embeddings.to('cpu')
        # delete entity_embeddings
        del self.entity_embeddings
        # delete all the unnecessary variables
        del self.not_nan
        del self.not_nme
        del not_nan
        del not_nme
        del self.model
        #del wiki_items_joined
        #del pages
        #del statements


    def __len__(self):
        # Find the number of rows that have wiki_url not NaN or --NME--
        return len(self.entity_df)
    
    def __getitem__(self, index):
        row = self.entity_df.iloc[index]
        context = self.context_text[index]
        full_mention = row['full_mention'].strip().lower()
        if self.train:
            return context, index, full_mention, row['item_id']
        else:
            return context, index, full_mention
        
    @staticmethod
    def collate_fn_train(batch, syntax_candidates_list, device='cuda'):
        tokenized_inputs_input_ids = []
        tokenized_inputs_attention_mask = []
        labels = []
        for i, data in enumerate(batch):
            context, index, full_mention, item_id = data
            if full_mention in anchor_to_candidate:
                candidate_ids = anchor_to_candidate[full_mention].copy()
            else:
                candidate_ids = []

            syntax_candidates = syntax_candidates_list[index]

            candidate_ids += [wiki_items.iloc[candidate_id['corpus_id']]['item_id'] for candidate_id in syntax_candidates]

            # Remove duplicates and limit to 8 candidates
            candidate_ids = list(set(candidate_ids))[:8]

            # do we have the ground truth in candidate_ids?
            try:
                label = candidate_ids.index(item_id)
            except ValueError:
                # skip this sample
                continue

            kb_text = ""  # Initialize kb_text as an empty string
            for candidate_id in candidate_ids:
                candidate_text = f"{context} [SEP] {full_mention}"
                if candidate_id in kb_explanations:
                    kb_texts = kb_explanations[candidate_id] # List of strings
                else:
                    kb_texts = [id_to_description[candidate_id]['description']]
                # while the total number of words of candidate_text and kb_texts is less than 512 or we still have kb_texts left we keep adding
                for kb_text in kb_texts:
                    # Check the token count before adding the next kb_text
                    token_count = len(tokenizer.tokenize(candidate_text))
                    kb_text_token_count = len(tokenizer.tokenize(kb_text))
                    if token_count + kb_text_token_count + 1 > 512:  # +1 for the additional [SEP] token
                        break  # Stop adding if the next kb_text will exceed the limit
                    # Add the kb_text with a [SEP] token
                    candidate_text += f" [SEP] {kb_text}"
                
                # Tokenize the candidate_text
                tokenized_candidate_text = tokenizer(candidate_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
                # Add the tokenized candidate_text to the list
                tokenized_inputs_input_ids.append(tokenized_candidate_text['input_ids'][0])
                tokenized_inputs_attention_mask.append(tokenized_candidate_text['attention_mask'][0])
                # If this is the correct candidate, label it as 1, otherwise label it as 0
                labels.append(1 if candidate_id == item_id else 0)

            num_candidates = len(candidate_ids)
            num_padding_needed = 8 - num_candidates
            if num_padding_needed > 0:
                # Generate padding for input_ids and attention_mask
                padding_input_ids = torch.zeros((num_padding_needed, 512), dtype=torch.long)
                padding_attention_mask = torch.zeros((num_padding_needed, 512), dtype=torch.long)
                # Append padding to the tokenized inputs
                tokenized_inputs_input_ids.extend(padding_input_ids)
                tokenized_inputs_attention_mask.extend(padding_attention_mask)
                # Append 0's to the labels to indicate these are padding (not real candidates)
                labels.extend([0] * num_padding_needed)
            

        # if labels is empty, return None
        if len(labels) == 0:
            return None, None, None
        # Tokenize all inputs

        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float, device=device)

        # Move tokenized inputs to device
        tokenized_inputs_input_ids = torch.stack(tokenized_inputs_input_ids).to(device)
        tokenized_inputs_attention_mask = torch.stack(tokenized_inputs_attention_mask).to(device)
        # tokenized_inputs_token_type_ids = tokenized_inputs['token_type_ids'].to(device)

        return tokenized_inputs_input_ids, tokenized_inputs_attention_mask, labels

    @staticmethod
    def collate_fn_test(batch, syntax_candidates_list, device='cuda'):
        tokenized_inputs_input_ids = []
        tokenized_inputs_attention_mask = []
        for i, data in enumerate(batch):
            context, index, full_mention = data
            if full_mention in anchor_to_candidate:
                candidate_ids = anchor_to_candidate[full_mention].copy()
            else:
                candidate_ids = []

            syntax_candidates = syntax_candidates_list[index]

            candidate_ids += [wiki_items.iloc[candidate_id['corpus_id']]['item_id'] for candidate_id in syntax_candidates]

            # Remove duplicates and limit to 8 candidates
            candidate_ids = list(set(candidate_ids))[:8]

            kb_text = ""  # Initialize kb_text as an empty string
            for candidate_id in candidate_ids:
                candidate_text = f"{context} [SEP] {full_mention}"
                kb_texts = kb_explanations[candidate_id] # List of strings
                # while the total number of words of candidate_text and kb_texts is less than 512 or we still have kb_texts left we keep adding
                for kb_text in kb_texts:
                    # Check the token count before adding the next kb_text
                    token_count = len(tokenizer.tokenize(candidate_text))
                    kb_text_token_count = len(tokenizer.tokenize(kb_text))
                    if token_count + kb_text_token_count + 1 > 512:  # +1 for the additional [SEP] token
                        break  # Stop adding if the next kb_text will exceed the limit
                    # Add the kb_text with a [SEP] token
                    candidate_text += f" [SEP] {kb_text}"
                
                # Tokenize the candidate_text
                tokenized_candidate_text = tokenizer(candidate_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
                # Add the tokenized candidate_text to the list
                tokenized_inputs_input_ids.append(tokenized_candidate_text['input_ids'][0])
                tokenized_inputs_attention_mask.append(tokenized_candidate_text['attention_mask'][0])
                # If this is the correct candidate, label it as 1, otherwise label it as 0

            num_candidates = len(candidate_ids)
            num_padding_needed = 8 - num_candidates
            if num_padding_needed > 0:
                # Generate padding for input_ids and attention_mask
                padding_input_ids = torch.zeros((num_padding_needed, 512), dtype=torch.long)
                padding_attention_mask = torch.zeros((num_padding_needed, 512), dtype=torch.long)
                # Append padding to the tokenized inputs
                tokenized_inputs_input_ids.extend(padding_input_ids)
                tokenized_inputs_attention_mask.extend(padding_attention_mask)
                # Append 0's to the labels to indicate these are padding (not real candidates)
            

        # Move tokenized inputs to device
        tokenized_inputs_input_ids = torch.stack(tokenized_inputs_input_ids).to(device)
        tokenized_inputs_attention_mask = torch.stack(tokenized_inputs_attention_mask).to(device)
        # tokenized_inputs_token_type_ids = tokenized_inputs['token_type_ids'].to(device)

        return tokenized_inputs_input_ids, tokenized_inputs_attention_mask,


class EntityClassifier(torch.nn.Module):
    def __init__(self, transformer_model, hidden_size, num_candidates=8, device='cuda'):
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

    def forward(self, tokenized_inputs_input_ids, tokenized_inputs_attention_mask, tokenized_inputs_token_type_ids):
        # Generate embeddings
        embeds = self.transformer(
            input_ids=tokenized_inputs_input_ids,
            attention_mask=tokenized_inputs_attention_mask,
        ).last_hidden_state.mean(dim=1)

        # Classify
        logits = self.classifier(embeds)
        # logits shape: (batch_size, num_candidates)

        return logits
        
def delete_corpus_embeds():
    global corpus_embeddings
    logger.log('Deleting corpus embeddings...')
    # Move it to cpu
    corpus_embeddings = corpus_embeddings.to('cpu')
    # Delete it
    del corpus_embeddings
    torch.cuda.empty_cache()