# Torch Dataset definition
from pydoc import doc
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
from transformers import BertTokenizer, BertModel
import os
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = './data/'
EXTRA_DATA_DIR =  DATA_DIR + 'extra/'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
SPECIAL_TOKENS = {"candidate_token": "[CANDIDATE]", "context_token": "[CONTEXT]", "mention_token": "[MENTION]"}
# Add special tokens to tokenizer
tokenizer.add_tokens(list(SPECIAL_TOKENS.values()))

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
    def collate_fn_train(batch, entity_embed_tensor, device='cuda'):
        # Unpack the batch data
        contexts, indexes, full_mentions, item_ids = zip(*batch)
        entity_embeddings = entity_embed_tensor[indexes, :]
        # move entity_embeddings to device
        

        context_texts = [SPECIAL_TOKENS['context_token'] + " " + context for context in contexts]
        # Tokenize the context texts
        contexts_tokenized = tokenizer(context_texts, padding=True, truncation=True, return_tensors='pt')

        full_mention_texts = [SPECIAL_TOKENS['mention_token'] + " " + full_mention for full_mention in full_mentions]
        # Tokenize the full_mention texts
        full_mentions_tokenized = tokenizer(full_mention_texts, padding=True, truncation=True, return_tensors='pt')

        syntax_candidate_ids_batch = util.semantic_search(entity_embeddings, corpus_embeddings, top_k=3, score_function=util.dot_score)
        valid_ids = []
        candidates_input_ids = []
        candidates_attention_mask = []
        candidates_token_type_ids = []
        labels = []
        for i, (full_mention, syntax_candidate_ids, label) in enumerate(zip(full_mentions, syntax_candidate_ids_batch, item_ids)):
            if full_mention in anchor_to_candidate:
                candidate_ids = anchor_to_candidate[full_mention].copy()
            else:
                candidate_ids = []

            candidate_ids += [wiki_items.iloc[candidate_id['corpus_id']]['item_id'] for candidate_id in syntax_candidate_ids] 

            # drop duplicates
            candidate_ids = list(set(candidate_ids))

            candidate_texts = [SPECIAL_TOKENS["candidate_token"] + " " + id_to_description[candidate_id]['description'] for candidate_id in candidate_ids]

            # Tokenize the candidate texts
            candidate_tokenized = tokenizer(candidate_texts, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
            # Padding candidates to ensure same number of candidates in each batch
            pad_size = 8 - candidate_tokenized.input_ids.size(0)
            if pad_size > 0:
                padded_ids = torch.full((pad_size, candidate_tokenized.input_ids.size(1)), tokenizer.pad_token_id, dtype=torch.long)
                candidate_tokenized.input_ids = torch.cat([candidate_tokenized.input_ids, padded_ids], dim=0)

                padded_mask = torch.full((pad_size, candidate_tokenized.attention_mask.size(1)), 0, dtype=torch.long)
                candidate_tokenized.attention_mask = torch.cat([candidate_tokenized.attention_mask, padded_mask], dim=0)

                # Pad token_type_ids if they exist
                if 'token_type_ids' in candidate_tokenized:
                    padded_token_type_ids = torch.full((pad_size, candidate_tokenized.token_type_ids.size(1)), 0, dtype=torch.long)
                    candidate_tokenized.token_type_ids = torch.cat([candidate_tokenized.token_type_ids, padded_token_type_ids], dim=0)
            candidates_input_ids.append(candidate_tokenized.input_ids)
            candidates_attention_mask.append(candidate_tokenized.attention_mask)
            candidates_token_type_ids.append(candidate_tokenized.token_type_ids)


            # Check if label is in candidate_ids
            try:
                label_index = candidate_ids.index(label)
                valid_ids.append(i)
            except ValueError:
                label_index = 0
            
            labels.append(label_index)

        # Only keep valid ids
        candidates_input_ids = torch.stack(candidates_input_ids, dim=0)[valid_ids]
        candidates_attention_mask = torch.stack(candidates_attention_mask, dim=0)[valid_ids]
        candidates_token_type_ids = torch.stack(candidates_token_type_ids, dim=0)[valid_ids]

        contexts_input_ids = contexts_tokenized.input_ids[valid_ids]
        contexts_attention_mask = contexts_tokenized.attention_mask[valid_ids]
        contexts_token_type_ids = contexts_tokenized.token_type_ids[valid_ids]

        full_mentions_input_ids = full_mentions_tokenized.input_ids[valid_ids]
        full_mentions_attention_mask = full_mentions_tokenized.attention_mask[valid_ids]
        full_mentions_token_type_ids = full_mentions_tokenized.token_type_ids[valid_ids]

        labels = [labels[i] for i in valid_ids]
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Move the rest to device
        candidates_input_ids = candidates_input_ids.to(device)
        candidates_attention_mask = candidates_attention_mask.to(device)
        candidates_token_type_ids = candidates_token_type_ids.to(device)

        contexts_input_ids = contexts_input_ids.to(device)
        contexts_attention_mask = contexts_attention_mask.to(device)
        contexts_token_type_ids = contexts_token_type_ids.to(device)

        full_mentions_input_ids = full_mentions_input_ids.to(device)
        full_mentions_attention_mask = full_mentions_attention_mask.to(device)
        full_mentions_token_type_ids = full_mentions_token_type_ids.to(device)
        
        return candidates_input_ids, candidates_attention_mask, candidates_token_type_ids, contexts_input_ids, contexts_attention_mask, contexts_token_type_ids, full_mentions_input_ids, full_mentions_attention_mask, full_mentions_token_type_ids, labels

class EntityClassifier(torch.nn.Module):
    def __init__(self, transformer_model, hidden_size, num_candidates=8, device='cuda'):
        super().__init__()
        self.transformer = BertModel.from_pretrained(transformer_model)
        self.transformer.resize_token_embeddings(len(tokenizer))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.transformer.config.hidden_size * 3, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, 1)  # Output is a score for each candidate
        )
        self.num_candidates = num_candidates
        self.device = device
        self.to(device)

    def forward(self, candidates_input_ids, candidates_attention_mask, candidates_token_type_ids, contexts_input_ids, contexts_attention_mask, contexts_token_type_ids, full_mentions_input_ids, full_mentions_attention_mask, full_mentions_token_type_ids):
        # Generate embeddings
        context_embeds = self.transformer(
            input_ids=contexts_input_ids,
            attention_mask=contexts_attention_mask,
            token_type_ids=contexts_token_type_ids
        ).last_hidden_state.mean(dim=1)
        mention_embeds = self.transformer(
            input_ids=full_mentions_input_ids,
            attention_mask=full_mentions_attention_mask,
            token_type_ids=full_mentions_token_type_ids
        ).last_hidden_state.mean(dim=1)
        # Concatenate embeddings
        combined_embeds = torch.cat((context_embeds, mention_embeds), dim=1) # (batch_size, 2 * hidden_size)

        # Prepare for storing candidate scores
        candidate_scores = torch.zeros(contexts_input_ids.size(0), self.num_candidates, device=self.device)
        # Loop over each candidate

        # candidate_input_ids: (batch_size, num_candidates, max_length)
        # candidate_attention_mask: (batch_size, num_candidates, max_length)
        # candidate_token_type_ids: (batch_size, num_candidates, max_length)
        # If we do: candidates_input_ids[:, i], we get the i-th candidate of each sample in the batch, shape: (batch_size, max_length)
        for i in range(self.num_candidates):
            # Extract embeddings for the i-th candidate of each sample
            candidate_embeds = self.transformer(
                input_ids=candidates_input_ids[:, i],
                attention_mask=candidates_attention_mask[:, i],
                token_type_ids=candidates_token_type_ids[:, i]
            ).last_hidden_state.mean(dim=1)

            # Concatenate embeddings and pass through classifier
            candidate_combined_embed = torch.cat((candidate_embeds, combined_embeds), dim=1) # shape: (batch_size, 3 * hidden_size)
            classifier_out = self.classifier(candidate_combined_embed)
            candidate_scores[:, i] = classifier_out.squeeze() # we do squeeze because the output is (batch_size, 1), but we want (batch_size)
        return candidate_scores
        
def delete_corpus_embeds():
    global corpus_embeddings
    logger.log('Deleting corpus embeddings...')
    # Move it to cpu
    corpus_embeddings = corpus_embeddings.to('cpu')
    # Delete it
    del corpus_embeddings
    torch.cuda.empty_cache()