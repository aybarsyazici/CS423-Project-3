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
import os

DATA_DIR = './data/'
EXTRA_DATA_DIR =  DATA_DIR + 'extra/'


def update_full_mentions(df):
    # Define a function to get the longest full_mention
    def get_longest_mention(group):
        return group.loc[group['full_mention'].str.len().idxmax()]

    # Group by 'doc_id' and 'token', then apply the function
    longest_mentions = df.groupby(['doc_id', 'token']).apply(get_longest_mention)

    # Reset index to turn the group identifiers into columns
    longest_mentions.reset_index(drop=True, inplace=True)

    longest_mentions.rename(columns={"full_mention": "full_mention_longest"}, inplace=True)

    # Merge the longest mentions back into the original DataFrame
    # This step will replace the full_mention in the original DataFrame
    #df = df.drop('full_mention', axis=1).merge(longest_mentions[['doc_id', 'token', 'full_mention']], on=['doc_id', 'token'], how='left')
    
    df = df.merge(longest_mentions[['doc_id', 'token', 'full_mention_longest']], on=['doc_id', 'token'], how='left')
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

logger.log('Loading item_id to statement embedding data...')
# with open(DATA_DIR + 'pkl/item_id_to_description_embedding.pkl', 'rb') as handle:
#     item_id_to_description_embedding = pickle.load(handle)
statement_embeddings = np.load('./bge_statement_features.npy')
#'./data/wiki_lite/statement_features_item_id_to_index.pickle'
with open('./data/wiki_lite/statement_bge_features_item_id_to_index.pickle', 'rb') as handle:
    statement_item_id_to_row = pickle.load(handle)

logger.log('Loading item_id to description embedding data...')
description_embeddings = np.load('./data/wiki_lite/bge_wiki_features.npy')
#'./data/wiki_lite/wiki_features_item_id_to_index.pickle'
with open('./data/wiki_lite/wiki_features_bge_item_id_to_index.pickle', 'rb') as handle:
    description_item_id_to_row = pickle.load(handle)

logger.log('Loading wikipedia title embedings...')
with open(DATA_DIR + 'pkl/final_corpus_embeddings.pt', 'rb') as handle:
    corpus_embeddings = torch.load(handle, map_location=torch.device('cuda'))

logger.log('Loading wikipedia items...')
wiki_items = pd.read_csv(DATA_DIR + 'wiki_lite/statements_grouped_filtered.csv')

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
        self.model = SentenceTransformer(model_name, device=device)
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
        logger.log(f'Now generating context embeddings...')
        self.train_data_no_doc_start['entity_loc'] = self.train_data_no_doc_start.groupby(['doc_id']).cumcount()
        self.not_nan = self.train_data_no_doc_start['wiki_url'].notna()
        self.not_nme = self.train_data_no_doc_start['wiki_url'] != '--NME--'
        self.entity_df = self.train_data_no_doc_start[self.not_nan & self.not_nme]
        # now for each entity in entity_df, create a context window, of total length 300, from both sides
        # save this to an array so that we retrieve it in __getitem__
        self.context_text = []
        self.context_for_semantic_search = []
        self.context_for_features = []
        for i, row in tqdm(self.entity_df.iterrows(), total=self.entity_df.shape[0]):
            doc_id = int(row['doc_id'])
            # get the whole document
            document = self.train_data_no_doc_start[self.train_data_no_doc_start['doc_id'] == doc_id]['token'].tolist()
            # get the entity location
            entity_loc = int(row['entity_loc'])
            left_index = max(entity_loc - 75, 0)
            right_index = min(entity_loc + 75, len(document) - 1)

            # Adjust if we are near the start or end of the document
            if left_index == 0:
                # Get more words from the right if possible
                right_index = min(entity_loc + 150 - (entity_loc - left_index), len(document) - 1)
            elif right_index == len(document) - 1:
                # Get more words from the left if possible
                left_index = max(entity_loc - (150 - (right_index - entity_loc)), 0)

            # Extract the context
            context_window = document[left_index:right_index + 1]
            self.context_text.append(' '.join(context_window))
            self.context_for_semantic_search.append(row['full_mention'])
            
            left_index = max(entity_loc - 50, 0)
            right_index = min(entity_loc + 50, len(document) - 1)

            # Adjust if we are near the start or end of the document
            if left_index == 0:
                # Get more words from the right if possible
                right_index = min(entity_loc + 100 - (entity_loc - left_index), len(document) - 1)
            elif right_index == len(document) - 1:
                # Get more words from the left if possible
                left_index = max(entity_loc - (100 - (right_index - entity_loc)), 0)

            # Extract the context
            context_window = document[left_index:right_index + 1]
            self.context_for_features.append(' '.join(context_window))

        self.context_embeddings = self.model.encode(self.context_text, show_progress_bar=True, convert_to_tensor=True)
        del self.context_text
        self.context_embeddings_for_semantic_search = self.model.encode(self.context_for_semantic_search, show_progress_bar=True, convert_to_tensor=True)
        del self.context_for_semantic_search
        self.context_embeddings_for_semantic_search = (self.context_embeddings + self.context_embeddings_for_semantic_search) / 2  # type: ignore
        del self.context_embeddings
        del self.model
        self.model = SentenceTransformer('BAAI/bge-base-en', device=device)
        self.context_embeddings = self.model.encode(self.context_for_features, show_progress_bar=True, convert_to_tensor=True)

        logger.log(f'Now generating entity embeddings...')
        self.entity_embeddings = self.model.encode(self.entity_df['full_mention'].to_list(), show_progress_bar=True, convert_to_tensor=True)
        self.entity_embeddings = self.entity_embeddings.to('cpu')
        print(f'Entity Length: {len(self.entity_embeddings)}')
        print(f'Entity shape: {self.entity_embeddings.shape}')
        logger.log(f'Now calculating syntactic neighbours...')
        self.syntax_candidates = util.semantic_search(self.context_embeddings_for_semantic_search, corpus_embeddings, top_k=5)
        
        # delete all the unnecessary variables
        del self.train_data
        del self.train_data_no_doc_start
        del self.not_nan
        del self.not_nme
        del self.model
        # del self.context_embeddings_for_semantic_search

        
        
    def __len__(self):
        # Find the number of rows that have wiki_url not NaN or --NME--
        return len(self.entity_df)
    
    def __getitem__(self, index):
        row = self.entity_df.iloc[index]
        full_mention = row['full_mention'].strip().lower()
        full_mention_longest = row['full_mention_longest'].strip().lower()

        if self.train:
            return index, full_mention, full_mention_longest, row['item_id']
        else:
            return index, full_mention, full_mention_longest

        
    @staticmethod
    def collate_fn_train(batch, entity_embeddings, context_embeddings, syntax_candidates_list, device='cuda'):
        batch_context_embeddings = []
        batch_entity_embeddings = []
        candidate_ids_batch = []
        candidate_description_embeddings_batch = []
        labels_batch = []
        for i, data in enumerate(batch):
            index, full_mention, full_mention_longest, item_id = data
            if full_mention in anchor_to_candidate:
                candidate_ids = anchor_to_candidate[full_mention].copy()
            else:
                candidate_ids = []

            
            syntax_candidates = syntax_candidates_list[index]

            candidate_ids += [wiki_items.iloc[candidate_id['corpus_id']]['source_item_id'] for candidate_id in syntax_candidates]

            # Remove duplicates
            candidate_ids = list(set(candidate_ids))

            # Shuffle the candidate_ids
            np.random.shuffle(candidate_ids)

            # do we have the ground truth in candidate_ids?
            try:
                label = candidate_ids.index(item_id)
            except ValueError:
                # skip this sample
                continue
            
            candidate_description_embeddings = [
                statement_embeddings[statement_item_id_to_row[candidate_id]] if candidate_id in statement_item_id_to_row
                else description_embeddings[description_item_id_to_row[candidate_id]]
                for candidate_id in candidate_ids
            ]

            # pad candidate_ids with 0s
            candidate_ids += [0] * (10 - len(candidate_ids))
            candidate_description_embeddings += [np.zeros(768)] * (10 - len(candidate_description_embeddings))
            # candidate description embeddings is a list of numpy arrays, convert it to numpy array
            candidate_description_embeddings = np.array(candidate_description_embeddings)
            
            batch_context_embeddings.append(context_embeddings[index])
            batch_entity_embeddings.append(entity_embeddings[index])
            candidate_ids_batch.append(candidate_ids)
            candidate_description_embeddings_batch.append(candidate_description_embeddings)

            labels_batch.append(label)

        # labels is a list of integers, convert it to a tensor
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        # batch_context_embeddings is a list of tensors thus we have to use stack
        batch_context_embeddings = torch.stack(batch_context_embeddings)
        batch_entity_embeddings = torch.stack(batch_entity_embeddings)
        # candidate_ids_batch is a list of lists, convert it to a tensor
        candidate_ids_batch = torch.tensor(candidate_ids_batch, dtype=torch.long)
        # candidate_description_embeddings_batch is a list of numpy arrays, convert to numpy array and then to tensor
        candidate_description_embeddings_batch = np.array(candidate_description_embeddings_batch)
        candidate_description_embeddings_batch = torch.tensor(candidate_description_embeddings_batch, dtype=torch.float32)

        # move everything to device
        batch_context_embeddings = batch_context_embeddings.to(device)
        batch_entity_embeddings = batch_entity_embeddings.to(device)
        candidate_ids_batch = candidate_ids_batch.to(device)
        candidate_description_embeddings_batch = candidate_description_embeddings_batch.to(device)
        labels_batch = labels_batch.to(device)

        return batch_context_embeddings, batch_entity_embeddings, candidate_ids_batch, candidate_description_embeddings_batch, labels_batch

    @staticmethod
    def collate_fn_test(batch, entity_embeddings, context_embeddings, syntax_candidates_list, device='cuda'):
        batch_context_embeddings = []
        batch_entity_embeddings = []
        candidate_ids_batch = []
        candidate_description_embeddings_batch = []
        for i, data in enumerate(batch):
            index, full_mention, full_mention_longest = data
            if full_mention in anchor_to_candidate:
                candidate_ids = anchor_to_candidate[full_mention].copy()
            else:
                candidate_ids = []

            # if full_mention_longest in anchor_to_candidate:
            #     candidate_ids += anchor_to_candidate[full_mention_longest].copy()    
            
            syntax_candidates = syntax_candidates_list[index]

            candidate_ids += [wiki_items.iloc[candidate_id['corpus_id']]['source_item_id'] for candidate_id in syntax_candidates]

            # Remove duplicates and limit to 8 candidates
            candidate_ids = list(set(candidate_ids))

            # Shuffle the candidate_ids
            np.random.shuffle(candidate_ids)

            candidate_description_embeddings = [
                statement_embeddings[statement_item_id_to_row[candidate_id]] if candidate_id in statement_item_id_to_row
                else description_embeddings[description_item_id_to_row[candidate_id]]
                for candidate_id in candidate_ids
            ]

            
            # pad candidate_ids with 0s
            candidate_ids += [0] * (10 - len(candidate_ids))
            candidate_description_embeddings += [np.zeros(768)] * (10 - len(candidate_description_embeddings))

            # candidate description embeddings is a list of numpy arrays, convert it to numpy array
            candidate_description_embeddings = np.array(candidate_description_embeddings)
            

            batch_context_embeddings.append(context_embeddings[index])
            batch_entity_embeddings.append(entity_embeddings[index])
            candidate_ids_batch.append(candidate_ids)
            candidate_description_embeddings_batch.append(candidate_description_embeddings)

        # batch_context_embeddings is a list of tensors thus we have to use stack
        batch_context_embeddings = torch.stack(batch_context_embeddings)
        batch_entity_embeddings = torch.stack(batch_entity_embeddings)
        # candidate_ids_batch is a list of lists, convert it to a tensor
        candidate_ids_batch = torch.tensor(candidate_ids_batch, dtype=torch.long)
        # candidate_description_embeddings_batch is a list of numpy arrays, convert to numpy array and then to tensor
        candidate_description_embeddings_batch = np.array(candidate_description_embeddings_batch)
        candidate_description_embeddings_batch = torch.tensor(candidate_description_embeddings_batch, dtype=torch.float32)

        # move everything to device
        batch_context_embeddings = batch_context_embeddings.to(device)
        batch_entity_embeddings = batch_entity_embeddings.to(device)
        candidate_ids_batch = candidate_ids_batch.to(device)
        candidate_description_embeddings_batch = candidate_description_embeddings_batch.to(device)

        return batch_context_embeddings, batch_entity_embeddings, candidate_ids_batch, candidate_description_embeddings_batch
    
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
        # candidate_description_embeddings = torch.tensor(candidate_description_embeddings, dtype=torch.float32, device=device)
        # Above gives error : ValueError: only one element tensors can be converted to Python scalars
        candidate_description_embeddings = torch.tensor(candidate_description_embeddings, dtype=torch.float32, device=device)

        labels = torch.tensor(labels, dtype=torch.long, device=device)

        return sentence_embeddings, entity_embeddings, candidate_ids, candidate_description_embeddings, labels


class EntityClassifier(torch.nn.Module):
    def __init__(self, device='cuda'):
        # This sequential layer will calculate a probability for each candidate entity, 
        # Since we will have CrossEntropyLoss as our loss function, we don't need to apply softmax
        # The input to this layer will be the concatenation of the sentence embedding, entity embedding, and candidate description embedding
        # The output will be single number, since it will be the probability of the candidate entity being the correct entity
        # We pass each candidate once and get the probability of it being the correct entity
        # Select the one with the highest probability
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(768 * 4, 768//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768//2, 1)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_context_embeddings, batch_entity_embeddings, candidate_ids_batch, candidate_description_embeddings_batch = x
        # The shapes are:
        # batch_context_embeddings: (batch_size, 384)
        # batch_entity_embeddings: (batch_size, 384)
        # candidate_ids_batch: (batch_size, 8)
        # candidate_description_embeddings_batch: (batch_size, 8, 384)
        context_entity_embed = torch.cat((batch_context_embeddings, batch_entity_embeddings), dim=1).to(self.device)
        mean_embed = (batch_context_embeddings + batch_entity_embeddings) / 2
        # The shape of context_entity_embed is (batch_size, 384 * 2)

        candidate_preds = torch.zeros((candidate_ids_batch.shape[0], candidate_ids_batch.shape[1]), device=self.device)
        for i in range(candidate_ids_batch.shape[1]):
            # Get the current candidate of all the batches
            candidate_description_embed = candidate_description_embeddings_batch[:,i,:]
            # The shape of candidate_description_embed is (batch_size, 384)
            # Concatenate the sentence_document_entity_embed and candidate_description_embed
            candidate_embed = torch.cat((context_entity_embed, candidate_description_embed), dim=1)
            difference_embed = torch.abs(mean_embed - candidate_description_embed)
            multiple_embed = mean_embed * candidate_description_embed
            candidate_embed = torch.cat((candidate_embed, difference_embed), dim=1)
            candidate_embed = torch.cat((candidate_embed, multiple_embed), dim=1)
            # The shape of candidate_embed is (batch_size, 384 * 3)
            # Pass it through the sequential layer
            candidate_pred = self.sequential(candidate_embed)
            # The shape of candidate_pred is (batch_size, 1)
            # we squeeze to get into shape (batch_size,)
            candidate_preds[:,i] = candidate_pred.squeeze()
        return candidate_preds


    # def forward(self, x):
    #     batch_context_embeddings, batch_entity_embeddings, candidate_ids_batch, candidate_description_embeddings_batch = x
    #     # The shapes are:
    #     # batch_context_embeddings: (batch_size, 384)
    #     # batch_entity_embeddings: (batch_size, 384)
    #     # candidate_ids_batch: (batch_size, 8)
    #     # candidate_description_embeddings_batch: (batch_size, 8, 384)
    #     context_entity_embed = torch.cat((batch_context_embeddings, batch_entity_embeddings), dim=1).to(self.device)
    #     # The shape of context_entity_embed is (batch_size, 384 * 2)

    #     candidate_preds = torch.zeros((candidate_ids_batch.shape[0], candidate_ids_batch.shape[1]), device=self.device)
    #     for i in range(candidate_ids_batch.shape[1]):
    #         # Get the current candidate of all the batches
    #         candidate_description_embed = candidate_description_embeddings_batch[:,i,:]
    #         # The shape of candidate_description_embed is (batch_size, 384)
    #         # Concatenate the sentence_document_entity_embed and candidate_description_embed
    #         candidate_embed = torch.cat((context_entity_embed, candidate_description_embed), dim=1)
    #         # The shape of candidate_embed is (batch_size, 384 * 3)
    #         # Pass it through the sequential layer
    #         candidate_pred = self.sequential(candidate_embed)
    #         # The shape of candidate_pred is (batch_size, 1)
    #         # we squeeze to get into shape (batch_size,)
    #         candidate_preds[:,i] = candidate_pred.squeeze()
    #     return candidate_preds

    
        
def delete_corpus_embeds():
    global corpus_embeddings
    logger.log('Deleting corpus embeddings...')
    # Move it to cpu
    # corpus_embeddings = corpus_embeddings.to('cpu')
    # Delete it
    del corpus_embeddings
    torch.cuda.empty_cache()