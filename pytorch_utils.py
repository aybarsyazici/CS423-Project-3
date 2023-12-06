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

logger.log('Loading item_id to statement embedding data...')
# with open(DATA_DIR + 'pkl/item_id_to_description_embedding.pkl', 'rb') as handle:
#     item_id_to_description_embedding = pickle.load(handle)
with open(DATA_DIR + 'pkl/statement_embeddings.pkl', 'rb') as handle:
    statement_embeddings = pickle.load(handle)
with open(DATA_DIR + 'pkl/statements_item_id_to_row.pkl', 'rb') as handle:
    statement_item_id_to_row = pickle.load(handle)

logger.log('Loading item_id to description embedding data...')
with open(DATA_DIR + 'pkl/title_description_embeddings.pkl', 'rb') as handle:
    description_embeddings = pickle.load(handle)
with open(DATA_DIR + 'pkl/title_description_embeddings_item_id_to_row_id.pkl', 'rb') as handle:
    description_item_id_to_row = pickle.load(handle)

logger.log('Loading wikipedia title embedings...')
with open(DATA_DIR + 'pkl/corpus_embeddings.pt', 'rb') as handle:
    corpus_embeddings = torch.load(handle)
    corpus_embeddings = corpus_embeddings.to('cuda')

logger.log('Loading wikipedia items...')
wiki_items = pd.read_csv(DATA_DIR + 'wiki_lite/wiki_items.csv')

class EntityDataSample:
    def __init__(self, mention, candidate_entities, label):
        self.mention = mention
        self.candidate_entities = candidate_entities
        self.label = label



class EntityDataset(Dataset):
    def __init__(self, train=True, model_name = 'all-MiniLM-L6-v2', device='cuda', DATA_DIR = './data/'):
        self.device = device
        self.train = train
        set_type = 'train' if train else 'test'
        logger.log(f'Loading {model_name} model & Generating sentence embeddings of {set_type} set...')
        self.model = SentenceTransformer(model_name, device='cuda')
        if train:
            self.train_data = pd.read_csv(DATA_DIR + 'train_data_preprocessed.csv')
        else:
            self.train_data = pd.read_csv(DATA_DIR + 'test.csv')
        self.train_data['sentence_id'] = (self.train_data['token'] == '.').cumsum()
        self.train_data['doc_id'] = self.train_data['token'].str.startswith('-DOCSTART-').cumsum()
        self.not_nan = self.train_data['wiki_url'].notna()
        self.not_nme = self.train_data['wiki_url'] != '--NME--'
        self.train_data['full_mention'] = self.train_data['full_mention'].fillna('')
        self.train_data = update_full_mentions(self.train_data)
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
        self.sentence_embeddings = self.model.encode([' '.join(tokens) for tokens in all_tokens], show_progress_bar=True)

        logger.log(f'Now generating Doc embeddings...')
        all_documents = []
        for doc_id in tqdm(train_data_no_doc_start['doc_id'].unique()):
            # Filter the DataFrame for the current document
            doc_data = train_data_no_doc_start[train_data_no_doc_start['doc_id'] == doc_id]
            
            # remove tokens with NaN values
            doc_data = doc_data[doc_data['token'].notna()]

            # Extract the token column and convert it to a list
            doc_tokens = doc_data['token'].tolist()
            
            # Append this list to the all_tokens list
            all_documents.append(doc_tokens)
        self.document_embeddings = self.model.encode([' '.join(tokens) for tokens in all_documents], show_progress_bar=True)

        logger.log(f'Now generating entity embeddings...')
        not_nan = self.train_data['wiki_url'].notna()
        not_nme = self.train_data['wiki_url'] != '--NME--'
        self.entity_embeddings = self.model.encode(self.train_data[not_nan & not_nme]['full_mention'].to_list(), show_progress_bar=True)

        if False:
            logger.log(f'Now loading statements and properties.')
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
        
        

        
        # delete all the unnecessary variables
        del self.train_data
        del self.not_nan
        del self.not_nme
        del self.model
        del not_nan
        del not_nme
        del all_tokens
        del train_data_no_doc_start

        #del wiki_items_joined
        #del pages
        #del statements


    def __len__(self):
        # Find the number of rows that have wiki_url not NaN or --NME--
        return len(self.entity_df)
    
    def __getitem__(self, index):
        row = self.entity_df.iloc[index]
        sentence_id = row['sentence_id']
        document_embed = self.document_embeddings[int(row['doc_id'])-1]
        sentence_embed = self.sentence_embeddings[sentence_id]
        entity_embed = self.entity_embeddings[index]
        full_mention = row['full_mention'].strip().lower()
        if self.train:
            return document_embed, sentence_embed, entity_embed, full_mention, row['item_id']
        else:
            return document_embed, sentence_embed, entity_embed, full_mention



    # def __getitemold__(self, index):
    #     row = self.entity_df.iloc[index]
    #     # get sentence_id
    #     sentence_id = row['sentence_id']
    #     # get sentence embedding
    #     sentence_embed = self.sentence_embeddings[sentence_id]
    #     # get entity embedding
    #     entity_embed = self.entity_embeddings[index]
    #     full_mention = row['full_mention']
    #     # get candidate entities from anchor
    #     if full_mention.strip().lower() in anchor_to_candidate:
    #         candidate_ids = anchor_to_candidate[full_mention.strip().lower()]
    #     else:
    #         candidate_ids = []
    #     # get the candidates
    #     # syntax_candidate_ids = util.semantic_search(entity_embed, corpus_embeddings, top_k=3, score_function=util.dot_score)[0]
    #     # candidate_ids = candidate_ids + [wiki_items.iloc[candidate_id['corpus_id']]['item_id'] for candidate_id in syntax_candidate_ids]
    #     # drop duplicates
    #     # candidate_ids = list(set(candidate_ids))
    #     candidate_description_embeddings = [item_id_to_description_embedding[candidate_id][1] for candidate_id in candidate_ids]
    #     # pad the candidate_description_embeddings and candidate_ids to length 8
    #     candidate_description_embeddings = candidate_description_embeddings + [np.zeros(384)] * (5 - len(candidate_description_embeddings))
    #     # pad candidate_ids with 0s
    #     candidate_ids = candidate_ids + [0] * (5 - len(candidate_ids))
    #     # get label (which candidate_id equals the item_id, if none, return 0)
    #     try:
    #         label = candidate_ids.index(row['item_id'])
    #     except:
    #         label = 0
    #     return sentence_embed, entity_embed, candidate_ids, np.array(candidate_description_embeddings), label
        
    @staticmethod
    def collate_fn_train(batch, device='cuda'):
        # Unpack the batch data
        document_embed, sentence_embeddings, entity_embeddings, full_mentions, item_ids = zip(*batch)

        # Convert to torch tensors
        document_embed = torch.tensor(document_embed, dtype=torch.float32, device=device)
        # document_embed = document_embed.to(device)
        sentence_embeddings = torch.tensor(sentence_embeddings, dtype=torch.float32, device=device)
        entity_embeddings = torch.tensor(entity_embeddings, dtype=torch.float32, device=device)

        # Batch semantic search
        # Perform the semantic search for all entity embeddings at once
        syntax_candidate_ids_batch = util.semantic_search(entity_embeddings, corpus_embeddings, top_k=3, score_function=util.dot_score)
        # Process the results of the semantic search
        candidate_ids_batch = []
        candidate_description_embeddings_batch = []
        labels = []
        valid_sample_indices = []
        for i, (full_mention, item_id, syntax_candidate_ids) in enumerate(zip(full_mentions, item_ids, syntax_candidate_ids_batch)):
            if full_mention in anchor_to_candidate:
                candidate_ids = anchor_to_candidate[full_mention].copy()
            else:
                candidate_ids = []
            
            candidate_ids += [wiki_items.iloc[candidate_id['corpus_id']]['item_id'] for candidate_id in syntax_candidate_ids if candidate_id['score'] > 0.7]

            # drop duplicates
            candidate_ids = list(set(candidate_ids))

            candidate_description_embeddings = [statement_embeddings[statement_item_id_to_row[candidate_id]] if candidate_id in statement_item_id_to_row else description_embeddings[description_item_id_to_row[candidate_id]] for candidate_id in candidate_ids]

            # pad the candidate_description_embeddings and candidate_ids to length 8
            candidate_description_embeddings += [np.zeros(384)] * (8 - len(candidate_description_embeddings))

            # pad candidate_ids with 0s
            candidate_ids += [0] * (8 - len(candidate_ids))

            try:
                label = candidate_ids.index(item_id)
                valid_sample_indices.append(i)
            except:
                label = 0

            candidate_ids_batch.append(candidate_ids)
            candidate_description_embeddings_batch.append(candidate_description_embeddings)
            labels.append(label)

        # Filter the batch data based on valid_sample_indices
        document_embed = document_embed[valid_sample_indices]
        sentence_embeddings = sentence_embeddings[valid_sample_indices]
        entity_embeddings = entity_embeddings[valid_sample_indices]
        # print(f'{document_embed.shape}, {sentence_embeddings.shape}, {entity_embeddings.shape}')
        candidate_ids_batch = [candidate_ids_batch[i] for i in valid_sample_indices]
        candidate_description_embeddings_batch = [candidate_description_embeddings_batch[i] for i in valid_sample_indices]
        labels = [labels[i] for i in valid_sample_indices]

        # convert to torch tensors
        candidate_ids_batch = torch.tensor(candidate_ids_batch, dtype=torch.long, device=device)
        candidate_description_embeddings_batch = torch.tensor(candidate_description_embeddings_batch, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        # print(f'{candidate_ids_batch.shape}, {candidate_description_embeddings_batch.shape}, {labels.shape}')

        return document_embed, sentence_embeddings, entity_embeddings, candidate_ids_batch, candidate_description_embeddings_batch, labels

    @staticmethod
    def collate_fn_test(batch, device='cuda'):
        # Unpack the batch data
        document_embed, sentence_embeddings, entity_embeddings, full_mentions = zip(*batch)

        # Convert to torch tensors
        sentence_embeddings = torch.tensor(sentence_embeddings, dtype=torch.float32, device=device)
        entity_embeddings = torch.tensor(entity_embeddings, dtype=torch.float32, device=device)
        document_embed = torch.tensor(document_embed, dtype=torch.float32, device=device)

        # Batch semantic search
        # Perform the semantic search for all entity embeddings at once
        syntax_candidate_ids_batch = util.semantic_search(entity_embeddings, corpus_embeddings, top_k=3, score_function=util.dot_score)
        # Process the results of the semantic search
        candidate_ids_batch = []
        candidate_description_embeddings_batch = []

        for i, (full_mention, syntax_candidate_ids) in enumerate(zip(full_mentions, syntax_candidate_ids_batch)):
            if full_mention in anchor_to_candidate:
                candidate_ids = anchor_to_candidate[full_mention].copy()
            else:
                candidate_ids = []

            candidate_ids += [wiki_items.iloc[candidate_id['corpus_id']]['item_id'] for candidate_id in syntax_candidate_ids if candidate_id['score'] > 0.7] 

            # drop duplicates
            candidate_ids = list(set(candidate_ids))

            candidate_description_embeddings = [statement_embeddings[statement_item_id_to_row[candidate_id]] if candidate_id in statement_item_id_to_row else description_embeddings[description_item_id_to_row[candidate_id]] for candidate_id in candidate_ids]

            # pad the candidate_description_embeddings and candidate_ids to length 8
            candidate_description_embeddings += [np.zeros(384)] * (8 - len(candidate_description_embeddings))

            # pad candidate_ids with 0s
            candidate_ids += [0] * (8 - len(candidate_ids))

            candidate_ids_batch.append(candidate_ids)
            candidate_description_embeddings_batch.append(candidate_description_embeddings)

        # convert to torch tensors
        candidate_ids_batch = torch.tensor(candidate_ids_batch, dtype=torch.long, device=device)
        candidate_description_embeddings_batch = torch.tensor(candidate_description_embeddings_batch, dtype=torch.float32, device=device)

        return document_embed, sentence_embeddings, entity_embeddings, candidate_ids_batch, candidate_description_embeddings_batch
    
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
            torch.nn.Linear(384 * 4, 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(384, 1)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        document_embeds, sentence_embeddings, entity_embeddings, candidate_ids, candidate_description_embeddings = x
        # The shapes are:
        # sentence_embeddings: (batch_size, 384)
        # entity_embeddings: (batch_size, 384)
        # candidate_ids: (batch_size, 5)
        # candidate_description_embeddings: (batch_size, 5, 384)
        # labels: (batch_size,)
        sentence_document_embed = torch.cat((sentence_embeddings, document_embeds), dim=1).to(self.device)
        sentence_document_entity_embed = torch.cat((sentence_document_embed, entity_embeddings), dim=1).to(self.device)
        # The shape of sentence_document_entity_embed is (batch_size, 384 * 3)

        candidate_preds = torch.zeros(candidate_ids.shape[0], candidate_ids.shape[1], dtype=torch.float32, device=self.device)
        for i in range(candidate_ids.shape[1]):
            # Get the current candidate of all the batches
            candidate_description_embed = candidate_description_embeddings[:,i,:]
            # The shape of candidate_description_embed is (batch_size, 384)
            # Concatenate the sentence_document_entity_embed and candidate_description_embed
            candidate_embed = torch.cat((sentence_document_entity_embed, candidate_description_embed), dim=1)
            # The shape of candidate_embed is (batch_size, 384 * 3)
            # Pass it through the sequential layer
            candidate_pred = self.sequential(candidate_embed)
            # The shape of candidate_pred is (batch_size, 1)
            # we squeeze to get into shape (batch_size,)
            candidate_preds[:,i] = candidate_pred.squeeze()
        return candidate_preds

        
