"""Load yelp data (training and eval)"""

import os
import sys
import collections
from typing import Tuple

import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset

def load_yelp_federated(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    max_seq_len: int = 64,
    split: float = 0.8,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/yelp',
    tokenizer_name: str = 'bert-base-uncased',
    save_dir: str = './dataset_cache/yelp/manual_save_500_0_1',
) -> Tuple[DataLoader, DataLoader]:
    """Construct a federated dataset from the centralized AG News.
    Sampling based on Dirichlet distribution over categories, following the paper
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/abs/1909.06335).
    Args:
      dirichlet_parameter: Parameter of Dirichlet distribution. Each client
        samples from this Dirichlet to get a multinomial distribution over
        classes. It controls the data heterogeneity of clients. If approaches 0,
        then each client only have data from a single category label. If
        approaches infinity, then the client distribution will approach IID
        partitioning.
      num_clients: The number of clients the examples are going to be partitioned
        on.
    Returns:
      A tuple of `torch.utils.data.DataLoader` representing unpreprocessed
      train data and test data.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    cache_dir = cache_dir + '-' + tokenizer_name
    raw_datasets = load_dataset("yelp_polarity", cache_dir=cache_dir)
    raw_datasets = raw_datasets.shuffle(seed=42)
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    inputs = np.hstack(
        (np.array(raw_datasets['train']['text']), 
         np.array(raw_datasets['test']['text'])),
        )
    labels = np.hstack(
        (np.array(raw_datasets['train']['label']), 
         np.array(raw_datasets['test']['label'])),
        )
    
    example_count = len(inputs)
        
    # train_clients = collections.OrderedDict()
    # test_clients = collections.OrderedDict()

    multinomial_vals = []
    
    # Each client has a multinomial distribution over classes drawn from a
    # Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(
            dirichlet_parameter
            * np.ones(
                num_labels,
            )
        )

        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)

    indices = []
    for k in range(num_labels):
        label_k = np.where(labels == k)[0]
        np.random.shuffle(label_k)
        indices.append(label_k)

    example_indices = np.array(indices)
    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(num_labels).astype(int)

    examples_per_client = int(example_count / num_clients)
    examples_per_label = int(example_count / num_labels)

    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, multinomial_vals[k, :]) == 1
            )[0][0]
            
            client_samples[k].append(
                example_indices[sampled_label, count[sampled_label]]
            )
            
            count[sampled_label] += 1
            
            if count[sampled_label] == examples_per_label:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                    multinomial_vals / multinomial_vals.sum(axis=1)[:, None]
                )

    for i in range(num_clients):
        client_name = str(i)
        x = inputs[np.array(client_samples[i])]
        y = (
            labels[np.array(client_samples[i])].astype("int64").squeeze()
        )
        
        tokenized_data = tokenizer(
            list(x), 
            max_length=max_seq_len,
            truncation=True,
            padding=True
        )
        
        all_input_ids = torch.tensor(tokenized_data['input_ids'], dtype=torch.int32)
        all_attention_mask = torch.tensor(tokenized_data['attention_mask'], dtype=torch.int32)
        all_labels = torch.tensor(y)
        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            all_token_type_ids = torch.tensor(tokenized_data['token_type_ids'], dtype=torch.int32)
        
        split_count = int(all_input_ids.shape[0] * split)
        
        train_input_ids = all_input_ids[:split_count, :]
        train_attention_mask = all_attention_mask[:split_count, :]
        train_labels = all_labels[:split_count]
        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            train_token_type_ids = all_token_type_ids[:split_count, :]
        
        test_input_ids = all_input_ids[split_count:, :]
        test_attention_mask = all_attention_mask[split_count:, :]
        test_labels = all_labels[split_count:]
        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            test_token_type_ids = all_token_type_ids[split_count:, :]
        
        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            train_data = TensorDataset(
                train_input_ids, train_attention_mask, train_labels, train_token_type_ids
            )
        else:
            train_data = TensorDataset(
                train_input_ids, train_attention_mask, train_labels
            )
        
        # train_clients[client_name] = train_data

        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            test_data = TensorDataset(
                test_input_ids, test_attention_mask, test_labels, test_token_type_ids
            )
        else:
            test_data = TensorDataset(
                test_input_ids, test_attention_mask, test_labels
            )
        # test_clients[client_name] = test_data
        
        torch.save(train_data.tensors, save_dir + '/train_dataset_' + client_name + '.pth')
        torch.save(test_data.tensors, save_dir + '/test_dataset_' + client_name + '.pth')

    # return train_clients, test_clients
    return train_data, test_data

# def get_yelp_federated(
#     num_clients: int = 500,
#     save_dir: str = './dataset_cache/yelp/manual_save_500_0_1',
# ):
#     print('Loading cached dataset')
#     train_clients = collections.OrderedDict()
#     test_clients = collections.OrderedDict()

#     for i in range(num_clients):
#         client_name = str(i)
        
#         loaded_tensors = torch.load(save_dir + '/train_dataset_' + client_name + '.pth')
#         train_clients[client_name] = TensorDataset(*loaded_tensors)
        
#         loaded_tensors = torch.load(save_dir + '/test_dataset_' + client_name + '.pth')
#         test_clients[client_name] = TensorDataset(*loaded_tensors)
        
#     return train_clients, test_clients

def get_yelp_federated(
    client_id: int = 0,
    train_client_batch_size: int = 8,
    test_client_batch_size: int = 100,
    save_dir: str = './dataset_cache/yelp/manual_save_500_0_1',
):
    loaded_tensors = torch.load(save_dir + '/train_dataset_' + str(client_id) + '.pth')
    train_data = TensorDataset(*loaded_tensors)
    train_dataloader = DataLoader(
        train_data,
        batch_size=train_client_batch_size,
        num_workers=2
    )
    
    loaded_tensors = torch.load(save_dir + '/test_dataset_' + str(client_id) + '.pth')
    test_data= TensorDataset(*loaded_tensors)
    test_dataloader = DataLoader(
        test_data,
        batch_size=test_client_batch_size,
        num_workers=2
    )
        
    return train_dataloader, test_dataloader

def get_federated_datasets(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    train_client_batch_size: int = 20,
    test_client_batch_size: int = 100,
    max_seq_len: int = 64,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/yelp',
    save_dir: str = './dataset_cache/yelp-bert-base-uncased/manual_save_500_1_0',
    tokenizer_name: str = 'bert-base-uncased',
):
    # save_dir = cache_dir + '-' + tokenizer_name + '/manual_save_' + str(num_clients) + '_' + str(dirichlet_parameter).replace('.', '_')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # yelp_train_all_data, yelp_test_all_data 
        sample_yelp_train_data, sample_yelp_test_data = load_yelp_federated(
            dirichlet_parameter = dirichlet_parameter,
            num_clients = num_clients,
            max_seq_len = max_seq_len,
            cache_dir = cache_dir,
            tokenizer_name = tokenizer_name,
            random_seed = random_seed,
            save_dir = save_dir,
        )
    else:
        sample_yelp_train_data, sample_yelp_test_data = get_yelp_federated(
            client_id = 0,
            train_client_batch_size = train_client_batch_size,
            test_client_batch_size = test_client_batch_size,
            save_dir = save_dir,
        )
    
    # yelp_train_dataloader_dict = {}
    # yelp_test_dataloader_dict = {}
    
    # for client in range(num_clients):
        
    sample_yelp_train_dataloader = DataLoader(
        sample_yelp_train_data,
        # yelp_train_all_data[str(client)],
        batch_size=train_client_batch_size,
        num_workers=2
    )
    
    sample_yelp_test_dataloader = DataLoader(
        sample_yelp_test_data,
        # yelp_test_all_data[str(client)],
        batch_size=test_client_batch_size,
        num_workers=2
    )
    
    return sample_yelp_train_dataloader, sample_yelp_test_dataloader
    # return yelp_train_dataloader_dict, yelp_test_dataloader_dict
    