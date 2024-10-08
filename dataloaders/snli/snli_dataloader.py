"""Load SNLI data (training and eval)"""

import os
import sys
import collections
from typing import Tuple

import torch
import numpy as np

from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

def load_snli_federated(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    max_seq_len: int = 64,
    split: float = 0.8,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/snli',
    tokenizer_name: str = 'bert-base-uncased',
    save_dir: str = './dataset_cache/snli/manual_save_500_0_1',
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
    raw_datasets = load_dataset('snli', cache_dir=cache_dir)
    raw_datasets = raw_datasets.shuffle(seed=42)
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True,
        )

    inputs_1 = np.hstack(
        (np.array(raw_datasets['train']['premise']), 
        np.array(raw_datasets['validation']['premise']), 
        np.array(raw_datasets['test']['premise']),
        ))
    inputs_2 = np.hstack(
        (np.array(raw_datasets['train']['hypothesis']), 
        np.array(raw_datasets['validation']['hypothesis']), 
        np.array(raw_datasets['test']['hypothesis']),
        ))
    labels = np.hstack(
        (np.array(raw_datasets['train']['label']), 
        np.array(raw_datasets['validation']['label']),
        np.array(raw_datasets['test']['label']),
        ))
    
    train_clients = collections.OrderedDict()
    test_clients = collections.OrderedDict()
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

    example_indices_0 = np.array(indices[0])
    example_indices_1 = np.array(indices[1])
    example_indices_2 = np.array(indices[2])
    examples_label_0 = len(example_indices_0)
    examples_label_1 = len(example_indices_1)
    examples_label_2 = len(example_indices_2)
    
    example_count = examples_label_0 + examples_label_1 + examples_label_2

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(num_labels).astype(int)

    examples_per_client = (int(example_count / num_clients))
    
    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, multinomial_vals[k, :]) == 1
            )[0][0]
            
            if sampled_label == 0:
                client_samples[k].append(
                    example_indices_0[count[sampled_label]]
                )
            elif sampled_label == 1:
                client_samples[k].append(
                    example_indices_1[count[sampled_label]]
                )
            elif sampled_label == 2:
                client_samples[k].append(
                    example_indices_2[count[sampled_label]]
                )
            
            count[sampled_label] += 1
            
            if count[sampled_label] == examples_label_0 and sampled_label == 0:
                multinomial_vals[:, 0] = 0.0
                multinomial_vals = (
                    multinomial_vals / multinomial_vals.sum(axis=1)[:, None]
                )
            elif count[sampled_label] == examples_label_1 and sampled_label == 1:
                multinomial_vals[:, 1] = 0.0
                multinomial_vals = (
                    multinomial_vals / multinomial_vals.sum(axis=1)[:, None]
                )
            elif count[sampled_label] == examples_label_2 and sampled_label == 2:
                multinomial_vals[:, 2] = 0.0
                multinomial_vals = (
                    multinomial_vals / multinomial_vals.sum(axis=1)[:, None]
                )

    for i in range(num_clients):
        client_name = str(i)
        x_1 = inputs_1[np.array(client_samples[i])]
        x_2 = inputs_2[np.array(client_samples[i])]
        y = (
            labels[np.array(client_samples[i])].astype("int64").squeeze()
        )
        
        tokenized_data = tokenizer(
            list(x_1), list(x_2), 
            max_length=max_seq_len,
            truncation=True,
            padding="max_length"
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
            token_type_ids = all_token_type_ids[:split_count, :]
        
        test_input_ids = all_input_ids[split_count:, :]
        test_attention_mask = all_attention_mask[split_count:, :]
        test_labels = all_labels[split_count:]
        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            test_token_type_ids = all_token_type_ids[split_count:, :]
        
        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            train_data = TensorDataset(
                train_input_ids, train_attention_mask, train_labels, token_type_ids
            )
        else:
            train_data = TensorDataset(
                train_input_ids, train_attention_mask, train_labels
            )
        train_clients[client_name] = train_data

        if 'distil' not in tokenizer_name and 'roberta' not in tokenizer_name:
            test_data = TensorDataset(
                test_input_ids, test_attention_mask, test_labels, test_token_type_ids
            )
        else:
            test_data = TensorDataset(
                test_input_ids, test_attention_mask, test_labels
            )
            
        test_clients[client_name] = test_data
        
        torch.save(train_data.tensors, save_dir + '/train_dataset_' + client_name + '.pth')
        torch.save(test_data.tensors, save_dir + '/test_dataset_' + client_name + '.pth')

    return train_clients, test_clients

def get_snli_federated(
    num_clients: int = 500,
    save_dir: str = './dataset_cache/snli/manual_save_500_0_1',
):
    print('Loading cached dataset')
    train_clients = collections.OrderedDict()
    test_clients = collections.OrderedDict()

    for i in range(num_clients):
        client_name = str(i)
        
        loaded_tensors = torch.load(save_dir + '/train_dataset_' + client_name + '.pth')
        train_clients[client_name] = TensorDataset(*loaded_tensors)
        
        loaded_tensors = torch.load(save_dir + '/test_dataset_' + client_name + '.pth')
        test_clients[client_name] = TensorDataset(*loaded_tensors)
        
    return train_clients, test_clients

def get_federated_datasets(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    train_client_batch_size: int = 16,
    test_client_batch_size: int = 100,
    max_seq_len: int = 64,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/snli',
    tokenizer_name: str = 'bert-base-uncased',
):
    # cache_dir = cache_dir + '-' + tokenizer_name
    save_dir = cache_dir + '-' + tokenizer_name + '/manual_save_' + str(num_clients) + '_' + str(dirichlet_parameter).replace('.', '_')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        snli_train_all_data, snli_test_all_data = load_snli_federated(
            dirichlet_parameter = dirichlet_parameter,
            num_clients = num_clients,
            cache_dir = cache_dir,
            max_seq_len = max_seq_len,
            tokenizer_name = tokenizer_name,
            random_seed = random_seed,
            save_dir = save_dir,
        )
    else:
        snli_train_all_data, snli_test_all_data = get_snli_federated(
            num_clients = num_clients,
            save_dir = save_dir,
        )
    
    snli_train_dataloader_dict = {}
    snli_test_dataloader_dict = {}
    
    for client in range(num_clients):
        
        snli_train_dataloader_dict[str(client)] = DataLoader(
            snli_train_all_data[str(client)],
            batch_size=train_client_batch_size,
            num_workers=2
        )
        
        snli_test_dataloader_dict[str(client)] = DataLoader(
            snli_test_all_data[str(client)],
            batch_size=test_client_batch_size,
            num_workers=2
        )
    
    return snli_train_dataloader_dict, snli_test_dataloader_dict
    