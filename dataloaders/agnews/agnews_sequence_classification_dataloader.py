"""Load AG News data (training and eval)"""
import os
import sys
import collections
from typing import Tuple

import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset

NUM_CLASSES = 4

def load_agnews_federated(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    max_seq_len: int = 64,
    split: float = 0.8,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/agnews-seqcls',
    tokenizer_name: str = 'bert-base-uncased',
    save_dir: str = './dataset_cache/agnews-seqcls/manual_save_1000_0_1',
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
    raw_datasets = load_dataset("ag_news", cache_dir=cache_dir)
    raw_datasets = raw_datasets.shuffle(seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # train_inputs = np.array(raw_datasets['train']['text'])
    # train_labels = np.array(raw_datasets['train']['label'])
    # test_inputs = np.array(raw_datasets['test']['text']) 
    # test_labels = np.array(raw_datasets['test']['label'])
    
    inputs = np.hstack(
        (np.array(raw_datasets['train']['text']), np.array(raw_datasets['test']['text'])),
        )
    labels = np.hstack(
        (np.array(raw_datasets['train']['label']), np.array(raw_datasets['test']['label'])),
        )
    
    # train_example_count = len(train_inputs)
    # test_example_count = len(test_inputs)
    example_count = len(inputs)
        
    train_clients = collections.OrderedDict()
    test_clients = collections.OrderedDict()

    # train_multinomial_vals = []
    # test_multinomial_vals = []
    multinomial_vals = []
    
    # Each client has a multinomial distribution over classes drawn from a
    # Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(
            dirichlet_parameter
            * np.ones(
                NUM_CLASSES,
            )
        )

        # train_multinomial_vals.append(proportion)
        # test_multinomial_vals.append(proportion)
        multinomial_vals.append(proportion)

    # train_multinomial_vals = np.array(train_multinomial_vals)
    # test_multinomial_vals = np.array(test_multinomial_vals)
    multinomial_vals = np.array(multinomial_vals)

    # train_example_indices = []
    # test_indices = []
    indices = []
    for k in range(NUM_CLASSES):
        # train_label_k = np.where(train_labels == k)[0]
        # np.random.shuffle(train_label_k)
        # train_example_indices.append(train_label_k)
        
        # test_label_k = np.where(test_labels == k)[0]
        # np.random.shuffle(test_label_k)
        # test_indices.append(test_label_k)
        
        label_k = np.where(labels == k)[0]
        np.random.shuffle(label_k)
        indices.append(label_k)

    # train_example_indices = np.array(train_example_indices)
    # test_indices = np.array(test_indices)
    example_indices = np.array(indices)

    # train_client_samples = [[] for _ in range(num_clients)]
    # test_client_samples = [[] for _ in range(num_clients)]
    client_samples = [[] for _ in range(num_clients)]
    # train_count = np.zeros(NUM_CLASSES).astype(int)
    # test_count = np.zeros(NUM_CLASSES).astype(int)
    count = np.zeros(NUM_CLASSES).astype(int)

    # train_examples_per_client = (int(train_example_count / num_clients))
    # test_examples_per_client = (int(test_example_count / num_clients))
    examples_per_client = (int(example_count / num_clients))
    
    # train_examples_per_label = int(train_example_count / NUM_CLASSES)
    # test_examples_per_label = int(test_example_count / NUM_CLASSES)
    examples_per_label = int(example_count / NUM_CLASSES)

    for k in range(num_clients):
        # for i in range(train_examples_per_client):
        for i in range(examples_per_client):
            # sampled_label = np.argwhere(
            #     np.random.multinomial(1, train_multinomial_vals[k, :]) == 1
            # )[0][0]
            sampled_label = np.argwhere(
                np.random.multinomial(1, multinomial_vals[k, :]) == 1
            )[0][0]
            
            # train_client_samples[k].append(
            #     train_example_indices[sampled_label, train_count[sampled_label]]
            # )
            client_samples[k].append(
                example_indices[sampled_label, count[sampled_label]]
            )
            
            # train_count[sampled_label] += 1
            count[sampled_label] += 1
            
            # if train_count[sampled_label] == train_examples_per_label:
            #     train_multinomial_vals[:, sampled_label] = 0
            #     train_multinomial_vals = (
            #         train_multinomial_vals / train_multinomial_vals.sum(axis=1)[:, None]
            #     )
            if count[sampled_label] == examples_per_label:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                    multinomial_vals / multinomial_vals.sum(axis=1)[:, None]
                )

        # for i in range(test_examples_per_client):
        #     sampled_label = np.argwhere(
        #         np.random.multinomial(1, test_multinomial_vals[k, :]) == 1
        #     )[0][0]
        #     test_client_samples[k].append(
        #         test_indices[sampled_label, test_count[sampled_label]]
        #     )
        #     test_count[sampled_label] += 1
        #     if test_count[sampled_label] == test_examples_per_label:
        #         test_multinomial_vals[:, sampled_label] = 0
        #         test_multinomial_vals = (
        #             test_multinomial_vals / test_multinomial_vals.sum(axis=1)[:, None]
        #         )

    for i in range(num_clients):
        client_name = str(i)
        # x_train = train_inputs[np.array(train_client_samples[i])]
        # y_train = (
        #     train_labels[np.array(train_client_samples[i])].astype("int64").squeeze()
        # )
        x = inputs[np.array(client_samples[i])]
        y = (
            labels[np.array(client_samples[i])].astype("int64").squeeze()
        )
        
        # tokenized_train_data = tokenizer(
        #     list(x_train), 
        #     max_length=max_seq_len,
        #     truncation=True,
        #     padding=True
        # )
        tokenized_data = tokenizer(
            list(x), 
            max_length=max_seq_len,
            truncation=True,
            padding=True
        )
        
        # input_ids = torch.tensor(tokenized_train_data['input_ids'], dtype=torch.int32)
        # token_type_ids = torch.tensor(tokenized_train_data['token_type_ids'])
        # attention_mask = torch.tensor(tokenized_train_data['attention_mask'], dtype=torch.int32)
        # labels = torch.tensor(y_train)
        all_input_ids = torch.tensor(tokenized_data['input_ids'], dtype=torch.int32)
        all_attention_mask = torch.tensor(tokenized_data['attention_mask'], dtype=torch.int32)
        all_labels = torch.tensor(y)
        
        if 'distil' not in tokenizer_name and \
            'roberta' not in tokenizer_name and \
            'gemma' not in tokenizer_name and \
            'llama' not in tokenizer_name:
            all_token_type_ids = torch.tensor(tokenized_data['token_type_ids'], dtype=torch.int32)
        
        split_count = int(all_input_ids.shape[0] * split)
        
        train_input_ids = all_input_ids[:split_count, :]
        train_attention_mask = all_attention_mask[:split_count, :]
        train_labels = all_labels[:split_count]
        
        if 'distil' not in tokenizer_name and \
            'roberta' not in tokenizer_name and \
            'gemma' not in tokenizer_name and \
            'llama' not in tokenizer_name:
            train_token_type_ids = all_token_type_ids[:split_count, :]
        
        test_input_ids = all_input_ids[split_count:, :]
        test_attention_mask = all_attention_mask[split_count:, :]
        test_labels = all_labels[split_count:]
        
        if 'distil' not in tokenizer_name and \
            'roberta' not in tokenizer_name and \
            'gemma' not in tokenizer_name and \
            'llama' not in tokenizer_name:
            test_token_type_ids = all_token_type_ids[split_count:, :]
        
        if 'distil' not in tokenizer_name and \
            'roberta' not in tokenizer_name and \
            'gemma' not in tokenizer_name and \
            'llama' not in tokenizer_name:
            train_data = TensorDataset(
                train_input_ids, train_attention_mask, train_labels, train_token_type_ids
            )
        else:
            train_data = TensorDataset(
                train_input_ids, train_attention_mask, train_labels
            )
            
        train_clients[client_name] = train_data

        # x_test = test_inputs[np.array(test_client_samples[i])]
        # y_test = test_labels[np.array(test_client_samples[i])].astype("int64").squeeze()
        
        # tokenized_test_data = tokenizer(
        #     list(x_test), 
        #     truncation=True,
        #     padding=True
        # )
        
        # input_ids = torch.tensor(tokenized_test_data['input_ids'], dtype=torch.int32)
        # token_type_ids = torch.tensor(tokenized_test_data['token_type_ids'])
        # attention_mask = torch.tensor(tokenized_test_data['attention_mask'], dtype=torch.int32)
        # labels = torch.tensor(y_test)
        
        if 'distil' not in tokenizer_name and \
            'roberta' not in tokenizer_name and \
            'gemma' not in tokenizer_name and \
            'llama' not in tokenizer_name:
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

def get_agnews_federated(
    num_clients: int = 500,
    save_dir: str = './dataset_cache/agnews-seqcls/manual_save_1000_0_1',
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
    train_client_batch_size: int = 20,
    test_client_batch_size: int = 100,
    max_seq_len: int = 64,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/agnews-seqcls',
    tokenizer_name: str = 'bert-base-uncased',
):
    save_dir = cache_dir + '-' + tokenizer_name + '/manual_save_' + str(num_clients) + '_' + str(dirichlet_parameter).replace('.', '_')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        agnews_train_all_data, agnews_test_all_data = load_agnews_federated(
            dirichlet_parameter = dirichlet_parameter,
            num_clients = num_clients,
            cache_dir = cache_dir,
            max_seq_len = max_seq_len,
            tokenizer_name = tokenizer_name,
            random_seed = random_seed,
            save_dir = save_dir,
        )
    else:
        agnews_train_all_data, agnews_test_all_data = get_agnews_federated(
            num_clients = num_clients,
            save_dir = save_dir,
        )
    
    agnews_train_dataloader_dict = {}
    agnews_test_dataloader_dict = {}
    
    for client in range(num_clients):
        
        agnews_train_dataloader_dict[str(client)] = DataLoader(
            agnews_train_all_data[str(client)],
            batch_size=train_client_batch_size,
            num_workers=2
        )
        
        agnews_test_dataloader_dict[str(client)] = DataLoader(
            agnews_test_all_data[str(client)],
            batch_size=test_client_batch_size,
            num_workers=2
        )
    
    return agnews_train_dataloader_dict, agnews_test_dataloader_dict
    