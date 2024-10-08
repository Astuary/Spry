"""Load SQuAD data (training and eval)"""
import os
import sys
import collections
from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm

from datasets import (
    load_dataset, 
    concatenate_datasets,
    load_from_disk
)
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer, 
    default_data_collator
)

def preprocess_function(examples, tokenizer):
    pad_on_right = tokenizer.padding_side == "right"
    
    # examples["question"] = examples["question"].strip()
    examples["question"] = [q.strip() for q in examples["question"]]
    # print('examples', examples)
    
    inputs = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        max_length=384,
        truncation="only_second" if pad_on_right else "only_first",
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )
    
    # print('inputs', inputs.keys())
    offset_mapping = inputs.pop("offset_mapping")
    special_tokens = inputs.pop("special_tokens_mask")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")

    inputs["start_positions"] = []
    inputs["end_positions"] = []
    inputs["is_impossible"] = []
    inputs["cls_index"] = []
    inputs["p_mask"] = []
    # print('inputs', inputs.keys())

    for i, offset in enumerate(offset_mapping):
        # print('\n')
        # print('i, offset', i, offset)
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        inputs["cls_index"].append(cls_index)
        
        # print('input_ids', input_ids)
        
        sequence_ids = inputs["token_type_ids"][i]
        # print('sequence_ids', sequence_ids)
        for k, s in enumerate(special_tokens[i]):
            if s:
                sequence_ids[k] = 3
        # print('sequence_ids', sequence_ids)
        context_idx = 1 if pad_on_right else 0
        # print('context_idx', context_idx)
        
        inputs["p_mask"].append(
            [
                0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                for k, s in enumerate(sequence_ids)
            ]
        )
        
        sample_index = sample_mapping[i]
        # print('sample_index', sample_index)
        answers = examples["answers"][sample_index]
        # print('answers', answers)
        # answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            inputs["start_positions"].append(cls_index)
            inputs["end_positions"].append(cls_index)
            inputs["is_impossible"].append(1.0)
        else:
            # print('In else')
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            # print('start_char', start_char)
            # print('end_char', end_char)

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != context_idx:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != context_idx:
                token_end_index -= 1
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offset[token_start_index][0] <= start_char and offset[token_end_index][1] >= end_char):
                inputs["start_positions"].append(cls_index)
                inputs["end_positions"].append(cls_index)
                inputs["is_impossible"].append(1.0)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offset) and offset[token_start_index][0] <= start_char:
                    token_start_index += 1
                inputs["start_positions"].append(token_start_index - 1)
                while offset[token_end_index][1] >= end_char:
                    token_end_index -= 1
                inputs["end_positions"].append(token_end_index + 1)
                inputs["is_impossible"].append(0.0)

    return inputs

def load_squad_federated(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    max_seq_len: int = 64,
    split: float = 0.8,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/squad',
    tokenizer_name: str = 'bert-base-uncased',
    save_dir: str = './dataset_cache/squad/manual_save_500_0_1',
) -> Tuple[DataLoader, DataLoader]:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    cache_dir = cache_dir + '-' + tokenizer_name
    raw_datasets = load_dataset('squad', cache_dir=cache_dir)
    raw_datasets = raw_datasets.shuffle(seed=42)
    
    train_examples = raw_datasets['train']
    test_examples = raw_datasets["validation"]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_title_counter = collections.Counter(train_examples['title'])
    test_title_counter = collections.Counter(test_examples['title'])
    
    pad_on_right = tokenizer.padding_side == "right"
    
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples['question'] = [q.lstrip() for q in examples['question']]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples['question' if pad_on_right  else 'context'],
            examples['context' if pad_on_right  else 'question'],
            truncation="only_second" if pad_on_right  else "only_first",
            max_length=384,
            stride=50,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
        special_tokens = tokenized_examples.pop("special_tokens_mask")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossible"] = []
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []
        # tokenized_examples["answers"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples["cls_index"].append(cls_index)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples["token_type_ids"][i]
            for k, s in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            context_idx = 1 if pad_on_right  else 0

            # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
            # The cls token gets 1.0 too (for predictions of empty answers).
            tokenized_examples["p_mask"].append(
                [
                    0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                    for k, s in enumerate(sequence_ids)
                ]
            )

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]
            # tokenized_examples["answers"].append(
            #     tokenizer(
            #         answers['text'], 
            #         max_length=384,
            #         padding="max_length",)['input_ids'][0]
            #     )
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossible"].append(1.0)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != context_idx:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_idx:
                    token_end_index -= 1
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["is_impossible"].append(1.0)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["is_impossible"].append(0.0)

        return tokenized_examples
    
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples['question'] = [q.lstrip() for q in examples['question']]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples['question' if pad_on_right else 'context'],
            examples['context' if pad_on_right else 'question'],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=384,
            stride=50,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    train_titles = np.array(train_examples['title'])
    test_titles = np.array(test_examples['title'])
    train_num_samples = train_titles.shape[0]
    test_num_samples = test_titles.shape[0]
    
    train_num_titles = len(train_title_counter.keys())
    test_num_titles = len(test_title_counter.keys())
        
    train_clients = collections.OrderedDict()
    test_clients = collections.OrderedDict()
    test_raw_examples_clients = collections.OrderedDict()
    train_multinomial_vals = []
    test_multinomial_vals = []
    
    # Each client has a multinomial distribution over classes drawn from a
    # Dirichlet.
    for i in range(num_clients):
        train_proportion = np.random.dirichlet(
            dirichlet_parameter
            * np.ones(
                train_num_titles,
            )
        )
        test_proportion = np.random.dirichlet(
            dirichlet_parameter
            * np.ones(
                test_num_titles,
            )
        )
        train_multinomial_vals.append(train_proportion)
        test_multinomial_vals.append(test_proportion)
        
    train_multinomial_vals = np.array(train_multinomial_vals)
    test_multinomial_vals = np.array(test_multinomial_vals)
    
    train_indices = []
    for k in list(train_title_counter.keys()):
        title_k = np.where(train_titles == k)[0]
        np.random.shuffle(title_k)
        train_indices.append(title_k)
        
    test_indices = []
    for k in list(test_title_counter.keys()):
        title_k = np.where(test_titles == k)[0]
        np.random.shuffle(title_k)
        test_indices.append(title_k)
    
    train_client_samples = [[] for _ in range(num_clients)]
    train_count = np.zeros(train_num_titles).astype(int)
    train_examples_per_client = (int(train_num_samples / num_clients))
    
    test_client_samples = [[] for _ in range(num_clients)]
    test_count = np.zeros(test_num_titles).astype(int)
    test_examples_per_client = (int(test_num_samples / num_clients))
    
    for k in range(num_clients):
        for i in range(train_examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, train_multinomial_vals[k, :]) == 1
            )[0][0]
            
            train_client_samples[k].append(
                train_indices[sampled_label][train_count[sampled_label]]
            )
            
            train_count[sampled_label] += 1
            
            if train_count[sampled_label] == train_title_counter[list(train_title_counter.keys())[sampled_label]]:
                train_multinomial_vals[:, sampled_label] = 0.0
                train_multinomial_vals = (
                    train_multinomial_vals / train_multinomial_vals.sum(axis=1)[:, None]
                )
        
        for i in range(test_examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, test_multinomial_vals[k, :]) == 1
            )[0][0]
            
            test_client_samples[k].append(
                test_indices[sampled_label][test_count[sampled_label]]
            )
            
            test_count[sampled_label] += 1
            
            if test_count[sampled_label] == test_title_counter[list(test_title_counter.keys())[sampled_label]]:
                test_multinomial_vals[:, sampled_label] = 0.0
                test_multinomial_vals = (
                    test_multinomial_vals / test_multinomial_vals.sum(axis=1)[:, None]
                )
           
    # TODO: Changes to seperate test from here               
    for i in tqdm(range(num_clients), desc="Clients #", position=0):
        client_name = str(i)
        # print(client_name, flush=True)
        train_clients[client_name] = train_examples.select(train_client_samples[i]).map(
            prepare_train_features, 
            batched=True, 
            # remove_columns=["question"],
            remove_columns=train_examples.column_names,
        )
        
        test_raw_examples_clients[client_name] = test_examples.select(test_client_samples[i])
        
        test_clients[client_name] = test_raw_examples_clients[client_name].map(
            prepare_validation_features, 
            batched=True, 
            remove_columns=test_examples.column_names,
        )
        
        # print(train_clients[client_name]['input_ids'])
        # print(len(train_clients[client_name]['input_ids']))
        # print(type(train_clients[client_name]['input_ids']))
        # print(len(train_clients[client_name]['input_ids'][0]))
        # print(type(train_clients[client_name]['input_ids'][0]))
        # print('---------------', flush=True)
        
        train_clients[client_name].save_to_disk(save_dir + '/train_dataset_' + client_name + '.hf')
        test_clients[client_name].save_to_disk(save_dir + '/test_dataset_' + client_name + '.hf')
        test_raw_examples_clients[client_name].save_to_disk(save_dir + '/test_raw_dataset_' + client_name + '.hf')
        # dummy = load_from_disk(save_dir + '/train_dataset_' + client_name + '.hf')
        # print(len(dummy['input_ids']))
        # print(type(dummy['input_ids']))
        # print(len(dummy['input_ids'][0]))
        # print(type(dummy['input_ids'][0]))
        # sys.exit(0)
    
    # test_common = test_examples.map(
    #     prepare_validation_features,
    #     batched=True,
    #     remove_columns=train_examples.column_names,
    #     desc="Running tokenizer on validation dataset",
    # )
    # test_common.save_to_disk(save_dir + '/test_dataset_common.hf')
    
    return train_clients, test_clients, test_raw_examples_clients

def get_squad_federated(
    num_clients: int = 500,
    tokenizer_name: str = 'bert-base-uncased',
    cache_dir: str = './dataset_cache/squad',
    save_dir: str = './dataset_cache/squad/manual_save_500_0_1',
):
    print('Loading cached dataset')
    train_clients = collections.OrderedDict()
    test_clients = collections.OrderedDict()
    test_raw_clients = collections.OrderedDict()

    for i in range(num_clients):
        client_name = str(i)
        train_clients[client_name] = load_from_disk(save_dir + '/train_dataset_' + client_name + '.hf')
        test_clients[client_name] = load_from_disk(save_dir + '/test_dataset_' + client_name + '.hf')
        test_raw_clients[client_name] = load_from_disk(save_dir + '/test_raw_dataset_' + client_name + '.hf')
        
    return train_clients, test_clients, test_raw_clients

def get_federated_datasets(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    train_client_batch_size: int = 16,
    test_client_batch_size: int = 16,
    max_seq_len: int = 64,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/squad',
    tokenizer_name: str = 'bert-base-uncased',
):
    save_dir = cache_dir + '-' + tokenizer_name + '/manual_save_' + str(num_clients) + '_' + str(dirichlet_parameter).replace('.', '_')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        squad_train_clients_dataset_dict, squad_test_clients_dataest_dict, squad_test_all_example_clients_data = load_squad_federated(
            dirichlet_parameter = dirichlet_parameter,
            num_clients = num_clients,
            cache_dir = cache_dir,
            max_seq_len = max_seq_len,
            tokenizer_name = tokenizer_name,
            random_seed = random_seed,
            save_dir = save_dir,
            split=1.0,
        )
    else:
        squad_train_clients_dataset_dict, squad_test_clients_dataest_dict, squad_test_all_example_clients_data = get_squad_federated(
            num_clients = num_clients,
            tokenizer_name = tokenizer_name,
            cache_dir = cache_dir,
            save_dir = save_dir,
        )
    
    squad_train_dataloader_dict = {}
    squad_test_dataloader_dict = {}
        
    for client in range(num_clients):
        squad_train_dataloader_dict[str(client)] = DataLoader(
            squad_train_clients_dataset_dict[str(client)],
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=train_client_batch_size,
            num_workers=2
        )
        
        squad_test_dataloader_dict[str(client)] = DataLoader(
            squad_test_clients_dataest_dict[str(client)].remove_columns(["example_id", "offset_mapping"]),
            collate_fn=default_data_collator,
            batch_size=test_client_batch_size,
            num_workers=2
        )
    
    return squad_train_dataloader_dict, squad_test_dataloader_dict, squad_test_clients_dataest_dict, squad_test_all_example_clients_data
    