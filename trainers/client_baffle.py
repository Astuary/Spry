"""Client-side implementation of Baffle"""
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
# print(sys.path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ["TRANSFORMERS_VERBOSITY"] = 'error'

import json
import math
import time
import copy
import random
import logging
import argparse
import subprocess as sp
from colorama import Fore
from functools import partial
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.func as fc
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
import functorch as ft

import flwr as fl
from flwr.common.logger import FLOWER_LOGGER
from flwr.server.app import ServerConfig

import dataloaders.agnews.agnews_sequence_classification_dataloader as agnews_sequence_classification_dataloader
import dataloaders.yahoo.yahoo_dataloader as yahoo_dataloader
import dataloaders.sst2.sst2_dataloader as sst2_dataloader
import dataloaders.squad.squad_dataloader as squad_dataloader
import dataloaders.mnli.mnli_dataloader as mnli_dataloader
import dataloaders.snli.snli_dataloader as snli_dataloader
import dataloaders.qnli.qnli_dataloader as qnli_dataloader
import dataloaders.cola.cola_dataloader as cola_dataloader
import dataloaders.yelp.yelp_dataloader as yelp_dataloader
import dataloaders.qqp.qqp_dataloader as qqp_dataloader

from trainers.server_fedyogi import FedYogi
from utils.metric_computation_qa import postprocess_qa_predictions

from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    EvalPrediction,
)

AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def text_classifier_train(net, train_data, test_data):
    net.train()
    optimizer = Adam(net.parameters(), lr=learning_rate)
        
    accumulated_loss = 0.0
    time_per_epoch = []
    time_per_iteration_per_epoch = []
    
    for e in range(epochs):
        start_time = time.perf_counter()
        sample_count = 0
        time_per_iteration = []
        
        for batch_index, batch in enumerate(train_data):
            iter_start_time = time.perf_counter()
            if 'distilbert' in model_name or 'roberta' in model_name:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'attention_mask': batch[1].to(DEVICE),
                    'labels': batch[2].to(DEVICE),
                }
            else:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'attention_mask': batch[1].to(DEVICE),
                    'labels': batch[2].to(DEVICE),
                    'token_type_ids': batch[3].to(DEVICE),
                }
            
            with torch.no_grad():
                original_parameter = net.state_dict()
                delta = OrderedDict()
                grad = OrderedDict()
                
                if finite_difference_format == "forward":
                    true_loss = net(**batch_dict).loss
                    
                    for name in original_parameter.keys():
                        delta[name] = torch.zeros(original_parameter[name].size()).to(DEVICE)
                        grad[name] = torch.zeros(original_parameter[name].size()).to(DEVICE)
                    
                    for i in range(K):
                        for name in delta.keys():
                            delta[name] = sigma * torch.randn(delta[name].size()).to(DEVICE)
                            original_parameter[name] += delta[name]
                        
                        net.load_state_dict(original_parameter, strict=False)
                        loss_add_delta = net(**batch_dict).loss
                        difference = loss_add_delta - true_loss
                        loss = (loss_add_delta + true_loss) / 2
                        
                        for name in delta.keys():
                            grad[name] += (difference / sigma) * (delta[name] / sigma)
                            original_parameter[name] -= delta[name]
                    
                    # return parameter
                    net.load_state_dict(original_parameter)
                    for name, p in net.named_parameters():
                        p.grad = grad[name] / K
                        
                elif finite_difference_format == "center":
                    for name in original_parameter.keys():
                        delta[name] = torch.zeros(original_parameter[name].size()).cuda()
                        grad[name] = torch.zeros(original_parameter[name].size()).cuda()
                    
                    for i in range(K):
                        for name in delta.keys():
                            delta[name] = sigma * torch.randn(delta[name].size()).cuda()
                            original_parameter[name] += delta[name]
                        
                        net.load_state_dict(original_parameter, strict=False)
                        loss_add_delta = net(**batch_dict).loss
                        
                        for name in delta.keys():
                            original_parameter[name] -= 2 * delta[name]
                        
                        net.load_state_dict(original_parameter, strict=False)
                        loss_minus_delta = net(**batch_dict).loss
                        difference = loss_add_delta - loss_minus_delta
                        loss = (loss_add_delta + loss_minus_delta) / 2
                        
                        for name in delta.keys():
                            grad[name] += (difference / (2 * sigma)) * (delta[name] / sigma)
                            original_parameter[name] += delta[name]
                    
                    net.load_state_dict(original_parameter)
                    for name, p in net.named_parameters():
                        p.grad = grad[name] / K
            
            """
            outputs = net(**batch_dict)
            loss = outputs.loss
            loss.backward()
            """
            
            accumulated_loss += loss.item()
            sample_count += batch[0].shape[0]
            
            optimizer.step()
            optimizer.zero_grad()
                
            iter_end_time = time.perf_counter()
            time_per_iteration.append(iter_end_time - iter_start_time)
            
        end_time = time.perf_counter()
        time_per_epoch.append(end_time - start_time)
        average_time_per_iteration = sum(time_per_iteration) / len(time_per_iteration)
        time_per_iteration_per_epoch.append(average_time_per_iteration)
    
    average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    average_time_per_iteration_per_epoch = sum(time_per_iteration_per_epoch) / len(time_per_iteration_per_epoch)
    _, personalized_accuracy, _ = text_classifier_test(net, test_data)

    return (accumulated_loss / sample_count, 
            sample_count, 
            personalized_accuracy, 
            average_time_per_epoch, 
            average_time_per_iteration_per_epoch
    )

def text_classifier_test(net, test_data):
    metric = load_metric("accuracy")
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    for batch in test_data:
        if 'distilbert' in model_name or 'roberta' in model_name:
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE),
                'labels': batch[2].to(DEVICE),
            }
        else:
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE),
                'labels': batch[2].to(DEVICE),
                'token_type_ids': batch[3].to(DEVICE),
            }
        
        with torch.no_grad():
            outputs = net(**batch_dict)
            
        logits = outputs.logits
        accumulated_loss += outputs.loss.item()
        sample_count += batch[0].shape[0]
        
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch_dict["labels"])
    
    accumulated_loss /= sample_count
    accuracy = metric.compute()["accuracy"]
    
    return accumulated_loss, accuracy, sample_count
    
def text_classifier_generalized_test(net, train_data, test_data):
    before_loss, before_accuracy, sample_count = text_classifier_test(net, test_data)
    after_loss, after_accuracy = before_loss, before_accuracy
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, sample_count


class TextClassifierClient(fl.client.NumPyClient):

    def __init__(self, cid, net):# -> None:
        self.net = net.to(DEVICE)
        self.cid = str(cid)
        # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('Model size: ', params)
    
    def get_parameters(self, config):
        trainable_keys = []
        for key, val in self.net.named_parameters():
            if val.requires_grad and peft_method == 'lora':
                trainable_keys.append(key)
            elif 'bias' in key and peft_method == 'bitfit':
                trainable_keys.append(key)

        return [val.cpu().numpy() for key, val in self.net.state_dict().items() if key in trainable_keys]

    def set_parameters(self, parameters):
        trainable_keys = []
        for key, val in self.net.named_parameters():
            if val.requires_grad and peft_method == 'lora':
                trainable_keys.append(key)
            elif 'bias' in key and peft_method == 'bitfit':
                trainable_keys.append(key)

        params_dict = zip(trainable_keys, parameters)
        # params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        # print(self.cid)
        train_data = train_dataloader_dict[self.cid]
        test_data = test_dataloader_dict[self.cid]
        self.set_parameters(parameters)

        (loss, count, personalized_accuracy, 
         average_time_per_epoch, 
         average_time_per_iteration_per_epoch) = text_classifier_train(self.net, train_data, test_data)
        return self.get_parameters(config), count, \
            {'loss': float(loss), 
             'personalized_accuracy': float(personalized_accuracy),
             'average_time_per_epoch': float(average_time_per_epoch),
             'average_time_per_iteration_per_epoch': float(average_time_per_iteration_per_epoch),
            }

    def evaluate(self, parameters, config):
        train_data = train_dataloader_dict[self.cid]
        test_data = test_dataloader_dict[self.cid]
        self.set_parameters(parameters)
        
        (before_loss, before_accuracy, 
         after_loss, after_accuracy, count) = text_classifier_generalized_test(self.net, train_data, test_data)
        return float(before_loss), count, \
            {'before_accuracy': float(before_accuracy), 
             'after_loss': float(after_loss),
             'after_accuracy': float(after_accuracy)}


def client_fn(cid: str):# -> fl.client.Client:
    # print('Picking Client ', cid, flush=True)
    # params = sum([np.prod(p.size()) for p in net.parameters()])
    # print('Model size: ', params)
        
    if dataset in ['agnews', 'yahoo', 'sst2', 'mnli', 'snli', 'qnli', 'cola', 'yelp', 'qqp']:
        net = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels = num_labels, 
                cache_dir = './models', 
            )  
        if peft_method == 'lora':
            if 'distilbert' in model_name:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    r=lora_r, lora_alpha=lora_alpha, 
                    target_modules = ['q_lin', 'v_lin'],
                    lora_dropout=0.1
                )
            elif 'albert' in model_name:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    r=lora_r, lora_alpha=lora_alpha, 
                    target_modules = ['query', 'value'],
                    lora_dropout=0.1
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    r=lora_r, lora_alpha=lora_alpha, 
                    lora_dropout=0.1
                )  
            net = get_peft_model(net, lora_config)
    
    elif dataset in 'squad':
        net = AutoModelForQuestionAnswering.from_pretrained(
                model_name, 
                cache_dir = './models', 
            )
        if peft_method == 'lora':
            lora_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS, 
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["ff.layer_1", "ff.layer_2"],
                lora_dropout=0.1,
            )
            net = get_peft_model(net, lora_config)
    
    if dataset == 'agnews': 
        return TextClassifierClient(cid, net)
    
    elif dataset == 'yahoo': 
        return TextClassifierClient(cid, net)
    
    elif dataset == 'sst2':
        net.config.label2id = {'negative': 0, 'positive': 1}
        net.config.id2label = {0: 'negative', 1: 'positive'}
        return TextClassifierClient(cid, net)
    
    elif dataset == 'squad':
        return QuestionAnswererClient(cid, net)
        
    elif dataset == 'mnli':
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        return TextClassifierClient(cid, net)
    
    elif dataset == 'snli':
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        return TextClassifierClient(cid, net)
    
    elif dataset == 'qnli':
        net.config.label2id = {'entailment': 0, 'not_entailment': 1}
        net.config.id2label = {0: 'entailment', 1: 'not_entailment'}
        return TextClassifierClient(cid, net)
    
    elif dataset == 'cola':
        net.config.label2id = {'unacceptable': 0, 'acceptable': 1}
        net.config.id2label = {0: 'unacceptable', 1: 'acceptable'}
        return TextClassifierClient(cid, net)
    
    elif dataset == 'yelp':
        return TextClassifierClient(cid, net)
    
    elif dataset == 'qqp':
        net.config.label2id = {'not_duplicate': 0, 'duplicate': 1}
        net.config.id2label = {0: 'not_duplicate', 1: 'duplicate'}
        
        return TextClassifierClient(cid, net)

def load_parameters_from_disk():# -> fl.common.Parameters:
    model_file_name = temp_dir + 'model.pth'

    if dataset in ['agnews', 'yahoo', 'sst2', 'mnli', 'snli', 'qnli', 'cola', 'yelp', 'qqp']:
        if peft_method == 'lora':
            if 'distilbert' in model_name:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    r=lora_r, lora_alpha=lora_alpha, 
                    target_modules = ['q_lin', 'v_lin'],
                    lora_dropout=0.1
                )
            elif 'albert' in model_name:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    r=lora_r, lora_alpha=lora_alpha, 
                    target_modules = ['query', 'value'],
                    lora_dropout=0.1
                )
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    r=lora_r, lora_alpha=lora_alpha, 
                    lora_dropout=0.1
                )
        
        net = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels,
                cache_dir = './models',
            )
        
        if peft_method == 'lora':
            net = get_peft_model(net, lora_config)
            
    elif dataset == 'squad' :
        if peft_method == 'lora':
            lora_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS, 
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["ff.layer_1", "ff.layer_2"],
                lora_dropout=0.1,
            )
        
        net = AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                cache_dir = './models',
            )
        
        if peft_method == 'lora':
            net = get_peft_model(net, lora_config)
    
    
    if dataset == 'sst2' :
        net.config.label2id = {'negative': 0, 'positive': 1}
        net.config.id2label = {0: 'negative', 1: 'positive'}
    
    elif dataset == 'mnli' :
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    elif dataset == 'snli' :
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    
    elif dataset == 'cola' :
        net.config.label2id = {'unacceptable': 0, 'acceptable': 1}
        net.config.id2label = {0: 'unacceptable', 1: 'acceptable'}
        
    elif dataset == 'qqp':
        net.config.label2id = {'not_duplicate': 0, 'duplicate': 1}
        net.config.id2label = {0: 'not_duplicate', 1: 'duplicate'}
    
    trainable_keys = []
    for key, val in net.named_parameters():
        if val.requires_grad and peft_method == 'lora':
            trainable_keys.append(key)
        elif 'bias' in key and peft_method == 'bitfit':
            trainable_keys.append(key)
            
    if not os.path.exists(model_file_name):
        return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for key, val in net.state_dict().items() if key in trainable_keys]
            ), rounds
    
    print("Loading: ", model_file_name)

    checkpoint = torch.load(model_file_name)
    net.load_state_dict(checkpoint['net_state_dict'], strict = False)
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    random.setstate(checkpoint['random_random_state'])

    print(Fore.YELLOW + f"Loading model weights from round #{checkpoint['round']}" + Fore.WHITE)

    return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for key, val in net.state_dict().items() if key in trainable_keys]
        ), rounds - checkpoint['round']

if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'Enter the dataset you want to train your algorithm on.')
    parser.add_argument('--model_name',)
    parser.add_argument('--random_seed',)
    parser.add_argument('--cpu_simultaneous_clients',)
    parser.add_argument('--gpu_simultaneous_clients',)
    parser.add_argument('--checkpoint_interval',)
    parser.add_argument('--num_clients',)
    parser.add_argument('--total_clients',)
    parser.add_argument('--rounds',)
    parser.add_argument('--total_rounds',)
    parser.add_argument('--dirichlet_distribution',)
    parser.add_argument('--epochs',)
    parser.add_argument('--batch_size',)
    parser.add_argument('--lr',)
    parser.add_argument('--max_seq_len',)
    parser.add_argument('--peft_method',)
    parser.add_argument('--lora_r',)
    parser.add_argument('--lora_alpha',)
    parser.add_argument('--K',)
    parser.add_argument('--sigma',)
    parser.add_argument('--finite_difference_format',)
    parser.add_argument('--experiment_name',)
    
    parser.add_argument('--checkpoint_dir',)
    parser.add_argument('--dataset_dir',)
    parser.add_argument('--temp_dir',)
    
    args = parser.parse_args()
    hyperparameters = {}

    dataset = args.dataset
    num_clients = int(args.num_clients)
    hyperparameters['num_clients'] = num_clients
    total_clients = int(args.total_clients)
    hyperparameters['total_clients'] = total_clients
    cpu_simultaneous_clients = int(args.cpu_simultaneous_clients)
    hyperparameters['cpu_simultaneous_clients'] = cpu_simultaneous_clients
    gpu_simultaneous_clients = int(args.gpu_simultaneous_clients)
    hyperparameters['gpu_simultaneous_clients'] = gpu_simultaneous_clients
    rounds = int(args.rounds)
    hyperparameters['rounds'] = rounds
    total_rounds = int(args.rounds)
    epochs = int(args.epochs)
    hyperparameters['epochs'] = epochs
    dirichlet_distribution = float(args.dirichlet_distribution)
    hyperparameters['dirichlet_distribution'] = dirichlet_distribution
    model_name = args.model_name
    hyperparameters['model_name'] = model_name
    peft_method = args.peft_method
    hyperparameters['peft_method'] = peft_method
    lora_r = int(args.lora_r)
    hyperparameters['lora_r'] = lora_r
    lora_alpha = int(args.lora_alpha)
    hyperparameters['lora_alpha'] = lora_alpha
    experiment_name = args.experiment_name
    hyperparameters['experiment_name'] = experiment_name
    checkpoint_interval = int(args.checkpoint_interval)
    hyperparameters['checkpoint_interval'] = checkpoint_interval
    batch_size = int(args.batch_size)
    hyperparameters['batch_size'] = batch_size
    learning_rate = float(args.lr)
    hyperparameters['learning_rate'] = learning_rate
    K = int(args.K)
    hyperparameters['K'] = K
    sigma = float(args.sigma)
    hyperparameters['sigma'] = sigma
    finite_difference_format = args.finite_difference_format
    hyperparameters['finite_difference_format'] = finite_difference_format
    max_seq_len = int(args.max_seq_len)
    hyperparameters['max_seq_len'] = max_seq_len
    random_seed = int(args.random_seed)
    hyperparameters['random_seed'] = random_seed
    
    checkpoint_dir = args.checkpoint_dir + experiment_name + '/'
    hyperparameters['checkpoint_dir'] = checkpoint_dir
    dataset_dir = args.dataset_dir
    hyperparameters['dataset_dir'] = dataset_dir
    temp_dir = args.temp_dir + experiment_name + '/'
    hyperparameters['temp_dir'] = temp_dir
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    if dataset == 'agnews':
        train_dataloader_dict, test_dataloader_dict = agnews_sequence_classification_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 4
    
    elif dataset == 'yahoo':
        train_dataloader_dict, test_dataloader_dict = yahoo_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name
            )
        
        num_labels = 10
        
    elif dataset == 'sst2':
        train_dataloader_dict, test_dataloader_dict = sst2_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 2
        
    elif dataset == 'squad':
        (train_dataloader_dict, test_dataloader_dict, 
         test_dataset_dict, test_examples_dict) = squad_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
    elif dataset == 'mnli':
        train_dataloader_dict, test_dataloader_dict = mnli_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 3
        
    elif dataset == 'snli':
        train_dataloader_dict, test_dataloader_dict = snli_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 3
        
    elif dataset == 'qnli':
        train_dataloader_dict, test_dataloader_dict = qnli_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 2
    
    elif dataset == 'cola':
        train_dataloader_dict, test_dataloader_dict = qnli_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 2
       
    elif dataset == 'yelp':
        train_dataloader_dict, test_dataloader_dict = yelp_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 2
       
    elif dataset == 'qqp':
        train_dataloader_dict, test_dataloader_dict = qqp_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 2
     
    hyperparameters['num_labels'] = num_labels
        
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    clients = [str(c) for c in range(total_clients)]

    num_cpus_needed = int(AVAILABLE_CPUS / cpu_simultaneous_clients) # int(AVAILABLE_CPUS / num_clients)
    num_gpus_needed = float(AVAILABLE_GPUS / gpu_simultaneous_clients)
    
    hyperparameters['num_cpus_needed'] = num_cpus_needed
    hyperparameters['num_gpus_needed'] = num_gpus_needed
    
    with open(checkpoint_dir + 'hyperparameter_snapshot.json', 'w+') as f:
        json.dump(hyperparameters, f)
    
    print('Experiment: ', experiment_name)
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  

    initial_parameters, rounds = load_parameters_from_disk()

    FLOWER_LOGGER.setLevel(logging.ERROR)

    strategy = FedYogi(
        fraction_fit = num_clients / len(clients),
        fraction_eval = num_clients / len(clients),
        min_fit_clients = num_clients,
        min_eval_clients = num_clients,
        min_available_clients = num_clients,
        dataset = dataset,
        client_algorithm = 'Baffle',
        hyperparameters = hyperparameters,
        initial_parameters = initial_parameters,
    )
    
    print(Fore.RED + "Availble Device: " + str(DEVICE) + ", Count: " + \
        str(AVAILABLE_GPUS) + ". CPUs, Count: " + str(AVAILABLE_CPUS) + '.' + Fore.WHITE, flush=True)

    config = ServerConfig()
    config.num_rounds = rounds

    simulation_start_time = datetime.now() 
    print('Starting at:', simulation_start_time)
    print('')

    fl.simulation.start_simulation(
        client_fn = client_fn,
        client_resources = {'num_cpus': num_cpus_needed, 
                            'num_gpus': num_gpus_needed},
        clients_ids = clients,
        config = config,
        strategy = strategy,
        ray_init_args = {'num_cpus': AVAILABLE_CPUS, 'num_gpus': AVAILABLE_GPUS}
    )
    
    simulation_end_time = datetime.now()
    simulation_duration = simulation_end_time - simulation_start_time
    
    duration_in_s = simulation_duration.total_seconds()
    days    = divmod(duration_in_s, 86400)
    hours   = divmod(days[1], 3600)
    minutes = divmod(hours[1], 60)
    seconds = divmod(minutes[1], 1)
    duration_info_string = "Simulation Duration: " + str(days[0]) + " days, "\
                            + str(hours[0]) + " hours, "\
                            + str(minutes[0]) + " minutes and "\
                            + str(seconds[0]) + " seconds"
    print(duration_info_string)
    
    with open(checkpoint_dir + 'simulation_duration.txt', 'w+') as f:
        f.write(duration_info_string)

    os.remove(temp_dir + 'model.pth')
        