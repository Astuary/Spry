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

import copy
import json
import random
import logging
import argparse
import threading
from colorama import Fore
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
import numpy as np
import matplotlib.pyplot as plt

import dataloaders.agnews.agnews_sequence_classification_dataloader as agnews_sequence_classification_dataloader
import dataloaders.yahoo.yahoo_dataloader as yahoo_dataloader
import dataloaders.sst2.sst2_dataloader as sst2_dataloader
import dataloaders.squad.squad_dataloader_t5 as squad_dataloader
import dataloaders.mnli.mnli_dataloader as mnli_dataloader
import dataloaders.snli.snli_dataloader as snli_dataloader
import dataloaders.qnli.qnli_dataloader as qnli_dataloader
import dataloaders.cola.cola_dataloader as cola_dataloader
import dataloaders.yelp.yelp_dataloader as yelp_dataloader
import dataloaders.qqp.qqp_dataloader as qqp_dataloader
import dataloaders.multirc.multirc_dataloader as multirc_dataloader

from utils.metric_computation_qa import postprocess_qa_predictions

from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    AutoModelForQuestionAnswering,
    EvalPrediction,
    GPTQConfig,
)

AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def agnews_train(net, batch):
    net.eval()
    
    batch_dict = {
        'input_ids': batch[0].to(DEVICE),
        'token_type_ids': batch[1].to(DEVICE),
        'attention_mask': batch[2].to(DEVICE),
        'labels': batch[3].to(DEVICE),
    }
    
    outputs = net(**batch_dict)
    loss = outputs.loss
    loss.backward()
        
    accumulated_loss = loss.item()
    sample_count = batch[0].shape[0]
    
    return accumulated_loss / sample_count, sample_count, {n: p.grad for n, p in net.named_parameters() if p.requires_grad}

def agnews_test(net, test_data):
    metric = load_metric("accuracy")
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    for batch in test_data:
        batch_dict = {
            'input_ids': batch[0].to(DEVICE),
            'token_type_ids': batch[1].to(DEVICE),
            'attention_mask': batch[2].to(DEVICE),
            'labels': batch[3].to(DEVICE),
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
    
def agnews_generalized_test(net, train_data, test_data):
    before_loss, before_accuracy, sample_count = agnews_test(net, test_data)
    after_loss, after_accuracy = before_loss, before_accuracy
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, sample_count

def agnews_trainer():
    print(Fore.GREEN +'Training begins.'+ Fore.WHITE + '\n')
    if peft_method == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=lora_r, lora_alpha=lora_alpha, 
            lora_dropout=0.1
        )
    
    net = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = num_labels, 
            cache_dir = './models', 
        ).to(DEVICE)
    # params = sum([np.prod(p.size()) for p in net.parameters()])
    # print('Model size: ', params)
    
    if peft_method == 'lora':
        net = get_peft_model(net, lora_config)

    net.train()
    optimizer = Adam(net.parameters(), lr=learning_rate)
    
    tracker_personalized_accuracy = {}
    tracker_generalized_accuracy_before = {}
    tracker_generalized_accuracy_after = {}
    tracker_loss = {}
    # tracker_cos_angle_grads = {}
    # tracker_cos_sim_grads = {}
    
    tracker_all_personalized_accuracies = {}
    tracker_all_generalized_accuracies_before = {}
    tracker_all_generalized_accuracies_after = {}
    tracker_all_losses = {}
    # tracker_all_cos_angle_grads = {}
    # tracker_all_cos_sim_grads = {}

    for r in range(rounds, total_rounds + 1):
        current_clients = [str(random.choice(clients)) for _ in range(num_clients)]

        # threads = []
        train_loss_total = 0.0
        test_accuracy_total = 0.0

        tracker_round_personalized_accuracies = []
        tracker_round_generalized_accuracies_before = []
        tracker_round_generalized_accuracies_after = []
        tracker_round_losses = []

        for current_batch_index in range(batch_count):
            g_c = None
            sample_count_total = 0
            # tracker_round_cos_sim = []
            # tracker_round_cos_angle = []
            # gradients_per_client = []
            
            for cid in current_clients:
                # train
                train_data = iter(train_dataloader_dict[cid])
                for batch_index, batch in enumerate(train_data):
                    if batch_index == current_batch_index:
                        break
            
                train_loss, sample_count, grads = agnews_train(net, batch)
                sample_count_total += sample_count
                
                # for _, w in weights.items():
                #     print(w[0][0], '*')
                #     break
                # gradients_per_client.append(grads)
                
                if g_c == None:
                    g_c = {name: grad.data for name, grad in grads.items()}
                else:
                    # w_c = [(layer.data * sample_count) + w_c_layer for layer, w_c_layer in zip(weights, w_c)]
                    for name, grad in grads.items():
                        g_c[name] += (grad.data)

                # personalized test
                if current_batch_index == batch_count - 1:
                    test_data = test_dataloader_dict[cid]  
                    valid_loss, accuracy, total_samples = agnews_test(net, test_data)
                    
                    train_loss_total += train_loss
                    # print(sample_count, sample_count_total)
                    # print(train_loss, train_loss_total)
                    test_accuracy_total += accuracy
                    
                    tracker_round_personalized_accuracies.append(accuracy)
                    tracker_round_losses.append(train_loss)

            # aggregate  
            aggregate_g = g_c
            
            # getting the cos similarity
            # for client_grad in gradients_per_client:
            #     cosine_similarities = []
            #     cosine_angles = []
            #     for name, grad in client_grad.items():
            #         flat_tensor_1 = torch.flatten(grad)
            #         flat_tensor_2 = torch.flatten(aggregate_g[name])
                    
            #         normalized_tensor_1 = F.normalize(flat_tensor_1, p=2, dim=-1)
            #         normalized_tensor_2 = F.normalize(flat_tensor_2, p=2, dim=-1)
                               
            #         cosine_similarity = F.cosine_similarity(normalized_tensor_1, normalized_tensor_2, dim=-1)    
            #         cosine_similarities.append(cosine_similarity.item())

            #         angle_radians = torch.acos(cosine_similarity)
            #         angle_degrees = torch.rad2deg(angle_radians)
            #         cosine_angles.append(angle_degrees.item())
                
            #     cosine_similarity_average = sum(cosine_similarities) / len(cosine_similarities)
            #     tracker_round_cos_sim.append(cosine_similarity_average)
                
            #     cosine_angle_average = sum(cosine_angles) / len(cosine_angles)
            #     tracker_round_cos_angle.append(cosine_angle_average)
            
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.grad = aggregate_g[n]
            
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss_total /= num_clients
        test_accuracy_total /= num_clients
        
        tracker_all_personalized_accuracies[r] = tracker_round_personalized_accuracies
        tracker_all_losses[r] = tracker_round_losses
        # tracker_all_cos_sim_grads[r] = tracker_round_cos_sim
        # tracker_all_cos_angle_grads[r] = tracker_round_cos_angle
        tracker_loss[r] = train_loss_total
        tracker_personalized_accuracy[r] = test_accuracy_total
        # tracker_cos_sim_grads[r] = sum(tracker_round_cos_sim) / len(tracker_round_cos_sim)
        # tracker_cos_angle_grads[r] = sum(tracker_round_cos_angle) / len(tracker_round_cos_angle)

        print(Fore.BLUE + f"{method}: Round {r} loss aggregated from client results: {train_loss_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method}: Round {r} personalized accuracy aggregated from client results: {test_accuracy_total}" + Fore.WHITE)

        # generalized test
        before_test_accuracy_total = 0.0
        after_test_accuracy_total = 0.0
        for c in range(num_clients):
            cid = random.choice(clients).strip()

            # test
            train_data = train_dataloader_dict[cid]
            test_data = test_dataloader_dict[cid]

            before_loss, before_accuracy, after_loss, after_accuracy, after_total = agnews_generalized_test(net, train_data, test_data)
            before_test_accuracy_total += before_accuracy
            after_test_accuracy_total += after_accuracy

            tracker_round_generalized_accuracies_before.append(before_accuracy)
            tracker_round_generalized_accuracies_after.append(after_accuracy)

        tracker_all_generalized_accuracies_before[r] = tracker_round_generalized_accuracies_before
        tracker_all_generalized_accuracies_after[r] = tracker_round_generalized_accuracies_after

        before_test_accuracy_total /= num_clients
        after_test_accuracy_total /= num_clients
        print(Fore.WHITE + f"{method}: Round {r} generalized accuracy (before training) aggregated from client results: {before_test_accuracy_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method}: Round {r} generalized accuracy (after training) aggregated from client results: {after_test_accuracy_total}\n" + Fore.WHITE)

        tracker_generalized_accuracy_before[r] = before_test_accuracy_total
        tracker_generalized_accuracy_after[r] = after_test_accuracy_total

        if r % checkpoint_interval == 0:
            with open(checkpoint_dir + '/all_personalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_all_personalized_accuracies, fp)
            with open(checkpoint_dir + '/all_before_generalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_all_generalized_accuracies_before, fp)
            with open(checkpoint_dir + '/all_after_generalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_all_generalized_accuracies_after, fp)
            with open(checkpoint_dir + '/all_losses.json', 'w+') as fp:
                json.dump(tracker_all_losses, fp)
            # with open(checkpoint_dir + '/all_cos_sim.json', 'w+') as fp:
            #     json.dump(tracker_all_cos_sim_grads, fp)
            # with open(checkpoint_dir + '/all_cos_angle.json', 'w+') as fp:
            #     json.dump(tracker_all_cos_angle_grads, fp)
            
            with open(checkpoint_dir + '/personalized_accuracy.json', 'w+') as fp:
                json.dump(tracker_personalized_accuracy, fp)
            with open(checkpoint_dir + '/before_generalized_accuracy.json', 'w+') as fp:
                json.dump(tracker_generalized_accuracy_before, fp)
            with open(checkpoint_dir + '/after_generalized_accuracy.json', 'w+') as fp:
                json.dump(tracker_generalized_accuracy_after, fp)
            with open(checkpoint_dir + '/loss.json', 'w+') as fp:
                json.dump(tracker_loss, fp)
            # with open(checkpoint_dir + '/cos_sim.json', 'w+') as fp:
            #     json.dump(tracker_cos_sim_grads, fp)
            # with open(checkpoint_dir + '/cos_angle.json', 'w+') as fp:
            #     json.dump(tracker_cos_angle_grads, fp)
                
            plt.figure(0)
            plt.plot(list(map(int, list(tracker_loss.keys()))), tracker_loss.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Training Loss')
            plt.title('Training Loss for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_losses.png')
            
            plt.figure(1)
            plt.plot(list(map(int, list(tracker_personalized_accuracy.keys()))), tracker_personalized_accuracy.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized Accuracy')
            plt.title('Personalized Accuracy for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_personalized_accuracy.png')
            
            plt.figure(2)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_before.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (Before)')
            plt.title('Generalized Accuracy (Before) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracy_before.png')
            
            plt.figure(3)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_after.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (After)')
            plt.title('Generalized Accuracy (After) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracy_after.png')
            
            # plt.figure(4)
            # plt.plot(list(map(int, list(tracker_cos_sim_grads.keys()))), tracker_cos_sim_grads.values(), c='#2978A0', label='Average')
            # plt.xlabel('Number of Rounds')
            # plt.ylabel('Cosine Similarity')
            # plt.title('Cosine Similarity for ' + method + ' Baseline')
            # plt.savefig(checkpoint_dir + '/plot_cos_sim.png')
            
            # plt.figure(5)
            # plt.plot(list(map(int, list(tracker_cos_angle_grads.keys()))), tracker_cos_angle_grads.values(), c='#2978A0', label='Average')
            # plt.xlabel('Number of Rounds')
            # plt.ylabel('Cosine Angle')
            # plt.title('Cosine Angle for ' + method + ' Baseline')
            # plt.savefig(checkpoint_dir + '/plot_cos_angle.png')

            torch.save({
                'round': r,
                'net_state_dict': net.state_dict(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'random_random_state': random.getstate(),
                }, temp_dir + 'model.pth')

def mnli_train(net, batch):
    net.eval()
    
    batch_dict = {
        'input_ids': batch[0].to(DEVICE),
        'attention_mask': batch[1].to(DEVICE),
        'labels': batch[2].to(DEVICE),
        # 'token_type_ids': batch[1].to(DEVICE),
    }
    
    outputs = net(**batch_dict)
    loss = outputs.loss
    loss.backward()
        
    accumulated_loss = loss.item()
    sample_count = batch[0].shape[0]
    
    return accumulated_loss / sample_count, sample_count, {n: p.grad for n, p in net.named_parameters() if p.requires_grad}

def mnli_test(net, test_data):
    metric = load_metric("accuracy")
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    for batch in test_data:
        batch_dict = {
            'input_ids': batch[0].to(DEVICE),
            'attention_mask': batch[1].to(DEVICE),
            'labels': batch[2].to(DEVICE),
            # 'token_type_ids': batch[1].to(DEVICE),
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
    
def mnli_generalized_test(net, train_data, test_data):
    before_loss, before_accuracy, sample_count = mnli_test(net, test_data)
    after_loss, after_accuracy = before_loss, before_accuracy
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, sample_count

def mnli_trainer():
    logging.info(Fore.GREEN +'Training begins.'+ Fore.WHITE + '\n')
    if peft_method == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=lora_r, lora_alpha=lora_alpha, 
            lora_dropout=0.1
        )
    
    net = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = num_labels, 
            cache_dir = './models', 
        ).to(DEVICE)
    
    if peft_method == 'lora':
        net = get_peft_model(net, lora_config)

    net.train()
    optimizer = Adam(net.parameters(), lr=learning_rate)
    
    tracker_personalized_accuracy = {}
    tracker_generalized_accuracy_before = {}
    tracker_generalized_accuracy_after = {}
    tracker_loss = {}
    
    tracker_all_personalized_accuracies = {}
    tracker_all_generalized_accuracies_before = {}
    tracker_all_generalized_accuracies_after = {}
    tracker_all_losses = {}

    for r in range(rounds, total_rounds + 1):
        current_clients = [str(random.choice(clients)) for _ in range(num_clients)]

        # threads = []
        train_loss_total = 0.0
        test_accuracy_total = 0.0

        tracker_round_personalized_accuracies = []
        tracker_round_generalized_accuracies_before = []
        tracker_round_generalized_accuracies_after = []
        tracker_round_losses = []

        # logger.info(f'Batch count: {batch_count}')
        for current_batch_index in range(batch_count):
            g_c = None
            sample_count_total = 0
            
            for cid in current_clients:
                # train
                train_data = iter(train_dataloader_dict[cid])
                for batch_index, batch in enumerate(train_data):
                    if batch_index == current_batch_index:
                        break
            
                train_loss, sample_count, grads = mnli_train(net, batch)
                sample_count_total += sample_count
                
                # for _, w in weights.items():
                #     print(w[0][0], '*')
                #     break
                # gradients_per_client.append(grads)
                
                if g_c == None:
                    g_c = {name: grad.data for name, grad in grads.items()}
                else:
                    # w_c = [(layer.data * sample_count) + w_c_layer for layer, w_c_layer in zip(weights, w_c)]
                    for name, grad in grads.items():
                        g_c[name] += (grad.data)

                # personalized test
                if current_batch_index == batch_count - 1:
                    test_data = test_dataloader_dict[cid]  
                    valid_loss, accuracy, total_samples = mnli_test(net, test_data)
                    
                    train_loss_total += train_loss
                    # print(sample_count, sample_count_total)
                    # print(train_loss, train_loss_total)
                    test_accuracy_total += accuracy
                    
                    tracker_round_personalized_accuracies.append(accuracy)
                    tracker_round_losses.append(train_loss)

            # aggregate  
            aggregate_g = {name: grad / len(current_clients) for name, grad in g_c.items()}
            
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.grad = aggregate_g[n]
            
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss_total /= num_clients
        test_accuracy_total /= num_clients
        
        tracker_all_personalized_accuracies[r] = tracker_round_personalized_accuracies
        tracker_all_losses[r] = tracker_round_losses
        tracker_loss[r] = train_loss_total
        tracker_personalized_accuracy[r] = test_accuracy_total

        print(Fore.BLUE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} loss aggregated from client results: {train_loss_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} personalized accuracy aggregated from client results: {test_accuracy_total}" + Fore.WHITE)

        # generalized test
        before_test_accuracy_total = 0.0
        after_test_accuracy_total = 0.0
        for c in range(num_clients):
            cid = random.choice(clients).strip()

            # test
            train_data = train_dataloader_dict[cid]
            test_data = test_dataloader_dict[cid]

            (before_loss, before_accuracy, 
             after_loss, after_accuracy, 
             after_total) = mnli_generalized_test(net, train_data, test_data)
            before_test_accuracy_total += before_accuracy
            after_test_accuracy_total += after_accuracy

            tracker_round_generalized_accuracies_before.append(before_accuracy)
            tracker_round_generalized_accuracies_after.append(after_accuracy)

        tracker_all_generalized_accuracies_before[r] = tracker_round_generalized_accuracies_before
        tracker_all_generalized_accuracies_after[r] = tracker_round_generalized_accuracies_after

        before_test_accuracy_total /= num_clients
        after_test_accuracy_total /= num_clients
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized accuracy (before training) aggregated from client results: {before_test_accuracy_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized accuracy (after training) aggregated from client results: {after_test_accuracy_total}\n" + Fore.WHITE)

        tracker_generalized_accuracy_before[r] = before_test_accuracy_total
        tracker_generalized_accuracy_after[r] = after_test_accuracy_total

        if r % checkpoint_interval == 0:
            with open(checkpoint_dir + '/all_personalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_all_personalized_accuracies, fp)
            with open(checkpoint_dir + '/all_before_generalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_all_generalized_accuracies_before, fp)
            with open(checkpoint_dir + '/all_after_generalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_all_generalized_accuracies_after, fp)
            with open(checkpoint_dir + '/all_losses.json', 'w+') as fp:
                json.dump(tracker_all_losses, fp)
            
            with open(checkpoint_dir + '/personalized_accuracy.json', 'w+') as fp:
                json.dump(tracker_personalized_accuracy, fp)
            with open(checkpoint_dir + '/before_generalized_accuracy.json', 'w+') as fp:
                json.dump(tracker_generalized_accuracy_before, fp)
            with open(checkpoint_dir + '/after_generalized_accuracy.json', 'w+') as fp:
                json.dump(tracker_generalized_accuracy_after, fp)
            with open(checkpoint_dir + '/loss.json', 'w+') as fp:
                json.dump(tracker_loss, fp)
                
            plt.figure(0)
            plt.plot(list(map(int, list(tracker_loss.keys()))), tracker_loss.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Training Loss')
            plt.title('Training Loss for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_losses.png')
            
            plt.figure(1)
            plt.plot(list(map(int, list(tracker_personalized_accuracy.keys()))), tracker_personalized_accuracy.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized Accuracy')
            plt.title('Personalized Accuracy for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_personalized_accuracy.png')
            
            plt.figure(2)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_before.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (Before)')
            plt.title('Generalized Accuracy (Before) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracy_before.png')
            
            plt.figure(3)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_after.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (After)')
            plt.title('Generalized Accuracy (After) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracy_after.png')
            
            torch.save({
                'round': r,
                'net_state_dict': net.state_dict(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'random_random_state': random.getstate(),
                }, temp_dir + 'model.pth')

def question_answerer_train(net, batch):
    net.train()
    
    accumulated_loss = 0.0
    
    batch = {
        'input_ids': batch['input_ids'].to(DEVICE),
        'attention_mask': batch['attention_mask'].to(DEVICE),
        'start_positions': batch['start_positions'].to(DEVICE),
        'end_positions': batch['end_positions'].to(DEVICE),
    }
    
    outputs = net(**batch)
    loss = outputs.loss
    loss.backward()
            
    accumulated_loss += loss.item()
    sample_count = batch['input_ids'].shape[0]
        
    torch.cuda.empty_cache()

    return accumulated_loss / sample_count, sample_count, {n: p.grad for n, p in net.named_parameters() if p.requires_grad}
    
def question_answerer_test(net, test_dataloader, test_dataset, test_examples):
    metric = load_metric("squad_v2")
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    all_start_logits = []
    all_end_logits = []
    
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step
            
            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat
    
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=10,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            prefix=stage,
        )
        
        # Format the result to the format the metric expects.
        # formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    
    for batch in test_dataloader:
        # batch_dict = {
        #     'input_ids': batch[0].to(DEVICE),
        #     'token_type_ids': batch[1].to(DEVICE),
        #     'attention_mask': batch[2].to(DEVICE),
        #     'start_positions': batch[3].to(DEVICE),
        #     'end_positions': batch[4].to(DEVICE),
        # }
        batch = {
            'input_ids': batch['input_ids'].to(DEVICE),
            'attention_mask': batch['attention_mask'].to(DEVICE),
        }
        
        with torch.no_grad():
            # outputs = net(**batch_dict)
            outputs = net(**batch)
            
        # print('loss', outputs.loss)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        all_start_logits.append(start_logits.cpu().numpy())
        all_end_logits.append(end_logits.cpu().numpy())
        
        # accumulated_loss += outputs.loss.item()
        sample_count += batch['input_ids'].shape[0]
        
    accumulated_loss /= sample_count
    max_len = max([x.shape[1] for x in all_start_logits])
    
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)
    
    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(test_examples, test_dataset, outputs_numpy)
    eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    
    return accumulated_loss, eval_metric['exact'], eval_metric['f1'], sample_count

def question_answerer_generalized_test(net, train_dataloader, test_dataloader, test_dataset, test_examples):
    before_loss, before_exact_match, before_f1, sample_count = question_answerer_test(net, test_dataloader, test_dataset, test_examples)
    # before_loss, before_accuracy, sample_count = question_answerer_test(net, test_data)
    after_loss, after_exact_match, after_f1 = before_loss, before_exact_match, before_f1
    return before_loss, before_exact_match, before_f1, after_loss, after_exact_match, after_f1, sample_count

def question_answerer_trainer():
    logging.info(Fore.GREEN +'Training begins.'+ Fore.WHITE + '\n')
    if peft_method == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS, 
            r=lora_r, lora_alpha=lora_alpha, 
            lora_dropout=0.1
        )
    
    gptq_config = GPTQConfig(bits=4, use_exllama=False,)
    net = AutoModelForQuestionAnswering.from_pretrained(
        "./models/models--4bit-autogptq-quantized-opt6B-v1" if 'opt-6' in model_name else "./models/models--4bit-autogptq-quantized-opt13B-v1", 
        local_files_only=True,
        quantization_config=gptq_config,
        # device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to(DEVICE)
    
    if peft_method == 'lora':
        net = get_peft_model(net, lora_config)

    net.train()
    if server_opt == 'sgd':
        optimizer = SGD(net.parameters(), lr=learning_rate)
    elif server_opt == 'adamw':
        optimizer = AdamW(net.parameters(), lr=learning_rate)
    elif server_opt == 'adam':
        optimizer = Adam(net.parameters(), lr=learning_rate)
    
    tracker_personalized_exact_matches = {}
    tracker_personalized_f1_score = {}
    tracker_generalized_exact_matches_before = {}
    tracker_generalized_f1_score_before = {}
    tracker_generalized_exact_matches_after = {}
    tracker_generalized_f1_score_after = {}
    tracker_loss = {}
    
    tracker_all_personalized_exact_matches = {}
    tracker_all_personalized_f1_score = {}
    tracker_all_generalized_exact_matches_before = {}
    tracker_all_generalized_f1_score_before = {}
    tracker_all_generalized_exact_matches_after = {}
    tracker_all_generalized_f1_score_after = {}
    tracker_all_losses = {}

    for r in range(rounds, total_rounds + 1):
        current_clients = [str(random.choice(clients)) for _ in range(num_clients)]

        # threads = []
        train_loss_total = 0.0
        test_exact_matches_total = 0.0
        test_f1_score_total = 0.0

        tracker_round_personalized_exact_matches = []
        tracker_round_personalized_f1_score = []
        tracker_round_generalized_exact_matches_before = []
        tracker_round_generalized_f1_score_before = []
        tracker_round_generalized_exact_matches_after = []
        tracker_round_generalized_f1_score_after = []
        tracker_round_losses = []

        logger.info(f'Batch count: {batch_count}')
        for current_batch_index in range(batch_count):
            g_c = None
            sample_count_total = 0
            
            for cid in current_clients:
                # train
                train_data = iter(train_dataloader_dict[cid])
                for batch_index, batch in enumerate(train_data):
                    if batch_index == current_batch_index:
                        break
            
                train_loss, sample_count, grads = question_answerer_train(net, batch)
                sample_count_total += sample_count
                
                # for _, w in weights.items():
                #     print(w[0][0], '*')
                #     break
                # gradients_per_client.append(grads)
                
                if g_c == None:
                    g_c = {name: grad.data for name, grad in grads.items()}
                else:
                    for name, grad in grads.items():
                        g_c[name] += (grad.data)

                # personalized test
                if current_batch_index == batch_count - 1:
                    test_dataloader = test_dataloader_dict[cid]  
                    test_dataset = test_dataset_dict[cid]
                    test_examples = test_examples_dict[cid]
                    valid_loss, exact_matches, f1_score, total_samples = question_answerer_test(net, test_dataloader, test_dataset, test_examples)
                    
                    train_loss_total += train_loss
                    # print(sample_count, sample_count_total)
                    # print(train_loss, train_loss_total)
                    test_exact_matches_total += exact_matches
                    test_f1_score_total += f1_score
                    
                    tracker_round_personalized_exact_matches.append(exact_matches)
                    tracker_round_personalized_f1_score.append(f1_score)
                    tracker_round_losses.append(train_loss)

            # aggregate  
            aggregate_g = {name: grad / len(current_clients) for name, grad in g_c.items()}
            
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.grad = aggregate_g[n]
            
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss_total /= num_clients
        test_exact_matches_total /= num_clients
        test_f1_score_total /= num_clients
        
        tracker_all_personalized_exact_matches[r] = tracker_round_personalized_exact_matches
        tracker_all_personalized_f1_score[r] = tracker_round_personalized_f1_score
        tracker_all_losses[r] = tracker_round_losses
        tracker_loss[r] = train_loss_total
        tracker_personalized_exact_matches[r] = test_exact_matches_total
        tracker_personalized_f1_score[r] = test_f1_score_total

        print(Fore.BLUE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} loss aggregated from client results: {train_loss_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} personalized exact matches aggregated from client results: {test_exact_matches_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} personalized f1 score aggregated from client results: {test_f1_score_total}" + Fore.WHITE)

        # generalized test
        before_test_exact_matches_total = 0.0
        before_test_f1_score_total = 0.0
        after_test_exact_matches_total = 0.0
        after_test_f1_score_total = 0.0
        for c in range(num_clients):
            cid = random.choice(clients).strip()

            # test
            train_dataloader = train_dataloader_dict[cid]
            test_dataloader = test_dataloader_dict[cid]
            test_dataset = test_dataset_dict[cid]
            test_examples = test_examples_dict[cid]

            (before_loss, before_exact_matches, before_f1_score, 
             after_loss, after_exact_matches, after_f1_score, 
             after_total) = question_answerer_generalized_test(net, train_dataloader, test_dataloader, test_dataset, test_examples)
            before_test_exact_matches_total += before_exact_matches
            before_test_f1_score_total += before_f1_score
            after_test_exact_matches_total += after_exact_matches
            after_test_f1_score_total += after_f1_score

            tracker_round_generalized_exact_matches_before.append(before_exact_matches)
            tracker_round_generalized_f1_score_before.append(before_f1_score)
            tracker_round_generalized_exact_matches_after.append(after_exact_matches)
            tracker_round_generalized_f1_score_after.append(after_f1_score)

        tracker_all_generalized_exact_matches_before[r] = tracker_round_generalized_exact_matches_before
        tracker_all_generalized_f1_score_before[r] = tracker_round_generalized_f1_score_before
        tracker_all_generalized_exact_matches_after[r] = tracker_round_generalized_exact_matches_after
        tracker_all_generalized_f1_score_after[r] = tracker_round_generalized_f1_score_after

        before_test_exact_matches_total /= num_clients
        before_test_f1_score_total /= num_clients
        after_test_exact_matches_total /= num_clients
        after_test_f1_score_total /= num_clients
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized exact matches (before training) aggregated from client results: {before_test_exact_matches_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized f1 score (before training) aggregated from client results: {before_test_f1_score_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized exact matches (after training) aggregated from client results: {after_test_exact_matches_total}\n" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized f1 score (after training) aggregated from client results: {after_test_f1_score_total}\n" + Fore.WHITE)

        tracker_generalized_exact_matches_before[r] = before_test_exact_matches_total
        tracker_generalized_f1_score_before[r] = before_test_f1_score_total
        tracker_generalized_exact_matches_after[r] = after_test_exact_matches_total
        tracker_generalized_f1_score_after[r] = after_test_f1_score_total

        if r % checkpoint_interval == 0:
            with open(checkpoint_dir + '/all_personalized_exact_matches.json', 'w+') as fp:
                json.dump(tracker_all_personalized_exact_matches, fp)
            with open(checkpoint_dir + '/all_personalized_f1_score.json', 'w+') as fp:
                json.dump(tracker_all_personalized_f1_score, fp)
            with open(checkpoint_dir + '/all_generalized_exact_matches_before_personalization.json', 'w+') as fp:
                json.dump(tracker_all_generalized_exact_matches_before, fp)
            with open(checkpoint_dir + '/all_generalized_f1_score_before_personalization.json', 'w+') as fp:
                json.dump(tracker_all_generalized_f1_score_before, fp)
            with open(checkpoint_dir + '/all_generalized_exact_matches_after_personalization.json', 'w+') as fp:
                json.dump(tracker_all_generalized_exact_matches_after, fp)
            with open(checkpoint_dir + '/all_generalized_f1_score_after_personalization.json', 'w+') as fp:
                json.dump(tracker_all_generalized_f1_score_after, fp)
            with open(checkpoint_dir + '/all_train_losses.json', 'w+') as fp:
                json.dump(tracker_all_losses, fp)
            
            with open(checkpoint_dir + '/personalized_exact_matches.json', 'w+') as fp:
                json.dump(tracker_personalized_exact_matches, fp)
            with open(checkpoint_dir + '/personalized_f1_score.json', 'w+') as fp:
                json.dump(tracker_personalized_f1_score, fp)
            with open(checkpoint_dir + '/generalized_exact_matches_before_personalization.json', 'w+') as fp:
                json.dump(tracker_generalized_exact_matches_before, fp)
            with open(checkpoint_dir + '/generalized_f1_score_before_personalization.json', 'w+') as fp:
                json.dump(tracker_generalized_f1_score_before, fp)
            with open(checkpoint_dir + '/generalized_exact_matches_after_personalization.json', 'w+') as fp:
                json.dump(tracker_generalized_exact_matches_after, fp)
            with open(checkpoint_dir + '/generalized_f1_score_after_personalization.json', 'w+') as fp:
                json.dump(tracker_generalized_f1_score_after, fp)
            with open(checkpoint_dir + '/train_losses.json', 'w+') as fp:
                json.dump(tracker_loss, fp)
                
            plt.figure(0)
            plt.plot(list(map(int, list(tracker_loss.keys()))), tracker_loss.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Training Loss')
            plt.title('Training Loss for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_train_losses.png')
            
            plt.figure(1)
            plt.plot(list(map(int, list(tracker_personalized_exact_matches.keys()))), tracker_personalized_exact_matches.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized Exact Matches')
            plt.title('Personalized Exact Matches for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_personalized_exact_matches.png')
            
            plt.figure(2)
            plt.plot(list(map(int, list(tracker_personalized_f1_score.keys()))), tracker_personalized_f1_score.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized F1 Score')
            plt.title('Personalized F1 Score for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_personalized_f1_score.png')
            
            plt.figure(3)
            plt.plot(list(map(int, list(tracker_generalized_exact_matches_before.keys()))), tracker_generalized_exact_matches_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Exact Matches (Before)')
            plt.title('Generalized Exact Matches (Before) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_exact_matches_before_personalization.png')
            
            plt.figure(4)
            plt.plot(list(map(int, list(tracker_generalized_f1_score_before.keys()))), tracker_generalized_f1_score_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized F1 Score (Before)')
            plt.title('Generalized F1 Score (Before) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_f1_score_before_personalization.png')
            
            plt.figure(5)
            plt.plot(list(map(int, list(tracker_generalized_exact_matches_after.keys()))), tracker_generalized_exact_matches_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Exact Matches (After)')
            plt.title('Generalized Exact Matches (After) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_exact_matches_after_personalization.png')
            
            plt.figure(6)
            plt.plot(list(map(int, list(tracker_generalized_f1_score_after.keys()))), tracker_generalized_f1_score_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized F1 Score (After)')
            plt.title('Generalized F1 Score (After) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_f1_score_after_personalization.png')
            
            torch.save({
                'round': r,
                'net_state_dict': net.state_dict(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'random_random_state': random.getstate(),
                }, temp_dir + 'model.pth')


def text_classifier_train(net, batch):
    net.train()
    
    if 'distilbert' in model_name \
        or 'roberta' in model_name \
        or 'llama' in model_name:
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
    
    outputs = net(**batch_dict)
    loss = outputs.loss
    loss.backward()
        
    accumulated_loss = loss.item()
    sample_count = batch[0].shape[0]
    
    return accumulated_loss / sample_count, sample_count, {n: p.grad for n, p in net.named_parameters() if p.requires_grad}

def text_classifier_test(net, test_data):
    metric = load_metric("accuracy")
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    for batch in test_data:
        if 'distilbert' in model_name \
        or 'roberta' in model_name \
        or 'llama' in model_name:
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

def text_classifier_trainer():
    logging.info(Fore.GREEN +'Training begins.'+ Fore.WHITE + '\n')
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
    
    if dataset in 'multirc':
        gptq_config = GPTQConfig(bits=4, use_exllama=False,)
        net = LlamaForSequenceClassification.from_pretrained(
            "./models/models--4bit-autogptq-quantized-llama2-v2", 
            local_files_only=True,
            quantization_config=gptq_config,
            # device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.float32,
        ).to(DEVICE)
            
        net.resize_token_embeddings(32001)
        net.config.pad_token = '<pad>'
        net.config.pad_token_id = 32000
    else:
        net = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = num_labels, 
            cache_dir = './models', 
        ).to(DEVICE)
    
    if peft_method == 'lora':
        net = get_peft_model(net, lora_config)

    net.train()
    if server_opt == 'sgd':
        optimizer = SGD(net.parameters(), lr=learning_rate)
    elif server_opt == 'adamw':
        optimizer = AdamW(net.parameters(), lr=learning_rate)
    elif server_opt == 'adam':
        optimizer = Adam(net.parameters(), lr=learning_rate)
    
    tracker_personalized_accuracy = {}
    tracker_generalized_accuracy_before = {}
    tracker_generalized_accuracy_after = {}
    tracker_loss = {}
    
    tracker_all_personalized_accuracies = {}
    tracker_all_generalized_accuracies_before = {}
    tracker_all_generalized_accuracies_after = {}
    tracker_all_losses = {}

    for r in range(rounds, total_rounds + 1):
        current_clients = [str(random.choice(clients)) for _ in range(num_clients)]

        # threads = []
        train_loss_total = 0.0
        test_accuracy_total = 0.0

        tracker_round_personalized_accuracies = []
        tracker_round_generalized_accuracies_before = []
        tracker_round_generalized_accuracies_after = []
        tracker_round_losses = []

        logger.info(f'Batch count: {batch_count}')
        for current_batch_index in range(batch_count):
            g_c = None
            sample_count_total = 0
            
            for cid in current_clients:
                # train
                train_data = iter(train_dataloader_dict[cid])
                for batch_index, batch in enumerate(train_data):
                    if batch_index == current_batch_index:
                        break
            
                train_loss, sample_count, grads = text_classifier_train(net, batch)
                sample_count_total += sample_count
                
                # for _, w in weights.items():
                #     print(w[0][0], '*')
                #     break
                # gradients_per_client.append(grads)
                
                if g_c == None:
                    g_c = {name: grad.data for name, grad in grads.items()}
                else:
                    for name, grad in grads.items():
                        g_c[name] += (grad.data)

                # personalized test
                if current_batch_index == batch_count - 1:
                    test_data = test_dataloader_dict[cid]  
                    valid_loss, accuracy, total_samples = text_classifier_test(net, test_data)
                    
                    train_loss_total += train_loss
                    # print(sample_count, sample_count_total)
                    # print(train_loss, train_loss_total)
                    test_accuracy_total += accuracy
                    
                    tracker_round_personalized_accuracies.append(accuracy)
                    tracker_round_losses.append(train_loss)

            # aggregate  
            aggregate_g = {name: grad / len(current_clients) for name, grad in g_c.items()}
            
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.grad = aggregate_g[n]
            
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss_total /= num_clients
        test_accuracy_total /= num_clients
        
        tracker_all_personalized_accuracies[r] = tracker_round_personalized_accuracies
        tracker_all_losses[r] = tracker_round_losses
        tracker_loss[r] = train_loss_total
        tracker_personalized_accuracy[r] = test_accuracy_total

        print(Fore.BLUE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} loss aggregated from client results: {train_loss_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} personalized accuracy aggregated from client results: {test_accuracy_total}" + Fore.WHITE)

        # generalized test
        before_test_accuracy_total = 0.0
        after_test_accuracy_total = 0.0
        for c in range(num_clients):
            cid = random.choice(clients).strip()

            # test
            train_data = train_dataloader_dict[cid]
            test_data = test_dataloader_dict[cid]

            (before_loss, before_accuracy, 
             after_loss, after_accuracy, 
             after_total) = text_classifier_generalized_test(net, train_data, test_data)
            before_test_accuracy_total += before_accuracy
            after_test_accuracy_total += after_accuracy

            tracker_round_generalized_accuracies_before.append(before_accuracy)
            tracker_round_generalized_accuracies_after.append(after_accuracy)

        tracker_all_generalized_accuracies_before[r] = tracker_round_generalized_accuracies_before
        tracker_all_generalized_accuracies_after[r] = tracker_round_generalized_accuracies_after

        before_test_accuracy_total /= num_clients
        after_test_accuracy_total /= num_clients
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized accuracy (before training) aggregated from client results: {before_test_accuracy_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{method + ' ' + dataset + ' ' + experiment_name}: Round {r} generalized accuracy (after training) aggregated from client results: {after_test_accuracy_total}\n" + Fore.WHITE)

        tracker_generalized_accuracy_before[r] = before_test_accuracy_total
        tracker_generalized_accuracy_after[r] = after_test_accuracy_total

        if r % checkpoint_interval == 0:
            with open(checkpoint_dir + '/all_personalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_all_personalized_accuracies, fp)
            with open(checkpoint_dir + '/all_generalized_accuracies_before_personalization.json', 'w+') as fp:
                json.dump(tracker_all_generalized_accuracies_before, fp)
            with open(checkpoint_dir + '/all_generalized_accuracies_after_personalization.json', 'w+') as fp:
                json.dump(tracker_all_generalized_accuracies_after, fp)
            with open(checkpoint_dir + '/all_train_losses.json', 'w+') as fp:
                json.dump(tracker_all_losses, fp)
            
            with open(checkpoint_dir + '/personalized_accuracies.json', 'w+') as fp:
                json.dump(tracker_personalized_accuracy, fp)
            with open(checkpoint_dir + '/generalized_accuracies_before_personalization.json', 'w+') as fp:
                json.dump(tracker_generalized_accuracy_before, fp)
            with open(checkpoint_dir + '/generalized_accuracies_after_personalization.json', 'w+') as fp:
                json.dump(tracker_generalized_accuracy_after, fp)
            with open(checkpoint_dir + '/train_losses.json', 'w+') as fp:
                json.dump(tracker_loss, fp)
                
            plt.figure(0)
            plt.plot(list(map(int, list(tracker_loss.keys()))), tracker_loss.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Training Loss')
            plt.title('Training Loss for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_train_losses.png')
            
            plt.figure(1)
            plt.plot(list(map(int, list(tracker_personalized_accuracy.keys()))), tracker_personalized_accuracy.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized Accuracy')
            plt.title('Personalized Accuracy for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_personalized_accuracies.png')
            
            plt.figure(2)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_before.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (Before)')
            plt.title('Generalized Accuracy (Before) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracies_before_personalization.png')
            
            plt.figure(3)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_after.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (After)')
            plt.title('Generalized Accuracy (After) for ' + method + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracies_after_personalization.png')
            
            torch.save({
                'round': r,
                'net_state_dict': net.state_dict(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'random_random_state': random.getstate(),
                }, temp_dir + 'model.pth')


def load_parameters_from_disk():# -> fl.common.Parameters:
    model_file_name = temp_dir + 'model.pth'

    if not os.path.exists(model_file_name):
        return None, 1
    
    print("Loading: ", model_file_name)

    checkpoint = torch.load(model_file_name)
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    random.setstate(checkpoint['random_random_state'])

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
        gptq_config = GPTQConfig(bits=4, use_exllama=False,)
        net = AutoModelForQuestionAnswering.from_pretrained(
            "./models/models--4bit-autogptq-quantized-13B-v1", 
            local_files_only=True,
            quantization_config=gptq_config,
            # device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        
        if peft_method == 'lora':
            lora_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS, 
                r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=0.1,
            )
            net = get_peft_model(net, lora_config)
    
    elif dataset == 'multirc':
        gptq_config = GPTQConfig(bits=4, use_exllama=False,)
        net = LlamaForSequenceClassification.from_pretrained(
            "./models/models--4bit-autogptq-quantized-llama2-v2", 
            local_files_only=True,
            quantization_config=gptq_config,
            # device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.float32,
            )
        
        net.resize_token_embeddings(32001)
        net.config.pad_token = '<pad>'
        net.config.pad_token_id = 32000
            
        if peft_method == 'lora':
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, 
                r=lora_r, 
                lora_alpha=lora_alpha, 
                lora_dropout=0.1
            )
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
        
    elif dataset == 'multirc' :
        net.config.label2id = {'False': 0, 'True': 1}
        net.config.id2label = {0: 'False', 1: 'True'}
        
    trainable_keys = []
    net.load_state_dict(checkpoint['net_state_dict'], strict = False)
    for key, val in net.named_parameters():
        if val.requires_grad and peft_method == 'lora':
            trainable_keys.append(key)
        elif 'bias' in key and peft_method == 'bitfit':
            trainable_keys.append(key)

    print(Fore.YELLOW + f"Loading model weights from round #{checkpoint['round']}" + Fore.WHITE)

    return [val.cpu().numpy() for key, val in net.state_dict().items() if key in trainable_keys], rounds - checkpoint['round']

if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'Enter the dataset you want to train your algorithm on.')
    parser.add_argument('--model_name',)
    parser.add_argument('--random_seed',)
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
    parser.add_argument('--server_opt',)
    parser.add_argument('--experiment_name',)
    
    parser.add_argument('--checkpoint_dir',)
    parser.add_argument('--dataset_dir',)
    parser.add_argument('--temp_dir',)
    
    args = parser.parse_args()
    hyperparameters = {}
    
    method = 'fedsgd'

    dataset = args.dataset
    num_clients = int(args.num_clients)
    hyperparameters['num_clients'] = num_clients
    total_clients = int(args.total_clients)
    hyperparameters['total_clients'] = total_clients
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
    server_opt = args.server_opt
    hyperparameters['server_opt'] = server_opt
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
         test_dataset_dict, test_examples_dict) = squad_dataloader.get_federated_qa_datasets(
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
        
    elif dataset == 'multirc':
        train_dataloader_dict, test_dataloader_dict = multirc_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            tokenizer_name=model_name,
            )
        
        num_labels = 2
     
    try:
        hyperparameters['num_labels'] = num_labels
    except:
        pass
    batch_count = len(train_dataloader_dict['1'])
        
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    clients = [str(c) for c in range(total_clients)]

    with open(checkpoint_dir + 'hyperparameter_snapshot.json', 'w+') as f:
        json.dump(hyperparameters, f)
    
    print('Experiment: ', experiment_name)
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  

    initial_parameters, rounds = load_parameters_from_disk()

    print(Fore.RED + "Availble Device: " + str(DEVICE) + ", Count: " + \
        str(AVAILABLE_GPUS) + ". CPUs, Count: " + str(AVAILABLE_CPUS) + '.' + Fore.WHITE, flush=True)

    simulation_start_time = datetime.now() 
    print('Starting at:', simulation_start_time)
    print('')

    if 'squad' == dataset:
        question_answerer_trainer()
    else:
        text_classifier_trainer()
    
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
        
        