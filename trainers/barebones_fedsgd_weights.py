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
import argparse
import threading
from colorama import Fore

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
import numpy as np
import matplotlib.pyplot as plt

import dataloaders.agnews.agnews_sequence_classification_dataloader as agnews_sequence_classification_dataloader
from configs.agnews_hyperparameters import *

from accelerate import Accelerator
from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def agnews_train(net, batch):
    optimizer = AdamW(net.parameters(), lr=learning_rate)
    net.train()
    
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
    
    optimizer.step()
    optimizer.zero_grad()
    
    return accumulated_loss / sample_count, sample_count, {n: p for n, p in net.named_parameters() if p.requires_grad}

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
    if DATASET == 'agnews_bert':
        if peft_method == 'lora':
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1
            )
        
        net = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels = 4, 
                cache_dir = './models', 
            ).to(DEVICE)
        # params = sum([np.prod(p.size()) for p in net.parameters()])
        # print('Model size: ', params)
        
        if peft_method == 'lora':
            net = get_peft_model(net, lora_config)

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

        for current_batch_index in range(batch_count):
            current_net = copy.deepcopy(net)
            w_c = None
            sample_count_total = 0
            
            for cid in current_clients:
                # train
                train_data = iter(agnews_train_dataloader_dict[cid])
                for batch_index, batch in enumerate(train_data):
                    if batch_index == current_batch_index:
                        break
            
                train_loss, sample_count, weights = agnews_train(net, batch)
                sample_count_total += sample_count
                
                if w_c == None:
                    w_c = {name: param.data * sample_count for name, param in weights.items()}
                else:
                    # w_c = [(layer.data * sample_count) + w_c_layer for layer, w_c_layer in zip(weights, w_c)]
                    for name, param in weights.items():
                        w_c[name] += (param.data * sample_count)

                # personalized test
                if current_batch_index == batch_count - 1:
                    test_data = agnews_test_dataloader_dict[cid]  
                    valid_loss, accuracy, total_samples = agnews_test(net, test_data)
                    
                    train_loss_total += train_loss
                    # print(sample_count, sample_count_total)
                    # print(train_loss, train_loss_total)
                    test_accuracy_total += accuracy
                    
                    tracker_round_personalized_accuracies.append(accuracy)
                    tracker_round_losses.append(train_loss)

                net = copy.deepcopy(current_net)
            
            # for _, w in w_c.items():
            #     print(w[0][0])
            #     print('sample_count_total', sample_count_total)
            #     break
            # aggregate  
            w_g = {name: param.data / sample_count_total for name, param in w_c.items()}

            # for _, w in w_g.items():
            #     print(w[0][0])
            #     print('---')
            #     break
            
            net.load_state_dict(w_g, strict = False)
            # for p, w in zip(net.parameters(), w_g):
            #     p.data = w

        train_loss_total /= num_clients
        test_accuracy_total /= num_clients
        
        tracker_all_personalized_accuracies[r] = tracker_round_personalized_accuracies
        tracker_all_losses[r] = tracker_round_losses
        tracker_loss[r] = train_loss_total
        tracker_personalized_accuracy[r] = test_accuracy_total

        print(Fore.BLUE + f"{METHOD}: Round {r} loss aggregated from client results: {train_loss_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{METHOD}: Round {r} personalized accuracy aggregated from client results: {test_accuracy_total}" + Fore.WHITE)

        # generalized test
        before_test_accuracy_total = 0.0
        after_test_accuracy_total = 0.0
        for c in range(num_clients):
            cid = random.choice(clients).strip()

            # test
            train_data = agnews_train_dataloader_dict[cid]
            test_data = agnews_test_dataloader_dict[cid]

            before_loss, before_accuracy, after_loss, after_accuracy, after_total = agnews_generalized_test(net, train_data, test_data)
            before_test_accuracy_total += before_accuracy
            after_test_accuracy_total += after_accuracy

            tracker_round_generalized_accuracies_before.append(before_accuracy)
            tracker_round_generalized_accuracies_after.append(after_accuracy)

        tracker_all_generalized_accuracies_before[r] = tracker_round_generalized_accuracies_before
        tracker_all_generalized_accuracies_after[r] = tracker_round_generalized_accuracies_after

        before_test_accuracy_total /= num_clients
        after_test_accuracy_total /= num_clients
        print(Fore.WHITE + f"{METHOD}: Round {r} generalized accuracy (before training) aggregated from client results: {before_test_accuracy_total}" + Fore.WHITE)
        print(Fore.WHITE + f"{METHOD}: Round {r} generalized accuracy (after training) aggregated from client results: {after_test_accuracy_total}\n" + Fore.WHITE)

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
            plt.title('Training Loss for ' + METHOD + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_losses.png')
            
            plt.figure(1)
            plt.plot(list(map(int, list(tracker_personalized_accuracy.keys()))), tracker_personalized_accuracy.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized Accuracy')
            plt.title('Personalized Accuracy for ' + METHOD + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_personalized_accuracy.png')
            
            plt.figure(2)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_before.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (Before)')
            plt.title('Generalized Accuracy (Before) for ' + METHOD + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracy_before.png')
            
            plt.figure(3)
            plt.plot(list(map(int, list(tracker_generalized_accuracy_after.keys()))), tracker_generalized_accuracy_before.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Generalized Accuracy (After)')
            plt.title('Generalized Accuracy (After) for ' + METHOD + ' Baseline')
            plt.savefig(checkpoint_dir + '/plot_generalized_accuracy_after.png')

            torch.save({
                'round': r,
                'net_state_dict': net.state_dict(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'random_random_state': random.getstate(),
                }, temp_dir + '/model.pth')

def load_parameters_from_disk():# -> fl.common.Parameters:
    model_file_name = temp_dir + 'model.pth'

    if not os.path.exists(model_file_name):
        return None, 1
    
    print("Loading: ", model_file_name)

    checkpoint = torch.load(model_file_name)
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    random.setstate(checkpoint['random_random_state'])

    if DATASET == 'agnews_bert':
        if peft_method == 'lora':
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1
            )
        
        net = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=4,
                cache_dir = './models',
            )
        
        if peft_method == 'lora':
            net = get_peft_model(net, lora_config)

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
    args = parser.parse_args()

    DATASET = args.dataset
    METHOD = 'FedSgd'
    
    if DATASET == 'agnews_bert':
        hyperparameters = eval(DATASET + '_' + METHOD.lower()) 
        
        num_clients = hyperparameters['num_clients']
        total_clients = hyperparameters['total_clients']
        cpu_simultaneous_clients = hyperparameters['cpu_simultaneous_clients']
        gpu_simultaneous_clients = hyperparameters['gpu_simultaneous_clients']
        rounds = hyperparameters['rounds']
        total_rounds = hyperparameters['rounds']
        epochs = hyperparameters['epochs']
        dirichlet_distribution = hyperparameters['dirichlet_distribution']
        model_name = hyperparameters['model_name']
        peft_method = hyperparameters['peft_method']
        lora_r = hyperparameters['lora_r']
        lora_alpha = hyperparameters['lora_alpha']
        experiment_name = peft_method + '_' + hyperparameters['experiment_name']
        checkpoint_interval = hyperparameters['checkpoint_interval']
        checkpoint_dir = hyperparameters['checkpoint_dir'] + experiment_name + '/'
        dataset_dir = hyperparameters['dataset_dir']
        temp_dir = hyperparameters['temp_dir'] + experiment_name + '/'
        batch_size = hyperparameters['batch_size']
        learning_rate = hyperparameters['lr']
        max_seq_len = hyperparameters['max_seq_len']
        random_seed = hyperparameters['random_seed']

        # TODO: Make sure that the fixed seeds reproduce the same clients, 
        # and other random states related to the model training 
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        agnews_train_dataloader_dict, agnews_test_dataloader_dict = agnews_sequence_classification_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            )
        
        batch_count = len(agnews_train_dataloader_dict['1'])
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        clients = [str(c) for c in range(total_clients)]

        num_cpus_needed = int(AVAILABLE_CPUS / cpu_simultaneous_clients) # int(AVAILABLE_CPUS / num_clients)
        num_gpus_needed = float(AVAILABLE_GPUS / gpu_simultaneous_clients) # Normal runs: divide by 6
        
        hyperparameters['num_cpus_needed'] = num_cpus_needed
        hyperparameters['num_gpus_needed'] = num_gpus_needed

        with open(checkpoint_dir + 'hyperparameter_snapshot.json', 'w+') as f:
            json.dump(hyperparameters, f)

    print('Experiment: ', experiment_name)
    
    if not os.path.exists(temp_dir + 'previous_models'):
        os.makedirs(temp_dir + 'previous_models')  

    initial_parameters, rounds = load_parameters_from_disk()
    
    print(Fore.RED + "Availble Device: " + str(DEVICE) + ", Count: " + \
        str(AVAILABLE_GPUS) + ". CPUs, Count: " + str(AVAILABLE_CPUS) + '.' + Fore.WHITE, flush=True)

    agnews_trainer()

    for f in os.listdir(temp_dir + 'previous_models'):
        os.remove(os.path.join(temp_dir + 'previous_models', f))
        