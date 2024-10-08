import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import json
import wandb
import torch
import psutil
import random
import logging
import subprocess as sp

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore
from typing import Optional, List, OrderedDict, Tuple, Dict
from flwr.common import Parameters, Scalar, MetricsAggregationFn, FitIns, EvaluateIns, FitRes, EvaluateRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from transformers import (
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    AutoModelForQuestionAnswering,
    # GPTQConfig,
)
# from transformers import logging as transformers_logging

from models.mezo_models import MODEL_TYPES
from models.mezo_modeling_roberta import RobertaConfig

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def get_cpu_memory():
    for process in [psutil.Process(pid) for pid in psutil.pids()]:
        try:
            process_name = process.name()
            if 'ray' == process_name[:3]:
                process_mem = process.memory_percent()
                process_cpu = process.cpu_percent(interval=0.05)
            else:
                continue
        except psutil.NoSuchProcess as e:
            pass
            # print(e.pid, "killed before analysis")
        else:
            print("Name:", process_name, "CPU%:", process_cpu, "MEM%:", process_mem)

def get_nvidia_smi_process_ids():
    try:
        # Run the nvidia-smi command and capture its output
        result = sp.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv'],
                        stdout=sp.PIPE, stderr=sp.PIPE, text=True, check=True)

        # Extract and return the process IDs as a list of integers
        process_ids = [pid for pid in result.stdout[3:].strip().split('\n')]
        return process_ids

    except sp.CalledProcessError as e:
        # Handle errors, such as nvidia-smi not found or other issues
        print(f"Error executing nvidia-smi: {e}")
        return None

class FedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
        hyperparameters,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        dataset: str = 'agnews',
        client_algorithm: str = 'FedMeZO',
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.dataset_name = dataset
        self.client_algorithm = client_algorithm

        self.hyperparameters = hyperparameters
        self.model_name = hyperparameters['model_name']
        self.experiment_name = hyperparameters['experiment_name']
        self.num_labels = hyperparameters['num_labels']
        self.checkpoint_dir = hyperparameters['checkpoint_dir']
        self.temp_dir = hyperparameters['temp_dir']
        self.checkpoint_interval = hyperparameters['checkpoint_interval']
        self.random_seed = hyperparameters['random_seed']
        
        wandb.init(
            project=dataset + '_federated', 
            name=client_algorithm + '_' + self.experiment_name,
            config=self.hyperparameters,
            dir=self.temp_dir
        )

        # random.seed(self.random_seed)
        # np.random.seed(self.random_seed)
        # torch.manual_seed(self.random_seed)
        # torch.cuda.manual_seed(self.random_seed)

        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.model_file_name = self.temp_dir + 'model.pth'
        self.loss_file_name = self.checkpoint_dir + 'train_losses.json'
        self.before_accu_file_name = self.checkpoint_dir + 'generalized_accuracies_before_personalization.json'
        self.after_accu_file_name = self.checkpoint_dir + 'generalized_accuracies_after_personalization.json'
        self.after_loss_file_name = self.checkpoint_dir + 'generalized_losses_after_personalization.json'
        self.average_time_per_epoch_file_name = self.checkpoint_dir + 'average_time_per_epoch.json'
        self.average_time_per_iteration_per_epoch_file_name = self.checkpoint_dir + 'average_time_per_iteration_per_epoch.json'
        
        if 'squad' in self.dataset_name:
            self.before_exact_match_file_name = self.checkpoint_dir + 'generalized_exact_matches_before_personalization.json'
            self.before_f1_file_name = self.checkpoint_dir + 'generalized_f1_before_personalization.json'
            self.after_exact_match_file_name = self.checkpoint_dir + 'generalized_exact_matches_after_personalization.json'
            self.after_f1_file_name = self.checkpoint_dir + 'generalized_f1_after_personalization.json'
            
        self.all_loss_file_name = self.checkpoint_dir + 'all_train_losses.json'
        self.all_before_accu_file_name = self.checkpoint_dir + 'all_generalized_accuracies_before_personalization.json'
        self.all_after_accu_file_name = self.checkpoint_dir + 'all_generalized_accuracies_after_personalization.json'
        self.all_after_loss_file_name = self.checkpoint_dir + 'all_generalized_losses_after_personalization.json'
        self.all_average_time_per_epoch_file_name = self.checkpoint_dir + 'all_average_time_per_epoch.json'
        self.all_average_time_per_iteration_per_epoch_file_name = self.checkpoint_dir + 'all_average_time_per_iteration_per_epoch.json'
        
        if 'squad' in self.dataset_name:
            self.all_before_exact_match_file_name = self.checkpoint_dir + 'all_generalized_exact_matches_before_personalization.json'
            self.all_before_f1_file_name = self.checkpoint_dir + 'all_generalized_f1_before_personalization.json'
            self.all_after_exact_match_file_name = self.checkpoint_dir + 'all_generalized_exact_matches_after_personalization.json'
            self.all_after_f1_file_name = self.checkpoint_dir + 'all_generalized_f1_after_personalization.json'
        
        self.plot_loss_file_name = self.checkpoint_dir + 'plot_train_losses.png'
        self.plot_before_accu_file_name = self.checkpoint_dir + 'plot_generalized_accuracies_before_personalization.png'
        self.plot_after_accu_file_name = self.checkpoint_dir + 'plot_generalized_accuracies_after_personalization.png'
        self.plot_after_loss_file_name = self.checkpoint_dir + 'plot_generalized_losses_after_personalization.png'
        
        if 'squad' in self.dataset_name:
            self.plot_before_exact_match_file_name = self.checkpoint_dir + 'plot_generalized_exact_matches_before_personalization.png'
            self.plot_before_f1_file_name = self.checkpoint_dir + 'plot_generalized_f1_before_personalization.png'
            self.plot_after_exact_match_file_name = self.checkpoint_dir + 'plot_generalized_exact_matches_after_personalization.png'
            self.plot_after_f1_file_name = self.checkpoint_dir + 'plot_generalized_f1_after_personalization.png'
        
        self.personalized_accu_file_name = self.checkpoint_dir + 'personalized_accuracies.json'
        self.personalized_all_accu_file_name = self.checkpoint_dir + 'all_personalized_accuracies.json'
        self.personalized_plot_accu_file_name = self.checkpoint_dir + 'plot_personalized_accuracies.png'
        
        if 'squad' in self.dataset_name:
            self.personalized_exact_match_file_name = self.checkpoint_dir + 'personalized_exact_matches.json'
            self.personalized_all_exact_match_file_name = self.checkpoint_dir + 'all_personalized_exact_matches.json'
            self.personalized_plot_exact_match_file_name = self.checkpoint_dir + 'plot_personalized_exact_matches.png'
            
            self.personalized_f1_file_name = self.checkpoint_dir + 'personalized_f1.json'
            self.personalized_all_f1_file_name = self.checkpoint_dir + 'all_personalized_f1.json'
            self.personalized_plot_f1_file_name = self.checkpoint_dir + 'plot_personalized_f1.png'
        
        self.sample_count_file_name = self.checkpoint_dir + 'sample_count.json'
        
        self.losses_dict = {}
        self.before_accuracies_dict = {}
        self.after_accuracies_dict = {}
        self.after_losses_dict = {}
        
        if 'squad' in self.dataset_name:
            self.before_exact_matches_dict = {}
            self.before_f1_dict = {}
            self.after_exact_matches_dict = {}
            self.after_f1_dict = {}
            
        self.average_time_per_epoch_dict = {}
        self.average_time_per_iteration_per_epoch_dict = {}

        self.all_losses_dict = {}
        self.all_before_accuracies_dict = {}
        self.all_after_accuracies_dict = {}
        self.all_after_losses_dict = {}
        
        if 'squad' in self.dataset_name:
            self.all_before_exact_matches_dict = {}
            self.all_before_f1_dict = {}
            self.all_after_exact_matches_dict = {}
            self.all_after_f1_dict = {}
            
        self.all_average_time_per_epoch_dict = {}
        self.all_average_time_per_iteration_per_epoch_dict = {}

        self.personalized_accuracies_dict = {}
        self.all_personalized_accuracies_dict = {}
        
        if 'squad' in self.dataset_name:
            self.personalized_exact_matches_dict = {}
            self.personalized_f1_dict = {}
            self.all_personalized_exact_matches_dict = {}
            self.all_personalized_f1_dict = {}

        self.sample_count_dict = {}
        self.round_offset = 0
        
        if os.path.exists(self.loss_file_name):
            with open(self.loss_file_name, 'r') as f:
                self.losses_dict = json.load(f)

            with open(self.all_loss_file_name, 'r') as f:
                self.all_losses_dict = json.load(f)

            if 'squad' in self.dataset_name:
                with open(self.before_exact_match_file_name, 'r') as f:
                    self.before_exact_matches_dict = json.load(f)
                with open(self.before_f1_file_name, 'r') as f:
                    self.before_f1_dict = json.load(f)
                with open(self.after_exact_match_file_name, 'r') as f:
                    self.after_exact_matches_dict = json.load(f)
                with open(self.after_f1_file_name, 'r') as f:
                    self.after_f1_dict = json.load(f)
            else:
                with open(self.before_accu_file_name, 'r') as f:
                    self.before_accuracies_dict = json.load(f)
                
                with open(self.after_accu_file_name, 'r') as f:
                    self.after_accuracies_dict = json.load(f)
            
            with open(self.after_loss_file_name, 'r') as f:
                self.after_losses_dict = json.load(f)
            
            with open(self.average_time_per_epoch_file_name, 'r') as f:
                self.average_time_per_epoch_dict = json.load(f)

            with open(self.average_time_per_iteration_per_epoch_file_name, 'r') as f:
                self.average_time_per_iteration_per_epoch_dict = json.load(f)

            if 'squad' in self.dataset_name:
                with open(self.all_before_exact_match_file_name, 'r') as f:
                    self.all_before_exact_matches_dict = json.load(f)
                with open(self.all_before_f1_file_name, 'r') as f:
                    self.all_before_f1_dict = json.load(f)
                with open(self.all_after_exact_match_file_name, 'r') as f:
                    self.all_after_exact_matches_dict = json.load(f)
                with open(self.all_after_f1_file_name, 'r') as f:
                    self.all_after_f1_dict = json.load(f)
            else:
                with open(self.all_before_accu_file_name, 'r') as f:
                    self.all_before_accuracies_dict = json.load(f)
                with open(self.all_after_accu_file_name, 'r') as f:
                    self.all_after_accuracies_dict = json.load(f)

            with open(self.all_after_loss_file_name, 'r') as f:
                self.all_after_losses_dict = json.load(f)
            
            with open(self.all_average_time_per_epoch_file_name, 'r') as f:
                self.all_average_time_per_epoch_dict = json.load(f)

            with open(self.all_average_time_per_iteration_per_epoch_file_name, 'r') as f:
                self.all_average_time_per_iteration_per_epoch_dict = json.load(f)

            if 'squad' in self.dataset_name:
                with open(self.personalized_exact_match_file_name, 'r') as f:
                    self.personalized_exact_matches_dict = json.load(f)
                with open(self.personalized_f1_file_name, 'r') as f:
                    self.personalized_f1_dict = json.load(f)
            else:
                with open(self.personalized_accu_file_name, 'r') as f:
                    self.personalized_accuracies_dict = json.load(f)
                with open(self.personalized_all_accu_file_name, 'r') as f:
                    self.all_personalized_accuracies_dict = json.load(f)

            with open(self.sample_count_file_name, 'r') as f:
                self.sample_count_dict = json.load(f)
            
            self.round_offset = max([int(i) for i in self.losses_dict.keys()])
            
        if self.dataset_name in ['agnews', 'yahoo', 'sst2', 'mnli', 'snli', 'qnli', 'cola', 'yelp', 'qqp']:
            config_kwargs = {
                'apply_lora': self.hyperparameters['apply_lora'],
                'lora_alpha': self.hyperparameters['lora_alpha'],
                'lora_r': self.hyperparameters['lora_r'],
            }
            
            config = RobertaConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                finetuning_task=self.dataset_name,
                cache_dir=self.hyperparameters['cache_dir'],
                **config_kwargs
            )
            model_fn = MODEL_TYPES[config.model_type]
            
            self.net = model_fn.from_pretrained(
                self.model_name,
                from_tf=bool(".ckpt" in self.model_name),
                config=config,
                cache_dir=self.hyperparameters['cache_dir'],
            )
            
            if self.hyperparameters['apply_lora']:
                for name, param in self.net.named_parameters():
                    if (name.startswith('roberta') and "lora" not in name) or (name.startswith('opt') and "lora" not in name):
                        param.requires_grad_(False)
            
            # print('in server')
            # for n, p in self.net.named_parameters():
            #     if p.requires_grad:
            #         print(n, p.shape, flush=True)
            
        elif self.dataset_name in ['multirc']:
            gptq_config = GPTQConfig(bits=4, use_exllama=False,)
            self.net = LlamaForSequenceClassification.from_pretrained(
                "./models/models--4bit-autogptq-quantized-llama2", 
                local_files_only=True,
                quantization_config=gptq_config,
                # device_map="auto"
            )#.to(DEVICE)
            
            self.net.resize_token_embeddings(32001)
            self.net.config.pad_token = '<pad>'
            self.net.config.pad_token_id = 32000
            
            if self.peft_method == 'lora':
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, 
                    r=self.lora_r, lora_alpha=self.lora_alpha, 
                    lora_dropout=0.1
                )
                self.net = get_peft_model(self.net, lora_config)

        elif 'squad' in self.dataset_name:
            gptq_config = GPTQConfig(bits=4, use_exllama=False,)
            self.net = AutoModelForQuestionAnswering.from_pretrained(
                "./models/models--4bit-autogptq-quantized-opt6B-v1" if 'opt-6' in self.model_name else "./models/models--4bit-autogptq-quantized-opt13B-v1", 
                local_files_only=True,
                quantization_config=gptq_config,
                # device_map="auto",
                attn_implementation="eager",
            )
            if self.peft_method == 'lora':
                lora_config = LoraConfig(
                    task_type=TaskType.QUESTION_ANS, 
                    r=self.lora_r, lora_alpha=self.lora_alpha,
                    lora_dropout=0.1,
                )
                self.net = get_peft_model(self.net, lora_config)
        
        if dataset == 'sst2':
            self.net.config.label2id = {'negative': 0, 'positive': 1}
            self.net.config.id2label = {0: 'negative', 1: 'positive'}
        
        elif dataset == 'mnli':
            self.net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            self.net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        elif dataset == 'snli':
            self.net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
            self.net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        elif dataset == 'qnli':
            self.net.config.label2id = {'entailment': 0, 'not_entailment': 1}
            self.net.config.id2label = {0: 'entailment', 1: 'not_entailment'}
        
        elif dataset == 'cola':
            self.net.config.label2id = {'unacceptable': 0, 'acceptable': 1}
            self.net.config.id2label = {0: 'unacceptable', 1: 'acceptable'}
            
        self.trainable_keys = []
        if self.hyperparameters['apply_lora']:
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    self.trainable_keys.append(name)
        
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_eval, min_fit_clients=min_fit_clients, min_evaluate_clients=min_eval_clients, min_available_clients=min_available_clients, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of training."""
        # print('Free memory per GPU: ', get_gpu_memory())
        # print('Free memory per CPU: ')
        # get_cpu_memory()
        # print('PIDs: ', get_nvidia_smi_process_ids())
        client_proxy_and_fitins = super().configure_fit(server_round, parameters, client_manager)
        
        for idx, i in enumerate(client_proxy_and_fitins):
            i[1].config['rnd'] = server_round

        return client_proxy_and_fitins

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of evaluation."""
        # print('Free memory per GPU: ', get_gpu_memory())
        # print('Free memory per CPU: ')
        # get_cpu_memory()
        # print('PIDs: ', get_nvidia_smi_process_ids())
        
        client_proxy_and_fitins = super().configure_evaluate(server_round, parameters, client_manager)

        for idx, i in enumerate(client_proxy_and_fitins):
            i[1].config['rnd'] = server_round

        return client_proxy_and_fitins

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        
        examples = [r.num_examples for _, r in results]
        average_time_per_epoch = [r.metrics["average_time_per_epoch"] for _, r in results]
        average_time_per_iteration_per_epoch = [r.metrics["average_time_per_iteration_per_epoch"] for _, r in results]

        # Aggregate and print custom metric
        loss_aggregated = sum(losses) / sum(examples)
        print("", flush=True)
        print(Fore.BLUE + f"{self.client_algorithm + ' ' + self.dataset_name + ' ' + self.experiment_name}: Round {rnd + self.round_offset} loss aggregated from client results: {loss_aggregated}" + Fore.WHITE, flush=True)

        if 'squad' in self.dataset_name:
            personalized_exact_matches = [r.metrics["personalized_exact_match"] * r.num_examples for _, r in results]
            personalized_f1 = [r.metrics["personalized_f1"] * r.num_examples for _, r in results]
            personalized_exact_matches_aggregated = sum(personalized_exact_matches) / sum(examples)
            personalized_f1_aggregated = sum(personalized_f1) / sum(examples)
            
            wandb.log({
                "Train Exact Matches": personalized_exact_matches_aggregated,
                "Train F1 Score": personalized_f1_aggregated,
            }, commit=False)
        
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} personalized exact matches aggregated from client results: {personalized_exact_matches_aggregated}" + Fore.WHITE, flush=True)
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} personalized f1 aggregated from client results: {personalized_f1_aggregated}" + Fore.WHITE, flush=True)
        else:
            personalized_accuracies = [r.metrics["personalized_accuracy"] * r.num_examples for _, r in results]
            personalized_accuracy_aggregated = sum(personalized_accuracies) / sum(examples)
        
            wandb.log({
                "Train Accuracy": personalized_accuracy_aggregated,
            }, commit=False)
        
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} personalized accuracy aggregated from client results: {personalized_accuracy_aggregated}" + Fore.WHITE, flush=True)

        all_losses = [r.metrics["loss"] for _, r in results]
        
        if 'squad' in self.dataset_name:
            all_personalized_exact_matches = [r.metrics["personalized_exact_match"] for _, r in results]
            all_personalized_f1 = [r.metrics["personalized_f1"] for _, r in results]
            self.personalized_exact_matches_dict[int(rnd + self.round_offset)] = personalized_exact_matches_aggregated
            self.personalized_f1_dict[int(rnd + self.round_offset)] = personalized_f1_aggregated
            self.all_personalized_exact_matches_dict[int(rnd + self.round_offset)] = all_personalized_exact_matches
            self.all_personalized_f1_dict[int(rnd + self.round_offset)] = all_personalized_f1
        else:
            all_personalized_accuracies = [r.metrics["personalized_accuracy"] for _, r in results]
            self.personalized_accuracies_dict[int(rnd + self.round_offset)] = personalized_accuracy_aggregated
            self.all_personalized_accuracies_dict[int(rnd + self.round_offset)] = all_personalized_accuracies
        
        wandb.log({
            "Train Loss": loss_aggregated, 
            }, commit=False)

        self.losses_dict[int(rnd + self.round_offset)] = loss_aggregated
        self.average_time_per_epoch_dict[int(rnd + self.round_offset)] = sum(average_time_per_epoch)/len(average_time_per_epoch)
        self.average_time_per_iteration_per_epoch_dict[int(rnd + self.round_offset)] = sum(average_time_per_iteration_per_epoch)/len(average_time_per_iteration_per_epoch)
        
        self.all_losses_dict[int(rnd + self.round_offset)] = all_losses
        self.all_average_time_per_epoch_dict[int(rnd + self.round_offset)] = average_time_per_epoch
        self.all_average_time_per_iteration_per_epoch_dict[int(rnd + self.round_offset)] = average_time_per_iteration_per_epoch

        # Aggregate updated local model parameters to update the global model
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        for n, p in zip(self.trainable_keys, aggregated_weights):
            wandb.log(
                {n + "_parameters": wandb.Histogram(p)},
                commit=False
            )

        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.loss_file_name, 'w') as f:
                json.dump(self.losses_dict, f)

            with open(self.all_loss_file_name, 'w') as f:
                json.dump(self.all_losses_dict, f)

            if 'squad' in self.dataset_name:
                with open(self.personalized_exact_match_file_name, 'w') as f:
                    json.dump(self.personalized_exact_matches_dict, f)
                with open(self.personalized_all_exact_match_file_name, 'w') as f:
                    json.dump(self.all_personalized_exact_matches_dict, f)
                with open(self.personalized_f1_file_name, 'w') as f:
                    json.dump(self.personalized_f1_dict, f)
                with open(self.personalized_all_f1_file_name, 'w') as f:
                    json.dump(self.all_personalized_f1_dict, f)
            else:
                with open(self.personalized_accu_file_name, 'w') as f:
                    json.dump(self.personalized_accuracies_dict, f)
                with open(self.personalized_all_accu_file_name, 'w') as f:
                    json.dump(self.all_personalized_accuracies_dict, f)
                
            with open(self.average_time_per_epoch_file_name, 'w') as f:
                json.dump(self.average_time_per_epoch_dict, f)

            with open(self.average_time_per_iteration_per_epoch_file_name, 'w') as f:
                json.dump(self.average_time_per_iteration_per_epoch_dict, f)

            with open(self.all_average_time_per_epoch_file_name, 'w') as f:
                json.dump(self.all_average_time_per_epoch_dict, f)

            with open(self.all_average_time_per_iteration_per_epoch_file_name, 'w') as f:
                json.dump(self.all_average_time_per_iteration_per_epoch_dict, f)

            plt.figure(0)
            plt.plot(list(map(int, list(self.losses_dict.keys()))), self.losses_dict.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Training Loss')
            plt.title('Training Loss for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_loss_file_name)

            if 'squad' in self.dataset_name:
                plt.figure(1)
                plt.plot(list(map(int, list(self.personalized_exact_matches_dict.keys()))), self.personalized_exact_matches_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Personalized Exact Matches')
                plt.title('Validation Exact Matches for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.personalized_plot_exact_match_file_name)
                
                plt.figure(6)
                plt.plot(list(map(int, list(self.personalized_f1_dict.keys()))), self.personalized_f1_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Personalized F1')
                plt.title('Validation F1 for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.personalized_plot_f1_file_name)
            else:
                plt.figure(1)
                plt.plot(list(map(int, list(self.personalized_accuracies_dict.keys()))), self.personalized_accuracies_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Personalized Accuracy')
                plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.personalized_plot_accu_file_name)

            print(Fore.GREEN + f"{self.client_algorithm}: Saving aggregated weights at round {rnd + self.round_offset}" + Fore.WHITE)
            
            params_dict = zip(self.trainable_keys, aggregated_weights)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if k in self.trainable_keys})
            self.net.load_state_dict(state_dict, strict=False)
            torch.save(
                {
                    'round': rnd + self.round_offset,
                    'net_state_dict': self.net.state_dict(),
                    'numpy_random_state': np.random.get_state(),
                    'torch_random_state': torch.get_rng_state(),
                    'random_random_state': random.getstate(),
                }, self.model_file_name
            )
            
        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ):# -> Tuple[Optional[float], Dict[str, Scalar]]:

        after_losses = [r.metrics["after_loss"] * r.num_examples for _, r in results]
        all_after_losses = [r.metrics["after_loss"] for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        # Aggregate and print custom metric
        after_loss_aggregated = sum(after_losses) / sum(examples)

        self.after_losses_dict[int(rnd + self.round_offset)] = after_loss_aggregated
        self.all_after_losses_dict[int(rnd + self.round_offset)] = all_after_losses
        
        wandb.log({
            "Test Generalized Loss": after_loss_aggregated,
            }, commit=False, step=rnd)

        # Weigh accuracy of each client by number of examples used
        if 'squad' in self.dataset_name:
            before_exact_matches = [r.metrics["before_exact_match"] * r.num_examples for _, r in results]
            before_f1 = [r.metrics["before_f1"] * r.num_examples for _, r in results]
            after_exact_matches = [r.metrics["after_exact_match"] * r.num_examples for _, r in results]
            after_f1 = [r.metrics["after_f1"] * r.num_examples for _, r in results]
        
            # Aggregate and print custom metric
            before_exact_match_aggregated = sum(before_exact_matches) / sum(examples)
            before_f1_aggregated = sum(before_f1) / sum(examples)
            after_exact_match_aggregated = sum(after_exact_matches) / sum(examples)
            after_f1_aggregated = sum(after_f1) / sum(examples)
            
            wandb.log({
                "Test Generalized Exact Matches": after_exact_match_aggregated,
                "Test Generalized F1 Score": after_f1_aggregated,
            }, commit=True, step=rnd)
            
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} generalized exact match (before personalization) aggregated from client results: {before_exact_match_aggregated}" + Fore.WHITE, flush=True)
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} generalized f1 (before personalization) aggregated from client results: {before_f1_aggregated}" + Fore.WHITE, flush=True)
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} generalized exact match (after personalization) aggregated from client results: {after_exact_match_aggregated}" + Fore.WHITE, flush=True)
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} generalized f1 (after personalization) aggregated from client results: {after_f1_aggregated}" + Fore.WHITE, flush=True)
            
            all_before_exact_matches = [r.metrics["before_exact_match"] for _, r in results]
            all_before_f1 = [r.metrics["before_f1"] for _, r in results]
            all_after_exact_matches = [r.metrics["after_exact_match"] for _, r in results]
            all_after_f1 = [r.metrics["after_f1"] for _, r in results]
            
            self.before_exact_matches_dict[int(rnd + self.round_offset)] = before_exact_match_aggregated
            self.before_f1_dict[int(rnd + self.round_offset)] = before_f1_aggregated
            self.after_exact_matches_dict[int(rnd + self.round_offset)] = after_exact_match_aggregated
            self.after_f1_dict[int(rnd + self.round_offset)] = after_f1_aggregated
            
            self.all_before_exact_matches_dict[int(rnd + self.round_offset)] = all_before_exact_matches
            self.all_before_f1_dict[int(rnd + self.round_offset)] = all_before_f1
            self.all_after_exact_matches_dict[int(rnd + self.round_offset)] = all_after_exact_matches
            self.all_after_f1_dict[int(rnd + self.round_offset)] = all_after_f1
        else:
            before_accuracies = [r.metrics["before_accuracy"] * r.num_examples for _, r in results]
            after_accuracies = [r.metrics["after_accuracy"] * r.num_examples for _, r in results]
        
            # Aggregate and print custom metric
            before_accuracy_aggregated = sum(before_accuracies) / sum(examples)
            after_accuracy_aggregated = sum(after_accuracies) / sum(examples)
            
            wandb.log({
                "Test Generalized Accuracy": after_accuracy_aggregated,
            }, commit=True, step=rnd)

            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} generalized accuracy (before personalization) aggregated from client results: {before_accuracy_aggregated}" + Fore.WHITE, flush=True)
            print(Fore.WHITE + f"{self.client_algorithm}: Round {rnd + self.round_offset} generalized accuracy (after personalization) aggregated from client results: {after_accuracy_aggregated}" + Fore.WHITE, flush=True)
        
            all_before_accuracies = [r.metrics["before_accuracy"] for _, r in results]
            all_after_accuracies = [r.metrics["after_accuracy"] for _, r in results]
        
            self.before_accuracies_dict[int(rnd + self.round_offset)] = before_accuracy_aggregated
            self.after_accuracies_dict[int(rnd + self.round_offset)] = after_accuracy_aggregated
            self.all_before_accuracies_dict[int(rnd + self.round_offset)] = all_before_accuracies
            self.all_after_accuracies_dict[int(rnd + self.round_offset)] = all_after_accuracies

        self.sample_count_dict[int(rnd + self.round_offset)] = examples

        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.after_loss_file_name, 'w') as f:
                json.dump(self.after_losses_dict, f)

            with open(self.all_after_loss_file_name, 'w') as f:
                json.dump(self.all_after_losses_dict, f)

            with open(self.sample_count_file_name, 'w') as f:
                json.dump(self.sample_count_dict, f)

            plt.figure(7)
            plt.plot(list(map(int, list(self.after_losses_dict.keys()))), self.after_losses_dict.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_after_loss_file_name)
            
            if 'squad' in self.dataset_name:
                with open(self.before_exact_match_file_name, 'w') as f:
                    json.dump(self.before_exact_matches_dict, f)
                with open(self.before_f1_file_name, 'w') as f:
                    json.dump(self.before_f1_dict, f)
                with open(self.after_exact_match_file_name, 'w') as f:
                    json.dump(self.after_exact_matches_dict, f)
                with open(self.after_f1_file_name, 'w') as f:
                    json.dump(self.after_f1_dict, f)

                with open(self.all_before_exact_match_file_name, 'w') as f:
                    json.dump(self.all_before_exact_matches_dict, f)
                with open(self.all_before_f1_file_name, 'w') as f:
                    json.dump(self.all_before_f1_dict, f)
                with open(self.all_after_exact_match_file_name, 'w') as f:
                    json.dump(self.all_after_exact_matches_dict, f)
                with open(self.all_after_f1_file_name, 'w') as f:
                    json.dump(self.all_after_f1_dict, f)
                
                plt.figure(2)
                plt.plot(list(map(int, list(self.before_exact_matches_dict.keys()))), list(self.before_exact_matches_dict.values()), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Validation Exact Matches')
                plt.title('Validation Exact Matches for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.plot_before_exact_match_file_name)

                plt.figure(3)
                plt.plot(list(map(int, list(self.after_exact_matches_dict.keys()))), self.after_exact_matches_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Validation Exact Matches')
                plt.title('Validation Exact Matches for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.plot_after_exact_match_file_name)
                
                plt.figure(4)
                plt.plot(list(map(int, list(self.before_f1_dict.keys()))), self.before_f1_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Validation F1')
                plt.title('Validation F1 for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.plot_before_f1_file_name)

                plt.figure(5)
                plt.plot(list(map(int, list(self.after_f1_dict.keys()))), self.after_f1_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Validation F1')
                plt.title('Validation F1 for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.plot_after_f1_file_name)
                
            else:
                with open(self.before_accu_file_name, 'w') as f:
                    json.dump(self.before_accuracies_dict, f)
                with open(self.after_accu_file_name, 'w') as f:
                    json.dump(self.after_accuracies_dict, f)

                with open(self.all_before_accu_file_name, 'w') as f:
                    json.dump(self.all_before_accuracies_dict, f)
                with open(self.all_after_accu_file_name, 'w') as f:
                    json.dump(self.all_after_accuracies_dict, f)

                plt.figure(2)
                plt.plot(list(map(int, list(self.before_accuracies_dict.keys()))), self.before_accuracies_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Validation Accuracy')
                plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.plot_before_accu_file_name)

                plt.figure(3)
                plt.plot(list(map(int, list(self.after_accuracies_dict.keys()))), self.after_accuracies_dict.values(), c='#2978A0', label='Average')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Validation Accuracy')
                plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.plot_after_accu_file_name)
                
        return super().aggregate_evaluate(rnd, results, failures)
    