"""Client-side implementation of FedAvg"""
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
from tqdm import tqdm
import subprocess as sp
from colorama import Fore
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD, AdamW

import flwr as fl
from flwr.common.logger import FLOWER_LOGGER
from flwr.server.app import ServerConfig

from trainers.server_fedmezo import FedAvg
from evaluate import load as load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    HfArgumentParser
)
from peft import LoraConfig, TaskType, get_peft_model

import dataloaders.mezo_agnews.agnews_sequence_classification_dataloader as agnews_sequence_classification_dataloader
import dataloaders.mezo_yahoo.yahoo_dataloader as yahoo_dataloader
import dataloaders.mezo_sst2.sst2_dataloader as sst2_dataloader
# import dataloaders.squad.squad_dataloader_t5 as squad_dataloader
import dataloaders.mezo_mnli.mnli_dataloader as mnli_dataloader
import dataloaders.mezo_snli.snli_dataloader as snli_dataloader
import dataloaders.qnli.qnli_dataloader as qnli_dataloader
import dataloaders.cola.cola_dataloader as cola_dataloader
import dataloaders.mezo_yelp.yelp_dataloader as yelp_dataloader

from models.mezo_models import MODEL_TYPES, resize_token_type_embeddings
from models.mezo_modeling_roberta import RobertaConfig

from utils.mezo_parsers import (
    ModelArguments,
    FederatedArguments,
    DynamicDataTrainingArguments,
    DynamicTrainingArguments,
    MyDataCollatorWithPadding
) 
from utils.mezo_trainer import Trainer
from utils.mezo_processors import num_labels_mapping, output_modes_mapping
from utils.mezo_metrics import *

AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def text_classifier_train(net, train_data, test_data, build_compute_metrics_fn):
    trainer_kwargs = {}
    
    training_args.num_train_epochs = epochs
    trainer = Trainer(
        model=net,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=build_compute_metrics_fn(test_data),
        data_collator=MyDataCollatorWithPadding(tokenizer),
        **trainer_kwargs
    )
    train_result = trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
    # print(len(train_data.support_examples))
    # print(len(train_data.query_examples))
    print(train_result, flush=True)

    # if 'eval_mnli/acc' not in train_result[0].metrics.keys():
    return (
        train_result[0].training_loss, 
        len(train_data.support_examples), 
        train_result[1],
        # train_result[0].metrics['eval_acc'],
        0.0,
        0.0,
    )
    # else:
    #     return (
    #         train_result[0].training_loss, 
    #         len(train_data.support_examples), 
    #         train_result[0].metrics['eval_mnli/acc'],
    #         0.0,
    #         0.0,
    #     )
    
def text_classifier_test(net, test_data, build_compute_metrics_fn):
    trainer_kwargs = {}
    
    trainer = Trainer(
        model=net,
        args=training_args,
        eval_dataset=test_data,
        compute_metrics=build_compute_metrics_fn(test_data),
        data_collator=MyDataCollatorWithPadding(tokenizer),
        **trainer_kwargs
    )
    
    output = trainer.evaluate(eval_dataset=test_data)
    eval_result = output.metrics

    if 'eval_mnli/acc' not in eval_result.keys():
        return eval_result['eval_loss'], eval_result['eval_acc'], len(test_data.query_examples)
    return eval_result['eval_loss'], eval_result['eval_mnli/acc'], len(test_data.query_examples)

def text_classifier_generalized_test(net, train_data, test_data, build_compute_metrics_fn):
    before_loss, before_accuracy, sample_count = text_classifier_test(net, test_data, build_compute_metrics_fn)
    after_loss, after_accuracy = before_loss, before_accuracy
    return before_loss, before_accuracy, after_loss, after_accuracy, sample_count


class TextClassifierClient(fl.client.NumPyClient):

    def __init__(self, cid, net, build_compute_metrics_fn):# -> None:
        self.net = net.to(DEVICE)
        self.cid = str(cid)
        self.build_compute_metrics_fn = build_compute_metrics_fn
        logger.setLevel(20)
        FLOWER_LOGGER.setLevel(20)
        # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('Model size: ', params)
    
    def get_parameters(self, config):
        trainable_keys = []
        for key, val in self.net.named_parameters():
            if val.requires_grad:
                trainable_keys.append(key)

        return [val.cpu().numpy() for key, val in self.net.state_dict().items() if key in trainable_keys]

    def set_parameters(self, parameters):
        trainable_keys = []
        for key, val in self.net.named_parameters():
            if val.requires_grad:
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
        
        # print('===============')
        # for n, p in self.net.named_parameters():
        #     if p.requires_grad:
        #         print(n, ': ', p.data[0][0])
        #         break
        
        self.net.label_word_list = torch.tensor(test_data.label_word_list).long().to(training_args.device)
        
        (loss, count, personalized_accuracy, 
         average_time_per_epoch, 
         average_time_per_iteration_per_epoch) = text_classifier_train(self.net, train_data, test_data, self.build_compute_metrics_fn)
        
        # for n, p in self.net.named_parameters():
        #     if p.requires_grad:
        #         print(n, ': ', p.data[0][0])
        #         break
        # print('----------------')
            
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
        
        before_loss, before_accuracy, after_loss, after_accuracy, count = text_classifier_generalized_test(self.net, train_data, test_data, self.build_compute_metrics_fn)
        return float(before_loss), count, \
            {'before_accuracy': float(before_accuracy), 
             'after_loss': float(after_loss),
             'after_accuracy': float(after_accuracy)}


def client_fn(cid: str):# -> fl.client.Client:
    # print('Picking Client ', cid, flush=True)
    
    # params = sum([np.prod(p.size()) for p in net.parameters()])
    # print('Model size: ', params)
          
    if dataset in ['agnews', 'yahoo', 'sst2', 'mnli', 'snli', 'qnli', 'cola', 'yelp']:
        net = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        
        if training_args.tie_emb:
            logger.warn("Tie embeddings. Only work for RoBERTa (in our code by default they are not tied)")
            net.tie_emb()
        
        if training_args.head_tuning:
            if net.config.model_type == "roberta":
                head_name = "lm_head"

            for n, p in net.named_parameters():
                if head_name not in n:
                    p.requires_grad = False 
                else:
                    logger.info(f"Only tuning {n}")
          
        if net.config.model_type == 'bert':           
            net.resize_token_embeddings(len(tokenizer))
            resize_token_type_embeddings(net, new_num_types=10, random_segment=model_args.random_segment) 
        
        net.model_args = model_args
        net.data_args = data_args
        net.tokenizer = tokenizer
        
        if model_args.apply_lora:
            for name, param in net.named_parameters():
                if (name.startswith('roberta') and "lora" not in name) or (name.startswith('opt') and "lora" not in name):
                    param.requires_grad_(False)
                    
    if dataset == 'agnews':
        build_compute_metrics_fn = build_sst2_compute_metrics_fn 
        return TextClassifierClient(cid, net, build_compute_metrics_fn)
    
    elif dataset == 'yahoo': 
        build_compute_metrics_fn = build_yahoo_compute_metrics_fn 
        net.config.label2id = {'Society': 0, 'Science': 1, 'Health': 2, 'Education': 3, 'Computers': 4, 'Sports': 5, 'Business': 6, 'Entertainment': 7, 'Family': 8, 'Politics': 9}
        net.config.id2label = {0: 'Society', 1: 'Science', 2: 'Health', 3: 'Education', 4: 'Computers', 5: 'Sports', 6: 'Business', 7: 'Entertainment', 8: 'Family', 9: 'Politics'}
        
        return TextClassifierClient(cid, net, build_compute_metrics_fn)
    
    elif dataset == 'sst2':
        build_compute_metrics_fn = build_sst2_compute_metrics_fn
        net.config.label2id = {'negative': 0, 'positive': 1}
        net.config.id2label = {0: 'negative', 1: 'positive'}
        # net.label_word_list = ['negative', 'positive']
        return TextClassifierClient(cid, net, build_compute_metrics_fn)
    
    elif dataset == 'squad':
        return QuestionAnswererClient(cid, net)
        
    elif dataset == 'mnli':
        build_compute_metrics_fn = build_mnli_compute_metrics_fn
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        # net.label_word_list = ['entailment', 'neutral', 'contradiction']
        return TextClassifierClient(cid, net, build_compute_metrics_fn)
    
    elif dataset == 'snli':
        build_compute_metrics_fn = build_snli_compute_metrics_fn
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        # net.label_word_list = ['entailment', 'neutral', 'contradiction']
        return TextClassifierClient(cid, net, build_compute_metrics_fn)
    
    elif dataset == 'qnli':
        net.config.label2id = {'entailment': 0, 'not_entailment': 1}
        net.config.id2label = {0: 'entailment', 1: 'not_entailment'}
        # net.label_word_list = ['entailment', 'not_entailment',]
        return TextClassifierClient(cid, net)
    
    elif dataset == 'cola':
        net.config.label2id = {'unacceptable': 0, 'acceptable': 1}
        net.config.id2label = {0: 'unacceptable', 1: 'acceptable'}
        # net.label_word_list = ['unacceptable', 'acceptable',]
        return TextClassifierClient(cid, net)
    
    elif dataset == 'yelp':
        build_compute_metrics_fn = build_yelp_compute_metrics_fn
        return TextClassifierClient(cid, net, build_compute_metrics_fn)
    
    elif dataset == 'qqp':
        build_compute_metrics_fn = build_qqp_compute_metrics_fn
        net.config.label2id = {'not_duplicate': 0, 'duplicate': 1}
        net.config.id2label = {0: 'not_duplicate', 1: 'duplicate'}
        
        return TextClassifierClient(cid, net, build_compute_metrics_fn)

def load_parameters_from_disk():# -> fl.common.Parameters:
    model_file_name = temp_dir + 'model.pth'
    
    if dataset in ['agnews', 'yahoo', 'sst2', 'mnli', 'snli', 'qnli', 'cola', 'yelp']:
        net = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        
    if training_args.tie_emb:
        logger.warn("Tie embeddings. Only work for RoBERTa (in our code by default they are not tied)")
        net.tie_emb()
    
    if training_args.head_tuning:
        if net.config.model_type == "roberta":
            head_name = "lm_head"

        for n, p in net.named_parameters():
            if head_name not in n:
                p.requires_grad = False 
            else:
                logger.info(f"Only tuning {n}")
        
    if net.config.model_type == 'bert':           
        net.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(net, new_num_types=10, random_segment=model_args.random_segment) 
    
    net.model_args = model_args
    net.data_args = data_args
    net.tokenizer = tokenizer
    
    trainable_keys = []
    if model_args.apply_lora:
        for name, param in net.named_parameters():
            if (name.startswith('roberta') and "lora" not in name) or (name.startswith('opt') and "lora" not in name):
                param.requires_grad_(False)
            else:
                trainable_keys.append(name)
        
    if dataset == 'sst2':
        net.config.label2id = {'negative': 0, 'positive': 1}
        net.config.id2label = {0: 'negative', 1: 'positive'}
    
    elif dataset == 'mnli':
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    
    elif dataset == 'snli':
        net.config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        net.config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    
    elif dataset == 'qnli':
        net.config.label2id = {'entailment': 0, 'not_entailment': 1}
        net.config.id2label = {0: 'entailment', 1: 'not_entailment'}
    
    elif dataset == 'cola':
        net.config.label2id = {'unacceptable': 0, 'acceptable': 1}
        net.config.id2label = {0: 'unacceptable', 1: 'acceptable'}
    
    elif dataset == 'yahoo':
        net.config.label2id = {'Society': 0, 'Science': 1, 'Health': 2, 'Education': 3, 'Computers': 4, 'Sports': 5, 'Business': 6, 'Entertainment': 7, 'Family': 8, 'Politics': 9}
        net.config.id2label = {0: 'Society', 1: 'Science', 2: 'Health', 3: 'Education', 4: 'Computers', 5: 'Sports', 6: 'Business', 7: 'Entertainment', 8: 'Family', 9: 'Politics'}
        
    
    if not os.path.exists(model_file_name):
        return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for key, val in net.state_dict().items() if key in trainable_keys]
            ), rounds
    
    print("Loading: ", model_file_name)

    checkpoint = torch.load(model_file_name)
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    random.setstate(checkpoint['random_random_state'])

    print(Fore.YELLOW + f"Loading model weights from round #{checkpoint['round']}" + Fore.WHITE)

    return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for key, val in net.state_dict().items() if key in trainable_keys]
        ), rounds - checkpoint['round']

if __name__  == '__main__':
    parser = HfArgumentParser((ModelArguments, FederatedArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, federated_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.task_name = data_args.task_name.lower()
    
    if training_args.sweep:
        now = datetime.now()
        dt_str = now.strftime('%m_%d_%H_%M_%S')
        training_args.output_dir = os.path.join(training_args.output_dir, dt_str)

    if training_args.kernel_formula == 'asymmetric_signgd':
        assert training_args.binary_classification, 'asymmetric solver not implemented for multi-class setting, use --binary_classification'

    if training_args.optimizer_variant != '':
        assert training_args.optimizer == 'sgd', 'variants on optimizer are only implemented for SGD'

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO #if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))
    
    training_args.local_rank = -1
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # logger.info("Training/evaluation parameters %s", training_args)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            new_sfc_template = data_args.sfc_prompt + ''
            old_template = old_template.replace('*cls*', '')
            old_template = old_template.replace('*bos*', '')
            if data_args.gpt3_in_context_head:
                new_template = new_template.replace('*cls*', '')
                new_template = new_template.replace('*bos*', '')

            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                if "opt" in model_args.model_name_or_path or "gpt" in model_args.model_name_or_path:
                    sub_template = sub_template + "*labelx_{}*".format(instance_id)
                else:
                    sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + data_args.gpt3_demo_separator + sub_template # Put context at the end
                    new_sfc_template = new_sfc_template + data_args.gpt3_demo_separator + sub_template
                else:
                    new_template = sub_template + data_args.gpt3_demo_separator + new_template # Put context at the beginning
                    new_sfc_template = sub_template + data_args.gpt3_demo_separator + new_sfc_template
            if data_args.gpt3_in_context_head:
                new_template = "*bos*" + new_template
                new_sfc_template = "*bos*" + new_sfc_template
            logger.info("| {} => {}".format(data_args.template, new_template))
            logger.info("New SFC template (in-context learning): {}".format(new_sfc_template))
            data_args.template = new_template
            if model_args.icl_sfc:
                data_args.icl_sfc_prompt = new_sfc_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    config_kwargs = {'apply_lora': model_args.apply_lora,
                    'lora_alpha': model_args.lora_alpha,
                    'lora_r': model_args.lora_r}
    
    if model_args.apply_lora:
        config = RobertaConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                **config_kwargs
                )
        
    if training_args.untie_emb:
        logger.warn("Untie embeddings and lm head")
        logger.warn("NOTE that this only works for OPT. By default RoBERTa model embeddings are already untied.")
        config.tie_word_embeddings = False
        
    if 'prompt' in model_args.few_shot_type:
        model_fn = MODEL_TYPES[config.model_type]
    elif model_args.few_shot_type == 'finetune':
        if training_args.from_linearhead:
            model_fn = MODEL_TYPES[config.model_type]
        else:
            model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    
    dataset = data_args.task_name
    method = 'FedMeZO'
    hyperparameters = {}
    
    hyperparameters['k'] = data_args.num_k
    hyperparameters['lr'] = training_args.learning_rate
    hyperparameters['epsilon'] = training_args.zero_order_eps
    hyperparameters['steps'] = training_args.max_steps
    hyperparameters['model'] = model_args.model_name_or_path
    hyperparameters['optimizer'] = training_args.optimizer
    hyperparameters['lora_r'] = model_args.lora_r
    hyperparameters['lora_alpha'] = model_args.lora_alpha
    hyperparameters['experiment_name'] = data_args.tag
    hyperparameters['cache_dir'] = model_args.cache_dir
    hyperparameters['apply_lora'] = model_args.apply_lora
        
    num_clients = federated_args.num_clients
    hyperparameters['num_clients'] = num_clients
    total_clients = federated_args.total_clients
    hyperparameters['total_clients'] = total_clients
    cpu_simultaneous_clients = federated_args.cpu_simultaneous_clients
    hyperparameters['cpu_simultaneous_clients'] = cpu_simultaneous_clients
    gpu_simultaneous_clients = federated_args.gpu_simultaneous_clients
    hyperparameters['gpu_simultaneous_clients'] = gpu_simultaneous_clients
    rounds = federated_args.rounds
    hyperparameters['rounds'] = rounds
    total_rounds = federated_args.rounds
    dirichlet_distribution = federated_args.dirichlet_distribution
    hyperparameters['dirichlet_distribution'] = dirichlet_distribution
    random_seed = federated_args.random_seed
    hyperparameters['random_seed'] = random_seed
    
    model_name = model_args.model_name_or_path
    hyperparameters['model_name'] = model_name
    temp_dir = federated_args.temp_dir + data_args.tag + '/'
    hyperparameters['temp_dir'] = temp_dir
    max_seq_len = data_args.max_seq_length
    hyperparameters['max_seq_len'] = max_seq_len
    batch_size = training_args.per_device_train_batch_size
    hyperparameters['batch_size'] = batch_size
    learning_rate = training_args.learning_rate
    hyperparameters['learning_rate'] = learning_rate
    epochs = federated_args.epochs
    hyperparameters['epochs'] = epochs
    checkpoint_dir = federated_args.checkpoint_dir + data_args.tag + '/'
    hyperparameters['checkpoint_dir'] = checkpoint_dir
    checkpoint_interval = federated_args.checkpoint_interval
    hyperparameters['checkpoint_interval'] = checkpoint_interval
    training_args.output_dir = checkpoint_dir
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        additional_special_tokens=[],
        cache_dir=model_args.cache_dir
        )
    tokenizer.model_type = config.model_type
    
    if dataset == 'agnews':
        train_dataloader_dict, test_dataloader_dict = agnews_sequence_classification_dataloader.get_federated_datasets(
            config,
            data_args,
            model_args,
            tokenizer,
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            )
        
        num_labels = 4
    
    elif dataset == 'yahoo':
        train_dataloader_dict, test_dataloader_dict = yahoo_dataloader.get_federated_datasets(
            config,
            data_args,
            model_args,
            tokenizer,
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            )
        
        num_labels = 10
        
    elif dataset == 'sst2':
        train_dataloader_dict, test_dataloader_dict = sst2_dataloader.get_federated_datasets(
            config,
            data_args,
            model_args,
            tokenizer,
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
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
            config,
            data_args,
            model_args,
            tokenizer,
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            )
        
        num_labels = 3
        
    elif dataset == 'snli':
        train_dataloader_dict, test_dataloader_dict = snli_dataloader.get_federated_datasets(
            config,
            data_args,
            model_args,
            tokenizer,
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
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
            config,
            data_args,
            model_args,
            tokenizer,
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            )
        
        num_labels = 2
        
    elif dataset == 'qqp':
        train_dataloader_dict, test_dataloader_dict = qqp_dataloader.get_federated_datasets(
            config,
            data_args,
            model_args,
            tokenizer,
            dirichlet_parameter=dirichlet_distribution,
            num_clients=total_clients, 
            train_client_batch_size=batch_size,
            max_seq_len=max_seq_len,
            )
        
        num_labels = 2
     
    model_args.num_labels = num_labels 
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
    
    print('Experiment: ', data_args.tag)
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  

    initial_parameters, rounds = load_parameters_from_disk()

    FLOWER_LOGGER.setLevel(20)

    strategy = FedAvg(
        fraction_fit = num_clients / len(clients),
        fraction_eval = num_clients / len(clients),
        min_fit_clients = num_clients,
        min_eval_clients = num_clients,
        min_available_clients = num_clients,
        dataset = dataset,
        client_algorithm = method,
        initial_parameters = initial_parameters,
        hyperparameters = hyperparameters,
    )
    
    print(Fore.RED + "Availble Device: " + str(DEVICE) + ", Count: " + \
        str(AVAILABLE_GPUS) + ". CPUs, Count: " + str(AVAILABLE_CPUS) + '.' + Fore.WHITE, flush=True)

    server_config = ServerConfig()
    server_config.num_rounds = rounds
    
    simulation_start_time = datetime.now() 
    print('Starting at:', simulation_start_time)
    print('')

    fl.simulation.start_simulation(
        client_fn = client_fn,
        client_resources = {'num_cpus': num_cpus_needed, 
                            'num_gpus': num_gpus_needed},
        clients_ids = clients,
        config = server_config,
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
        