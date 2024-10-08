
# Thinking Forward: Memory-Efficient Federated Finetuning of Language Models (NeurIPS 2024)

Spry is a federated learning algorithm that enables finetuning LLMs using Forward-mode Auto Differentiation; to achieve low memory footprint, high accuracy, and fast convergence.

  

Paper link: https://openreview.net/forum?id=dGQtja9X2C 


```

@inproceedings{

anonymous2024thinking,

title={Thinking Forward: Memory-Efficient Federated Finetuning of Language Models},

author={Anonymous},

booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},

year={2024},

url={https://openreview.net/forum?id=dGQtja9X2C}

}

```

  

# Directory Structure

```

📁 Spry

├── 📁 dataloaders

│ ├── 📁 agnews

│ │ └── 🐍 agnews_sequence_classification_dataloader.py

│ └── 📁 ...

├── 📁 dataset_cache

│ ├── 📁 agnews-seqcls-bert-large

│ │ ├── 📁 manual_save_1000_0_1

│ │ │ └── 💽 test_dataset_0.pth

│ │ └── 📁 manual_save_1000_1_0

│ └── 📁 ...

├── 📁 models

│ ├── 📁 models--bert-large-uncased

│ │ └── \<dataset related files\>

│ └── 📁 ...

├── 📁 results

│ ├── 📁 agnews

│ │ ├── 📁 Spry

│ │ └── 📁 ...

│ └── 📁 ...

├── 📁 trainers

│ ├── 🐍 client_spry.py

│ ├── 🐍 server_spry.py

│ └── 🐍 ...

├── 🛠️ run_federated_job.sh

└── 📄 requirements.txt

```

  

## Directory Descriptions

  

1.  `./dataloaders`

Dataloaders for all the datasets are provided in this directory. The dataloader files include code to

(a) Download the central dataset (most of them would be based on Huggingface datasets).

(b) Split it into $c$ clients for the federated learning setting.

(c) Simulate heterogeneity across client dataset splits, based on Dirichlet distribution.

  

The dataloader methods would be called from `./trainers/client_<method_name>.py`.

  

2.  `./dataset_cache`

Downloaded centralized dataset would be cached in this folder. The split datasets would also be cached in this directory, categorized by the Dirichlet distribution concentration parameter $\alpha$.

  

3.  `./models`

All the models used in this work are available on Huggingface. The downloaded pre-trained weights are cached in this directory.

  

The quantized model weights (code in `./trainers/client_<method_name>.py`) would also be cached here.

  

4.  `./results`

The log files of each run, the automatically generated plots, the detailed performance numbers would be stored in this directory, under `./results/<dataset_name>/<method_name>/<unique_experiment_name>/`.

  

5.  `./trainers`

This directory containes client and server files for all the training simulations of a federated learning setting.

(a) `server_<method_name>.py` has methods to orchestrate creation of client objects, calls to client `fit` and `evaluate` methods, aggregating received results from clients.

(b) `client_<method_name>.py` is the starting point of all experiments. It contains trainer loops, and a client object, which the server will create to generate client instances for each round.

  

# Run Spry and its Baselines

The starting point of the methods and their file names are listed here:

1. Spry - `./trainers/client_spry.py`

2. FedAvg - `./trainers/client_fedavg.py`

3. FedYogi - `./trainers/client_fedyogi.py`

4. FedSgd - `./trainers/barebones_fedsgd_grads.py`

5. FwdLLM+ - `./trainers/client_fwdllm_plus.py`

6. Baffle+ - `./trainers/client_baffle_plus.py`

7. FedMeZO - `./trainers/client_fedmezo.py`

  

A bash script `./run_job.sh` is given to play around with hyperparameters and run the Python script.

  

Experiments can be run with a minimal change of values of `dataset` and `method` variables.