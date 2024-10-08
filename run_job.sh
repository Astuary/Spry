dataset=yelp
method=fedfgd
experiment_count=02

random_seed=0
cpu_simultaneous_clients=4
gpu_simultaneous_clients=4
checkpoint_interval=10

rounds=1500
dirichlet_distribution=1.0

epochs=1
finetuning_epochs=5
batch_size=8
max_seq_len=64

peft_method=bitfit
if echo "$peft_method" | grep -q "lora"; then
    lora_r=1
    lora_alpha=1
elif echo "$peft_method" | grep -q "bitfit"; then
    lora_r=0
    lora_alpha=0
elif echo "$peft_method" | grep -q "classifier"; then
    lora_r=0
    lora_alpha=0
elif echo "$peft_method" | grep -q "ia3"; then
    lora_r=0
    lora_alpha=0
fi

if echo "$dataset" | grep -q "agnews"; then
    model_name=roberta-large
    # model_name=bert-base-uncased
    num_clients=100
    total_clients=1000
    lr=1e-3
elif echo "$dataset" | grep -q "sst2"; then
    model_name=distilbert-base-uncased
    # model_name=bert-base-uncased
    # model_name=bert-large-uncased
    num_clients=10
    total_clients=100
    lr=1e-4
elif echo "$dataset" | grep -q "snli"; then
    model_name=bert-large-uncased
    num_clients=10
    total_clients=1000
    lr=1e-4
elif echo "$dataset" | grep -q "mnli"; then
    model_name=bert-large-uncased
    num_clients=10
    total_clients=1000
    lr=1e-4
elif echo "$dataset" | grep -q "yahoo"; then
    model_name=distilbert-base-uncased
    num_clients=10
    total_clients=1000
    lr=1e-4
elif echo "$dataset" | grep -q "yelp"; then
    model_name=albert-large-v2
    num_clients=10
    total_clients=1000
    lr=1e-4
elif echo "$dataset" | grep -q "qqp"; then
    model_name=bert-base-uncased
    num_clients=10
    total_clients=100
    lr=1e-4
elif echo "$dataset" | grep -q "squad6"; then
    # model_name=google-t5/t5-11B
    model_name=facebook/opt-6.7B
    # model_name=EleutherAI/gpt-j-6b
    max_seq_len=384
    num_clients=10
    total_clients=500
    lr=1e-4
    if echo "$method" | grep -q "fedavg"; then
        cpu_simultaneous_clients=2
        gpu_simultaneous_clients=2
    elif echo "$method" | grep -q "fedyogi"; then
        cpu_simultaneous_clients=2
        gpu_simultaneous_clients=2
    fi
elif echo "$dataset" | grep -q "squad13"; then
    # model_name=google-t5/t5-11B
    model_name=facebook/opt-13B
    # model_name=EleutherAI/gpt-j-6b
    max_seq_len=384
    num_clients=10
    total_clients=500
    batch_size=4
    lr=1e-4
    if echo "$method" | grep -q "fedavg"; then
        cpu_simultaneous_clients=2
        gpu_simultaneous_clients=2
    elif echo "$method" | grep -q "fedyogi"; then
        cpu_simultaneous_clients=2
        gpu_simultaneous_clients=2
    fi
elif echo "$dataset" | grep -q "multirc"; then
    # model_name=/datasets/ai/llama2/huggingface/llama-2-7b
    model_name=/datasets/ai/llama2/llama-2-7b
    num_clients=10
    total_clients=100
    lr=1e-4

    if echo "$method" | grep -q "fedavg"; then
        cpu_simultaneous_clients=2
        gpu_simultaneous_clients=2
        lr=1e-4
    elif echo "$method" | grep -q "fedyogi"; then
        cpu_simultaneous_clients=2
        gpu_simultaneous_clients=2
    fi
fi

if echo "$method" | grep -q "spry"; then
    if echo "$dataset" | grep -q "agnews"; then
        lr=1e-4
    elif echo "$dataset" | grep -q "yelp"; then
        echo "Yelp Albert-large doesn't work with split because of too few layers. Just run FedFGD."
        exit
    fi

    perturbation_count=1
    perturbation_var=1
    fixed_seed=False
    classifier_share=False

    per_client_layer_count=2

    client_opt=adamw
    server_opt=yogi

    checkpoint_dir=./results/${dataset}/Spry/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/Spry/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${finetuning_epochs}_ft_epochs_${lr}_lr_llama2_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
    elif echo "$dataset" | grep -q "squad6"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${finetuning_epochs}_ft_epochs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
        dataset=squad
    elif echo "$dataset" | grep -q "squad13"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${finetuning_epochs}_ft_epochs_${lr}_lr_opt13_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
        dataset=squad
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${finetuning_epochs}_ft_epochs_${per_client_layer_count}_layers_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
    fi
elif echo "$method" | grep -q "fedavgsplit"; then
    classifier_share=False

    client_opt=adamw

    checkpoint_dir=./results/${dataset}/FedAvgSplit/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/FedAvgSplit/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama2_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    elif echo "$dataset" | grep -q "squad"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    fi
elif echo "$method" | grep -q "fedyogisplit"; then
    classifier_share=False

    client_opt=adamw

    checkpoint_dir=./results/${dataset}/FedYogiSplit/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/FedYogiSplit/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama2_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    elif echo "$dataset" | grep -q "squad"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    fi
elif echo "$method" | grep -q "fedfgd"; then
    if echo "$dataset" | grep -q "agnews"; then
        lr=1e-4
    fi

    perturbation_count=1
    perturbation_var=1
    fixed_seed=False

    client_opt=adamw
    server_opt=yogi

    checkpoint_dir=./results/${dataset}/FedFgd/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/FedFgd/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama2_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
    elif echo "$dataset" | grep -q "squad6"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
        dataset=squad
    elif echo "$dataset" | grep -q "squad13"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt13_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
        dataset=squad
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${client_opt}_${server_opt}_opts_${experiment_count}
    fi
elif echo "$method" | grep -q "baffle_plus"; then
    K=10
    sigma=1e-4
    lr=1e-4
    finite_difference_format=center
    
    if echo "$dataset" | grep -q "yelp"; then
        lr=1e-5
    fi

    client_opt=adamw

    checkpoint_dir=./results/${dataset}/BafflePlus/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/BafflePlus/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama2_${peft_method}_${lora_r}_r_${K}_K_${sigma}_sigma_${finite_difference_format}_fd_${experiment_count}
    elif echo "$dataset" | grep -q "squad6"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${K}_K_${sigma}_sigma_${finite_difference_format}_fd_${experiment_count}
        dataset=squad
    elif echo "$dataset" | grep -q "squad13"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt13_${peft_method}_${lora_r}_r_${K}_K_${sigma}_sigma_${finite_difference_format}_fd_${experiment_count}
        dataset=squad
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${K}_K_${sigma}_sigma_${finite_difference_format}_fd_${experiment_count}
    fi
elif echo "$method" | grep -q "baffle"; then
    K=10
    sigma=1e-4
    lr=1e-4
    finite_difference_format=center
    
    if echo "$dataset" | grep -q "snli"; then
        model_name=roberta-base
    elif echo "$dataset" | grep -q "yahoo"; then
        model_name=roberta-base
    elif echo "$dataset" | grep -q "yelp"; then
        lr=1e-5
    fi

    checkpoint_dir=./results/${dataset}/Baffle/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/Baffle/
    
    experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${K}_K_${sigma}_sigma_${finite_difference_format}_fd_${experiment_count}
elif echo "$method" | grep -q "fwdllm_plus"; then
    perturbation_count=20
    if echo "$dataset" | grep -q "squad"; then
        perturbation_count=1
    fi
    perturbation_var=1e+0
    perturbation_scale=1e-2
    fixed_seed=False
    var_threshold=1e+2
    lr=1e-4

    rounds=3000

    client_opt=adamw

    checkpoint_dir=./results/${dataset}/FwdLLMPlus/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/FwdLLMPlus/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${perturbation_scale}_pscale_${var_threshold}_var_thr_${client_opt}_opts_${experiment_count}
    elif echo "$dataset" | grep -q "squad6"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${perturbation_scale}_pscale_${var_threshold}_var_thr_${client_opt}_opts_${experiment_count}
        dataset=squad
    elif echo "$dataset" | grep -q "squad13"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt13_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${perturbation_scale}_pscale_${var_threshold}_var_thr_${client_opt}_opts_${experiment_count}
        dataset=squad
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${perturbation_scale}_pscale_${var_threshold}_var_thr_${client_opt}_opts_${experiment_count}
    fi
elif echo "$method" | grep -q "fwdllm"; then
    perturbation_count=1
    perturbation_var=1e+0
    perturbation_scale=1e-2
    fixed_seed=False
    var_threshold=1e+2
    lr=1e-4

    rounds=3000

    cpu_simultaneous_clients=1
    gpu_simultaneous_clients=1

    client_opt=sgd

    checkpoint_dir=./results/${dataset}/FwdLLM/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/FwdLLM/
    
    if [[ 'llama' =~ $model_name ]]; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${perturbation_scale}_pscale_${var_threshold}_var_thr_${client_opt}_opts_${experiment_count}
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${fixed_seed}_seed_${perturbation_count}_pcount_${perturbation_var}_pvar_${perturbation_scale}_pscale_${var_threshold}_var_thr_${client_opt}_opts_${experiment_count}
    fi
elif echo "$method" | grep -q "fedavg"; then
    client_opt=adamw
    
    checkpoint_dir=./results/${dataset}/FedAvg/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/FedAvg/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama2_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    elif echo "$dataset" | grep -q "squad6"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
        dataset=squad
    elif echo "$dataset" | grep -q "squad13"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt13_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
        dataset=squad
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    fi
elif echo "$method" | grep -q "fedyogi"; then
    client_opt=adamw

    checkpoint_dir=./results/${dataset}/FedYogi/
    dataset_dir=/dataloaders/${dataset}/
    temp_dir=/scratch/workspace/kpanchal_umass_edu-felicity/${dataset}/FedYogi/
    
    if echo "$dataset" | grep -q "multirc"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_llama2_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    elif echo "$dataset" | grep -q "squad6"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt6_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
        dataset=squad
    elif echo "$dataset" | grep -q "squad13"; then
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_opt13_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
        dataset=squad
    else
        experiment_name=${num_clients}_from_${total_clients}_${dirichlet_distribution//.}_dir_${batch_size}_bs_${lr}_lr_${model_name:0:4}_${peft_method}_${lora_r}_r_${client_opt}_opt_${experiment_count}
    fi
fi

echo "$dataset | $method"
echo "$experiment_name"

if echo "$method" | grep -q "spry"; then
    python3 ./trainers/client_${method}_transformers.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution --finetuning_epochs $finetuning_epochs\
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --perturbation_count $perturbation_count --perturbation_var $perturbation_var \
        --no-fixed_seed --no-classifier_share --per_client_layer_count $per_client_layer_count\
        --client_opt $client_opt --server_opt server_opt \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "fedavgsplit"; then
    python3 ./trainers/client_${method}_transformers.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds --no-classifier_share \
        --dirichlet_distribution $dirichlet_distribution --client_opt $client_opt \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "fedyogisplit"; then
    python3 ./trainers/client_${method}_transformers.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds --no-classifier_share \
        --dirichlet_distribution $dirichlet_distribution --client_opt $client_opt \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "fedfgd"; then
    python3 ./trainers/client_${method}_auto_diff_transformers_v2.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --perturbation_count $perturbation_count --perturbation_var $perturbation_var \
        --no-fixed_seed --client_opt $client_opt --server_opt server_opt \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "baffle_plus"; then
    python3 ./trainers/client_${method}.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --client_opt $client_opt \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --K $K --sigma $sigma --finite_difference_format $finite_difference_format\
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "baffle"; then
    python3 ./trainers/client_${method}.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --K $K --sigma $sigma --finite_difference_format $finite_difference_format\
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "fwdllm_plus"; then
    python3 -W ignore ./trainers/client_${method}.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --perturbation_count $perturbation_count --perturbation_var $perturbation_var \
        --perturbation_scale $perturbation_scale --var_threshold $var_threshold \
        --no-fixed_seed --client_opt $client_opt --server_opt server_opt \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "fwdllm"; then
    python3 -W ignore ./trainers/client_${method}.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --perturbation_count $perturbation_count --perturbation_var $perturbation_var \
        --perturbation_scale $perturbation_scale --var_threshold $var_threshold \
        --no-fixed_seed --client_opt $client_opt --server_opt server_opt \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "fedavg"; then
    python3 ./trainers/client_${method}_transformers_v2.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution --client_opt $client_opt \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
elif echo "$method" | grep -q "fedyogi"; then
    python3 ./trainers/client_${method}_transformers.py \
        --dataset $dataset --model_name $model_name --random_seed $random_seed \
        --cpu_simultaneous_clients $cpu_simultaneous_clients --gpu_simultaneous_clients $gpu_simultaneous_clients \
        --checkpoint_interval $checkpoint_interval --num_clients $num_clients \
        --total_clients $total_clients --rounds $rounds \
        --dirichlet_distribution $dirichlet_distribution --client_opt $client_opt \
        --epochs $epochs --batch_size $batch_size --lr $lr --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --checkpoint_dir $checkpoint_dir --dataset_dir $dataset_dir \
        --temp_dir $temp_dir --experiment_name $experiment_name
fi
