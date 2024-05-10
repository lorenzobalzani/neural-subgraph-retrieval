#!/bin/bash

# Set default command
command="main.py \
    --pretrained_model_name michiyasunaga/BioLinkBERT-base \
    --model_huggingface_hub_url neural-subgraph-retrieval/umls-nrp-BioLinkBERT-base \
    --dataset_huggingface_hub_url neural-subgraph-retrieval/umls-nrp-dataset \
    --n_epochs 10 \
    --train_batch_size 16 \
    --eval_batch_size 8"

# Check if nvidia-smi command is available and executable
if command -v nvidia-smi &>/dev/null; then
    num_gpus=$(nvidia-smi --list-gpus | wc -l)

    if [ "$num_gpus" -gt 1 ]; then
      command="accelerate launch --config_file accelerate_config.yaml $command"
    else
      command="python3.9 $command"
    fi
else
    command="python3.9 $command"
fi

# Execute the command
eval "$command"
