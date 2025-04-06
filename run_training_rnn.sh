#!/bin/bash
#SBATCH -w mauao
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --job-name=embedding-hypernetwork

hostname
date

env_python=/nfs-share/as3623/projects/L65-nat/llm-counting-benchmark/.venv/bin/python3

TOKENIZERS_PARALLELISM=false $env_python train_rnn.py \
    --model_dir "/nfs-share/as3623/models/Llama-3.2-1B/" \
    --batch_size 256 \
    --epochs 100 \
    --learning_rate 3e-3 \
    --device cuda \
    --save_dir checkpoints \
    --use_wandb \
    --input_dim 2048 \
    --hidden_dim 4096 \
    --output_dim 2048 \
    --num_layers 4 \
    --rnn_type "lstm" \
    --dropout 0.1;