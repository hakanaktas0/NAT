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
    --mode 'train' \
    --model_dir "/nfs-share/as3623/models/Llama-3.2-1B/" \
    --batch_size 256 \
    --epochs 400 \
    --learning_rate 5e-3 \
    --use_cosine_scheduler \
    --device cuda \
    --save_dir checkpoints \
    --use_wandb \
    --input_dim 2048 \
    --hidden_dim 1024 \
    --output_dim 2048 \
    --num_layers 4 \
    --rnn_type "lstm" \
    --dropout 0.1 \
    --combined_loss_alpha 1;
