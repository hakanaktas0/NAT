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
    --mode "val" \
    --trained_model_path "/nfs-share/as3623/projects/L65-nat/NAT/checkpoints-20250407_172609/best_model.pt" \
    --model_dir "/nfs-share/as3623/models/Llama-3.2-1B/" \
    --device cuda \
    --input_dim 2048 \
    --hidden_dim 1024 \
    --output_dim 2048 \
    --num_layers 4 \
    --rnn_type "lstm";
