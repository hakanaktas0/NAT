#!/bin/bash
#SBATCH -w mauao
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --job-name=embedding-hypernetwork


env_python=/nfs-share/as3623/projects/L65-nat/llm-counting-benchmark/.venv/bin/python3

TOKENIZERS_PARALLELISM=false $env_python train_rnn.py