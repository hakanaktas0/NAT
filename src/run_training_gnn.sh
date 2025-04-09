#!/bin/bash
#SBATCH -w ngongotaha
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=priority
#SBATCH --job-name=tokenizing-gnn

hostname
date

env_python=/nfs-share/as3623/projects/L65-nat/llm-counting-benchmark/.venv/bin/python3

TOKENIZERS_PARALLELISM=false $env_python run.py