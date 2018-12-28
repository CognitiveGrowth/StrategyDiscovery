#!/usr/bin/env bash
#SBATCH --job-name=short
#SBATCH --output=short.out
#SBATCH --time=30
#SBATCH --cpus-per-task=6

source activate mcrl
echo `which python`
python train.py
