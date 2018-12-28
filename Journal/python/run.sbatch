#!/usr/bin/env bash
#SBATCH --job-name=foobar
#SBATCH --output=foobar%a.out
#SBATCH --array=1-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24

source activate mcrl
echo `which python`
python train.py
