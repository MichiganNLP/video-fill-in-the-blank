#!/usr/bin/env bash

#SBATCH --job-name=text_only_ft
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu

source scripts/great_lakes/init.source
python -u scripts/run_model.py --gpus 1 --num-workers 4 --train --batch-size 64 --epochs 50 "$*"
