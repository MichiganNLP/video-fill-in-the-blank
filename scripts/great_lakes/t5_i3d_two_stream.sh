#!/usr/bin/env bash

#SBATCH --job-name=two_stream
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=5:00:00

source scripts/great_lakes/init.source
python -u scripts/run_model.py --use-visual --two-stream --train --gpus 1 --num-workers 4 --batch-size 64 "$*"
