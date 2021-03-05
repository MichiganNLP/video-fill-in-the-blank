#!/usr/bin/env bash

#SBATCH --job-name=text_only_ft
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

source scripts/great_lakes/init.source
python -m scripts.run_model --gpus 1 --generation-early-stopping --no-repeat-ngram-size 2 \
  --num-workers 4 --train --batch-size 96 --epochs 50
