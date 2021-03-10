#!/usr/bin/env bash

#SBATCH --job-name=text_only
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

source scripts/great_lakes/init.source

t5_small_greedy_command="python -m scripts.run_model --gpus=1 --max-length=10 --model=t5-small \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=1204"
echo "evaluating ${t5_small_greedy_command}"
eval "${t5_small_greedy_command}"

t5_base_greedy_command="python -m scripts.run_model --gpus=1 --max-length=10 --model=t5-base \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=512"
echo "evaluating ${t5_base_greedy_command}"
eval "${t5_base_greedy_command}"

t5_large_greedy_command="python -m scripts.run_model --gpus=1 --max-length=10 --model=t5-large \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=256"
echo "evaluating ${t5_large_greedy_command}"
eval "${t5_large_greedy_command}"

t5_3b_greedy_command="python -m scripts.run_model --gpus=1 --max-length=10 --model=t5-3b \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=16"
echo "evaluating ${t5_3b_greedy_command}"
eval "${t5_3b_greedy_command}"

#t5_11b_greedy_command="python -m scripts.run_model --gpus=1 --max-length=10 --model=t5-11b \
#--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=8"
#echo "evaluating ${t5_11b_greedy_command}"
#eval "${t5_11b_greedy_command}"
