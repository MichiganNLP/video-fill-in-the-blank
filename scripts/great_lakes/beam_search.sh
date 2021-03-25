#!/usr/bin/env bash

#SBATCH --job-name=beam_search
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

source scripts/great_lakes/init.source

greedy_batch_size=${1:-512}
command="python -u scripts/run_model.py --gpus 1 --num-workers 4 ${*:2}"

$command --batch-size "$greedy_batch_size" --no-repeat-ngram-size 2

set -x

for beam_size in 2 4 8; do
  for only_noun_phrase in 0 1; do
    for early_stopping in 0 1; do
      for no_repeat_n_gram in 0 1; do
        curr_command="${command} --beam-size ${beam_size} --batch-size $((greedy_batch_size / beam_size))"
        if [[ "$only_noun_phrase" == 1 ]]; then
          curr_command+=" --only-noun-phrases"
        fi
        if [[ "$early_stopping" == 1 ]]; then
          curr_command+=" --generation-early-stopping"
        fi
        if [[ "$no_repeat_n_gram" == 1 ]]; then
          curr_command+=" --no-repeat-ngram-size 2"
        fi
        $curr_command
      done
    done
  done
done
