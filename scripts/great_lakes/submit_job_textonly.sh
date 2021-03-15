#!/usr/bin/env bash

#SBATCH --job-name=text_only
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

source scripts/great_lakes/init.source

greedy_batch_size=512

set -x

python -m scripts.run_model --gpus 1 --num-workers 4 --batch-size "$greedy_batch_size" "$*"
python -m scripts.run_model --gpus 1 --no-repeat-ngram-size 2 --num-workers 4 --batch-size "$greedy_batch_size" "$*"

command="python -m scripts.run_model --gpus 1 --num-workers 4 $*"
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
