#!/usr/bin/env bash

#SBATCH --job-name=textonly
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

echo Started!
echo Hooking
eval "$(conda shell.bash hook)"
echo Sourcing
conda activate lqam
echo Sourced

echo evaluting greedy search
echo evaluating "python -m scripts.run_model --gpus=1 --max-length=10 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=512"
python -m scripts.run_model --gpus=1 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=512

echo evaluting beam search
command="python -m scripts.run_model --gpus=1 --max-length=10 --num-workers=4"
for beam_size in 2 4 8
do
  for only_noun_phrase in 0 1
  do
    for early_stopping in 0 1
    do
      for no_repeat_n_gram in 0 1
      do
        curr_command="${command} --beam-size=${beam_size} --batch-size=$((512 / beam_size))"
        if [[ "$only_noun_phrase" == 1 ]]; then
          curr_command+=" --only-noun-phrases"
        fi
        if [[ "$early_stopping" == 1 ]]; then
          curr_command+=" --generation-early-stopping"
        fi
        if [[ "$no_repeat_n_gram" == 1 ]]; then
          curr_command+=" --no-repeat-ngram-size=2"
        fi
        echo "evaluating $curr_command"
        eval "$curr_command"
      done
    done
  done
done