#!/usr/bin/env bash

#SBATCH --job-name=text_only_ft
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

#source scripts/great_lakes/init.source

model_path=''

print_usage() {
  printf "Usage: using flag '-p MODEL_PATH' to load your model for testing"
}

while getopts 'p:' flag; do
  case "${flag}" in
    p) model_path="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

if [ -z "$1" ]
  then
    print_usage
    exit 1
fi

echo evaluting beam search
command="python -m scripts.run_model --gpus=1 --max-length=10 --num-workers=4 --checkpoint_path $model_path"

for beam_size in 2 4 8; do
  for only_noun_phrase in 0 1; do
    for early_stopping in 0 1; do
      for no_repeat_n_gram in 0 1; do
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