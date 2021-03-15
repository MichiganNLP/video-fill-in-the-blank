#!/usr/bin/env bash

#SBATCH --job-name=text_only
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

source scripts/great_lakes/init.source

# t5-11b and google/t5-v1_1-xxl don't fit in a V100-16Gb.
t5=(t5-small t5-base t5-large t5-3b)
t5_google=(google/t5-v1_1-small google/t5-v1_1-base google/t5-v1_1-large google/t5-v1_1-xl)
batch_sizes=(1024 512 256 16)

command="python -m scripts.run_model --gpus 1 --generation-early-stopping --no-repeat-ngram-size 2 --num-workers 4 $*"

set -x

for i in "${!t5[@]}"; do
  $command --model "${t5[i]}" --batch-size "${batch_sizes[i]}"
done

for i in "${!t5_google[@]}"; do
  $command --model "${t5_google[i]}" --batch-size "${batch_sizes[i]}"
done
