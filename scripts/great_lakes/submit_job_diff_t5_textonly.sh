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

t5=(t5-small t5-base t5-large t5-3b t5-11b)
t5_google=(google/t5-v1_1-small google/t5-v1_1-base google/t5-v1_1-large google/t5-v1_1-xl google/t5-v1_1-xxl)
batch_sizes=(1024 512 256 16)

command="python -m scripts.run_model --gpus=1 --max-length=10 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4"

for i in "${!t5[@]}"; do
    curr_command=$command
    curr_command+=" --model=${t5[i]} --batch-size=${batch_sizes[i]}"
    echo "evaluating $curr_command"
    eval "$curr_command"
done

for i in "${!t5_google[@]}"; do
    curr_command=$command
    curr_command+=" --model=${t5_google[i]} --batch-size=${batch_sizes[i]}"
    echo "evaluating $curr_command"
    eval "$curr_command"
done