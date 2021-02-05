#!/usr/bin/env bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=textonly_ft
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu
# The application(s) to execute along with its input arguments and options:
echo Started!
# Use your own conda because Great Lakes ones are old and thus problematic.
# source ~/.bashrc
echo Hooking
eval "$(conda shell.bash hook)"
echo Sourcing
conda activate lqam
echo Sourced

python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --train --batch_size=96
