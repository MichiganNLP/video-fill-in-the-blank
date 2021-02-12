#!/bin/bash

#SBATCH --job-name=t5_i3d
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=5:00:00

echo Started!
# Use your own conda because Great Lakes ones are old and thus problematic.
echo Hooking
eval "$(conda shell.bash hook)"
echo Sourcing
conda activate lqam
echo Sourced
export
PYTHONPATH=. python -u scripts/run_model.py --use-visual --train --gpus 1 --num-workers 3 --batch-size 64
echo Done
