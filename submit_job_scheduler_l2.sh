#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=scheduler_weight_decay
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32000m 
#SBATCH --gres=gpu:1
#SBATCH --time=03-5:00:00
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
python -u train_mm_scheduler_l2.py
echo done
