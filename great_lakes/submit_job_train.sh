#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40000m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=05-5:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu
#SBATCH --output=/home/liunan/train.out
#SBATCH --error=/home/liunan/train.err
# The application(s) to execute along with its input arguments and options:
echo Started!
# Use your own conda because Great Lakes ones are old and thus problematic.
# source ~/.bashrc
echo Hooking
eval "$(conda shell.bash hook)"
echo Sourcing
conda activate /home/liunan/anaconda3/envs/lqam
echo Sourced
python -u /home/liunan/LifeQA-methodology/train.py > /home/liunan/LifeQA-methodology/train.txt
echo done