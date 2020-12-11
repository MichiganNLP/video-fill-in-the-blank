#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=extract_object_detection_features
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000m 
#SBATCH --gres=gpu:1
#SBATCH --time=01-5:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu
# The application(s) to execute along with its input arguments and options:

# Use your own conda because Great Lakes ones are old and thus problematic.
# source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate lqam
python -u normal_bbox.py