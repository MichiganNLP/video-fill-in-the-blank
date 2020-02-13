#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=baseline_test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=3:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=standard
# The application(s) to execute along with its input arguments and options:
echo Started!
# Use your own conda because Great Lakes ones are old and thus problematic.
source ~/.bashrc
echo Hooking
eval "$(conda shell.bash hook)"
echo Sourcing
conda activate lqam
echo Sourced
python -u train.py
echo done
