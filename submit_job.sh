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
module load python3.6-anaconda
echo hook:
eval "$(conda shell.bash hook)"
echo Sourcing
conda activate NLPenv0
echo Sourced
python train.py
echo done
