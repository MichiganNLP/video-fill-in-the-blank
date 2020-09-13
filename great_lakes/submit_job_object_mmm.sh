#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=lightning_mm_model
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000m 
#SBATCH --gres=gpu:1
#SBATCH --time=02-5:00:00
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
python -u ../multi_modal_model_lightning.py --data-path /scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/multimodal_model --num-workers 4 --max-token-num 1 --visual-size 12544
echo done
