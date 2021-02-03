#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=lightning_mm_t5_model
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000m 
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
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
PYTHONPATH=/home/ruoyaow/LifeQA-methodology
export PYTHONPATH
python -u ../run_model.py --train --gpus 1 --has-visual --num-workers 8 --default-root-dir /scratch/mihalcea_root/mihalcea1/shared_data/qgen/VATEX/multimodal_model/ --visual-data-path /scratch/mihalcea_root/mihalcea1/shared_data/qgen/VATEX/I3D_video_features
echo done
