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
export http_proxy="http://proxy.arc-ts.umich.edu:3128/"
export https_proxy="http://proxy.arc-ts.umich.edu:3128/"
export ftp_proxy="http://proxy.arc-ts.umich.edu:3128/"
export no_proxy="localhost,127.0.0.1,.localdomain,.umich.edu"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export FTP_PROXY="${ftp_proxy}"
export NO_PROXY="${no_proxy}"
PYTHONPATH=. python -u scripts/run_model.py --use-visual --train --gpus 1 --num-workers 3 --batch-size 64
echo Done
