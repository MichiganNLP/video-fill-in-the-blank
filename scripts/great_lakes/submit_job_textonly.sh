#!/usr/bin/env bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=textonly
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
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --batch-size=96
echo evaluating "python -m scripts.evaluate_text_only_baseline --batch-size=96 --gpus=1 --max-length=10 --generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4"

command="python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --num-workers=4"

for beam_size in 2 4 8
do
  for only_noun_phrase in 0 1
  do
    batch_size=$((512 / beam_size))
    new_command="${command} --beam-size=${beam_size} --batch-size=${batch_size}"
    if [[ "$only_noun_phrase" == 1 ]]; then
      new_command="${new_command} --only-noun-phrases"
    fi
    for early_stopping in 0 1
    do
      if [[ "$early_stopping" == 1 ]]; then
        new_command="${new_command} --generation-early-stopping"
      fi
      for no_repeat_n_gram in 0 1
      do
        if [[ "$no_repeat_n_gram" == 1 ]]; then
          new_command="${new_command} --no-repeat-ngram-size=2"
        fi
        echo "evaluating $new_command"
        eval "$new_command"
      done
    done
  done
done
