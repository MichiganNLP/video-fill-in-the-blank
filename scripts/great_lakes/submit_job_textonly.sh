#!/usr/bin/env bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=textonly
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2gb
#SBATCH --ntasks=
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-5:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu
#SBATCH --output=LifeQA-methodology/textonly.out
#SBATCH --error=LifeQA-methodology/textonly.err
# The application(s) to execute along with its input arguments and options:
echo Started!
# Use your own conda because Great Lakes ones are old and thus problematic.
# source ~/.bashrc
echo Hooking
eval "$(conda shell.bash hook)"
echo Sourcing
conda activate lqam
echo Sourced
cd LifeQA-methodology || exit
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4
echo Greedy - done!

python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=2 \
--no-repeat-ngram-size=2 --num-workers=4 --only-noun-phrases
echo Beam Size = 2 and --generation-early-stopping flag is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=4 \
--no-repeat-ngram-size=2 --num-workers=4 --only-noun-phrases
echo Beam Size = 4 and --generation-early-stopping flag is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=8 \
--no-repeat-ngram-size=2 --num-workers=4 --only-noun-phrases
echo Beam Size = 8 and --generation-early-stopping flag is not set - done!

python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=2 \
--generation-early-stopping --num-workers=4 --only-noun-phrases
echo Beam Size = 2 and --no-repeat-ngram-size flag is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=4 \
--generation-early-stopping --num-workers=4 --only-noun-phrases
echo Beam Size = 4 and --no-repeat-ngram-size flag is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=8 \
--generation-early-stopping --num-workers=4 --only-noun-phrases
echo Beam Size = 8 and --no-repeat-ngram-size flag is not set - done!

python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=2 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --only-noun-phrases
echo Beam Size = 2 - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=4 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --only-noun-phrases
echo Beam Size = 4 - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=8 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4 --only-noun-phrases
echo Beam Size = 8 - done!

# NO NP FILTERS
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=2 \
--no-repeat-ngram-size=2 --num-workers=4
echo Beam Size = 2 and --generation-early-stopping flag --only-noun-phrases is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=4 \
--no-repeat-ngram-size=2 --num-workers=4
echo Beam Size = 4 and --generation-early-stopping flag --only-noun-phrases is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=8 \
--no-repeat-ngram-size=2 --num-workers=4
echo Beam Size = 8 and --generation-early-stopping flag --only-noun-phrases is not set - done!

python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=2 \
--generation-early-stopping --num-workers=4
echo Beam Size = 2 and --no-repeat-ngram-size flag --only-noun-phrases is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=4 \
--generation-early-stopping --num-workers=4
echo Beam Size = 4 and --no-repeat-ngram-size flag --only-noun-phrases is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=8 \
--generation-early-stopping --num-workers=4
echo Beam Size = 8 and --no-repeat-ngram-size flag --only-noun-phrases is not set - done!

python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=2 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4
echo Beam Size = 2 --only-noun-phrases is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=4 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4
echo Beam Size = 4 --only-noun-phrases is not set - done!
python -m scripts.evaluate_text_only_baseline --gpus=1 --max-length=10 --beam-size=8 \
--generation-early-stopping --no-repeat-ngram-size=2 --num-workers=4
echo Beam Size = 8 --only-noun-phrases is not set - done!
echo "done"
