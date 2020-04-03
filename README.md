# lqam

## Setup

With Conda:

```bash
conda create -n lqam -c pytorch --file environment.txt
conda activate lqam
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

## Training

```bash
./multi_modal_model_lightning.py --data-path /scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/train.pkl
```
