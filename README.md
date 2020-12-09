# lqam

## Setup

If you are a macOS user, then you need to comment out the CUDA-related lines in `environment.yml`:

```yaml
dependencies:
  # ...
  - cudatoolkit=...
  - cudnn==...
```

With Conda:

```bash
conda env create -f environment.yml
conda activate lqam
# FIXME: are we still using the following?
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

Put the data under `data/`. For example, in Great Lakes:

```bash
ln -s /scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions data
```

## Training

```bash
./multi_modal_model_lightning.py
```

### Good optimizations

```bash
./multi_modal_model_lightning.py \
  --gpu-count 1 \
  --num-workers $(nproc) \
  --pin-memory
```
