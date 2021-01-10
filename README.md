# LifeQA Methodology

This repo contains the annotation scheme, results and methods for the LifeQA project "qgen".

## Setup

1. If you are a macOS user, then you need to comment out the CUDA-related lines in `environment.yml`:

    ```yaml
    dependencies:
      # ...
      - cudatoolkit=...
      - cudnn==...
    ```

2. With Conda installed:

    ```bash
    conda env create
    conda activate lqam
    ```

3. To execute any Python script under [`scripts/`](scripts), run it from the project root directory (where this file is)
and prepend `PYTHONPATH=.` to the command execution (or do `export PYTHONPATH=$PWD` once per Bash session).

4. Put the data under `data/`. For example, in Great Lakes:

    ```bash
    ln -s /scratch/mihalcea_root/mihalcea1/shared_data/qgen/VATEX data
    ```

## Annotation

### Visualize the annotation results

Run:

```bash
./scripts/analyze_annotation_results.py --show-metrics < INPUT_CSV_FILE > OUTPUT_TXT_FILE
```

## Download the YouTube videos

Given a file with the YouTube video IDs, one per line:

```bash
youtube-dl -f "best[ext=mp4]/best" -o "videos/%(id)s.%(ext)s" --batch-file FILE
```

## Evaluation

```bash
./scripts/evaluate_text_only_baseline.py
```
