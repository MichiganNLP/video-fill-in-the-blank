# LifeQA Methodology

This repo contains the annotation scheme, results and methods for the LifeQA project "qgen".

## Setup

1. If you are a macOS user, then you need to comment out the CUDA-related lines in `environment.yml`:

    ```yaml
    dependencies:
      # ...
      - cudatoolkit=...
      - cudnn=...
    ```

2. With Conda installed:

    ```bash
    conda env create
    conda activate lqam
    spacy download en_core_web_trf
    export PYTHONPATH=.
    ```

3. Put the data under `data/`. For example, in Great Lakes it can be a symlink:

    ```bash
    ln -s /scratch/mihalcea_root/mihalcea1/shared_data/qgen/VATEX data
    ```

NB: the scripts mentioned in the rest of this README generally accept many options. Try using the `--help` (or `-h`) 
option or looking at their code for more information.

## Dataset

You can skip these steps if the data is already generated.

1. Generate the blanked captions, for each
[VATEX split JSON file](https://eric-xw.github.io/vatex-website/download.html) (replacing the variables with values):

    ```bash
    ./scripts/generate_dataset_from_vatex.py $VATEX_JSON_FILE_OR_URL > $GENERATED_CSV_FILE
    ```

2. Create a list of the available videos:

    ```bash
    csvcut -c video_id $GENERATED_CSV_FILE \
      | sed 1d \
      | sort \
      | uniq \
      | ./scripts/list_available_videos.py > $AVAILABLE_VIDEO_IDS_FILE
    ```

3. Filter the CSV file based on the available videos:

    ```bash
    csvjoin \
      -c video_id \
      $GENERATED_CSV_FILE \
      <(echo video_id && cat $AVAILABLE_VIDEO_IDS_FILE) \
      > $GENERATED_CSV_FILTERED_FILE
    ```

## Annotation

See [Annotation](annotation.md).

## Download the videos

In case you want to download the video, given a file with one YouTube video ID per line (such as 
`$AVAILABLE_VIDEO_IDS_FILE`):

```bash
youtube-dl -f "best[ext=mp4]/best" -o "videos/%(id)s.%(ext)s" --batch-file FILE
```

## Training

```bash
./scripts/run_model.py --train
```

See the available options using the flag `--help`.

## Evaluation

Evaluate the T5 text-only baseline:

```bash
./scripts/run_model.py
```

See the available options using the flag `--help`.
