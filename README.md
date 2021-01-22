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
    python -m spacy download en_core_web_trf
    export PYTHONPATH=$PWD
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
    ./scripts/generate_dataset.py $VATEX_JSON_FILE_OR_URL > $GENERATED_CSV_FILE
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

In case you need to have data annotated through Amazon Mechanical Turk.

### Preparing the annotation

#### Remove already used instances

If you want to release a new annotation batch, but you want to avoid repeating previous instances, then you can 
create a new file that doesn't consider the ones previously used:

```bash
csvsql \
  --query "select *
           from input as i
             left join already_used as a
               on (i.video_id = a.video_id
                   and i.video_start_time = a.video_start_time
                   and i.video_end_time = a.video_end_time)
           where a.video_id is null" \
  input.csv \
  already_used.csv > output.csv
```

#### Create the annotation input CSV file

```bash
./scripts/generate_annotation_input.py $GENERATED_CSV_FILTERED_FILE > $MTURK_INPUT_CSV_FILE
```

If you just want to take a random portion, do:

```bash
./scripts/generate_annotation_input.py \
    --hit-count $HIT_COUNT \
    $GENERATED_CSV_FILTERED_FILE \
    > $MTURK_INPUT_CSV_FILE
```

### Using the annotation results

Visualize the annotation results:

```bash
./scripts/analyze_annotation_results.py --show-metrics INPUT_CSV_FILE_OR_URL > OUTPUT_TXT_FILE
```

## Download the videos

In case you want to download the video, given a file with one YouTube video ID per line (such as 
`$AVAILABLE_VIDEO_IDS_FILE`):

```bash
youtube-dl -f "best[ext=mp4]/best" -o "videos/%(id)s.%(ext)s" --batch-file FILE
```

## Training

TODO

## Evaluation

T5 text-only baseline:

```bash
./scripts/evaluate_text_only_baseline.py
```

## Woker Bonus

If you want to reward workers with **extra** bonus, prepare a `./scripts/bonus_info.json` file with the following fields:

```json
{
	"workerIds": list[string],
	"BonusAmounts": list[string],
	"Reasons": list[string],
	"UniqueRequestTokens": list[string]
}
```
Then execute the following command

```bash
./scripts/bonus_mturk.py
```

You could refer to:

 https://blog.mturk.com/tutorial-a-beginners-guide-to-crowdsourcing-ml-training-data-with-python-and-mturk-d8df4bdf2977

and 

https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk.html?highlight=sendbonus#MTurk.Client.send_bonus

for more information.

