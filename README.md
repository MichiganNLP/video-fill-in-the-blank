# FIBER data and code

This repo contains the data and code for the ACL 2022 paper [FIBER: Fill-in-the-Blanks as a Challenging Video 
Understanding Evaluation Framework](https://aclanthology.org/2022.acl-long.209/).

## Download the data

* [train](https://www.dropbox.com/s/lc3e1ave94hz9tu/train.json)
* [validation](https://www.dropbox.com/s/t1dpotaz2sjjtxk/val.json)
* [test](https://www.dropbox.com/s/2nr7kooprjti975/test.json)

The video features are available in [VATEX download page](https://eric-xw.github.io/vatex-website/download.html). 
You should download the train, validation, and public test I3D video features and extract them in some folder (e.g., 
`data/I3D_video_features`).

## Setup

With Conda installed (or preferably [mamba](https://github.com/mamba-org/mamba) for more speed), run:

```bash
conda env create
conda activate lqam
spacy download en_core_web_trf
```

> NB: the scripts mentioned in the rest of this README generally accept many options. Try using the `--help` (or `-h`) 
option or looking at their code for more information.

## Dataset Blank Generation

Follow these steps if you wanna re-generate the blanks' dataset. Note you probably don't have to run this.

1. Generate the blanked captions, for each
[VATEX split JSON file](https://eric-xw.github.io/vatex-website/download.html) (replacing the variables with values):

    ```bash
    ./scripts/generate_dataset_from_vatex.py $VATEX_JSON_FILE_OR_URL > $GENERATED_JSON_FILE
    ```

2. Create a list of the available videos (you first need to set the env var `GOOGLE_API_KEY` that can use the 
   YouTube API):

    ```bash
    jq --raw-output '.[] | .video_id' $GENERATED_JSON_FILE \
      | sed 1d \
      | sort \
      | uniq \
      | ./scripts/list_available_videos.py > $AVAILABLE_VIDEO_IDS_FILE
    ```

3. Filter the JSON file based on the available videos:

    ```bash
    jq \
      --compact-output \
      --slurpfile ids <((echo '[' && sed 's/.*/"&"/g' < $AVAILABLE_VIDEO_IDS_FILE | paste -s -d, - && echo ']') | jq .) \
      '[ JOIN(INDEX($ids[][]; .); .video_id)[] | select(.[1] != null) | .[0] ]' \
      $GENERATED_JSON_FILE > $GENERATED_JSON_FILTERED_FILE
    ```

## Data Annotation

See [Annotation](annotation.md). Note you probably don't need to do this.

## Download the videos

In case you want to download the video (which you likely won't because there are video features already available), 
given a file with one YouTube video ID per line (such as `$AVAILABLE_VIDEO_IDS_FILE`):

```bash
youtube-dl -f "best[ext=mp4]/best" -o "videos/%(id)s.%(ext)s" --batch-file FILE
```

## Training

```bash
./scripts/run_model.py --train
```

## Evaluation

Evaluate the T5 text-only baseline:

```bash
./scripts/run_model.py
```

## Citation

```bibtex
@inproceedings{castro-etal-2022-fill,
    title = "{FIBER}: Fill-in-the-Blanks as a Challenging Video Understanding Evaluation Framework",
    author = "Castro, Santiago  and
      Wang, Ruoyao  and
      Huang, Pingxuan  and
      Stewart, Ian  and
      Ignat, Oana  and
      Liu, Nan  and
      Stroud, Jonathan C.  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.209/",
    pages = "2925--2940"
}
```

## Analyses from the paper

Some analyses from the paper can be found under [`notebooks/`](notebooks), [`scripts/`](scripts), and in [this Google
Colab](https://colab.research.google.com/drive/1aNEg5meD9o8hjewtNO0dvPo55zPfTFXu?usp=sharing).
