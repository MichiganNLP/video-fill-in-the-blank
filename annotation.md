# Annotation

Follow these steps in case you need to have data annotated through Amazon Mechanical Turk.

## Preparing the annotation

### Remove already used instances

If you want to release a new annotation batch, but you want to avoid repeating previous instances, then you can 
create a new file that doesn't consider the ones previously used:

```bash
jq \
  --slurpfile used $ALREADY_USED_JSON_FILE \
  '[ JOIN(INDEX($used[][]; .video_id); .video_id)[] | select(.[1] == null) | .[0] ]' \
  $GENERATED_JSON_FILTERED_FILE \
  > $GENERATED_JSON_FILTERED_FILE
```

You can remove already used instances from multiple files by substituting `$ALREADY_USED_JSON_FILE` from the last
command with: `<(jq --null-input '[inputs] | flatten' $ALREADY_USED_JSON_FILE1 $ALREADY_USED_JSON_FILE2 ...)`.

### Subsample

If you want to select a random sample of certain size (e.g., 1000; first make sure you have at least that many), run:

```bash
jq \
  --compact-output \
  .[] \
  $GENERATED_JSON_FILTERED_FILE \
  | shuf -n 1000 \
  | jq --compact-output --null-input '[inputs]' \
  > $GENERATED_JSON_FILTERED_FILE
```

### Create the annotation input CSV file

```bash
./scripts/generate_annotation_input.py $GENERATED_JSON_FILTERED_FILE > $MTURK_INPUT_CSV_FILE
```

## Previewing the annotation web page

Run:

```bash
python -m http.server
```

Then open
[the annotation page locally](http://localhost:8000/annotation_page/amt_testing_page.html?templatePagePath=annotation.html&dataPath=../$MTURK_INPUT_CSV_FILE)
.

## Visualize the annotation results

```bash
./scripts/analyze_annotation_results.py --show-metrics $ANNOTATION_RESULTS_CSV_FILE_OR_URL > $OUTPUT_TXT_FILE
```

## Prepare for a randomly sampled manual review

```bash
./scripts/prepare_to_review_workers.py $ANNOTATION_RESULTS_CSV_FILE_OR_URL > $OUTPUT_CSV_FILE
```

## Incorporate the review

TODO

## Pay the bonuses

This should be done once all the instances have been annotated as for the bonus type 2 we need to have a HIT 
completely annotated. We could send partial payments over time, however for simplicity we send them altogether at 
the end.

1. First calculate the bonuses:

```bash
./script/calculate_bonuses.py $ANNOTATION_RESULTS_CSV_FILE_OR_URL > $OUTPUT_JSONL_FILE
```

2. Make sure you have the AWS credentials set up. For this, run `aws configure` and set the key ID and secret.
   
3. CAUTION: this operation will really pay and can't be undone. Please, manually review the file 
   before running it. To pay all the bonuses, run:

```bash
tr '\n' '\0' < $OUTPUT_JSONL_FILE | xargs -0 -L 1 aws mturk send-bonus --cli-input-json
```
