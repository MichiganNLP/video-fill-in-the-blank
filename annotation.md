# Annotation

Follow these steps in case you need to have data annotated through Amazon Mechanical Turk.

## 1. Prepare the annotation

### 1.1 Remove already used instances

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

### 1.2 Subsample

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

### 1.3 Create the annotation input CSV file

Run:

```bash
./scripts/generate_annotation_input.py $GENERATED_JSON_FILTERED_FILE > $MTURK_INPUT_CSV_FILE
```

## 2. Previewing the annotation web page

Run:

```bash
python -m http.server
```

Then open
[the annotation page locally](http://localhost:8000/annotation_page/amt_testing_page.html?templatePagePath=annotation.html&dataPath=../$MTURK_INPUT_CSV_FILE)
.

## 3. Annotate with AMT

Use Amazon Mechanical Turk to annotate the instances and download the resulting CSV file.

## 4. Visualize the annotation results

Run:

```bash
./scripts/analyze_annotation_results.py --show-metrics $ANNOTATION_RESULTS_CSV_FILE_OR_URL > $OUTPUT_TXT_FILE
```

## 5. Prepare for a randomly sampled manual review

Run:

```bash
./scripts/prepare_to_review_workers.py $ANNOTATION_RESULTS_CSV_FILE_OR_URL > $OUTPUT_CSV_FILE
```

## 6. Incorporate the review

Run:

```bash
./scripts/generate_review_results.py $ANNOTATION_RESULTS_CSV_FILE_OR_URL $REVIEWED_CSV_FILE
```

Then create a copy of `$ANNOTATION_RESULTS_CSV_FILE_OR_URL`, and reject the workers that are part of the output. For 
this, for each rejected one, put the following reason in the "Reject" column:

> Unfortunately we are rejecting the assignment as many of the answers are wrong. If you have questions, please contact
> us.

For the rest, put an "x" in the "Approve" column. Then upload it to AMT.

## 7. Generate the dataset from the annotations

Once the review has been sent (and processed) by AMT, download the results again, and run:

```bash
./scripts/generate_dataset_from_annotations.py $ANNOTATION_RESULTS_CSV_FILE_OR_URL > $DATASET_JSON_FILE
```

You should replace the dataset file in the Dropbox folder with this new one.

## 8. Pay the bonuses

This should be done once all the instances have been annotated as for the bonus type 2 we need to have a HIT 
completely annotated. We could send partial payments over time, however for simplicity we send them altogether at 
the end.

1. First calculate the bonuses:

```bash
./scripts/calculate_bonuses.py $ANNOTATION_RESULTS_CSV_FILE_OR_URL > $OUTPUT_JSONL_FILE
```

2. Make sure you have the AWS credentials set up. For this, run `aws configure` and set the key ID and secret.
   
3. CAUTION: this operation will really pay and can't be undone. Please, manually review the file 
   before running it. To pay all the bonuses, run:

```bash
tr '\n' '\0' < $OUTPUT_JSONL_FILE | xargs -0 -L 1 aws mturk send-bonus --cli-input-json
```

If you want to pay only some bonuses from the file, you can run the same command, and the already paid ones are 
going to be skipped. However, it will make the command fail. A trick is to make every command spawn by `xargs` to not 
fail by using `sh -c` and appending `|| true`. However, by using `sh`, the `xargs` args are going to be expanded. So 
then they need to be escaped. You can put them in between single quotes (because double quotes strings may allow 
shell expansions). Finally, for this, we need to escape the single quotes. Doing `\'` doesn't work. A way of 
escaping is by ending the single quotes, and immediately adding a single quote in between double quotes, then 
continue with the single-quoted string. All in all, you can do: 

```bash
tr '\n' '\0' < $OUTPUT_JSONL_FILE \
  | sed "s/'/'\"'\"'/g" \
  | xargs -0 -L 1 -I{} bash -c "aws mturk send-bonus --cli-input-json '{}' || true"
```
