# Annotation

In case you need to have data annotated through Amazon Mechanical Turk.

## Preparing the annotation

### Remove already used instances

If you want to release a new annotation batch, but you want to avoid repeating previous instances, then you can 
create a new file that doesn't consider the ones previously used:

```bash
csvsql \
  --query "select *
           from filename as i
             left join already_used_filename as a
               on (i.video_id = a.video_id
                   and i.video_start_time = a.video_start_time
                   and i.video_end_time = a.video_end_time)
           where a.video_id is null" \
  $GENERATED_CSV_FILTERED_FILE \
  --no-inference \
  $ALREADY_USED_CSV_FILE > $GENERATED_CSV_FILTERED_FILE
```

You can remove already used instances from multiple files by substituting `$ALREADY_USED_CSV_FILE` from the last
command with: `<(csvstack $ALREADY_USED_CSV_FILE1 $ALREADY_USED_CSV_FILE2 ...)`.

### Subsample

If you want to select a random sample of certain size (e.g., 1000; first make sure you have at least that many), run:

```bash
csvsql \
  --query "select *
           from filename
           order by random()
           limit 1000" \
  --no-inference \
  $GENERATED_CSV_FILTERED_FILE > $GENERATED_CSV_FILTERED_FILE
```

If you want to both remove some already used instances and subsample, note you can combine the last 2 commands.

### Create the annotation input CSV file

```bash
./scripts/generate_annotation_input.py $GENERATED_CSV_FILTERED_FILE > $MTURK_INPUT_CSV_FILE
```

If you just want to take a random portion, do:

```bash
./scripts/generate_annotation_input.py $GENERATED_CSV_FILTERED_FILE > $MTURK_INPUT_CSV_FILE
```

## Previewing the annotation web page

Run:

```bash
python -m http.server
```

Then open
http://localhost:8000/annotation_page/amt_testing_page.html?templatePagePath=annotation.html&dataPath=../$MTURK_INPUT_CSV_FILE

## Using the annotation results

Visualize the annotation results:

```bash
./scripts/analyze_annotation_results.py --show-metrics ANNOTATION_RESULTS_CSV_FILE_OR_URL > OUTPUT_TXT_FILE
```

## Prepare for a randomly sampled manual review

```bash
./scripts/prepare_to_review_workers.py ANNOTATION_RESULTS_CSV_FILE_OR_URL > OUTPUT_CSV_FILE
```

## Paying bonuses

If you want to pay bonuses, prepare a CSV file with the fields `uuid`, `worker_id`, `bonus_amount`, `assignment_id`, and 
`reason`. The `uuid` field is used to uniquely identify this bonus payment. It's optional, but useful, so it's
recommended that you generate one randomly. Then run:

```bash
./scripts/pay_mturk_bonus.py BONUS_CSV_FILE
```

Use the `--production` flag to use it for production (real money) and not in the sandbox mode.
