# LifeQA-MTurk

This repository includes codes related to the LifeQA-MTurk tasks.

## Setup

See [LifeQA-methodology's setup](https://github.com/MichiganNLP/LifeQA-methodology/#setup).

To execute any Python script under [`scripts/`](scripts), run it from the project root directory (where this file is)
and prepend `PYTHONPATH=.` to the command execution (or do `export PYTHONPATH=$PWD` once per Bash session).

## Visualize the annotation results

Run:

```bash
PYTHONPATH=. ./scripts/analyze_annotation_results.py --show-metrics < INPUT_CSV_FILE > OUTPUT_TXT_FILE
```
