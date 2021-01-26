#!/usr/bin/env python
import argparse
import sys

import pandas as pd

from lqam.annotations import QUESTIONS_PER_HIT
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path
from lqam.util.iterable_utils import chunks


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("questions_path_or_url", metavar="QUESTIONS_CSV_FILE_OR_URL", nargs="?", default="-")
    parser.add_argument("--questions-per-hit", type=int, default=QUESTIONS_PER_HIT)
    args = parser.parse_args()

    args.input = sys.stdin if args.questions_path_or_url == "-" else cached_path(args.questions_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    hits_df = pd.DataFrame([
        {
            k: v
            for i, row in enumerate(hit_rows, start=1)
            for k, v in [(f"video{i}_id", row.video_id),
                         (f"video{i}_start_time", row.video_start_time),
                         (f"video{i}_end_time", row.video_end_time),
                         (f"question{i}", row.masked_caption),
                         (f"label{i}", row.label)]
        }
        for hit_rows in chunks(df.itertuples(), args.questions_per_hit)
    ])

    print(hits_df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
