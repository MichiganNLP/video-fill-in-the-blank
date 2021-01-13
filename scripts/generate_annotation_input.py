#!/usr/bin/env python
import argparse
import itertools
import json
import sys

import pandas as pd

from lqam.annotations import QUESTIONS_PER_HIT
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path
from lqam.util.iterable_utils import chunks


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()

    parser.add_argument("questions_path_or_url", metavar="QUESTIONS_CSV_FILE_OR_URL", nargs="?", default="-")

    parser.add_argument("--hit-count", type=int)

    parser.add_argument("--questions-per-hit", type=int, default=QUESTIONS_PER_HIT)

    parser.add_argument("--already-used-indices-path")
    parser.add_argument("--already-used-indices-split", choices=["train", "val", "test"])

    args = parser.parse_args()

    args.input = sys.stdin if args.questions_path_or_url == "-" else cached_path(args.questions_path_or_url)

    args.question_count = args.hit_count * args.questions_per_hit if args.hit_count else None

    return args


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    assert args.question_count is None or len(df) >= args.question_count

    if args.already_used_indices_path:
        assert args.already_used_indices_split

        with open(args.already_used_indices_path) as file:
            already_used_indices = set(json.load(file)[args.already_used_indices_split])
    else:
        already_used_indices = frozenset()

    selected_indices = []
    for i in itertools.count():
        if len(selected_indices) == (args.question_count or len(df)):
            break
        if i not in already_used_indices:
            selected_indices.append(i)

    assert (args.question_count is None and len(selected_indices) % args.questions_per_hit) \
           or len(selected_indices) == args.question_count

    hits = [
        {
            f"video{i}_id": df.iloc[instance_i]["videoID"][:-14],
            f"video{i}_start_time": int(df.iloc[instance_i]["videoID"][-13:-7]),
            f"video{i}_end_time": int(df.iloc[instance_i]["videoID"][-6:]),
            f"question{i}": df.iloc[instance_i]["masked caption"],
            f"label{i}": df.iloc[instance_i]["label"],
        }
        for hit_instance_indices in chunks(selected_indices, args.questions_per_hit)
        for i, instance_i in enumerate(hit_instance_indices, start=1)
    ]

    print(pd.DataFrame(hits).to_csv(index=False))


if __name__ == "__main__":
    main()
