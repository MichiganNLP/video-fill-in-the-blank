#!/usr/bin/env python
import argparse
import json
import sys

import pandas as pd

from lqam.annotations import QUESTIONS_PER_HIT
from lqam.util import grouper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("questions_path", metavar="QUESTIONS_CSV_FILE", nargs="?", default="-")

    parser.add_argument("--hit-count", type=int)

    parser.add_argument("--questions-per-hit", type=int, default=QUESTIONS_PER_HIT)

    parser.add_argument("--already-used-indices-path", default="already_used_annotation_indices.json")
    parser.add_argument("--already-used-indices-split", choices=["train", "val", "test"], default="val")

    args = parser.parse_args()

    args.input = sys.stdin if args.questions_path == "-" else args.questions_path

    args.question_count = args.hit_count * args.questions_per_hit if args.hit_count else None

    return args


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    assert args.question_count is None or len(df) >= args.question_count

    with open(args.already_used_indices_path) as file:
        already_used_indices = set(json.load(file)[args.already_used_indices_split])

    selected_indices = [i for i in range(args.question_count) if i not in already_used_indices]
    assert (args.question_count is None and len(selected_indices) % args.questions_per_hit) \
           or len(selected_indices) == args.question_count

    hits = [
        {
            f"video{i}_id": df.iloc[instance_i]["videoID"][:-14],
            f"video{i}_start_time": int(df.iloc[instance_i]["videoID"][-13:-7]),
            f"video{i}_end_time": int(df.iloc[instance_i]["videoID"][-6:]),
            f"question{i}": df.iloc[instance_i]["masked caption"].replace("<extra_id_0>", "[MASK]"),
            f"label{i}": df.iloc[instance_i]["label"],
        }
        for hit_instance_indices in grouper(args.questions_per_hit, selected_indices)
        for i, instance_i in enumerate(hit_instance_indices, start=1)
    ]

    print(pd.DataFrame(hits).to_csv(index=False))


if __name__ == "__main__":
    main()
