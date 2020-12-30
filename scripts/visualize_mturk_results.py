#!/usr/bin/env python
# coding: utf-8
import argparse
import sys

import pandas as pd

from lqam_data import format_answer, hits_to_instances, parse_hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mturk_results_path", metavar="MTURK_RESULTS_FILE", default="-")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_ = sys.stdin if args.mturk_results_path == "-" else args.mturk_results_path
    hits = parse_hits(input_)
    instances = hits_to_instances(hits)

    for id_, instance in instances.items():
        instance["answers"] = {worker_id: [format_answer(answer) for answer in answers]
                               for worker_id, answers in instance["answers"].items()}

        df = pd.DataFrame(instance["answers"].values(),
                          index=pd.Index(instance["answers"].keys(), name="Worker ID"))
        df.columns = [f"Ans. {j + 1}" for j in range(len(df.columns))]
        df[df.isna()] = ""
        # Convert the index into a column. Otherwise, the index name and column names are output in different lines.
        df = df.reset_index()
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
            formatted_question_answers = df.to_string(index=False)

        print(f"""\
ID: {id_}
Question: {instance["question"].replace("[MASK]", "_____")}
Video URL: {instance["video_url"]}
Std. answer: {instance[f"label"]}
Worker answers:
{formatted_question_answers}
""")

    print()
    print("*** HIT-level information ***")
    print()

    for hit_id, hit in hits.items():
        print(f"HIT ID: {hit_id}")
        comments_map = {worker_id: answers["comments"]
                        for worker_id, answers in hit["answers"].items()
                        if answers["comments"]}

        if comments_map:
            print("Comments:")
            for worker_id, comment in comments_map.items():
                print(f"{worker_id:>14}: {comment}")
        else:
            print("No comments.")

        print()


if __name__ == "__main__":
    main()
