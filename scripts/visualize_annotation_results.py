#!/usr/bin/env python
# coding: utf-8
import argparse
import sys

import pandas as pd

from lqam_data import format_answer, hits_to_instances, parse_hits
from lqam_data.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mturk_results_path", metavar="MTURK_RESULTS_FILE", default="-")

    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--ignore-zero-scores", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert not args.ignore_zero_scores or args.compute_metrics, "The flag --ignore-zero-scores needs the flag " \
                                                                "--compute-metrics to be specified as well."

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

        if args.compute_metrics:
            ff1s, precisions, recalls, decision_scores, (
                std_ff1, std_precision, std_recall, std_decision_score) = compute_metrics(instance["answers"].values(),
                                                                                          instance["label"],
                                                                                          args.ignore_zero_scores)
            df.insert(0, "FF1", ff1s)
            df.insert(1, "Pre", precisions)
            df.insert(2, "Rec", recalls)
            df.insert(3, "Dec", decision_scores)
            pd.options.display.float_format = lambda x: f"{x * 100: >3.0f}"

            std_answer_metrics_str = (f"(FF1 {std_ff1 * 100:.0f}, Pre {std_precision * 100:.0f}, Rec"
                                      f" {std_recall * 100:.0f}, Dec {std_decision_score * 100:.0f})")
        else:
            std_answer_metrics_str = ""

        # Convert the index into a column. Otherwise, the index name and column names are output in different lines.
        df = df.reset_index()
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
            formatted_question_answers = df.to_string(index=False)

        print(f"""\
ID: {id_}
Question: {instance["question"].replace("[MASK]", "_____")}
Video URL: {instance["video_url"]}
Std. answer: {instance[f"label"]} {std_answer_metrics_str}
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
