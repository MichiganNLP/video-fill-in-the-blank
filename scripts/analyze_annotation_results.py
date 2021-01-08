#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
from collections import defaultdict

import pandas as pd

from lqam_data import format_answer, hits_to_instances, parse_hits
from lqam_data.metrics import compute_answer_level_metrics, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_results_path", metavar="ANNOTATION_RESULTS_FILE", nargs="?", default="-")

    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--ignore-zero-scores", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert not args.ignore_zero_scores or args.compute_metrics, "The flag --ignore-zero-scores needs the flag " \
                                                                "--compute-metrics to be specified as well."

    input_ = sys.stdin if args.annotation_results_path == "-" else args.annotation_results_path
    hits = parse_hits(input_)
    instances = hits_to_instances(hits)

    worker_stats = defaultdict(lambda: defaultdict(int))

    for id_, instance in instances.items():
        instance["answers"] = {worker_id: [format_answer(answer) for answer in answers]
                               for worker_id, answers in instance["answers"].items()}

        df = pd.DataFrame(instance["answers"].values(),
                          index=pd.Index(instance["answers"].keys(), name="Worker ID"))
        df.columns = [f"Ans. {j + 1}" for j in range(len(df.columns))]
        df[df.isna()] = ""

        pd.options.display.float_format = lambda x: f"{x: >3.0f}"

        if args.compute_metrics:
            ff1s, precisions, recalls, decision_scores, (
                std_ff1, std_precision, std_recall, std_decision_score), ignored_workers = compute_metrics(
                instance["answers"].values(), instance["label"], args.ignore_zero_scores)
            df.insert(0, "FF1", ff1s * 100)
            df.insert(1, "Pre", precisions * 100)
            df.insert(2, "Rec", recalls * 100)
            df.insert(3, "Dec", decision_scores * 100)

            aggregated_metrics_str = (f"\nAvg.: FF1 {ff1s.mean() * 100:.0f}, Pre {precisions.mean() * 100:.0f}, Rec"
                                      f" {recalls.mean() * 100:.0f}, Dec {decision_scores.mean() * 100:.0f}")

            std_answer_metrics_str = (f" (FF1 {std_ff1 * 100:.0f}, Pre {std_precision * 100:.0f}, Rec"
                                      f" {std_recall * 100:.0f}, Dec {std_decision_score * 100:.0f})")

            answer_level_metrics = compute_answer_level_metrics(instance["answers"], instance["label"], ignored_workers)

            for worker_id, answer_stats in answer_level_metrics.items():
                worker_stats[worker_id]["questions"] += 1
                worker_stats[worker_id]["answers"] += len(answer_stats)
                worker_stats[worker_id]["total_ff1"] += next(iter(answer_stats.values()))["f1"]
                worker_stats[worker_id]["total_f1"] += sum(m["f1"] for m in answer_stats.values())
                worker_stats[worker_id]["total_em"] += sum(m["em"] for m in answer_stats.values())
                worker_stats[worker_id]["total_np"] += sum(m["np"] for m in answer_stats.values())

            answer_df = pd.DataFrame(((w, a, m["f1"] * 100, m["em"], m["np"])
                                      for w, aa in answer_level_metrics.items()
                                      for a, m in aa.items()),
                                     columns=["Worker ID", "Answer", "F1", "EM", "NP?"])

            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
                answer_level_metrics_str = f"\nAnswer-level metrics:\n{answer_df.to_string(index=False)}"
        else:
            std_answer_metrics_str = ""
            aggregated_metrics_str = ""
            answer_level_metrics_str = ""

        # Convert the index into a column. Otherwise, the index name and column names are output in different lines.
        df = df.reset_index()

        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
            formatted_question_answers = df.to_string(index=False)

        print(f"""\
ID: {id_}
Question: {instance["question"].replace("[MASK]", "_____")}
Video URL: {instance["video_url"]}
Std. answer: {instance[f"label"]}{std_answer_metrics_str}
Worker answers:
{formatted_question_answers}{aggregated_metrics_str}{answer_level_metrics_str}
""")

    print()
    print("*** HIT-level information ***")
    print()

    for hit_id, hit in hits.items():
        print(f"HIT ID: {hit_id}")

        comments_map = {worker_id: answers["comments"]
                        for worker_id, answers in hit["answers"].items()
                        if answers["comments"]}
        print("Comments:" if comments_map else "No comments.")
        for worker_id, comment in comments_map.items():
            print(f"{worker_id:>14}: {comment}")

        print()

    if worker_stats:
        print()
        print("*** Worker-level metrics ***")
        print()

        summary_worker_stats = {}
        for worker_id in worker_stats:
            summary_worker_stats[worker_id] = {
                "Q": worker_stats[worker_id]["questions"],
                "A/Q": worker_stats[worker_id]["answers"] / worker_stats[worker_id]["questions"],
                "FF1": 100 * worker_stats[worker_id]["total_ff1"] / worker_stats[worker_id]["questions"],
                "F1": 100 * worker_stats[worker_id]["total_f1"] / worker_stats[worker_id]["answers"],
                "EM": 100 * worker_stats[worker_id]["total_em"] / worker_stats[worker_id]["answers"],
                "NP?": 100 * worker_stats[worker_id]["total_np"] / worker_stats[worker_id]["answers"],
            }

        worker_df = pd.DataFrame.from_dict(summary_worker_stats, orient="index").sort_values(["NP?", "EM", "F1"],
                                                                                             ascending=False)
        worker_df.index.name = "Worker ID"
        worker_df = worker_df.reset_index()

        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
            print(worker_df.to_string(index=False))

    # TODO: worker info per question.
    # TODO: aggregated total stats


if __name__ == "__main__":
    main()
