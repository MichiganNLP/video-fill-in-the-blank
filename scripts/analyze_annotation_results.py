#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from lqam.annotations.metrics import compute_annotation_metrics, compute_answer_level_annotation_metrics
from lqam.annotations.postprocessing import format_answer, hits_to_instances, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path_or_url", metavar="ANNOTATION_RESULTS_FILE_OR_URL", nargs="?",
                        default="-")

    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--ignore-zero-scores", action="store_true")

    args = parser.parse_args()

    assert not args.ignore_zero_scores or args.compute_metrics, "The flag --ignore-zero-scores needs the flag " \
                                                                "--compute-metrics to be specified as well."

    args.input = sys.stdin if args.annotation_results_path_or_url == "-" \
        else cached_path(args.annotation_results_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    hits = parse_hits(args.input)
    instances = hits_to_instances(hits)

    worker_stats = defaultdict(lambda: defaultdict(int))

    ff1s_list = []

    for id_, instance in instances.items():
        instance["answers_by_worker"] = {worker_id: [format_answer(answer) for answer in answers]
                                         for worker_id, answers in instance["answers_by_worker"].items()}

        df = pd.DataFrame(instance["answers_by_worker"].values(),
                          index=pd.Index(instance["answers_by_worker"].keys(), name="Worker ID"))
        df.columns = [f"Ans. {j + 1}" for j in range(len(df.columns))]
        df[df.isna()] = ""

        pd.options.display.float_format = lambda x: f"{x: >3.0f}"

        there_are_answers = any(worker_answers for worker_answers in instance["answers_by_worker"].values())

        if args.compute_metrics and there_are_answers:
            ff1s, precisions, recalls, decision_scores, (
                std_ff1, std_precision, std_recall, std_decision_score), ignored_workers = compute_annotation_metrics(
                instance["answers_by_worker"].values(), instance["label"], args.ignore_zero_scores)
            df.insert(0, "FF1", ff1s * 100)
            df.insert(1, "Pre", precisions * 100)
            df.insert(2, "Rec", recalls * 100)
            df.insert(3, "Dec", decision_scores * 100)

            aggregated_metrics_str = (f"\nAvg.: FF1 {ff1s.mean() * 100:.0f}, Pre {precisions.mean() * 100:.0f}, Rec"
                                      f" {recalls.mean() * 100:.0f}, Dec {decision_scores.mean() * 100:.0f}")

            std_answer_metrics_str = (f" (FF1 {std_ff1 * 100:.0f}, Pre {std_precision * 100:.0f}, Rec"
                                      f" {std_recall * 100:.0f}, Dec {std_decision_score * 100:.0f})")

            answer_level_metrics = compute_answer_level_annotation_metrics(instance["question"],
                                                                           instance["answers_by_worker"],
                                                                           instance["label"], ignored_workers)

            for worker_id, answer_stats in answer_level_metrics.items():
                worker_stats[worker_id]["questions"] += 1
                worker_stats[worker_id]["answers_by_worker"] += len(answer_stats)
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

            ff1s_list.append(torch.from_numpy(ff1s))
        else:
            std_answer_metrics_str = ""
            aggregated_metrics_str = "\n\nWARNING: this question has no answers. Ignored for the metrics computation." \
                if args.compute_metrics and not there_are_answers else ""
            answer_level_metrics_str = ""

        # Convert the index into a column. Otherwise, the index name and column names are output in different lines.
        df = df.reset_index()

        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
            formatted_question_answers = df.to_string(index=False)

        print(f"""\
ID: {id_}
Question: {instance["question"]}
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
                        for worker_id, answers in hit["answers_by_worker"].items()
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

    if args.compute_metrics:
        print()
        print("*** Aggregated metrics ***")
        print()

        # Note the questions may have a different number of workers because it may be an unfinished annotation.
        ff1s_matrix = pad_sequence(ff1s_list, batch_first=True, padding_value=np.nan).numpy()
        ff1s_questions = np.nanmean(ff1s_matrix, axis=1)  # Because it's macro avg, we first avg each question.

        print(f"Question-level workers' first answer macro avg. F1 (FF1): {100 * ff1s_questions.mean():.0f}%")

        if worker_stats:
            total_stats = {k: sum(w_stats[k] for w_stats in worker_stats.values())
                           for k in next(iter(worker_stats.values()))}

            print(f"Avg. answers per question: {total_stats['answers'] / total_stats['questions']:.2f}")
            print(f"Answer-level avg. F1 Score: {100 * total_stats['total_f1'] / total_stats['answers']:.0f}%")
            print(f"Answer-level avg. Exact Match (EM): {100 * total_stats['total_em'] / total_stats['answers']:.0f}%")
            print(f"Answer-level avg. Noun Phrases (NP): {100 * total_stats['total_np'] / total_stats['answers']:.0f}%")


if __name__ == "__main__":
    main()
