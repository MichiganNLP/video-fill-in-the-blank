#!/usr/bin/env python
import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from lqam.annotations.metrics import compute_answer_level_metrics
from lqam.annotations.postprocessing import format_answer, hits_to_instances, \
    instance_can_be_used_in_annotated_dataset, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path_or_url", metavar="ANNOTATION_RESULTS_FILE_OR_URL", nargs="?",
                        default="-")
    parser.add_argument("--compute-metrics", action="store_true")
    args = parser.parse_args()

    args.input = sys.stdin if args.annotation_results_path_or_url == "-" \
        else cached_path(args.annotation_results_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    hits = parse_hits(args.input)
    instances = hits_to_instances(hits)

    metric_value_list = defaultdict(list)
    worker_stats = defaultdict(lambda: defaultdict(int))

    for instance_id, instance in instances.items():
        instance["answers_by_worker"] = {worker_id: [format_answer(answer) for answer in answers]
                                         for worker_id, answers in instance["answers_by_worker"].items()}

        df = pd.DataFrame(instance["answers_by_worker"].values(),
                          index=pd.Index(instance["answers_by_worker"].keys(), name="Worker ID"))
        df.columns = [f"Ans. {j + 1}" for j in range(len(df.columns))]
        df[df.isna()] = ""

        pd.options.display.float_format = lambda x: f"{x: >3.0f}"

        instance_can_be_used = instance_can_be_used_in_annotated_dataset(instance)

        if args.compute_metrics and instance_can_be_used:
            # TODO: compute metrics after filtering NP.

            worker_answer_metrics = compute_answer_level_metrics(instance["question"], instance["answers_by_worker"],
                                                                 instance["label"])

            metric_keys = next(iter(next(iter(worker_answer_metrics.values())).values()))

            worker_first_answer_metrics = [
                # There could be a missing worker if all their answers are empty after normalizing.
                next(iter(worker_answer_metrics.get(worker_id, {"": defaultdict(lambda: np.nan)}).values()))
                for worker_id in instance["answers_by_worker"]
            ]

            worker_first_answer_metric_arrays = {
                k: np.asarray([metrics[k] for metrics in worker_first_answer_metrics])
                for k in metric_keys
            }

            df.insert(0, "FF1", worker_first_answer_metric_arrays["f1"] * 100)
            df.insert(1, "FEM", worker_first_answer_metric_arrays["em"])

            ff1_mean = np.nanmean(worker_first_answer_metric_arrays["f1"])
            fem_mean = np.nanmean(worker_first_answer_metric_arrays["em"])

            aggregated_metrics_str = f"\nAvg.: FF1 {ff1_mean * 100:.0f}, FEM {fem_mean * 100:.0f}"

            std_answer_metrics = next(iter(worker_answer_metrics["std_answer"].values()))
            std_answer_metrics_str = (f" (F1 {std_answer_metrics['f1'] * 100:.0f},"
                                      f" EM {std_answer_metrics['em']})")

            for worker_id, answer_metrics in worker_answer_metrics.items():
                worker_stats[worker_id]["questions"] += 1
                worker_stats[worker_id]["answers"] += len(answer_metrics)
                worker_stats[worker_id]["total_ff1"] += next(iter(answer_metrics.values()))["f1"]
                worker_stats[worker_id]["total_f1"] += sum(m["f1"] for m in answer_metrics.values())
                worker_stats[worker_id]["total_fem"] += next(iter(answer_metrics.values()))["em"]
                worker_stats[worker_id]["total_em"] += sum(m["em"] for m in answer_metrics.values())
                worker_stats[worker_id]["total_np"] += sum(m["np"] for m in answer_metrics.values())

            answer_df = pd.DataFrame(((worker_id, answer, m["f1"] * 100, m["em"], m["np"])
                                      for worker_id, answer_metrics in worker_answer_metrics.items()
                                      for answer, m in answer_metrics.items()),
                                     columns=["Worker ID", "Answer", "F1", "EM", "NP?"])

            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
                answer_level_metrics_str = f"\nAnswer-level metrics:\n{answer_df.to_string(index=False)}"

            for k, v in worker_first_answer_metric_arrays.items():
                metric_value_list[k].append(torch.from_numpy(v))
        else:
            std_answer_metrics_str = ""
            aggregated_metrics_str = "\n\nWARNING: this question wouldn't be used in a final dataset (i.e., " \
                                     "no valid answers or unavailable video). Ignored for the metrics computation." \
                if args.compute_metrics and not instance_can_be_used else ""
            answer_level_metrics_str = ""

        # Convert the index into a column. Otherwise, the index name and column names are output in different lines.
        df = df.reset_index()

        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
            formatted_question_answers = df.to_string(index=False)

        print(f"""\
ID: {instance_id}
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

        worker_df = pd.DataFrame.from_dict({
            worker_id: {
                "Q": w_stats["questions"],
                "A/Q": w_stats["answers"] / w_stats["questions"],
                "FF1": 100 * w_stats["total_ff1"] / w_stats["questions"],
                "F1": 100 * w_stats["total_f1"] / w_stats["answers"],
                "FEM": 100 * w_stats["total_fem"] / w_stats["questions"],
                "EM": 100 * w_stats["total_em"] / w_stats["answers"],
                "NP?": 100 * w_stats["total_np"] / w_stats["answers"],
            }
            for worker_id, w_stats in worker_stats.items()
        }, orient="index").sort_values(["NP?", "EM", "F1"], ascending=False)
        worker_df.index.name = "Worker ID"
        worker_df = worker_df.reset_index()

        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0):
            print(worker_df.to_string(index=False))

    if args.compute_metrics:
        print()
        print("*** Aggregated metrics ***")
        print()

        print(f"Questions: {len(instances)}")
        print(f"Questions that can be used: {len(next(iter(metric_value_list.values())))}")

        # Note the questions may have a different number of workers because it may be an unfinished annotation.
        ff1s_matrix = pad_sequence(metric_value_list["f1"], batch_first=True, padding_value=np.nan).numpy()
        ff1s_questions = np.nanmean(ff1s_matrix, axis=1)  # Because it's macro avg, we first avg each question.

        fems_matrix = pad_sequence(metric_value_list["em"], batch_first=True, padding_value=np.nan).numpy()
        fems_questions = np.nanmean(fems_matrix, axis=1)

        print("Question-level workers' first answer macro avg. F1 (FF1):"
              f" {100 * ff1s_questions.mean():.1f}% +/- {100 * ff1s_questions.std():.1f}%")
        print("Question-level workers' first answer macro avg. EM (FEM):"
              f" {100 * fems_questions.mean():.1f}% +/- {100 * fems_questions.std():.1f}%")

        if worker_stats:
            total_stats = {k: sum(w_stats[k] for w_stats in worker_stats.values())
                           for k in next(iter(worker_stats.values()))}
            f1_mean = total_stats["total_f1"] / total_stats["answers"]
            em_mean = total_stats["total_em"] / total_stats["answers"]
            f1_std_dev = np.std([w_stats["total_f1"] / w_stats["answers"] for w_stats in worker_stats.values()])
            em_std_dev = np.std([w_stats["total_em"] / w_stats["answers"] for w_stats in worker_stats.values()])

            print(f"Avg. answers per question: {total_stats['answers'] / total_stats['questions']:.2f}")
            print(f"Answer-level avg. F1 Score: {100 * f1_mean:.1f}% +/- {100 * f1_std_dev:.1f}%")
            print(f"Answer-level avg. Exact Match (EM): {100 * em_mean:.1f}% +/- {100 * em_std_dev:.1f}%")
            print(f"Answer-level avg. Noun Phrases (NP): {100 * total_stats['total_np'] / total_stats['answers']:.1f}%")


if __name__ == "__main__":
    main()
