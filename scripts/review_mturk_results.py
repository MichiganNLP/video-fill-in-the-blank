#!/usr/bin/env python
import argparse
import sys
import textwrap

from lqam_data import hits_to_instances, parse_hits
from lqam_data.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mturk_results_path", metavar="MTURK_RESULTS_FILE", default="-")
    parser.add_argument("--ignore-zero-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_ = sys.stdin if args.mturk_results_path == "-" else args.mturk_results_path
    instances = hits_to_instances(parse_hits(input_))
    print(textwrap.dedent("""\
    Format:

    Question
    Video

    Precision Recall Decision score (0.67 * precision + recall) Standard answer

    Precision Recall Decision score (0.67 * precision + recall) [Worker 1 answers]
    Precision Recall Decision score (0.67 * precision + recall) [Worker 2 answers]
    …

    ---
    """))

    for instance in instances.values():
        ff1s, precisions, recalls, decision_scores, (std_precision, std_recall,
                                                     std_decision_score) = compute_metrics(instance["answers"].values(),
                                                                                           instance["label"],
                                                                                           args.ignore_zero_scores)

        print(textwrap.dedent(f"""\
                {instance["question"].replace("[MASK]", "_____")}
                {instance["video_url"]}
    
                pre rec dec
                {std_precision * 100: >3.0f} {std_recall * 100: >3.0f} {std_decision_score * 100: >3.0f}\
        {instance["label"]}
                """))

        for ff1, precision, recall, decision_score, worker_answers in zip(ff1s, precisions, recalls, decision_scores,
                                                                          instance["answers"].values()):
            print(f"{ff1 * 100: >3.0f} {precision * 100: >3.0f} {recall * 100: >3.0f}"
                  f" {decision_score * 100: >3.0f} {worker_answers}")

        print()
        print("---")
        print()


if __name__ == "__main__":
    main()
