#!/usr/bin/env python
import argparse
import random
import sys

import pandas as pd
import pytorch_lightning as pl

from lqam.annotations import MIN_ACCEPTABLE_ANSWERS_PER_QUESTION, REVIEW_SAMPLE_SIZE_PER_WORKER
from lqam.annotations.postprocessing import compute_instances_by_worker_id, hits_to_instances, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path_or_url", metavar="ANNOTATION_RESULTS_FILE_OR_URL", nargs="?",
                        default="-")
    parser.add_argument("--sample-size", type=int, default=REVIEW_SAMPLE_SIZE_PER_WORKER)
    parser.add_argument("--min-good-answers-per-question", type=float, default=MIN_ACCEPTABLE_ANSWERS_PER_QUESTION)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    args.annotation_results_path = sys.stdin if args.annotation_results_path_or_url == "-" \
        else cached_path(args.annotation_results_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed)

    hits = parse_hits(args.annotation_results_path, include_accepted=False)
    instances = hits_to_instances(hits)
    instances_by_worker_id = compute_instances_by_worker_id(instances, compute_np_answers=True)

    for worker_id, worker_instances in list(instances_by_worker_id.items()):
        np_answers_per_question = \
            sum(len(instance["np_answers"]) for instance in worker_instances) / len(worker_instances)
        if np_answers_per_question < args.min_good_answers_per_question:
            del instances_by_worker_id[worker_id]

    for worker_id in instances_by_worker_id:
        if len(instances_by_worker_id[worker_id]) > args.sample_size:
            instances_by_worker_id[worker_id] = random.sample(instances_by_worker_id[worker_id], args.sample_size)

    df = pd.DataFrame([
        {
            "worker_id": worker_id,
            **{k: instance[k] for k in ["video_id", "video_start_time", "video_end_time", "video_url", "question",
                                        "label"]},
            "answer": answer,
        }
        for worker_id, worker_instances in instances_by_worker_id.items()
        for instance in worker_instances
        for answer in instance["answers"]
    ])

    df["reviewer"] = ""
    df["correct?"] = ""

    print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
