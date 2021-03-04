#!/usr/bin/env python
import argparse
import random
import sys
from collections import defaultdict
from typing import Iterable

import pandas as pd
import pytorch_lightning as pl

from lqam.annotations import AUTO_APPROVE_WORKER_ID_DENY_LIST, MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_1, \
    MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_2, \
    MIN_QUESTION_COUNT_FOR_THRESHOLD_2, REVIEW_SAMPLE_SIZE_PER_WORKER
from lqam.annotations.postprocessing import compute_instances_by_worker_id, hits_to_instances, parse_hits
from lqam.core.metrics import normalize_answer
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path_or_url", metavar="ANNOTATION_RESULTS_FILE_OR_URL", nargs="?",
                        default="-")
    parser.add_argument("--sample-size", type=int, default=REVIEW_SAMPLE_SIZE_PER_WORKER)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    args.annotation_results_path = sys.stdin if args.annotation_results_path_or_url == "-" \
        else cached_path(args.annotation_results_path_or_url)

    return args


def get_already_accepted_workers(annotation_results_path: str) -> Iterable[str]:
    statuses_by_worker = defaultdict(set)

    hits = parse_hits(annotation_results_path, include_rejected=True)

    for hit in hits.values():
        for worker_id, status in hit["status_by_worker"].items():
            statuses_by_worker[worker_id].add(status)

    for worker_id, status_set in statuses_by_worker.items():
        if "rejected" not in status_set and "accepted" in status_set:
            yield worker_id


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed)

    hits = parse_hits(args.annotation_results_path, include_accepted=False)
    instances = hits_to_instances(hits)
    instances_by_worker_id = compute_instances_by_worker_id(instances, compute_np_answers=True)

    q_count_by_worker = {worker_id: len(worker_instances)
                         for worker_id, worker_instances in instances_by_worker_id.items()}

    for worker_id, worker_instances in list(instances_by_worker_id.items()):
        q_count = q_count_by_worker[worker_id]

        np_answers_per_question = \
            sum(len(instance["np_answers"]) for instance in worker_instances) / len(worker_instances)

        if (q_count < MIN_QUESTION_COUNT_FOR_THRESHOLD_2
            and np_answers_per_question < MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_1) \
                or np_answers_per_question < MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_2:
            del instances_by_worker_id[worker_id]

    for worker_id in instances_by_worker_id:
        if len(instances_by_worker_id[worker_id]) > args.sample_size:
            instances_by_worker_id[worker_id] = random.sample(instances_by_worker_id[worker_id], args.sample_size)

    worker_ids_to_auto_accept = {worker_id
                                 for worker_id in get_already_accepted_workers(args.annotation_results_path)
                                 if worker_id not in AUTO_APPROVE_WORKER_ID_DENY_LIST}

    df = pd.DataFrame([
        {
            "worker_id": worker_id,
            **{k: instance[k] for k in ["video_id", "video_start_time", "video_end_time", "video_url", "question",
                                        "label"]},
            "answer": answer,
            "reviewer": "",
            # We can already annotate some:
            "correct?": "y" if (normalize_answer(answer) == normalize_answer(instance["label"])
                                or ("unavailable" in answer.lower() and "video" in answer.lower())
                                or worker_id in worker_ids_to_auto_accept) else "",
        }
        for worker_id, worker_instances in instances_by_worker_id.items()
        for instance in worker_instances
        for answer in instance["answers"]
    ])

    print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
