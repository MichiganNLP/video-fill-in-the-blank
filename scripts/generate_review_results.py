#!/usr/bin/env python
import argparse

import pandas as pd

from lqam.annotations import MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_1, MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_2, \
    MIN_QUESTION_COUNT_FOR_THRESHOLD_2, REVIEW_SAMPLE_SIZE_PER_WORKER
from lqam.annotations.postprocessing import compute_instances_by_worker_id, hits_to_instances, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path", metavar="ANNOTATION_RESULTS_FILE_OR_URL", type=cached_path)
    parser.add_argument("reviewed_answers_path", metavar="REVIEWED_ANSWERS_FILE_OR_URL", type=cached_path)
    parser.add_argument("--sample-size-used", type=int, default=REVIEW_SAMPLE_SIZE_PER_WORKER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hits = parse_hits(args.annotation_results_path, include_accepted=False)
    instances = hits_to_instances(hits)
    instances_by_worker_id = compute_instances_by_worker_id(instances, compute_np_answers=True)

    q_count_by_worker = {worker_id: len(worker_instances)
                         for worker_id, worker_instances in instances_by_worker_id.items()}

    np_answers_per_question_by_worker = {
        worker_id: sum(len(instance["np_answers"]) for instance in worker_instances) / len(worker_instances)
        for worker_id, worker_instances in instances_by_worker_id.items()
    }

    # We should consider the instance count from the worker at the moment we sampled, because in the sample there
    # could be fewer than the desired sample size either because the worker annotated fewer HITs or because some
    # instances ended up with no answers from the worker after filtering out the non-noun-phrase answers.
    # So then when we estimate the quality we consider the sample size that should have been actually considered
    # before the filtering.
    q_count_considered_by_worker = {worker_id: min(args.sample_size_used, instance_count)
                                    for worker_id, instance_count in q_count_by_worker.items()}

    df = pd.read_csv(args.reviewed_answers_path, converters={"correct?": lambda s: s and s.lower() in {"y", "true"}})
    df["video_question_id"] = list(zip(df.video_id, df.video_start_time, df.video_end_time, df.question))

    correct_answers_by_worker = df[df["correct?"]].groupby("worker_id")["answer"].count().sort_index()

    correct_answers_per_considered_q_by_worker = {
        worker_id: (
                correct_answers_by_worker.get(
                    # If it's not found then it means it wasn't sampled because the quality was low after filtering out
                    # non-NP answers. We re-compute it as we need it for some cases then.
                    worker_id, np_answers_per_question_by_worker[worker_id] * instance_count_considered) /
                instance_count_considered
        )
        for worker_id, instance_count_considered in q_count_considered_by_worker.items()
    }

    for worker_id, correct_answers_per_considered_q in correct_answers_per_considered_q_by_worker.items():
        q_count = q_count_by_worker[worker_id]

        if correct_answers_per_considered_q < MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_1:
            if correct_answers_per_considered_q < MIN_ACCEPTABLE_ANSWERS_PER_QUESTION_2 \
                    or q_count < MIN_QUESTION_COUNT_FOR_THRESHOLD_2:
                print(worker_id, correct_answers_per_considered_q, q_count)
            else:
                print(f"WARNING: the worker {worker_id} should be approved but the worker isn't doing a great job. "
                      f"The worker is in between the 2 thresholds ({correct_answers_per_considered_q}, {q_count}) "
                      f"and annotated at least than {MIN_QUESTION_COUNT_FOR_THRESHOLD_2} questions.")


if __name__ == "__main__":
    main()
