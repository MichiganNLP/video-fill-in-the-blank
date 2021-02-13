#!/usr/bin/env python
import argparse
from collections import defaultdict

import pandas as pd

from lqam.annotations import MAX_COVERED_ANSWERS_PER_QUESTION_BONUS_TYPE_1, MIN_ANSWERS_PER_QUESTION, \
    PAY_PER_ANSWER_BONUS_TYPE_1, QUESTIONS_PER_HIT
from lqam.annotations.postprocessing import compute_instances_by_worker_id, hits_to_instances, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path

MIN_ANSWERS_PER_HIT = QUESTIONS_PER_HIT * MIN_ANSWERS_PER_QUESTION

MAX_COVERED_ANSWERS_PER_HIT_BONUS_TYPE_1 = QUESTIONS_PER_HIT * MAX_COVERED_ANSWERS_PER_QUESTION_BONUS_TYPE_1


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path", metavar="ANNOTATION_RESULTS_FILE_OR_URL", type=cached_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hits = parse_hits(args.annotation_results_path)
    instances = hits_to_instances(hits)

    # assert all(instance["status"] == "approved" for instance in instances)  # FIXME

    instances_by_worker_id = compute_instances_by_worker_id(instances, compute_np_answers=True)

    correct_answers_by_worker_assignment = defaultdict(lambda: defaultdict(int))
    for worker_id, worker_instances in instances_by_worker_id.items():
        for instance in worker_instances:
            correct_answers_by_worker_assignment[worker_id][instance["assignment_id"]] += len(instance["np_answers"])

    # TODO: for each hit and worker, compute how many NP answers the worker has. Then subtract 10, multiply by a
    #  number and set a max (bonus type I). Then, check which worker(s) has the max number of NP answers (bonus type
    #  II). Save these amounts (and build the "reason" string as well) and then use them.
    # TODO: Skip the workers that don't receive a bonus (or maybe pay them 1 cent and thank them for participating?).
    # TODO: bonus type 3: exceptional workers? or manually?

    bonus_type_1_amount = {worker_id: {assignment_id: (min(max(0, correct_answer_count - MIN_ANSWERS_PER_HIT),
                                                           MAX_COVERED_ANSWERS_PER_HIT_BONUS_TYPE_1)
                                                       * PAY_PER_ANSWER_BONUS_TYPE_1)
                                       for assignment_id, correct_answer_count in worker_assignments.items()}
                           for worker_id, worker_assignments in correct_answers_by_worker_assignment.items()}

    assignment_to_hit = {assignment_id: hit_id
                         for hit_id, hit in hits.items()
                         for assignment_id in hit["assignment_ids"].values()}

    correct_answers_by_hit_worker = defaultdict(lambda: defaultdict(int))
    for worker_id, worker_correct_answers_by_assignment in correct_answers_by_worker_assignment.items():
        for assignment_id, correct_answer_count in worker_correct_answers_by_assignment.items():
            correct_answers_by_hit_worker[assignment_to_hit[assignment_id]][worker_id] += correct_answer_count

    bonus_type_2_amount = {hit_id: {max(hit_correct_answers_by_worker.items(), key=lambda t: t[1])[0]: 0.2}
                           for hit_id, hit_correct_answers_by_worker in correct_answers_by_hit_worker.items()}

    df = pd.DataFrame(
        {
            # We pay all the bonuses altogether at once for each worker-assignment-ID pair. So we use this pair as the
            # UUID to avoid duplicate payments.
            "uuid": f"{worker_id}-{assignment_id}",
            "worker_id": worker_id,
            "bonus_amount": 0.01,  # TODO
            "assignment_id": assignment_id,
            "reason": "",  # TODO
        } for hit in hits.values() for worker_id, assignment_id in hit["assignment_ids"].items()
    )

    print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
