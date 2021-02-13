#!/usr/bin/env python
import argparse
from collections import defaultdict

import pandas as pd

from lqam.annotations import MAX_COVERED_ANSWERS_PER_QUESTION_BONUS_TYPE_1, MIN_ANSWERS_PER_QUESTION, \
    PAY_PER_ANSWER_BONUS_TYPE_1, PAY_PER_HIT_BONUS_TYPE_2, QUESTIONS_PER_HIT
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

    correct_answers_by_assignment = defaultdict(int)
    for worker_instances in instances_by_worker_id.values():
        for instance in worker_instances:
            correct_answers_by_assignment[instance["assignment_id"]] += len(instance["np_answers"])

    bonus_type_1_amount = {assignment_id: (min(max(0, correct_answer_count - MIN_ANSWERS_PER_HIT),
                                               MAX_COVERED_ANSWERS_PER_HIT_BONUS_TYPE_1) * PAY_PER_ANSWER_BONUS_TYPE_1)
                           for assignment_id, correct_answer_count in correct_answers_by_assignment.items()}

    best_assignment_by_hit = {hit_id: max(((assignment_id, correct_answers_by_assignment[assignment_id])
                                           for assignment_id in hit["assignment_ids"].values()), key=lambda t: t[1])[0]
                              for hit_id, hit in hits.items()}

    bonus_type_2_amount = {assignment_id: PAY_PER_HIT_BONUS_TYPE_2 for assignment_id in best_assignment_by_hit.values()}

    assignments_by_worker = defaultdict(list)
    for hit in hits.values():
        for worker_id, assignment_id in hit["assignment_ids"].items():
            assignments_by_worker[worker_id].append(assignment_id)

    pay_per_worker = {worker_id: sum(bonus_type_1_amount[assignment_id] + bonus_type_2_amount[assignment_id]
                                     for assignment_id in assignment_ids)
                      for worker_id, assignment_ids in assignments_by_worker.items()}

    # We give a thank you note to those workers that don't receive any bonus money. This is also in part a way to let
    # them know they didn't qualify for any bonus. Otherwise, there would be radio silence.
    #
    # We're working with floats, so it may not be exactly 0. So we compare with something lower than 0.005 which
    # would be in turn rounded to 0.
    #
    # We need to indicate the bonus assignment ID, so we just pick any.
    participation_bonus_amount = {next(iter(instances_by_worker_id[worker_id]))["assignment_id"]: 0.01
                                  for worker_id, amount in pay_per_worker.items()
                                  if amount < 0.005}

    worker_by_assignment = {assignment_id: worker_id
                            for hit in hits.values()
                            for worker_id, assignment_id in hit["assignment_ids"].items()}

    # TODO: build the "reason" string as well
    # TODO: should use decimal for money?
    # TODO: round to cents.

    {
        {
            "assignment_id": assignment_id,
            "worker_id": worker_id,
            "bonus_amount": ,
        } for assignment_id, worker_id in worker_by_assignment.items()
    }

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
