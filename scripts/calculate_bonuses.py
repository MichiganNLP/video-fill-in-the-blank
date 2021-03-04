#!/usr/bin/env python
import argparse
import json
import sys
from collections import defaultdict

from lqam.annotations import MAX_COVERED_ANSWERS_PER_QUESTION_BONUS_TYPE_1, MIN_ANSWERS_PER_QUESTION, \
    PAY_PER_ANSWER_BONUS_TYPE_1, PAY_PER_HIT_BONUS_TYPE_2, QUESTIONS_PER_HIT
from lqam.annotations.postprocessing import compute_instances_by_worker_id, hits_to_instances, parse_hits
from lqam.util.argparse_with_defaults import ArgumentParserWithDefaults
from lqam.util.file_utils import cached_path

MIN_PAYABLE_AMOUNT = 0.005  # The min that gets rounded up to a cent, which is actually the min.

MIN_ANSWERS_PER_HIT = QUESTIONS_PER_HIT * MIN_ANSWERS_PER_QUESTION

MAX_COVERED_ANSWERS_PER_HIT_BONUS_TYPE_1 = QUESTIONS_PER_HIT * MAX_COVERED_ANSWERS_PER_QUESTION_BONUS_TYPE_1


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("annotation_results_path_or_url", metavar="ANNOTATION_RESULTS_FILE_OR_URL", nargs="?",
                        default="-")
    args = parser.parse_args()

    args.input = sys.stdin if args.annotation_results_path_or_url == "-" \
        else cached_path(args.annotation_results_path_or_url)

    return args


def main() -> None:
    args = parse_args()

    hits_by_id = parse_hits(args.input)
    instances_by_id = hits_to_instances(hits_by_id)

    assert all(all(status == "approved" for status in instance["status_by_worker"].values())
               for instance in instances_by_id.values()), \
        "All assignments should have been reviewed already (approved or rejected)."

    instances_by_worker_id = compute_instances_by_worker_id(instances_by_id, compute_np_answers=True)

    correct_answers_by_assignment = defaultdict(int)
    for worker_instances in instances_by_worker_id.values():
        for instance in worker_instances:
            correct_answers_by_assignment[instance["assignment_id"]] += len(instance["np_answers"])

    # We could use `Decimal` to handle money because floats can introduce errors after many calculations. However,
    # here we're doing quite simple calculations. So it's not necessary.

    bonus_type_1_amount = {assignment_id: (min(max(0, correct_answer_count - MIN_ANSWERS_PER_HIT),
                                               MAX_COVERED_ANSWERS_PER_HIT_BONUS_TYPE_1) * PAY_PER_ANSWER_BONUS_TYPE_1)
                           for assignment_id, correct_answer_count in correct_answers_by_assignment.items()}

    best_assignment_by_hit = {hit_id: max(((assignment_id, correct_answers_by_assignment[assignment_id])
                                           for assignment_id in hit["assignment_id_by_worker"].values()),
                                          key=lambda t: t[1])[0]
                              for hit_id, hit in hits_by_id.items()}

    bonus_type_2_amount = {assignment_id: PAY_PER_HIT_BONUS_TYPE_2 for assignment_id in best_assignment_by_hit.values()}

    assignments_by_worker = defaultdict(list)
    for hit in hits_by_id.values():
        for worker_id, assignment_id in hit["assignment_id_by_worker"].items():
            assignments_by_worker[worker_id].append(assignment_id)

    pay_per_worker = {worker_id: sum(bonus_type_1_amount.get(assignment_id, 0)
                                     + bonus_type_2_amount.get(assignment_id, 0)
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
                                  if amount < MIN_PAYABLE_AMOUNT}

    worker_by_assignment = {assignment_id: worker_id
                            for hit in hits_by_id.values()
                            for worker_id, assignment_id in hit["assignment_id_by_worker"].items()}

    considered_assignment_ids = {assignment_id
                                 for assignment_id in worker_by_assignment
                                 if (bonus_type_1_amount.get(assignment_id, 0)
                                     + bonus_type_2_amount.get(assignment_id, 0)
                                     + participation_bonus_amount.get(assignment_id, 0)) >= MIN_PAYABLE_AMOUNT}

    for assignment_id in considered_assignment_ids:
        print(json.dumps({
            # We pay all the bonuses altogether at once for each assignment ID (and it's unique). So we use it as
            # the UUID to avoid duplicate payments.
            "UniqueRequestToken": assignment_id,
            "AssignmentId": assignment_id,
            "WorkerId": worker_by_assignment[assignment_id],
            "BonusAmount": str(round(bonus_type_1_amount.get(assignment_id, 0)  # The API expects a string amount.
                                     + bonus_type_2_amount.get(assignment_id, 0)
                                     + participation_bonus_amount.get(assignment_id, 0), ndigits=2)),
            "Reason": (""
                       + (f"You receive ${bonus_type_1_amount[assignment_id]} because you provided a total of "
                          f"{correct_answers_by_assignment[assignment_id]} correct answers in this HIT. "
                          if bonus_type_1_amount.get(assignment_id, 0) >= MIN_PAYABLE_AMOUNT else "")
                       + (f"You receive ${bonus_type_2_amount[assignment_id]} because you were the worker with "
                          f"the highest number of correct answers in the HIT. "
                          if bonus_type_2_amount.get(assignment_id, 0) >= MIN_PAYABLE_AMOUNT else "")
                       + ("We send this one-cent bonus to be able to communicate with you (there's no other way "
                          "as far as we know). We wanted to let you know that unfortunately you didn't qualify for any "
                          "bonus for this task. We thank you a lot for your participation! "
                          if assignment_id in participation_bonus_amount else "")
                       + "If you have any question or concern, please contact us."),
        }))


if __name__ == "__main__":
    main()
