#!/usr/bin/env python
import argparse
import itertools
import sys
import textwrap
from collections import defaultdict
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from lqam_data import normalize_answer, order_worker_answers_by_question, parse_hits


def compute_decision_score(precision: float, recall: float) -> float:
    return recall + 0.67 * precision


# TODO: how to deal with repeated words?
def compute_token_level_f1(a: Set[str], b: Set[str]) -> float:
    true_positives = len(a & b)
    false_count_in_a = len(a - b)
    false_count_in_b = len(b - a)
    return true_positives / (true_positives + (false_count_in_a + false_count_in_b) / 2)


def tokenize(s: str) -> Iterator[str]:
    return s.split()


def compute_token_level_f1_many(answer: Iterator[str], ground_truths: Iterator[Iterator[str]]) -> float:
    answer = set(answer)
    return max(compute_token_level_f1(answer, set(g)) for g in ground_truths)


def compute_metrics(question_answers: Sequence[Iterable[str]], std_answer: str,
                    ignored_workers: Optional[Sequence[bool]] = None) -> Tuple[Sequence[float], Sequence[float],
                                                                               Sequence[float], Sequence[float]]:
    assert len(question_answers) > 1

    ignored_workers = ignored_workers or defaultdict(lambda: False)

    ff1s = []
    precisions = []
    recalls = []
    decision_scores = []

    for i, worker_question_answers in enumerate(question_answers):
        if ignored_workers[i]:
            ff1 = precision = recall = 0
        else:
            assert worker_question_answers

            other_workers_answers = (question_answers[j]
                                     for j in range(len(question_answers))
                                     if j != i and not ignored_workers[j])
            other_answers = {answer
                             for other_worker_answers in other_workers_answers
                             for answer in other_worker_answers} | {std_answer}

            first_answer_tokens = tokenize(next(iter(worker_question_answers)))
            ff1 = compute_token_level_f1_many(first_answer_tokens, (tokenize(answer) for answer in other_answers))

            worker_question_answers = set(worker_question_answers)

            true_positives = len(worker_question_answers & other_answers)
            precision = true_positives / len(worker_question_answers)
            recall = true_positives / len(other_answers)

        ff1s.append(ff1)
        precisions.append(precision)
        recalls.append(recall)
        decision_scores.append(compute_decision_score(precision, recall))

    return ff1s, precisions, recalls, decision_scores


def print_metrics(hits: Mapping[str, Any],
                  ignore_zero_scores: bool = False) -> Tuple[Sequence[Sequence[Sequence[Iterable[str]]]],
                                                             Sequence[Sequence[Sequence[float]]],
                                                             Sequence[Sequence[Sequence[float]]],
                                                             Sequence[Sequence[Sequence[float]]],
                                                             Sequence[Sequence[Sequence[int]]],
                                                             Sequence[Sequence[Sequence[float]]]]:
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

    hits_answers = []

    hits_precisions = []
    hits_recalls = []
    hits_decision_scores = []

    hits_answer_lengths = []

    hits_std_answer_metrics = []

    for hit in hits:
        hit_answers = (order_worker_answers_by_question(worker_answers_map)
                       for worker_answers_map in hit["Answer.taskAnswers"])
        hit_answers = [[[normalize_answer(answer) for answer in worker_question_answers]
                        for worker_question_answers in worker_answers]
                       for worker_answers in hit_answers]

        hit_answers = [[worker_answers[i] for worker_answers in hit_answers] for i in range(len(hit_answers[0]))]
        hits_answers.append(hit_answers)

        hit_precisions = []
        hit_recalls = []
        hit_decision_scores = []

        hit_answer_lengths = []

        hit_std_answer_metrics = []

        for i in itertools.count(start=1):
            if not (question := hit.get(f"Input.question{i}")):
                break

            question_answers = hit_answers[i - 1]

            std_answer = normalize_answer(hit[f"Input.label{i}"])

            ff1s, precisions, recalls, decision_scores = compute_metrics(question_answers, std_answer)

            if ignore_zero_scores:
                ignored_workers = [d == 0 for d in decision_scores]
                ff1s, precisions, recalls, decision_scores = compute_metrics(question_answers, std_answer,
                                                                             ignored_workers)
            else:
                ignored_workers = [False for _ in question_answers]

            question_answers_flat = [answer
                                     for worker_answers, is_worker_ignored in zip(question_answers, ignored_workers)
                                     if not is_worker_ignored
                                     for answer in worker_answers]

            std_precision = float(std_answer in question_answers_flat)
            std_recall = sum(answer == std_answer for answer in question_answers_flat) / len(question_answers_flat)
            std_decision_score = compute_decision_score(std_precision, std_recall)

            video_id = hit[f"Input.video{i}_id"]
            start_time = hit[f"Input.video{i}_start_time"]
            end_time = hit[f"Input.video{i}_end_time"]

            print(textwrap.dedent(f"""\
            {question.replace("[MASK]", "_____")}
            https://www.youtube.com/embed/{video_id}?start={start_time}&end={end_time}
            
            pre rec dec
            {std_precision * 100: >3.0f} {std_recall * 100: >3.0f} {std_decision_score * 100: >3.0f}\
 {std_answer}
            """))

            for j in range(len(question_answers)):
                print(f"{ff1s[j] * 100: >3.0f} {precisions[j] * 100: >3.0f} {recalls[j] * 100: >3.0f}"
                      f" {decision_scores[j] * 100: >3.0f} {question_answers[j]}")

            print()
            print("---")
            print()

            hit_precisions.append(precisions)
            hit_recalls.append(recalls)
            hit_decision_scores.append(decision_scores)

            hit_answer_lengths.append([len(answer) for answer in question_answers])

            hit_std_answer_metrics.append((std_precision, std_recall, std_decision_score))

        hits_precisions.append(np.stack(hit_precisions))
        hits_recalls.append(np.stack(hit_recalls))
        hits_decision_scores.append(np.stack(hit_decision_scores))

        hits_answer_lengths.append(np.stack(hit_answer_lengths))

        hits_std_answer_metrics.append(np.stack(hit_std_answer_metrics))

    return (hits_answers, hits_precisions, hits_recalls, hits_decision_scores, hits_answer_lengths,
            hits_std_answer_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mturk_results_path", metavar="MTURK_RESULTS_FILE", default="-")
    parser.add_argument("--ignore-zero-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_ = sys.stdin if args.mturk_results_path == "-" else args.mturk_results_path
    hits = parse_hits(input_)

    print_metrics(hits, ignore_zero_scores=args.ignore_zero_scores)


if __name__ == "__main__":
    main()
