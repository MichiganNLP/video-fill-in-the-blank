from collections import defaultdict
from typing import Iterable, Iterator, Optional, Sequence, Set, Tuple

from lqam_data import normalize_answer


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


def _compute_metrics_once(
        answers: Sequence[Iterable[str]], std_answer: str, ignored_workers: Optional[Sequence[bool]] = None
) -> Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Tuple[float, float, float, float]]:
    ignored_workers = ignored_workers or [False for _ in answers]

    ff1s = []
    precisions = []
    recalls = []
    decision_scores = []

    for i, worker_question_answers in enumerate(answers):
        if ignored_workers[i]:
            ff1 = precision = recall = 0
        else:
            assert worker_question_answers

            other_workers_answers = (answers[j]
                                     for j in range(len(answers))
                                     if j != i and not ignored_workers[j])
            other_answers = {answer
                             for other_worker_answers in other_workers_answers
                             for answer in other_worker_answers} | {std_answer}

            first_answer_tokens = tokenize(next(iter(worker_question_answers)))
            ff1 = compute_token_level_f1_many(first_answer_tokens, (tokenize(answer) for answer in other_answers))

            worker_question_answers_set = set(worker_question_answers)

            true_positives = len(worker_question_answers_set & other_answers)
            precision = true_positives / len(worker_question_answers_set)
            recall = true_positives / len(other_answers)

        ff1s.append(ff1)
        precisions.append(precision)
        recalls.append(recall)
        decision_scores.append(compute_decision_score(precision, recall))

    # Not a set because we want to keep the counts.
    answers_flat = [answer
                    for worker_answers, is_worker_ignored in zip(answers, ignored_workers)
                    if not is_worker_ignored
                    for answer in worker_answers]

    std_answer_tokens = tokenize(std_answer)
    std_ff1 = compute_token_level_f1_many(std_answer_tokens, (tokenize(answer) for answer in answers_flat))

    std_precision = float(std_answer in answers_flat)
    std_recall = sum(answer == std_answer for answer in answers_flat) / len(answers_flat)
    std_decision_score = compute_decision_score(std_precision, std_recall)

    return ff1s, precisions, recalls, decision_scores, (std_ff1, std_precision, std_recall, std_decision_score)


def compute_metrics(
        answers: Iterator[Iterable[str]], std_answer: str, ignore_zero_scores: bool = False
) -> Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Tuple[float, float, float, float]]:
    """Computes the metrics for an instance.

    If `ignore_zero_scores`, then it computes the scores again but ignores the workers whose decision score is 0.
    """
    answers = list(answers)

    assert len(answers) > 1

    answers = [[normalize_answer(answer) for answer in worker_answers] for worker_answers in answers]

    std_answer = normalize_answer(std_answer)

    ff1s, precisions, recalls, decision_scores, std_answer_metrics = _compute_metrics_once(answers, std_answer)

    if ignore_zero_scores:
        ignored_workers = [d == 0 for d in decision_scores]
        ff1s, precisions, recalls, decision_scores, std_answer_metrics = _compute_metrics_once(answers,
                                                                                               std_answer,
                                                                                               ignored_workers)

    return ff1s, precisions, recalls, decision_scores, std_answer_metrics
