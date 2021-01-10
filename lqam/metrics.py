import re
import string
from collections import defaultdict
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pytorch_lightning as pl
import spacy.tokens
import torch
from overrides import overrides

from lqam.noun_phrases import is_noun_phrase_like

# TODO: change the JavaScript code to match this exact punctuation:
RE_A_AN_THE_OR_PUNCTUATION = re.compile(rf"\b(?:an?|the)\b|[{re.escape(string.punctuation)}]")
RE_MULTIPLE_SPACES = re.compile(r"\s{2,}")

SPACY_MODEL = spacy.load("en_core_web_lg")  # I detected fewer errors with it than with "en_core_web_sm".


def normalize_answer(answer: str) -> str:
    """Should correspond to the JavaScript function `normalizeAnswerToLookForRepetitions`.

    Useful when looking for repetitions or computing measures.
    """
    return RE_MULTIPLE_SPACES.sub(" ", RE_A_AN_THE_OR_PUNCTUATION.sub("", answer.lower())).strip()


def compute_decision_score(precision: float, recall: float) -> float:
    return recall + 0.67 * precision


# TODO: how to deal with repeated words?
def compute_token_level_f1(a: Set[str], b: Set[str]) -> float:
    true_positives = len(a & b)
    false_count_in_a = len(a - b)
    false_count_in_b = len(b - a)
    return true_positives / (true_positives + (false_count_in_a + false_count_in_b) / 2)


def _tokenize(s: str) -> Iterator[str]:
    return s.split()


def compute_token_level_f1_many(answer: Iterator[str], ground_truths: Iterable[Iterable[str]]) -> float:
    answer = set(answer)
    return max(compute_token_level_f1(answer, set(g)) for g in ground_truths)


def _compute_annotation_metrics_once(
        answers: Sequence[Iterable[str]], std_answer: str, ignored_workers: Optional[Sequence[bool]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
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

            first_answer_tokens = _tokenize(next(iter(worker_question_answers)))
            ff1 = compute_token_level_f1_many(first_answer_tokens, (_tokenize(answer) for answer in other_answers))

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

    std_answer_tokens = _tokenize(std_answer)
    std_ff1 = compute_token_level_f1_many(std_answer_tokens, (_tokenize(answer) for answer in answers_flat))

    std_precision = float(std_answer in answers_flat)
    std_recall = sum(answer == std_answer for answer in answers_flat) / len(answers_flat)
    std_decision_score = compute_decision_score(std_precision, std_recall)

    return np.stack(ff1s), np.stack(precisions), np.stack(recalls), np.stack(decision_scores), (
        std_ff1, std_precision, std_recall, std_decision_score)


def compute_annotation_metrics(
        answers: Iterator[Iterable[str]], std_answer: str, ignore_zero_scores: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float], Sequence[bool]]:
    """Computes the metrics for an instance.

    If `ignore_zero_scores`, then it computes the scores again but ignores the workers whose decision score is 0.
    """
    answers = list(answers)

    assert len(answers) > 1

    answers = [[normalize_answer(answer) for answer in worker_answers] for worker_answers in answers]

    std_answer = normalize_answer(std_answer)

    ff1s, precisions, recalls, decision_scores, std_answer_metrics = _compute_annotation_metrics_once(answers,
                                                                                                      std_answer)

    if ignore_zero_scores:
        ignored_workers = [d == 0 for d in decision_scores]
        ff1s, precisions, recalls, decision_scores, std_answer_metrics = _compute_annotation_metrics_once(
            answers, std_answer, ignored_workers)
    else:
        ignored_workers = [False for _ in answers]

    return ff1s, precisions, recalls, decision_scores, std_answer_metrics, ignored_workers


def compute_answer_level_annotation_metrics(answers_map: Mapping[str, Sequence[str]], std_answer_tokens: str,
                                            ignored_workers: Optional[Sequence[bool]] = None
                                            ) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    ignored_workers = ignored_workers or [False for _ in answers_map]

    # `frozenset` so it's immutable thus hashable.
    answer_processed_map = {worker_id: [(answer, n := normalize_answer(answer), frozenset(_tokenize(n)))
                                        for answer in worker_answers]
                            for worker_id, worker_answers in answers_map.items()}

    std_answer_normalized = normalize_answer(std_answer_tokens)
    std_answer_tokens = frozenset(_tokenize(std_answer_normalized))

    results = defaultdict(lambda: defaultdict(dict))

    for worker_id, worker_answers in answer_processed_map.items():
        other_workers_answers = [other_worker_answers
                                 for i, (other_worker_id,
                                         other_worker_answers) in enumerate(answer_processed_map.items())
                                 if other_worker_id != worker_id and not ignored_workers[i]]
        other_answer_tokens = {tokens
                               for other_worker_answers in other_workers_answers
                               for _, _, tokens in other_worker_answers} | {std_answer_tokens}
        other_normalized_answers = {normalized_answer
                                    for other_worker_answers in other_workers_answers
                                    for _, normalized_answer, _ in other_worker_answers} | {std_answer_normalized}

        for answer, normalized_answer, tokens in worker_answers:
            results[worker_id][answer]["f1"] = compute_token_level_f1_many(tokens, other_answer_tokens)
            results[worker_id][answer]["em"] = any(normalized_answer == other_normalized_answer
                                                   for other_normalized_answer in other_normalized_answers)

    # Check if they are noun phrases:

    answers_flat = {f"{prefix}{answer}"
                    for worker_answers in answers_map.values()
                    for answer in worker_answers
                    for prefix in ["", "the "]}

    np_map = {answer: is_noun_phrase_like(doc)
              for answer, doc in zip(answers_flat, SPACY_MODEL.pipe(answers_flat, batch_size=64, n_process=2))}

    for worker_id, worker_answers in answers_map.items():
        for answer in worker_answers:
            results[worker_id][answer]["np"] = np_map[answer] or np_map[f"the {answer}"]

    return results


def exact_match(label1: str, label2: str) -> bool:
    return normalize_answer(label1) == normalize_answer(label2)


class AlmostExactMatchAccuracy(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @overrides
    def update(self, preds: Sequence[str], targets: Sequence[str]) -> None:  # noqa
        assert len(preds) == len(targets)
        self.correct += sum(exact_match(pred, target) for pred, target in zip(preds, targets))
        self.total += len(targets)

    @overrides
    def compute(self) -> float:
        return self.correct / self.total
