from __future__ import annotations

import itertools
import re
import string
from collections import Iterable, Iterator

RE_A_AN_THE_OR_PUNCTUATION = re.compile(rf"\b(?:an?|the)\b|[{re.escape(string.punctuation)}]")
RE_MULTIPLE_SPACES = re.compile(r"\s{2,}")


def normalize_answer(answer: str) -> str:
    """Should correspond to the JavaScript function `normalizeAnswerToLookForRepetitions`.

    Useful when looking for repetitions or computing measures.
    """
    return RE_MULTIPLE_SPACES.sub(" ", RE_A_AN_THE_OR_PUNCTUATION.sub("", answer.lower())).strip()


def tokenize_answer_to_compute_metrics(normalized_answer: str) -> Iterator[str]:
    return normalized_answer.split()


def compute_token_level_f1(answer1_tokens: set[str], answer2_tokens: set[str]) -> float:
    # Note it ignore the repeated words.
    true_positives = len(answer1_tokens & answer2_tokens)
    false_count_in_1 = len(answer1_tokens - answer2_tokens)
    false_count_in_2 = len(answer2_tokens - answer1_tokens)
    return true_positives / (true_positives + (false_count_in_1 + false_count_in_2) / 2)


def compute_token_level_f1_many(predicted_answer_tokens: Iterable[str],
                                ground_truth_tokens_iterable: Iterable[Iterable[str]]) -> float:
    predicted_answer_tokens = set(predicted_answer_tokens)
    return max(compute_token_level_f1(predicted_answer_tokens, set(ground_truth_tokens))
               for ground_truth_tokens in ground_truth_tokens_iterable)


def exact_match(unnormalized_answer1: str, unnormalized_answer2: str) -> bool:
    return normalize_answer(unnormalized_answer1) == normalize_answer(unnormalized_answer2)


def exact_match_many(unnormalized_predicted_answer: str, unnormalized_ground_truth_answers: Iterable[str]) -> bool:
    return any(exact_match(unnormalized_predicted_answer, unnormalized_ground_truth_answer)
               for unnormalized_ground_truth_answer in unnormalized_ground_truth_answers)


def flatten_additional_answers(additional_answers: Iterable[Iterable[str]]) -> set[str]:
    return {answer for worker_answers in additional_answers for answer in worker_answers}


def flatten_all_answers(
        labels: Iterable[str], additional_answers_batch: Iterable[Iterable[Iterable[str]]] | None = None
) -> Iterable[Iterable[str]]:
    additional_answers_batch = additional_answers_batch or itertools.repeat([])
    return [flatten_additional_answers(additional_answers) | {label}
            for additional_answers, label in zip(additional_answers_batch, labels)]
