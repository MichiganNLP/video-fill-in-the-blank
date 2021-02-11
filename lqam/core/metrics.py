import re
import string
from typing import Iterable, Iterator, Set

RE_A_AN_THE_OR_PUNCTUATION = re.compile(rf"\b(?:an?|the)\b|[{re.escape(string.punctuation)}]")
RE_MULTIPLE_SPACES = re.compile(r"\s{2,}")


def normalize_answer(answer: str) -> str:
    """Should correspond to the JavaScript function `normalizeAnswerToLookForRepetitions`.

    Useful when looking for repetitions or computing measures.
    """
    return RE_MULTIPLE_SPACES.sub(" ", RE_A_AN_THE_OR_PUNCTUATION.sub("", answer.lower())).strip()


def tokenize_answer_to_compute_metrics(normalized_answer: str) -> Iterator[str]:
    return normalized_answer.split()


# TODO: how to deal with repeated words?
def compute_token_level_f1(answer1_tokens: Set[str], answer2_tokens: Set[str]) -> float:
    true_positives = len(answer1_tokens & answer2_tokens)
    false_count_in_1 = len(answer1_tokens - answer2_tokens)
    false_count_in_2 = len(answer2_tokens - answer1_tokens)
    return true_positives / (true_positives + (false_count_in_1 + false_count_in_2) / 2)


def compute_token_level_f1_many(answer_tokens: Iterator[str], ground_truths_tokens: Iterable[Iterable[str]]) -> float:
    answer_tokens = set(answer_tokens)
    return max(compute_token_level_f1(answer_tokens, set(g)) for g in ground_truths_tokens)


def exact_match(unnormalized_answer1: str, unnormalized_answer2: str) -> bool:
    return normalize_answer(unnormalized_answer1) == normalize_answer(unnormalized_answer2)

def exact_match_many(unnormalized_answer1: str, unnormalized_answer2_list: Iterator[str]) -> bool:
    for answer in unnormalized_answer2_list:
        if exact_match(unnormalized_answer1, answer):
            return True
    return False