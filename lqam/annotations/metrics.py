import string
from collections import defaultdict
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np

from lqam.core.metrics import compute_token_level_f1_many, normalize_answer, tokenize_answer_to_compute_metrics
from lqam.core.noun_phrases import create_spacy_model, is_noun_phrase_or_n_bar

SPACY_MODEL = create_spacy_model(prefer_gpu=True)


def strip_punctuation(s: str) -> str:
    # See https://stackoverflow.com/a/266162/1165181
    return s.translate(str.maketrans("", "", string.punctuation)).strip()


def _compute_annotation_metrics_once(answers: Sequence[Iterable[str]],
                                     std_answer: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    ff1s = []
    fems = []

    for i, worker_question_answers in enumerate(answers):
        if worker_question_answers:  # It could happen because of some weird issues with the annotation interface.
            other_workers_answers = (answers[j] for j in range(len(answers)) if j != i)
            other_answers = {answer
                             for other_worker_answers in other_workers_answers
                             for answer in other_worker_answers} | {std_answer}

            first_answer = next(iter(worker_question_answers))
            first_answer_tokens = tokenize_answer_to_compute_metrics(first_answer)
            ff1 = compute_token_level_f1_many(first_answer_tokens, (tokenize_answer_to_compute_metrics(answer)
                                                                    for answer in other_answers))

            fem = float(first_answer in other_answers)
        else:
            ff1 = fem = np.nan

        ff1s.append(ff1)
        fems.append(fem)

    # Not a set because we want to keep the counts.
    answers_flat = [answer for worker_answers in answers for answer in worker_answers]

    std_answer_tokens = tokenize_answer_to_compute_metrics(std_answer)
    std_ff1 = compute_token_level_f1_many(std_answer_tokens, (tokenize_answer_to_compute_metrics(answer)
                                                              for answer in answers_flat))

    std_fem = float(std_answer in answers_flat)

    return np.stack(ff1s), np.stack(fems), (std_ff1, std_fem)


def compute_annotation_metrics(answers: Iterator[Iterable[str]],
                               std_answer: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Computes the metrics for an instance.

    If `ignore_zero_scores`, then it computes the scores again but ignores the workers whose decision score is 0.
    """
    answers = list(answers)

    assert len(answers) > 1

    answers = [[n for answer in worker_answers if (n := normalize_answer(answer))] for worker_answers in answers]

    std_answer = normalize_answer(std_answer)

    ff1s, fems, std_answer_metrics = _compute_annotation_metrics_once(answers, std_answer)

    return ff1s, fems, std_answer_metrics


def compute_np_value_by_answer(question: str, answers_map: Mapping[str, Sequence[str]]) -> Mapping[str, bool]:
    # Workers can add extra punctuation, and this messes up with the parsing. So we remove it.
    # We could use `alignment_mode="contract"` if there's extra punctuation, however this doesn't prevent the parsing
    # from failing. So we remove the punctuation altogether.
    answers_flat = {(answer, strip_punctuation(answer))
                    for worker_answers in answers_map.values()
                    for answer in worker_answers}

    question_with_answers = (question.replace("_____", clean_answer) for _, clean_answer in answers_flat)

    return {answer: bool(clean_answer) and is_noun_phrase_or_n_bar(doc.char_span((start := question.index("_____")),
                                                                                 start + len(clean_answer)))
            for (answer, clean_answer), doc in zip(answers_flat, SPACY_MODEL.pipe(question_with_answers))}


def compute_answer_level_annotation_metrics(question: str, answers_map: Mapping[str, Sequence[str]],
                                            std_answer: str) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    # `frozenset` so it's immutable thus hashable.
    answer_processed_map = {worker_id: [(answer, normalized_answer,
                                         frozenset(tokenize_answer_to_compute_metrics(normalized_answer)))
                                        for answer in worker_answers
                                        # There's some rare cases in which an answer is empty.
                                        if (normalized_answer := normalize_answer(answer))]
                            for worker_id, worker_answers in answers_map.items()}

    # We apply the filtering we just did to the original answers map.
    answers_map = {worker_id: [answer for answer, _, _ in worker_answers]
                   for worker_id, worker_answers in answer_processed_map.items()}

    std_answer_normalized = normalize_answer(std_answer)
    std_answer = frozenset(tokenize_answer_to_compute_metrics(std_answer_normalized))

    results = defaultdict(lambda: defaultdict(dict))

    for worker_id, worker_answers in answer_processed_map.items():
        other_workers_answers = [other_worker_answers
                                 for i, (other_worker_id,
                                         other_worker_answers) in enumerate(answer_processed_map.items())
                                 if other_worker_id != worker_id]
        other_answer_tokens = {tokens
                               for other_worker_answers in other_workers_answers
                               for _, _, tokens in other_worker_answers} | {std_answer}
        other_normalized_answers = {normalized_answer
                                    for other_worker_answers in other_workers_answers
                                    for _, normalized_answer, _ in other_worker_answers} | {std_answer_normalized}

        for answer, normalized_answer, tokens in worker_answers:
            results[worker_id][answer]["f1"] = compute_token_level_f1_many(tokens, other_answer_tokens)
            results[worker_id][answer]["em"] = any(normalized_answer == other_normalized_answer
                                                   for other_normalized_answer in other_normalized_answers)

    np_map = compute_np_value_by_answer(question, answers_map)

    for worker_id, worker_answers in answers_map.items():
        for answer in worker_answers:
            results[worker_id][answer]["np"] = np_map[answer]

    return results
