import itertools
import string
from collections import defaultdict
from typing import Any, Iterable, Mapping

from lqam.core.metrics import compute_token_level_f1_many, normalize_answer, tokenize_answer_to_compute_metrics
from lqam.core.noun_phrases import create_spacy_model, is_noun_phrase_or_n_bar

SPACY_MODEL = create_spacy_model(prefer_gpu=True)


def strip_punctuation(s: str) -> str:
    # See https://stackoverflow.com/a/266162/1165181
    return s.translate(str.maketrans("", "", string.punctuation)).strip()


def compute_np_value_by_answer(question: str, answers_map: Mapping[str, Iterable[str]]) -> Mapping[str, bool]:
    answers_flat = {answer for worker_answers in answers_map.values() for answer in worker_answers}

    # Workers can add extra punctuation, and this messes up with the parsing. So we remove it.
    # We could use `alignment_mode="contract"` if there's extra punctuation, however this doesn't prevent the parsing
    # from failing. So we remove the punctuation altogether.
    # We don't completely normalize because it may perform slightly worse when checking NPs.
    answers_flat = {(answer, strip_punctuation(answer)) for answer in answers_flat}

    question_with_answers = (question.replace("_____", clean_answer) for _, clean_answer in answers_flat)

    return {answer: clean_answer and is_noun_phrase_or_n_bar(doc.char_span((start := question.index("_____")),
                                                                           start + len(clean_answer)))
            for (answer, clean_answer), doc in zip(answers_flat, SPACY_MODEL.pipe(question_with_answers))}


def compute_answer_level_metrics(question: str, answers_map: Mapping[str, Iterable[str]],
                                 std_answer: str) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    # `frozenset` so it's immutable thus hashable.
    answer_processed_map = {worker_id: [(answer, normalized_answer,
                                         frozenset(tokenize_answer_to_compute_metrics(normalized_answer)))
                                        for answer in worker_answers
                                        # There's some rare cases in which an answer is empty.
                                        if (normalized_answer := normalize_answer(answer))]
                            for worker_id, worker_answers in itertools.chain(answers_map.items(),
                                                                             [("std_answer", [std_answer])])}

    # We apply the filtering we just did to the original answers map.
    answers_map = {worker_id: [answer for answer, _, _ in worker_answers]
                   for worker_id, worker_answers in answer_processed_map.items()}

    results = defaultdict(lambda: defaultdict(dict))

    np_map = compute_np_value_by_answer(question, answers_map)

    for worker_id, worker_answers in answers_map.items():
        for answer in worker_answers:
            if (is_np := np_map[answer]) or worker_id == "std_answer":
                results[worker_id][answer]["np"] = is_np

    answer_processed_map = {worker_id: [(answer, normalized_answer, tokens)
                                        for answer, normalized_answer, tokens in worker_answers
                                        if np_map[answer] or worker_id == "std_answer"]
                            for worker_id, worker_answers in answer_processed_map.items()}

    for worker_id, worker_answers in answer_processed_map.items():
        other_workers_answers = [other_worker_answers
                                 for other_worker_id, other_worker_answers in answer_processed_map.items()
                                 if other_worker_id != worker_id]
        other_workers_answer_tokens = {tokens
                                       for other_worker_answers in other_workers_answers
                                       for _, _, tokens in other_worker_answers}
        other_normalized_answers = {normalized_answer
                                    for other_worker_answers in other_workers_answers
                                    for _, normalized_answer, _ in other_worker_answers}

        for answer, normalized_answer, tokens in worker_answers:
            if other_normalized_answers:
                results[worker_id][answer]["f1"] = compute_token_level_f1_many(tokens, other_workers_answer_tokens)
                results[worker_id][answer]["em"] = any(normalized_answer == other_normalized_answer
                                                       for other_normalized_answer in other_normalized_answers)
            else:
                import numpy as np
                results[worker_id][answer]["f1"] = results[worker_id][answer]["em"] = np.nan

    return results
