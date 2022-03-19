from collections import Iterator, Sequence
from typing import Literal

import spacy
import torch
from transformers import PretrainedConfig

from lqam.core.noun_phrases import is_noun_phrase_or_n_bar
from lqam.util.iterable_utils import chunks


def compute_answer_probs(logits: torch.Tensor, answer_ids: torch.Tensor, model_config: PretrainedConfig,
                         ignore_eos_token: bool = False) -> torch.Tensor:
    """Computes the probability of the given answer token using the logits.

    `logits` has shape (N, L, V) and dtype float.
    `answer_ids` has shape (N, L) and dtype int.

    Returned tensor has shape (N, L) and dtype float.
    """
    if model_config.decoder_start_token_id is not None \
            and (answer_ids[:, 0] == model_config.decoder_start_token_id).all():  # noqa
        answer_ids = answer_ids[:, 1:]

    N, L = answer_ids.shape

    probs = logits.softmax(dim=-1)
    answer_probs = probs[torch.arange(N)[:, None], torch.arange(L)[None], answer_ids]

    if model_config.pad_token_id is not None:
        answer_probs[answer_ids == model_config.pad_token_id] = 1

    if ignore_eos_token and model_config.eos_token_id is not None:
        answer_probs[answer_ids == model_config.eos_token_id] = 1

    return answer_probs


def compute_answer_prob(answer_probs: torch.Tensor) -> torch.Tensor:
    """Computes the joint probability of the given answer.

    `answer_probs` has shape (N, L) and dtype float.

    Returned tensor has shape (N,) and dtype float.
    """
    # There should be just a few factors, so the product should be numerically stable.
    return answer_probs.prod(dim=-1)


def arg_noun_phrase(spacy_model: spacy.language.Language, questions: Sequence[str],
                    answers: Sequence[Sequence[str]],
                    span_alignment_mode: Literal["strict", "contract", "expand"] = "strict") -> Iterator[int]:
    """Yields the position of the first noun phrase for every chunk of answers in `answers`. If a chunk has no noun
    phrase then it returns 0 for it (the first position).
    """
    docs_flatten = spacy_model.pipe(question.replace("_____", answer)
                                    for question, answers_instance in zip(questions, answers)
                                    for answer in answers_instance)

    answers_per_instance = (sum(1 for _ in answers_instance) for answers_instance in answers)
    docs = chunks(docs_flatten, answers_per_instance)

    for docs_instance, question, answers_instance in zip(docs, questions, answers):
        for i, (doc, answer) in enumerate(zip(docs_instance, answers_instance)):
            start = question.index("_____")
            end = start + len(answer)
            if is_noun_phrase_or_n_bar(doc.char_span(start, end, alignment_mode=span_alignment_mode)):
                yield i
                break
        else:
            yield 0
