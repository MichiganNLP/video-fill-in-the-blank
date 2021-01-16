from typing import Iterable

import spacy
import torch
from transformers import PretrainedConfig

from lqam.core.noun_phrases import is_noun_phrase_or_n_bar
from lqam.util.iterable_utils import chunks


def compute_answer_prob(logits: torch.Tensor, answer_ids: torch.Tensor, model_config: PretrainedConfig,
                        ignore_eos_token: bool = False) -> torch.Tensor:
    """Computes the joint probability of the given answer using the logits.

    `logits` has shape (N, L, V) and dtype float.
    `answer_ids` has shape (N, L) and dtype int.

    Returned tensor has shape (N,) and dtype float.
    """
    if model_config.decoder_start_token_id is not None \
            and (answer_ids[:, 0] == model_config.decoder_start_token_id).all():  # noqa
        answer_ids = answer_ids[:, 1:]

    N, L = answer_ids.shape

    probs = logits.softmax(dim=-1)
    probs_answer_ids = probs[torch.arange(N)[:, None], torch.arange(L)[None], answer_ids]

    if model_config.pad_token_id is not None:
        probs_answer_ids[answer_ids == model_config.pad_token_id] = 1

    if ignore_eos_token and model_config.eos_token_id is not None:
        probs_answer_ids[answer_ids == model_config.eos_token_id] = 1

    # There should be just a few factors, so the product should be numerically stable.
    return probs_answer_ids.prod(dim=-1)


def arg_noun_phrase(spacy_model: spacy.language.Language, generated_answers: Iterable[str],
                    num_return_sequences: int) -> Iterable[int]:
    """Computes the positions of the first noun phrase for every chunk of `num_return_sequences` in
    `generated_answers`. If a chunk has no noun phrase then it returns 0 for it (the first position).
    """
    return [
        next((i for i, generated_doc in enumerate(generated_docs_instance) if is_noun_phrase_or_n_bar(generated_doc)), 0)
        for generated_docs_instance in chunks(spacy_model.pipe(generated_answers), num_return_sequences)
    ]
