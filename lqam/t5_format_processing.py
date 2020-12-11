"""
Transform the output of a T5 pretrained model.

T5 was trained with input-output pairs such as:

input: "The <extra_id_0> walks in <extra_id_1> park."
expected output: "<extra_id_0> cute dog <extra_id_1> the <extra_id_2>"

The output has the syntax: `([EXTRA TOKEN ID] [TOKEN] ...) ...`. It's like a mapping, where the extra token ID is the
key and the list of tokens next to it its value.
It's possible that it contains extra token IDs that aren't present in the input text, especially at the end.

See https://huggingface.co/transformers/model_doc/t5.html for more info.
"""
import itertools
from typing import Iterator, Mapping, Optional

import torch
from transformers import PreTrainedTokenizerBase

from lqam import iterable_utils

# Note T5 has, as far as I know, 2 tokenizer implementations, and specific-class.
# For simplification and because the way of knowing if a token is extra or not is quite custom,
# we just use `PreTrainedTokenizerBase` as the tokenizer type.

# Another consideration is to avoid doing id-token-string conversions as much as we can.
# So the input and output types for these functions are the ones that are most convenient to minimize
# these conversations from inside the functions but also from outside of them (at least for the use-case that they were
# created).

TYPE_BLANK_MAP = Mapping[int, torch.Tensor]


def is_extra_token(token_id: int, tokenizer: PreTrainedTokenizerBase) -> bool:
    # It should work for any tokenizer with extra IDs.
    return token_id >= tokenizer.vocab_size - tokenizer._extra_ids


def compute_blank_map_instance(generated_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase,
                               masked_caption_ids: Optional[torch.Tensor] = None) -> TYPE_BLANK_MAP:
    # Use a primitive type for the key so `in` and `__getitem__` work.
    extra_id_to_position = {id_.item(): i for i, id_ in enumerate(generated_ids) if is_extra_token(id_, tokenizer)}
    extra_id_to_position[tokenizer.eos_token_id] = len(generated_ids)

    return {extra_id: generated_ids[extra_id_to_position[extra_id] + 1:extra_id_to_position[next_extra_id]]
            for extra_id, next_extra_id in iterable_utils.pairwise(extra_id_to_position)
            if masked_caption_ids is None or (extra_id == masked_caption_ids).any()}  # noqa


def compute_blank_map(generated_ids: Iterator[torch.Tensor], tokenizer: PreTrainedTokenizerBase,
                      masked_caption_ids: Optional[torch.Tensor] = None) -> Iterator[TYPE_BLANK_MAP]:
    """
    Converts the output of a T5-like pretrained model into a mapping.

    `generated_ids` is a 2D tensor.

    `masked_caption_ids` is an optional 2D tensor useful to remove any spurious extra token.
    """
    masked_caption_ids = itertools.repeat(None) if masked_caption_ids is None else masked_caption_ids
    for generated_ids_instance, masked_caption_ids_instance in zip(generated_ids, masked_caption_ids):
        yield compute_blank_map_instance(generated_ids_instance, tokenizer, masked_caption_ids_instance)
