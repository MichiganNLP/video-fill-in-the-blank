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
import re
from typing import Iterator, List, Mapping, Optional

from transformers import PreTrainedTokenizerBase

from lqam import iterable_utils

# Note T5 has, as far as I know, 2 tokenizer implementations, and specific-class.
# For simplification and because the way of knowing if a token is extra or not is quite custom,
# we just use `PreTrainedTokenizerBase` as the tokenizer type.

# Another consideration is to avoid doing id-token-string conversions as much as we can.
# So the input and output types for these functions are the ones that are most convenient to minimize
# these conversations from inside the functions but also from outside of them (at least for the use-case that they were
# created).

# Note some `Tokenizer`'s methods explicitly use lists, as opposed to other iterable types.
TYPE_BLANK_MAP = Mapping[str, List[str]]

RE_EXTRA_ID = re.compile(r"<extra_id_\d+>")


def is_extra_token(token: str) -> bool:
    return RE_EXTRA_ID.match(token) is not None


def compute_blank_map(generated_ids: Iterator[int], tokenizer: PreTrainedTokenizerBase,
                      input_tokens: Optional[Iterator[str]] = None) -> TYPE_BLANK_MAP:
    """
    Converts the output of a T5-like pretrained model into a mapping.

    `generated_ids` is a 1D tensor.

    The `input_text` is optional and it's useful to remove any spurious extra token.
    """
    tokens = tokenizer.convert_ids_to_tokens(list(generated_ids))

    extra_token_ids = {token: i for i, token in enumerate(tokens) if is_extra_token(token)}
    extra_token_ids[tokenizer.eos_token] = len(tokens)

    input_tokens = set(input_tokens) if input_tokens else None

    return {extra_token: tokens[extra_token_ids[extra_token] + 1:extra_token_ids[next_extra_token]]
            for extra_token, next_extra_token in iterable_utils.pairwise(extra_token_ids)
            if input_tokens is None or extra_token in input_tokens}


def fill_in_the_blanks(input_tokens: Iterator[str], blank_map: TYPE_BLANK_MAP) -> Iterator[str]:
    for token in input_tokens:
        yield from blank_map.get(token, [token])
