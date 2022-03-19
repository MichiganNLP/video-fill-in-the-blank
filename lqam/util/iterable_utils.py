from __future__ import annotations

import itertools
from collections import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


# From https://stackoverflow.com/a/5434936/1165181
def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# Some ideas copied from `spacy.util.minibatch`.
def chunks(iterable: Iterable[T], n: int | Iterable[int]) -> Iterator[tuple[T, ...]]:
    iterator = iter(iterable)

    sizes = n if hasattr(n, "__iter__") else itertools.repeat(n)
    sizes_iterator = iter(sizes)

    # We could return an iterator instead of a tuple, but maybe it's consumed later and the next chunk is yield.
    # In that scenario, there would be a bug because the main iterator wasn't consumed, so the chunks would be wrong.
    # So we force the consumption of the iterator here.
    #
    # The default value for `next` here is incredibly necessary, as it may otherwise raise a StopIteration.
    # StopIteration behavior inside generators and coroutines has changed since Python 3.7, and they're converted
    # into runtime errors, so they aren't interpreted as the iteration end anymore.
    # See https://stackoverflow.com/a/51701040/1165181
    while chunk := tuple(itertools.islice(iterator, next(sizes_iterator, 0))):
        yield chunk
