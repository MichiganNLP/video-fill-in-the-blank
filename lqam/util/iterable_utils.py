import itertools
from typing import Iterable, Iterator, Tuple, TypeVar, Union

T = TypeVar("T")


# From https://stackoverflow.com/a/5434936/1165181
def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# Some ideas copied from `spacy.util.minibatch`.
def chunks(iterable: Iterable[T], n: Union[int, Iterable[int]]) -> Iterator[Tuple[T, ...]]:
    iterator = iter(iterable)

    sizes = n if hasattr(n, "__iter__") else itertools.repeat(n)
    sizes_iterator = iter(sizes)

    # We could return an iterator instead of a tuple, but maybe it's consumed later and the next chunk is yield.
    # In that scenario, there would be a bug because the main iterator wasn't consumed, so the chunks would be wrong.
    # So we force the consumption of the iterator here.
    while chunk := tuple(itertools.islice(iterator, next(sizes_iterator))):
        yield chunk
