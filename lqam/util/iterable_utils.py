import itertools
from typing import Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")


# From https://stackoverflow.com/a/5434936/1165181
def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def chunks(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    iterator = iter(iterable)
    # We could return an iterator instead of a tuple, but maybe it's consumed later and the next chunk is yield.
    # In that scenario, there would be a bug because the main iterator wasn't consumed, so the chunks would be wrong.
    # So we force the consumption of the iterator here.
    while chunk := tuple(itertools.islice(iterator, n)):
        yield chunk
