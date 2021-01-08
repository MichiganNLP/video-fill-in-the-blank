import itertools
from typing import Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")


def grouper(n: int, iterable: Iterable[T]) -> Iterator[Tuple[T, ...]]:
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
