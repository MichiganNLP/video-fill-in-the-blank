# Update: there are many issues going on with this class, so we just don't use it (it's not worth it to fix it for now).
# See the fix-mes below.
# We don't use the one that's in torchtext because it substitutes `DataLoader` with "iterators" but they don't
# provide some important things such as `num_workers`.
# And there's also the
# [AllenNLP one](https://github.com/allenai/allennlp/blob/f2a5331/allennlp/data/samplers/bucket_batch_sampler.py),
# but it's tied to their classes.
#
# File initially copied from https://github.com/pytorch/text/blob/33797c7/torchtext/experimental/utils/samplers.py
# (pytorch/text#859).
from __future__ import annotations

import math
from collections import Callable, Iterator, Sequence
from heapq import heappop, heappush
from typing import TypeVar

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

T = TypeVar("T")


class BucketBatchSampler(Sampler):
    """Defines a batch sampler that batches examples of similar lengths together and
    minimizes amount of padding needed. This BatchSampler works by initially taking a large
    steps (multiplied by 100) and then sort the data according to `seq_len_fn`.
    Arguments:
        data_source: data source to sample from.
        seq_len_fn: function to return the current length of the sequence.
        batch_size: size of mini-batch.
            Default: 32
        shuffle: data_source will be wrapped with RandomSampler if set to ``True``,
            otherwise, SequentialSampler. Default: True
            FIXME: shuffles only the 100-size blocks of batches, but not inside them.
                It'd be good to see the AllenNLP one, and the torchtext one to keep just one.
    Example:
        >>> dummy = [torch.tensor(range(1, torch.randint(2, 11, ()).item())) for _ in range(10)]
        >>> def tensor_seq_len_fn(row):
        ...     return row.size(0)
        >>> list(BucketBatchSampler(dummy, tensor_seq_len_fn, batch_size=5, shuffle=False))
        [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9]]
        >>> list(BucketBatchSampler(dummy, tensor_seq_len_fn, batch_size=5))
        [[9, 2, 4, 3, 1], [8, 7, 5, 6], [0]]
    """

    def __init__(self, data_source: Sequence[T], seq_len_fn: Callable[[T], int], batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False) -> None:
        super().__init__(data_source)
        if isinstance(data_source, IterableDataset):
            raise TypeError("Currently does not support IterableDataset!")

        self.data_source = data_source
        self.seq_len_fn = seq_len_fn
        self.batch_size = batch_size
        self.sampler = RandomSampler(data_source) if shuffle else SequentialSampler(data_source)
        self.drop_last = drop_last

        self.sample_count = 100

    def __iter__(self) -> Iterator[int]:
        mini_batch = []
        for i in self.sampler:
            if len(mini_batch) % (self.batch_size * self.sample_count) == 0:
                yield from self._batch(mini_batch)
                mini_batch = []
            heappush(mini_batch, (self.seq_len_fn(self.data_source[i]), i))

        if mini_batch:
            for batch in self._batch(mini_batch):
                if not self.drop_last or len(batch) == self.batch_size:
                    yield batch  # FIXME: make sure this `drop_last` behavior is fine.

    def __len__(self) -> int:
        len_float = (len(self.data_source) + self.batch_size - 1) / self.batch_size  # FIXME: is this fine?
        return math.floor(len_float) if self.drop_last else math.ceil(len_float)

    def _batch(self, mini_batch: list[tuple[int, int]]) -> Iterator[list[int]]:
        total_iter = (len(mini_batch) + self.batch_size - 1) // self.batch_size  # FIXME: consider drop_last.
        for _ in range(total_iter):
            max_steps = min(self.batch_size, len(mini_batch))
            yield [heappop(mini_batch)[1] for _ in range(max_steps)]  # Return ordered data
