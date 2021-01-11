from typing import Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides

from lqam.core.metrics import exact_match


class AlmostExactMatchAccuracy(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @overrides
    def update(self, preds: Sequence[str], targets: Sequence[str]) -> None:  # noqa
        assert len(preds) == len(targets)
        self.correct += sum(exact_match(pred, target) for pred, target in zip(preds, targets))
        self.total += len(targets)

    @overrides
    def compute(self) -> float:
        return self.correct / self.total
