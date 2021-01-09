import re
import string
from typing import Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides

RE_DET = re.compile(rf"\b(?:an?|the)\b|[{re.escape(string.punctuation)}]")


def _normalize_label(label: str) -> str:
    return RE_DET.sub("", label.lower()).strip()


def exact_match(label1: str, label2: str) -> bool:
    return _normalize_label(label1) == _normalize_label(label2)


class AlmostExactMatchAccuracy(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    @overrides
    def update(self, preds: Sequence[str], targets: Sequence[str]) -> None:  # noqa
        assert len(preds) == len(targets)
        self.correct += sum(exact_match(pred, target) for pred, target in zip(preds, targets))
        self.total += len(targets)

    @overrides
    def compute(self) -> float:
        return self.correct / self.total
