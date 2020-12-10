import re
import string

import pytorch_lightning as pl
import torch

RE_DET = re.compile(rf"\b(?:an?|the)\b|[{re.escape(string.punctuation)}]")


def _normalize_label(label: str) -> str:
    return RE_DET.sub("", label.lower()).strip()


def exact_match(label1: str, label2: str) -> bool:
    return _normalize_label(label1) == _normalize_label(label2)


class AlmostExactMatchAccuracy(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # noqa
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += (preds == target).sum()  # noqa
        self.total += target.numel()

    def compute(self) -> float:
        return self.correct.float() / self.total
