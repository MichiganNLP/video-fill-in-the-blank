from typing import Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides

from lqam.core.metrics import exact_match
from lqam.core.metrics import compute_token_level_f1_many
from lqam.core.metrics import exact_match_many


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

class F1Scores(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _split_answers(self, additional_answers: Sequence[str]) -> Sequence[Sequence[str]]:
        return [ans.split() for ans in additional_answers]

    @overrides
    def update(self, preds: Sequence[str], labels: Sequence[str], additional_answers_list: Sequence[Sequence[str]]) -> None:  # noqa
        assert len(preds) == len(additional_answers_list)
        # for debug
        # FIXME: If we don't need to use additional answers alone in other cases, we can put gt into them at the data reading step 
        batch_sum = sum(compute_token_level_f1_many(pred.split(), self._split_answers(set(additional_answers+[label]))) 
                        for pred, label, additional_answers in zip(preds, labels, additional_answers_list))
        self.score_sum += batch_sum
        # self.score_sum += sum(compute_token_level_f1_many(pred, self._split_answers(additional_answers)) 
        #                         for pred, additional_answers in zip(preds, additional_answers_list))
        self.total += len(additional_answers_list)

    @overrides
    def compute(self) -> float:
        return self.score_sum / self.total

class AlmostExactMatchAccuracyAdditionAnswers(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @overrides
    def update(self, preds: Sequence[str], targets: Sequence[Sequence[str]]) -> None:  # noqa
        assert len(preds) == len(targets)
        self.correct += sum(exact_match_many(pred, target) for pred, target in zip(preds, targets))
        self.total += len(targets)

    @overrides
    def compute(self) -> float:
        return self.correct / self.total