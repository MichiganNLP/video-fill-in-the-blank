from typing import Sequence, Optional

import pytorch_lightning as pl
import torch
from overrides import overrides

from lqam.core.metrics import exact_match, compute_token_level_f1_many, exact_match_many, tokenize_answer_to_compute_metrics, flatten_answers

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

    def _split_answers(self, answers: Sequence[str]) -> Sequence[Sequence[str]]:
        return [tokenize_answer_to_compute_metrics(answer) for answer in answers]

    @overrides
    def update(self, preds: Sequence[str], labels: Sequence[str], additional_answers_list: Optional[Sequence[Sequence[Sequence[str]]]]= None) -> None:  # noqa
        assert len(preds) == len(labels)
        answers_list = flatten_answers(labels, additional_answers_list)
        self.score_sum += sum(compute_token_level_f1_many(tokenize_answer_to_compute_metrics(pred), self._split_answers(set(answers))) 
                        for pred, answers in zip(preds, answers_list))
        self.total += len(labels)

    @overrides
    def compute(self) -> float:
        return self.score_sum / self.total

class AlmostExactMatchAccuracyAdditionAnswers(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @overrides
    def update(self, preds: Sequence[str], labels: Sequence[str], additional_answers_list: Optional[Sequence[Sequence[Sequence[str]]]] = None) -> None:  # noqa
        assert len(preds) == len(labels)
        answers_list = flatten_answers(labels, additional_answers_list)
        self.correct += sum(exact_match_many(pred, answers) for pred, answers in zip(preds, answers_list))
        self.total += len(labels)

    @overrides
    def compute(self) -> float:
        return self.correct / self.total