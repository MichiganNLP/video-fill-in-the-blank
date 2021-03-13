from typing import Iterable, Iterator, Optional, Sequence

import pytorch_lightning as pl
import torch
from overrides import overrides

from lqam.core.metrics import compute_token_level_f1_many, exact_match_many, flatten_all_answers, \
    tokenize_answer_to_compute_metrics


class F1ScoreMany(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _tokenize_answers(answers: Iterable[str]) -> Iterator[Iterator[str]]:
        yield from (tokenize_answer_to_compute_metrics(answer) for answer in answers)

    @overrides
    def update(self, preds: Sequence[str], labels: Sequence[str],
               additional_answers_batch: Optional[Sequence[Sequence[Sequence[str]]]] = None) -> None:
        assert len(preds) == len(labels)
        answers_batch = flatten_all_answers(labels, additional_answers_batch)
        self.score_sum += sum(compute_token_level_f1_many(tokenize_answer_to_compute_metrics(pred),
                                                          self._tokenize_answers(answers))
                              for pred, answers in zip(preds, answers_batch))
        self.total += len(labels)

    @overrides
    def compute(self) -> torch.Tensor:
        return self.score_sum / self.total


class ExactMatchAccuracyMany(pl.metrics.Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @overrides
    def update(self, preds: Sequence[str], labels: Sequence[str],
               additional_answers_batch: Optional[Sequence[Sequence[Sequence[str]]]] = None) -> None:  # noqa
        assert len(preds) == len(labels)
        answers_batch = flatten_all_answers(labels, additional_answers_batch)
        self.correct += sum(exact_match_many(pred, answers) for pred, answers in zip(preds, answers_batch))
        self.total += len(labels)

    @overrides
    def compute(self) -> torch.Tensor:
        return self.correct / self.total
